import { readFileSync } from "node:fs";
import path from "node:path";
import { expect, test, type Page } from "@playwright/test";

interface ExportedSpread {
  seeds: number[];
  reportedSeed: number;
  prAuc: number[];
  mean: number;
  min: number;
  max: number;
}

interface ExportedRun {
  method: string;
  featureSet: string;
  seed: number;
  prAuc: number;
  date: string;
  hyperparameters: { seedSpread?: ExportedSpread };
}

interface AxeViolation {
  id: string;
  nodes: Array<{ target: unknown[] }>;
}

interface AxeWindow {
  axe: {
    run: (context: Document, options: { runOnly: { type: "tag"; values: string[] } }) => Promise<{ violations: AxeViolation[] }>;
  };
}

interface ExportedNode {
  id: string;
  label: "illicit" | "licit" | "unknown";
  seedRank: number | null;
  seedRankGnn: number | null;
  hop: number | null;
  hopGnn: number | null;
  scoreGnn: number | null;
}

const DATA_DIR = path.resolve(__dirname, "../../../data/k3");
const METRICS = JSON.parse(readFileSync(path.join(DATA_DIR, "metrics.json"), "utf8")) as {
  runs: ExportedRun[];
  evaluation: { prevalence: number; randomPrAucExpected: number; randomPrAucQ95: number };
};
const MARGIN = METRICS.evaluation.randomPrAucQ95 - METRICS.evaluation.randomPrAucExpected;
const AXE_SOURCE = readFileSync(require.resolve("axe-core/axe.min.js"), "utf8");
const AXE_TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];
const AXE_WIDTHS = [390, 768, 1280];
const SHORT_NAME: Record<string, string> = {
  zscore: "Z-Scores",
  iforest: "Isolation Forest",
  hgb: "Gradient Boosting",
  gcn: "GCN",
  graphsage: "GraphSAGE",
  mlp: "MLP (ohne Kanten)",
};
const NEIGHBOURHOOD_SHARE = "der größere Teil des Vorsprungs der Graph Neural Networks aus der Nachbarschaft";
const EDGELESS_SHARE = "Der größere Teil des Abstands zur besseren Baseline entsteht auf diesem Netz aber schon ohne Kanten";
const EDGELESS_METHODS = ["hgb", "mlp"];
const INK = "rgb(15, 27, 45)";
const NODES = JSON.parse(readFileSync(path.join(DATA_DIR, "nodes.json"), "utf8")) as {
  scoreGnnMethod: "gcn" | "graphsage" | null;
  nodes: ExportedNode[];
};
const FIRST_SEED = NODES.nodes.find((node) => node.seedRank === 1);
const FIRST_GNN_SEED = NODES.nodes.find((node) => node.seedRankGnn === 1);
const GNN_ONLY_SEED = NODES.nodes.find((node) => node.seedRankGnn !== null && node.hop === null);
const LEARNED_METHODS = ["hgb", "gcn", "graphsage", "mlp"];
const NEURAL_METHODS = ["gcn", "graphsage", "mlp"];
const GNN_LABEL = { gcn: "GCN", graphsage: "GraphSAGE" } as const;
const PENDING_TEXT = "noch nicht gemessen";
const LABEL_PHRASE = { illicit: "Label illegal", licit: "Label legal", unknown: "ohne Label" } as const;

function germanDecimal(value: number): string {
  return value.toFixed(4).replace(".", ",").replace(/^-/, "\u2212");
}

function germanPercent(value: number, digits: number): string {
  return `${(value * 100).toFixed(digits).replace(".", ",")}\u00a0%`;
}

function germanDate(iso: string): string {
  const [year, month, day] = iso.slice(0, 10).split("-");
  return `${day}.${month}.${year}`;
}

function germanList(items: string[]): string {
  if (items.length <= 1) return items.join("");
  return `${items.slice(0, -1).join(", ")} und ${items[items.length - 1]}`;
}

function runOf(method: string): ExportedRun | undefined {
  return METRICS.runs.find((run) => run.method === method);
}

function bestOf(runs: readonly ExportedRun[]): ExportedRun | undefined {
  return runs.reduce<ExportedRun | undefined>((best, run) => (!best || run.prAuc > best.prAuc ? run : best), undefined);
}

async function chooseSeedSet(page: Page, set: "iforest" | "gnn"): Promise<void> {
  const button = page.getByTestId(`seed-set-${set}`);
  await button.click();
  await expect(button).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByTestId("top-nodes")).toHaveAttribute("data-seed-set", set);
}

function isAbove(value: number, reference: number): boolean {
  return value - reference > MARGIN;
}

function watchRequests(page: Page, baseURL: string | undefined): string[] {
  const origin = new URL(baseURL ?? "http://localhost:3000").origin;
  const violations: string[] = [];
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (url.protocol === "data:" || url.protocol === "blob:") return;
    if (url.origin !== origin || request.method() !== "GET") {
      violations.push(`${request.method()} ${request.url()}`);
    }
  });
  return violations;
}

function watchErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(`pageerror: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(`console: ${message.text()}`);
  });
  return errors;
}

async function cspViolations(page: Page): Promise<string[]> {
  return page.evaluate(() => (window as unknown as { cspViolations?: string[] }).cspViolations ?? []);
}

async function completePageText(page: Page): Promise<string> {
  await expect(page.getByTestId("graph-view")).toHaveAttribute("data-state", "ready", { timeout: 30_000 });
  if (FIRST_SEED) {
    await page.locator(`[data-testid="top-node-row"][data-node-id="${FIRST_SEED.id}"]`).click();
    await expect(page.getByTestId("node-detail-id")).toHaveText(FIRST_SEED.id);
    if (FIRST_SEED.scoreGnn !== null) await expect(page.getByTestId("node-detail-gnn")).toBeVisible();
  }
  const closed = page.locator("main details:not([open])");
  await closed.evaluateAll((elements) => {
    for (const element of elements) (element as HTMLDetailsElement).open = true;
  });
  await expect(closed).toHaveCount(0);
  const visible = await page.getByRole("main").innerText();
  let gnnState = "";
  if (NODES.scoreGnnMethod !== null) {
    await chooseSeedSet(page, "gnn");
    gnnState = await page.getByTestId("netzradar-explorer").innerText();
    gnnState += `\n${(await page.getByTestId("graph-canvas").getAttribute("aria-label")) ?? ""}`;
    await chooseSeedSet(page, "iforest");
  }
  const attributes = await page.evaluate(() =>
    [...document.querySelectorAll("main [aria-label], main [alt], main [title], head title, head meta[content]")].map(
      (element) =>
        [
          element.getAttribute("aria-label"),
          element.getAttribute("alt"),
          element.getAttribute("title"),
          element.getAttribute("content"),
          element.tagName === "TITLE" ? element.textContent : null,
        ]
          .filter((value): value is string => value !== null)
          .join("\n"),
    ),
  );
  return [visible, gnnState, ...attributes].join("\n");
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    const target = window as unknown as { cspViolations: string[] };
    target.cspViolations = [];
    document.addEventListener("securitypolicyviolation", (event) => {
      target.cspViolations.push(`${event.violatedDirective} ${event.blockedURI}`);
    });
  });
});

test("NetzRadar page renders headline, metrics, graph and node details without leaving the origin", async ({
  page,
  baseURL,
}) => {
  const violations = watchRequests(page, baseURL);
  const errors = watchErrors(page);
  await page.goto("/projects/netzradar");
  await page.waitForLoadState("load");

  await expect(page.getByRole("heading", { level: 1, name: "NetzRadar" })).toBeVisible();
  await expect(page.getByRole("heading", { level: 1, name: "NetzRadar" })).toHaveCSS("font-family", /Cormorant/);
  await expect(page.getByText("Projekt 03 · Graph-ML", { exact: true })).toBeVisible();

  const metricsTable = page.getByTestId("metrics-table");
  await expect(metricsTable).toBeVisible();
  for (const run of METRICS.runs) {
    await expect(page.getByTestId(`pr-auc-${run.method}`)).toHaveText(germanDecimal(run.prAuc));
  }
  await expect(page.getByTestId("headline-figures")).toContainText(SHORT_NAME.hgb ?? "");
  await expect(page.getByTestId("metrics-row-random")).toContainText(
    germanDecimal(METRICS.evaluation.randomPrAucExpected),
  );
  await expect(page.getByTestId("metrics-row-prevalence")).toContainText(germanDecimal(METRICS.evaluation.prevalence));
  await expect(page.getByTestId("assessment")).toContainText(
    `(Prävalenz im Test: ${germanPercent(METRICS.evaluation.prevalence, 2)})`,
  );
  for (const method of LEARNED_METHODS) {
    const row = page.getByTestId(`metrics-row-${method}`);
    if (METRICS.runs.some((run) => run.method === method)) {
      await expect(row).not.toContainText(PENDING_TEXT);
    } else {
      await expect(row).toContainText(PENDING_TEXT);
    }
  }
  if (LEARNED_METHODS.every((method) => METRICS.runs.some((run) => run.method === method))) {
    await expect(page.getByText("noch nicht gemessen")).toHaveCount(0);
    await expect(page.getByText("noch nicht berechnet")).toHaveCount(0);
    await expect(page.getByTestId("homophily-caveat")).toBeVisible();
    await expect(page.getByTestId("search-table")).toBeVisible();
  }
  const chart = page.getByTestId("pr-curve").locator("svg.recharts-surface").first();
  await expect(chart).toBeVisible();
  await expect(chart).not.toHaveAttribute("role", "application");
  await expect(chart).not.toHaveAttribute("tabindex", "0");
  const legendItems = page.getByTestId("pr-curve-legend").locator("li");
  await expect(legendItems).toHaveCount(METRICS.runs.length + 1);
  for (const item of await legendItems.all()) {
    await expect(item).toHaveCSS("color", INK);
  }
  await expect(page.getByTestId("pr-curve-legend").locator("svg[aria-hidden='true']")).toHaveCount(
    METRICS.runs.length + 1,
  );

  const graph = page.getByTestId("graph-view");
  await expect(graph).toHaveAttribute("data-state", "ready", { timeout: 30_000 });
  await expect(page.getByTestId("graph-fallback")).toHaveCount(0);
  await expect(page.getByTestId("graph-canvas").locator("canvas").first()).toBeVisible();

  expect(FIRST_SEED).toBeDefined();
  const firstRow = page.getByTestId("top-node-row").first();
  await expect(firstRow).toHaveAttribute("data-node-id", FIRST_SEED?.id ?? "");
  if (NODES.scoreGnnMethod === null) {
    await expect(page.getByTestId("top-nodes-gnn-header")).toHaveCount(0);
  } else {
    const label = GNN_LABEL[NODES.scoreGnnMethod];
    await expect(page.getByTestId("top-nodes-gnn-header")).toHaveText(`Score GNN (${label})`);
    await expect(firstRow.getByTestId("top-node-gnn")).toHaveText(germanDecimal(FIRST_SEED?.scoreGnn ?? Number.NaN));
  }
  await firstRow.click();
  await expect(page.getByTestId("node-detail-id")).toHaveText(FIRST_SEED?.id ?? "");
  await expect(page.getByTestId("node-detail-label")).toHaveText(LABEL_PHRASE[FIRST_SEED?.label ?? "unknown"]);
  await expect(page.getByTestId("node-detail")).toContainText("1 von");
  await expect(firstRow).toHaveAttribute("aria-current", "true");
  if (NODES.scoreGnnMethod === null) {
    await expect(page.getByTestId("node-detail")).toContainText("noch nicht berechnet");
  } else {
    await expect(page.getByTestId("node-detail")).toContainText(`Score GNN (${GNN_LABEL[NODES.scoreGnnMethod]})`);
    await expect(page.getByTestId("node-detail-gnn")).toHaveText(germanDecimal(FIRST_SEED?.scoreGnn ?? Number.NaN));
  }

  await page.getByTestId("top-node-row").nth(1).getByRole("button").press("Enter");
  await expect(page.getByTestId("node-detail-id")).toHaveText(
    (await page.getByTestId("top-node-row").nth(1).getAttribute("data-node-id")) ?? "",
  );

  const selectedId = (await page.getByTestId("node-detail-id").textContent()) ?? "";
  const neighbor = page.getByTestId("node-detail").getByRole("listitem").first().getByRole("button");
  const neighborId = ((await neighbor.textContent()) ?? "").split(",")[0]?.trim() ?? "";
  await neighbor.press("Enter");
  await expect(page.getByTestId("node-detail-id")).toHaveText(neighborId);
  expect(neighborId).not.toBe(selectedId);
  await expect(page.locator("#node-detail-title")).toBeFocused();

  await page.getByRole("button", { name: "Auswahl aufheben" }).press("Enter");
  await expect(page.getByTestId("node-detail-id")).toHaveCount(0);
  await expect(page.locator("#node-detail-title")).toBeFocused();

  expect(violations).toEqual([]);
  expect(errors).toEqual([]);
  expect(await cspViolations(page)).toEqual([]);
});

test("assessment sentences follow the differences in metrics.json", async ({ page }) => {
  await page.goto("/projects/netzradar");
  const zscore = runOf("zscore");
  const iforest = runOf("iforest");
  const gcn = runOf("gcn");
  const graphsage = runOf("graphsage");
  const graphRuns = [gcn, graphsage].filter((run): run is ExportedRun => run !== undefined);
  const baselines = [zscore, iforest].filter((run): run is ExportedRun => run !== undefined);
  const baseline = baselines.reduce<ExportedRun | undefined>((best, run) => (!best || run.prAuc > best.prAuc ? run : best), undefined);

  const below = METRICS.runs
    .filter((run) => run.prAuc <= METRICS.evaluation.randomPrAucQ95)
    .map((run) => SHORT_NAME[run.method] ?? run.method);
  const random = page.getByTestId("random-comparison");
  if (below.length === 0) {
    await expect(random).toContainText("Alle Verfahren liegen über dem 95-%-Quantil zufälliger Rangfolgen");
  } else if (below.length < METRICS.runs.length) {
    await expect(random).toContainText(`Alle Verfahren außer ${germanList(below)} liegen über dem 95-%-Quantil`);
  } else {
    await expect(random).toContainText("Kein Verfahren liegt über dem 95-%-Quantil");
  }
  await expect(random).toContainText(`(${germanDecimal(METRICS.evaluation.randomPrAucQ95)})`);

  if (zscore && iforest) {
    const difference = iforest.prAuc - zscore.prAuc;
    const pair = page.getByTestId("baseline-pair");
    await expect(pair).toContainText(germanDecimal(Math.abs(difference)));
    if (Math.abs(difference) > MARGIN) {
      await expect(pair).toContainText(difference > 0 ? "über den robusten Z-Scores" : "unter den robusten Z-Scores");
    } else {
      await expect(pair).toContainText("nicht belastbar");
    }
  }

  if (baseline && graphRuns.length > 0) {
    const gap = page.getByTestId("gnn-vs-baseline");
    await expect(gap).toContainText(`PR-AUC ${germanDecimal(baseline.prAuc)}`);
    if (graphRuns.some((run) => run.featureSet !== baseline.featureSet)) {
      await expect(gap).toContainText("Dieser Abstand mischt mehrere Effekte");
      await expect(gap).toContainText(baseline.featureSet === "local" ? "zusätzlich die Graphmaße" : "keine Graphmaße");
    } else {
      await expect(gap).toContainText("Dieser Abstand mischt zwei Effekte");
    }
  }

  if (gcn && graphsage) {
    const difference = graphsage.prAuc - gcn.prAuc;
    const amount = germanDecimal(Math.abs(difference));
    const expected = isAbove(graphsage.prAuc, gcn.prAuc)
      ? `GraphSAGE liegt um ${amount} vor GCN.`
      : isAbove(gcn.prAuc, graphsage.prAuc)
        ? `GCN liegt um ${amount} vor GraphSAGE.`
        : `GCN und GraphSAGE liegen nur ${amount} auseinander`;
    await expect(page.getByTestId("gcn-vs-graphsage")).toContainText(expected);
  }

  const edgeless = EDGELESS_METHODS.map(runOf).filter((run): run is ExportedRun => run !== undefined);
  const reference = bestOf(edgeless);
  if (reference && baseline && graphRuns.length > 0) {
    const decomposition = page.getByTestId("decomposition");
    const supervision = reference.prAuc - baseline.prAuc;
    for (const run of edgeless) await expect(decomposition).toContainText(germanDecimal(run.prAuc));
    if (Math.abs(supervision) > MARGIN) {
      await expect(decomposition).toContainText(germanDecimal(Math.abs(supervision)));
    }
    const referenceName = reference.method === "hgb" ? "Gradient Boosting" : "MLP";
    for (const run of graphRuns) {
      await expect(decomposition).toContainText(`${SHORT_NAME[run.method]} ${germanDecimal(run.prAuc)} (`);
      const amount = germanDecimal(Math.abs(run.prAuc - reference.prAuc));
      await expect(decomposition).toContainText(
        isAbove(run.prAuc, reference.prAuc)
          ? `um ${amount} über dem ${referenceName}`
          : isAbove(reference.prAuc, run.prAuc)
            ? `um ${amount} unter dem ${referenceName}`
            : `nur ${amount} vom ${referenceName} entfernt`,
      );
    }
    const gain = Math.max(supervision, 0);
    const ahead = graphRuns.every((run) => isAbove(run.prAuc, reference.prAuc));
    const share = ahead && graphRuns.every((run) => run.prAuc - reference.prAuc - gain > MARGIN);
    const edgelessShare = ahead && graphRuns.every((run) => gain - (run.prAuc - reference.prAuc) > MARGIN);
    if (share) {
      await expect(decomposition).toContainText(NEIGHBOURHOOD_SHARE);
    } else {
      await expect(decomposition).not.toContainText(NEIGHBOURHOOD_SHARE);
    }
    if (edgelessShare) {
      await expect(decomposition).toContainText(EDGELESS_SHARE);
      for (const run of graphRuns) {
        await expect(decomposition).toContainText(`${germanDecimal(run.prAuc - baseline.prAuc)} bei ${SHORT_NAME[run.method]}`);
      }
    } else {
      await expect(decomposition).not.toContainText(EDGELESS_SHARE);
    }
    if (reference.method === "hgb") {
      await expect(decomposition).toContainText("Gegenüber dem Gradient Boosting wechselt neben den Kanten auch das Verfahren");
    }
    const lead = graphRuns.every(
      (run) => isAbove(run.prAuc, baseline.prAuc) && edgeless.every((other) => isAbove(run.prAuc, other.prAuc)),
    );
    await expect(page.locator("#homophily-caveat-title")).toHaveText(
      lead
        ? "Vorbehalt: Der Vorsprung ist zum Teil eingebaut"
        : "Vorbehalt: Ein Vorsprung der Nachbarschaft wäre zum Teil eingebaut",
    );
  }
});

async function axeViolations(page: Page): Promise<string[]> {
  return page.evaluate(async (tags) => {
    const result = await (window as unknown as AxeWindow).axe.run(document, { runOnly: { type: "tag", values: tags } });
    return result.violations.map((violation) => `${violation.id}: ${violation.nodes.map((node) => node.target.join(" ")).join(" | ")}`);
  }, AXE_TAGS);
}

for (const width of AXE_WIDTHS) {
  test(`NetzRadar page has no axe violations at ${width} px with both start sets`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await page.goto("/projects/netzradar");
    await expect(page.getByTestId("graph-view")).toHaveAttribute("data-state", "ready", { timeout: 30_000 });
    await page.waitForFunction(() =>
      document.getAnimations().every((animation) => !(animation instanceof CSSTransition) || animation.playState !== "running"),
    );
    await page.addScriptTag({ content: AXE_SOURCE });
    expect(await axeViolations(page)).toEqual([]);
    if (NODES.scoreGnnMethod !== null) {
      await chooseSeedSet(page, "gnn");
      await page.waitForFunction(() =>
        document.getAnimations().every((animation) => !(animation instanceof CSSTransition) || animation.playState !== "running"),
      );
      expect(await axeViolations(page)).toEqual([]);
    }
  });
}

test("start page links to the NetzRadar page", async ({ page }) => {
  await page.goto("/");
  const link = page.getByTestId("project-netzradar").getByRole("link", { name: /^Live ansehen/ });
  await expect(link).toHaveAttribute("href", "/projects/netzradar");
  await link.click();
  await expect(page).toHaveURL(/\/projects\/netzradar$/);
  await expect(page.getByRole("heading", { level: 1, name: "NetzRadar" })).toBeVisible();
});

test("status names the run date and the page avoids internal jargon", async ({ page }) => {
  await page.goto("/projects/netzradar");
  const dates = [...new Set(METRICS.runs.map((run) => run.date))].sort();
  const first = germanDate(dates[0] ?? "");
  const last = germanDate(dates[dates.length - 1] ?? "");
  const status = page.getByTestId("status-text");
  await expect(status).toContainText(`Stand der Ergebnisse: ${first === last ? first : `${first} bis ${last}`}.`);
  await expect(status).toContainText(
    "Offen: Case-Study, Läufe mit gestörter Nachbarschaft, Baseline mit gemittelten Nachbarmerkmalen.",
  );
  await expect(status.getByRole("link", { name: "„Einordnung der Ergebnisse“" })).toHaveAttribute("href", "#einordnung");
  const text = await completePageText(page);
  expect(text).not.toMatch(/Schritt \d/);
  expect(text).not.toMatch(/\b(il)?licit\b/i);
  await expect(page.getByTestId("search-table")).toContainText("(Verhältnis aus den Labels)");
  await expect(page.getByRole("main")).toContainText(
    "Deutlich stärker weicht der größte Ausgangsanteil der Fan-out-Verteiler ab, und runde Beträge sind bei Musterknoten doppelt so häufig.",
  );
  await expect(page.getByRole("main")).toContainText(
    "Fan-in-Sammler haben als Summe vieler Zubringer einen höheren Betrag und einen kleineren Variationskoeffizienten der Eingangsbeträge (die Beträge der Zubringer streuen weniger).",
  );
  await expect(page.getByRole("main")).not.toContainText("Einzelne Merkmale weichen stärker ab");
});

test("labels and scores keep separate terms", async ({ page }) => {
  await page.goto("/projects/netzradar");
  const text = await completePageText(page);
  expect(text).toContain("Median und MAD reagieren kaum auf einzelne Ausreißer");
  if (NODES.scoreGnnMethod !== null) {
    expect(text).toContain("Logit für Label illegal minus Logit für Label legal");
    expect(text).toContain("w bei Label illegal");
  }
  expect(text).not.toMatch(/auffällig/i);
  expect(text).not.toMatch(/\bunbekannt\b/i);
  expect(text).not.toMatch(/\bals (il)?legal\b|\balles (il)?legal\b|\bKlassen? (il)?legal\b|\bLogit (il)?legal\b|einstuf/);
  await expect(page.getByTestId("graph-legend").locator("h3")).toHaveText(
    "Legende: Label im Datensatz (keine rechtliche Bewertung)",
  );
  await expect(page.getByTestId("graph-legend").locator("li span.font-medium")).toHaveText([
    "illegal",
    "legal",
    "ohne Label",
  ]);
  await expect(page.getByTestId("label-table").locator("tbody th")).toHaveText(["illegal", "legal", "ohne Label"]);
  await expect(page.getByTestId("terms").locator("dt")).toHaveText(["Label", "Score", "markiert"]);
  const accuracy = page.getByTestId("accuracy-definition");
  await expect(accuracy).toContainText("gelten als markiert");
  await expect(accuracy).toContainText("mit Label illegal");
  await expect(accuracy).toContainText("Ein Modell, das keinen Knoten markiert, erreicht");
  await expect(page.getByTestId("graph-canvas")).toHaveAttribute(
    "aria-label",
    /Farbe und Form nach Label \(illegal, legal, ohne Label\)/,
  );
});

test("section links jump to the assessment and the limits", async ({ page }) => {
  await page.goto("/projects/netzradar");
  const nav = page.getByRole("navigation", { name: "Abschnitte" });
  await expect(nav.getByRole("link")).toHaveText([
    "Überblick",
    "Daten",
    "Netzwerk",
    "Ergebnisse",
    "Methodik",
    "Einordnung",
    "Grenzen",
  ]);
  const targets = await nav
    .getByRole("link")
    .evaluateAll((links) => links.map((link) => link.getAttribute("href")?.slice(1) ?? ""));
  const sectionOrder = await page
    .locator("main section[id]")
    .evaluateAll((sections) => sections.map((section) => section.id));
  const positions = targets.map((id) => sectionOrder.indexOf(id));
  expect(positions.every((position) => position >= 0)).toBe(true);
  expect(positions).toEqual([...positions].sort((a, b) => a - b));

  await page.getByRole("link", { name: "Zur Einordnung der Ergebnisse" }).click();
  await expect(page).toHaveURL(/#einordnung$/);
  await expect(page.getByRole("heading", { level: 2, name: "Einordnung der Ergebnisse" })).toBeInViewport();

  await page.evaluate(() => window.scrollTo(0, 0));
  await nav.getByRole("link", { name: "Grenzen" }).focus();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/#grenzen$/);
  await expect(page.getByRole("heading", { level: 2, name: "Grenzen" })).toBeInViewport();
});

test("start node table is one tab stop with arrow keys inside", async ({ page }) => {
  await page.goto("/projects/netzradar");
  const buttons = page.getByTestId("top-nodes-table").getByRole("button");
  await expect(buttons.first()).toHaveAttribute("tabindex", "0");
  await expect(page.getByTestId("top-nodes-table").locator("button[tabindex='0']")).toHaveCount(1);
  await buttons.first().focus();
  await page.keyboard.press("ArrowDown");
  await expect(buttons.nth(1)).toBeFocused();
  await expect(buttons.nth(1)).toHaveAttribute("tabindex", "0");
  await page.keyboard.press("End");
  await expect(buttons.last()).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("node-detail-id")).toHaveText((await buttons.last().textContent()) ?? "");
  await expect(page.getByTestId("top-nodes-table").locator("button[tabindex='0']")).toHaveCount(1);
});

test("sticky detail panel fits a 1280 x 720 screen and the method index opens its section", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.goto("/projects/netzradar");
  await expect(page.getByTestId("graph-view")).toHaveAttribute("data-state", "ready", { timeout: 30_000 });
  await page.getByTestId("top-node-row").first().click();
  const reset = page.getByRole("button", { name: "Auswahl aufheben" });
  await reset.scrollIntoViewIfNeeded();
  await expect(reset).toBeInViewport();
  const panel = await page.getByTestId("node-detail").evaluate((element) => {
    const box = element.parentElement?.getBoundingClientRect();
    return { top: box?.top ?? 0, bottom: box?.bottom ?? 0 };
  });
  expect(panel.bottom - panel.top).toBeLessThanOrEqual(720);

  const index = page.getByTestId("method-index");
  const target = page.locator("#methodik-gcn-graphsage");
  await expect(target).not.toHaveAttribute("open", /.*/);
  await index.getByRole("link", { name: "GCN und GraphSAGE" }).click();
  await expect(page).toHaveURL(/#methodik-gcn-graphsage$/);
  await expect(target).toHaveAttribute("open", "");
  await expect(page.getByTestId("chain-example")).toBeVisible();
});

test("phone layout keeps the label table complete and the graph controls off the graph", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/projects/netzradar");
  const region = page.getByRole("region", { name: "Labels im gesamten Netz" });
  const size = await region.evaluate((element) => ({ scroll: element.scrollWidth, client: element.clientWidth }));
  expect(size.scroll).toBeLessThanOrEqual(size.client);

  await expect(page.getByTestId("graph-hint")).toContainText(
    "auf Touch-Geräten mit zwei Fingern zoomen und verschieben. Außerhalb des Graphen scrollen Sie die Seite.",
  );
  await expect(page.getByTestId("graph-view")).toHaveAttribute("data-state", "ready", { timeout: 30_000 });
  const canvas = await page.getByTestId("graph-canvas").boundingBox();
  const controls = await page.getByTestId("graph-controls").boundingBox();
  expect(canvas).not.toBeNull();
  expect(controls).not.toBeNull();
  expect(controls?.y ?? 0).toBeGreaterThanOrEqual((canvas?.y ?? 0) + (canvas?.height ?? 0));

  const metrics = page.getByRole("region", { name: "Kennzahlen je Verfahren auf den Testknoten mit Label" });
  await metrics.scrollIntoViewIfNeeded();
  const viewport = await metrics.boundingBox();
  const mainValue = await page.getByTestId(`pr-auc-${METRICS.runs[0]?.method ?? "zscore"}`).boundingBox();
  expect(viewport).not.toBeNull();
  expect(mainValue).not.toBeNull();
  expect((mainValue?.x ?? 0) + (mainValue?.width ?? 0)).toBeLessThanOrEqual((viewport?.x ?? 0) + (viewport?.width ?? 0));
});

test.describe("without JavaScript", () => {
  test.use({ javaScriptEnabled: false });

  test("the graph area explains that it needs JavaScript instead of loading forever", async ({ page }) => {
    await page.goto("/projects/netzradar");
    const notice = page.getByTestId("graph-noscript");
    await expect(notice).toBeVisible();
    await expect(notice).toHaveText("Die Graph-Ansicht braucht JavaScript; Kennzahlen und Tabellen stehen unten.");
    await expect(page.getByTestId("graph-loading")).toHaveCount(0);
    await expect(page.getByTestId("graph-view")).not.toContainText("Graph wird geladen");
    await expect(page.getByTestId("metrics-table")).toBeVisible();
  });
});

test("metrics table shows the spread over the seeds and explains a spread of 0", async ({ page }) => {
  await page.goto("/projects/netzradar");
  const spreads = METRICS.runs.flatMap((run) => (run.hyperparameters.seedSpread ? [{ run, spread: run.hyperparameters.seedSpread }] : []));
  expect(spreads.length).toBeGreaterThan(0);
  const seedCount = spreads[0]?.spread.seeds.length ?? 0;
  await expect(page.getByTestId("spread-header")).toHaveText(`Spanne über ${seedCount} Seeds`);
  for (const { run, spread } of spreads) {
    const cell = page.getByTestId(`spread-${run.method}`);
    const range = page.getByTestId(`spread-range-${run.method}`);
    expect(spread.prAuc[spread.seeds.indexOf(run.seed)]).toBe(run.prAuc);
    if (spread.max > spread.min) {
      await expect(range).toHaveText(`${germanDecimal(spread.min)}–${germanDecimal(spread.max)}`);
      await expect(cell).toContainText(`Mittel ${germanDecimal(spread.mean)}`);
    } else {
      await expect(range).toHaveText(germanDecimal(spread.min));
      await expect(cell).toContainText(`in allen ${spread.seeds.length} Seeds gleich`);
    }
  }
  for (const run of METRICS.runs.filter((entry) => !entry.hyperparameters.seedSpread)) {
    await expect(page.getByTestId(`spread-${run.method}`)).toBeVisible();
    await expect(page.getByTestId(`spread-range-${run.method}`)).toHaveCount(0);
  }
  const hgb = runOf("hgb");
  if (hgb?.hyperparameters.seedSpread) {
    await expect(page.getByTestId("spread-hgb")).toContainText("kein Zufallsanteil");
    await expect(page.getByTestId("spread-note")).toContainText(
      "Beim Gradient Boosting ist die Spanne 0: Ohne Stichprobe von Zeilen oder Merkmalen und ohne Early Stopping zieht das Verfahren nichts zufällig",
    );
    await expect(page.getByTestId("seed-limit")).toContainText("beim Gradient Boosting ohne Zufallsanteil 0");
  }
  await expect(page.getByTestId("spread-note")).toContainText("Ein Maß für die Unsicherheit durch die Daten ist die Spanne bei keinem Verfahren.");
  await expect(page.getByTestId("supervision-limit")).toContainText("Ein Random Forest und ein größerer Suchraum des Gradient Boosting");
});

test("method texts keep the gradient boosting apart from the neural methods", async ({ page }) => {
  await page.goto("/projects/netzradar");
  const openSection = async (id: string) => {
    const section = page.locator(`#${id}`);
    await section.evaluate((element) => {
      (element as HTMLDetailsElement).open = true;
    });
    return section.innerText();
  };
  const neural = METRICS.runs.filter((run) => NEURAL_METHODS.includes(run.method));
  if (neural.length > 1) {
    const text = await openSection("methodik-message-passing");
    const start = text.indexOf("Hier haben die neuronalen Verfahren");
    const end = text.indexOf("Optimierungsschritt.", start);
    expect(start).toBeGreaterThanOrEqual(0);
    expect(end).toBeGreaterThan(start);
    const sentence = text.slice(start, end);
    expect(sentence).toContain(
      `Hier haben die neuronalen Verfahren ${germanList(neural.map((run) => SHORT_NAME[run.method] ?? run.method))} dieselbe Form:`,
    );
    expect(sentence).not.toContain("Gradient Boosting");
    expect(text).not.toContain("gelernten Verfahren");
  }
  if (runOf("hgb")) {
    const text = await openSection("methodik-hgb");
    expect(text).toContain("mit dem Startwert F0, den Log-Odds des Anteils der Trainingszeilen mit Label illegal");
    expect(text).toContain(
      "Dieser Abstand mischt aber den Wechsel von Bäumen zu neuronalen Netzen mit der Wirkung der Nachbarschaft",
    );
    expect(text).not.toContain("zeigt deshalb strenger");
  }
  const plans = page.getByTestId("dataset-plans");
  await expect(plans).toContainText(
    "Ausgewertet ist bisher keiner von beiden. Die öffentliche Demo auf dieser Seite bleibt das synthetische Netz",
  );
  await expect(plans).not.toContainText("Elliptic wird nur lokal");
});

test("excerpt text counts both start sets from nodes.json", async ({ page }) => {
  test.skip(NODES.scoreGnnMethod === null, "nodes.json has no GNN start set");
  await page.goto("/projects/netzradar");
  const count = (value: number) => value.toLocaleString("de-DE");
  const iforest = NODES.nodes.filter((node) => node.hop !== null).length;
  const gnn = NODES.nodes.filter((node) => node.hopGnn !== null).length;
  const both = NODES.nodes.filter((node) => node.hop !== null && node.hopGnn !== null).length;
  const text = page.getByTestId("excerpt-text");
  await expect(text).toContainText(`Der Ausschnitt ist die Vereinigung beider Nachbarschaften: ${count(NODES.nodes.length)} Knoten`);
  await expect(text).toContainText(
    `${count(iforest)} Knoten liegen in der Nachbarschaft der Startknoten nach Isolation-Forest-Score, ${count(gnn)} in der nach GNN-Score, ${count(both)} in beiden`,
  );
  await expect(page.getByRole("main")).not.toContainText("Der Ausschnitt selbst ist nach dem Isolation-Forest-Score gewählt");
});

test("start set switch changes table and graph by mouse and keyboard", async ({ page }) => {
  test.skip(NODES.scoreGnnMethod === null, "nodes.json has no GNN start set");
  await page.goto("/projects/netzradar");
  await expect(page.getByTestId("graph-view")).toHaveAttribute("data-state", "ready", { timeout: 30_000 });
  const label = GNN_LABEL[NODES.scoreGnnMethod ?? "gcn"];
  const group = page.getByRole("group", { name: "Startknoten nach Score" });
  const iforestButton = group.getByRole("button", { name: "Isolation Forest" });
  const gnnButton = group.getByRole("button", { name: `GNN (${label})` });
  await expect(iforestButton).toHaveAttribute("aria-pressed", "true");
  await expect(gnnButton).toHaveAttribute("aria-pressed", "false");
  await expect(page.getByTestId("top-nodes-title")).toHaveText("Die 50 Testknoten mit dem höchsten Isolation-Forest-Score");
  await expect(page.getByTestId("top-node-row").first()).toHaveAttribute("data-node-id", FIRST_SEED?.id ?? "");
  await expect(page.getByTestId("graph-view")).toHaveAttribute("data-seed-set", "iforest");

  await iforestButton.focus();
  await page.keyboard.press("Tab");
  await expect(gnnButton).toBeFocused();
  await page.keyboard.press("Space");
  await expect(gnnButton).toHaveAttribute("aria-pressed", "true");
  await expect(iforestButton).toHaveAttribute("aria-pressed", "false");
  await expect(gnnButton).toBeFocused();
  await expect(page.getByTestId("graph-view")).toHaveAttribute("data-seed-set", "gnn");
  await expect(page.getByTestId("top-nodes-title")).toHaveText(`Die 50 Testknoten mit dem höchsten GNN-Score (${label})`);
  await expect(page.getByTestId("top-nodes-gnn-header")).toHaveAttribute("aria-sort", "descending");
  await expect(page.getByTestId("top-node-row").first()).toHaveAttribute("data-node-id", FIRST_GNN_SEED?.id ?? "");
  await expect(page.getByTestId("top-node-rank")).toHaveText(Array.from({ length: 50 }, (_, index) => String(index + 1)));
  await expect(page.getByTestId("seed-set-note")).toContainText("0 ist dabei keine kalibrierte Schwelle");
  await expect(page.getByTestId("graph-canvas")).toHaveAttribute("aria-label", new RegExp(`Startknoten nach GNN-Score \\(${label}\\)`));
  await expect(page.getByTestId("top-nodes-table").locator("button[tabindex='0']")).toHaveCount(1);

  if (GNN_ONLY_SEED) {
    await page.locator(`[data-testid="top-node-row"][data-node-id="${GNN_ONLY_SEED.id}"]`).click();
    const seeds = page.getByTestId("node-detail-seeds");
    const value = (term: string) => seeds.locator("div", { has: page.locator("dt", { hasText: term }) }).locator("dd");
    await expect(page.getByTestId("node-detail-id")).toHaveText(GNN_ONLY_SEED.id);
    await expect(value("Rang nach Isolation-Forest-Score")).toHaveText("kein Startknoten");
    await expect(value(`Rang nach GNN-Score (${label})`)).toHaveText(`${GNN_ONLY_SEED.seedRankGnn} von 50`);
    await expect(seeds.locator("dd").nth(1)).toHaveText("mehr als 2 Kanten");
    await expect(seeds.locator("dd").nth(3)).toHaveText("0 Kanten");
  }

  await iforestButton.focus();
  await page.keyboard.press("Enter");
  await expect(iforestButton).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByTestId("top-node-row").first()).toHaveAttribute("data-node-id", FIRST_SEED?.id ?? "");
  await expect(page.getByTestId("seed-set-note")).not.toContainText("kalibrierte Schwelle");
});
