import { readFileSync } from "node:fs";
import path from "node:path";
import { expect, test, type Page } from "@playwright/test";

interface ExportedRun {
  method: string;
  prAuc: number;
}

interface ExportedNode {
  id: string;
  seedRank: number | null;
}

const DATA_DIR = path.resolve(__dirname, "../../../data/k3");
const METRICS = JSON.parse(readFileSync(path.join(DATA_DIR, "metrics.json"), "utf8")) as {
  runs: ExportedRun[];
  evaluation: { prevalence: number; randomPrAucExpected: number };
};
const INK = "rgb(17, 17, 17)";
const NODES = JSON.parse(readFileSync(path.join(DATA_DIR, "nodes.json"), "utf8")) as { nodes: ExportedNode[] };
const FIRST_SEED = NODES.nodes.find((node) => node.seedRank === 1);

function germanDecimal(value: number): string {
  return value.toFixed(4).replace(".", ",");
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
  await expect(page.getByText("Projekt K3", { exact: true })).toBeVisible();

  const metricsTable = page.getByTestId("metrics-table");
  await expect(metricsTable).toBeVisible();
  for (const run of METRICS.runs) {
    await expect(page.getByTestId(`pr-auc-${run.method}`)).toHaveText(germanDecimal(run.prAuc));
  }
  await expect(page.getByTestId("metrics-row-random")).toContainText(
    germanDecimal(METRICS.evaluation.randomPrAucExpected),
  );
  await expect(page.getByTestId("metrics-row-prevalence")).toContainText(germanDecimal(METRICS.evaluation.prevalence));
  for (const method of ["gcn", "graphsage"]) {
    if (!METRICS.runs.some((run) => run.method === method)) {
      await expect(page.getByTestId(`metrics-row-${method}`)).toContainText("Schritt 4, noch nicht gemessen");
    }
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
  await firstRow.click();
  await expect(page.getByTestId("node-detail-id")).toHaveText(FIRST_SEED?.id ?? "");
  await expect(page.getByTestId("node-detail")).toContainText("1 von");
  await expect(firstRow).toHaveAttribute("aria-current", "true");

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

test("start page links to the NetzRadar page", async ({ page }) => {
  await page.goto("/");
  const link = page.getByTestId("project-netzradar").getByRole("link");
  await expect(link).toHaveAttribute("href", "/projects/netzradar");
  await link.click();
  await expect(page).toHaveURL(/\/projects\/netzradar$/);
  await expect(page.getByRole("heading", { level: 1, name: "NetzRadar" })).toBeVisible();
});
