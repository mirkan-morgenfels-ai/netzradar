import { expect, test, type Page } from "@playwright/test";

const LEGAL_PAGES = [
  { path: "/impressum", heading: "Impressum" },
  { path: "/datenschutz", heading: "Datenschutzerklärung" },
  { path: "/nutzungsbedingungen", heading: "Nutzungsbedingungen" },
];
const LEGAL_UPDATED = "07.10.2026";
const REPO_BASE = "https://github.com/mirkan-morgenfels-ai";
const NETZRADAR_REPO = `${REPO_BASE}/netzradar`;
const SITE_URL = (process.env.NEXT_PUBLIC_SITE_URL?.trim() || "https://netzradar.vercel.app").replace(/\/+$/, "");
const HOME_TITLE = "Projekte · Mirkan Deniz Günkaya";
const PROJECT_TITLE = "NetzRadar – Anomalie-Erkennung in Transaktionsnetzwerken";
const REFLOW_PATHS = ["/", "/projects/netzradar", ...LEGAL_PAGES.map((legal) => legal.path)];

async function metaContent(page: Page, selector: string): Promise<string | null> {
  return page.locator(selector).first().getAttribute("content");
}

test("start page lists the three projects", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1, name: "Projekte" })).toBeVisible();
  await expect(page.getByTestId(/^project-/)).toHaveCount(3);

  for (const slug of ["depotdoktor", "kontoklar"]) {
    const link = page.getByTestId(`project-${slug}`).getByRole("link", { name: /^Zum Projekt/ });
    await expect(link).toHaveAttribute("href", /^https:\/\//);
    await expect(link).toHaveAttribute("rel", "noopener noreferrer");
    await expect(link).not.toHaveAttribute("target", /.+/);
  }

  await expect(
    page.getByTestId("project-netzradar").getByRole("link", { name: /^Zum Projekt/ }),
  ).toHaveAttribute("href", "/projects/netzradar");

  for (const slug of ["depotdoktor", "kontoklar", "netzradar"]) {
    const repo = page.getByTestId(`project-${slug}`).getByRole("link", { name: /^Quellcode/ });
    await expect(repo).toHaveAttribute("href", new RegExp(`^${REPO_BASE}/${slug}$`));
    await expect(repo).toHaveAttribute("rel", "noopener noreferrer");
    await expect(repo).not.toHaveAttribute("target", /.+/);
  }
});

test("main navigation and skip link are present", async ({ page }) => {
  await page.goto("/projects/netzradar");
  const nav = page.getByRole("navigation", { name: "Hauptnavigation" });
  const links = nav.getByRole("list").getByRole("link");
  await expect(links).toHaveCount(4);
  await expect(links.nth(0)).toHaveText("Start");
  await expect(links.nth(1)).toHaveAccessibleName("DepotDoktor (externe Seite)");
  await expect(links.nth(2)).toHaveAccessibleName("KontoKlar (externe Seite)");
  await expect(links.nth(3)).toHaveText("NetzRadar");
  for (const index of [1, 2]) {
    await expect(links.nth(index)).toHaveAttribute("href", /^https:\/\//);
    await expect(links.nth(index)).toHaveAttribute("rel", "noopener noreferrer");
    await expect(links.nth(index)).not.toHaveAttribute("target", /.+/);
  }
  await expect(links.nth(3)).toHaveAttribute("aria-current", "page");
  await expect(links.nth(0)).not.toHaveAttribute("aria-current", /.+/);
  await expect(page.locator("a.skip-link")).toHaveAttribute("href", "#main");

  await page.goto("/");
  await expect(nav.getByRole("link", { name: "Start" })).toHaveAttribute("aria-current", "page");
  await expect(nav.getByRole("link", { name: "NetzRadar", exact: true })).not.toHaveAttribute("aria-current", /.+/);
});

test("footer and project page link the public repository", async ({ page }) => {
  await page.goto("/");
  const footerLink = page.getByTestId("footer-repo-link");
  await expect(footerLink).toHaveAttribute("href", NETZRADAR_REPO);
  await expect(footerLink).toHaveAttribute("rel", "noopener noreferrer");
  await expect(footerLink).toHaveAccessibleName("Quellcode auf GitHub (externe Seite)");
  await expect(page.locator("footer")).toContainText("Quellcode auf GitHub (externe Seite) (MIT-Lizenz)");

  await page.goto("/projects/netzradar");
  await expect(page.getByTestId("project-repo-link")).toHaveAttribute("href", NETZRADAR_REPO);
  await expect(page.getByTestId("limits-readme-link")).toHaveAttribute("href", `${NETZRADAR_REPO}#readme`);
  await expect(page.getByTestId("limits-runs-link")).toHaveAttribute("href", `${NETZRADAR_REPO}/tree/main/docs/runs`);
  await expect(page.getByTestId("status-runs-link")).toHaveAttribute("href", `${NETZRADAR_REPO}/tree/main/docs/runs`);
  await expect(page.getByTestId("generator-source-link")).toHaveAttribute(
    "href",
    `${NETZRADAR_REPO}/blob/main/services/k3-train/src/k3_train/synth.py`,
  );
  const external = page.locator(`a[href^="${REPO_BASE}"]`);
  for (const link of await external.all()) {
    await expect(link).toHaveAttribute("rel", "noopener noreferrer");
  }
  await expect(page.locator("a[target]")).toHaveCount(0);

  for (const path of ["/impressum", "/nutzungsbedingungen"]) {
    await page.goto(path);
    await expect(page.getByRole("main").getByRole("link", { name: /^MIT-Lizenz/ })).toHaveAttribute(
      "href",
      `${NETZRADAR_REPO}/blob/main/LICENSE`,
    );
    await expect(page.getByRole("main")).toContainText("Ausgenommen sind die Exporte des eigenen synthetischen Netzes in data/k3/");
  }

  await page.goto("/datenschutz");
  await expect(page.getByRole("main")).toContainText("Quellcode-Repositories bei GitHub (GitHub, Inc., USA)");
  await expect(page.getByRole("main")).not.toContainText("JSON-Dateien");
});

for (const legal of LEGAL_PAGES) {
  test(`legal page ${legal.path} renders`, async ({ page }) => {
    await page.goto(legal.path);
    await expect(page.getByRole("heading", { level: 1, name: legal.heading })).toBeVisible();
    await expect(page.getByText(`Stand: ${LEGAL_UPDATED}`, { exact: true })).toBeVisible();
  });
}

test("no page overflows horizontally at 320 px", async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 640 });
  for (const path of REFLOW_PATHS) {
    await page.goto(path);
    await page.waitForLoadState("load");
    const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
    expect(scrollWidth, path).toBeLessThanOrEqual(320);
  }
});

test("unknown paths answer 404 with a noindex title", async ({ page }) => {
  const response = await page.goto("/gibt-es-nicht");
  expect(response?.status()).toBe(404);
  await expect(page.getByRole("heading", { level: 1, name: "Seite nicht gefunden" })).toBeVisible();
  await expect(page).toHaveTitle("Seite nicht gefunden · NetzRadar");
  const robots = await page.locator('meta[name="robots"]').evaluateAll((metas) =>
    metas.map((meta) => meta.getAttribute("content") ?? ""),
  );
  expect(robots.length).toBeGreaterThan(0);
  for (const content of robots) expect(content).toContain("noindex");
});

test("pages carry canonical links, link previews and distinct titles", async ({ page, request }) => {
  const titles: Record<string, string> = {};
  for (const path of ["/", "/projects/netzradar", ...LEGAL_PAGES.map((legal) => legal.path)]) {
    await page.goto(path);
    titles[path] = await page.title();
    const canonical = await page.locator('link[rel="canonical"]').getAttribute("href");
    expect(canonical, path).not.toBeNull();
    expect(new URL(canonical ?? "").origin, path).toBe(SITE_URL);
    expect(new URL(canonical ?? "").pathname, path).toBe(path);
    expect(await metaContent(page, 'meta[property="og:url"]'), path).toBe(canonical);
    expect(await metaContent(page, 'meta[property="og:site_name"]'), path).toBe("NetzRadar");
    expect(await metaContent(page, 'meta[name="twitter:card"]'), path).toBe("summary_large_image");
    const image = await metaContent(page, 'meta[property="og:image"]');
    expect(image, path).toMatch(/^https:\/\//);
    expect(image, path).not.toContain("localhost");
    expect(new URL(image ?? "").origin, path).toBe(SITE_URL);
    expect(new URL(image ?? "").search, path).toMatch(/^\?v=\d+$/);
    expect(await metaContent(page, 'meta[property="og:image:type"]'), path).toBe("image/png");
    const local = await request.get(new URL(image ?? "").pathname + new URL(image ?? "").search);
    expect(local.status(), path).toBe(200);
    expect(local.headers()["content-type"], path).toContain("image/png");
  }
  expect(titles["/"]).toBe(HOME_TITLE);
  expect(titles["/projects/netzradar"]).toBe(PROJECT_TITLE);
  expect(titles["/projects/netzradar"]).not.toBe(titles["/"]);
  expect(titles["/impressum"]).toBe("Impressum · NetzRadar");
});

test("sitemap and robots.txt describe the public pages", async ({ request }) => {
  const sitemap = await request.get("/sitemap.xml");
  expect(sitemap.status()).toBe(200);
  const xml = await sitemap.text();
  for (const path of ["/projects/netzradar", ...LEGAL_PAGES.map((legal) => legal.path)]) {
    expect(xml).toContain(`<loc>${SITE_URL}${path}</loc>`);
  }
  const robots = await request.get("/robots.txt");
  expect(robots.status()).toBe(200);
  const text = await robots.text();
  expect(text).not.toContain("/api/");
  expect(text).toContain(`Sitemap: ${SITE_URL}/sitemap.xml`);
  const appleIcon = await request.get("/apple-icon");
  expect(appleIcon.status()).toBe(200);
  expect(appleIcon.headers()["content-type"]).toContain("image/png");
});

test("pages only send same-origin GET requests", async ({ page, baseURL }) => {
  const origin = new URL(baseURL ?? "http://localhost:3000").origin;
  const violations: string[] = [];
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (url.protocol === "data:" || url.protocol === "blob:") return;
    if (url.origin !== origin || request.method() !== "GET") {
      violations.push(`${request.method()} ${request.url()}`);
    }
  });

  for (const path of ["/", ...LEGAL_PAGES.map((legal) => legal.path)]) {
    await page.goto(path);
    await page.waitForLoadState("load");
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  }

  expect(violations).toEqual([]);
});
