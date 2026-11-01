import { expect, test, type Locator, type Page } from "@playwright/test";

const LEGAL_PAGES = [
  { path: "/impressum", heading: "Impressum" },
  { path: "/datenschutz", heading: "Datenschutzerklärung" },
  { path: "/nutzungsbedingungen", heading: "Nutzungsbedingungen" },
];
const LEGAL_UPDATED = "01.11.2026";
const REPO_BASE = "https://github.com/mirkan-morgenfels-ai";
const NETZRADAR_REPO = `${REPO_BASE}/netzradar`;
const SITE_URL = (process.env.NEXT_PUBLIC_SITE_URL?.trim() || "https://netzradar.mirkandeniz.de").replace(/\/+$/, "");
const LEGACY_HOST = "netzradar.vercel.app";
const PORTFOLIO_URL = "https://mirkandeniz.de";
const PROJECT_TITLE = "NetzRadar – Anomalie-Erkennung in Transaktionsnetzwerken";
const REFLOW_PATHS = ["/projects/netzradar", ...LEGAL_PAGES.map((legal) => legal.path)];

async function metaContent(page: Page, selector: string): Promise<string | null> {
  return page.locator(selector).first().getAttribute("content");
}

test("start address opens NetzRadar directly", async ({ page, request }) => {
  const response = await request.get("/", { maxRedirects: 0 });
  expect(response.status()).toBe(308);
  expect(response.headers()["location"]).toBe("/projects/netzradar");
  await page.goto("/");
  await expect(page).toHaveURL(/\/projects\/netzradar$/);
  await expect(page.getByRole("heading", { level: 1 })).toHaveCSS("font-family", /Cormorant/);
});

test("the former Vercel address forwards to the subdomain of mirkandeniz.de", async ({ request }) => {
  test.skip(new URL(SITE_URL).host === LEGACY_HOST, "NEXT_PUBLIC_SITE_URL zeigt noch auf die alte Adresse");
  const project = await request.get("/projects/netzradar?quelle=alt", { headers: { host: LEGACY_HOST }, maxRedirects: 0 });
  expect(project.status()).toBe(308);
  expect(project.headers()["location"]).toBe(`${SITE_URL}/projects/netzradar?quelle=alt`);
  const root = await request.get("/", { headers: { host: LEGACY_HOST }, maxRedirects: 0 });
  expect(root.status()).toBe(308);
  expect(root.headers()["location"]).toBe(SITE_URL);
});

test("main navigation, way back to mirkandeniz.de and skip link are present", async ({ page }) => {
  await page.goto("/projects/netzradar");
  const nav = page.getByRole("navigation", { name: "Hauptnavigation" });
  const links = nav.getByRole("list").getByRole("link");
  await expect(links).toHaveCount(4);
  await expect(links.nth(0)).toHaveAccessibleName("DepotDoktor (externe Seite)");
  await expect(links.nth(1)).toHaveAccessibleName("KontoKlar (externe Seite)");
  await expect(links.nth(2)).toHaveText("NetzRadar");
  await expect(links.nth(3)).toHaveAccessibleName("Alle Projekte (externe Seite)");
  await expect(links.nth(0)).toHaveAttribute("href", "https://depotdoktor.mirkandeniz.de/projects/depotdoktor");
  await expect(links.nth(1)).toHaveAttribute("href", "https://kontoklar.mirkandeniz.de/projects/kontoklar");
  await expect(links.nth(3)).toHaveAttribute("href", `${PORTFOLIO_URL}/projekte`);
  for (const index of [0, 1, 3]) {
    await expect(links.nth(index)).toHaveAttribute("rel", "noopener noreferrer");
    await expect(links.nth(index)).not.toHaveAttribute("target", /.+/);
  }
  await expect(links.nth(2)).toHaveAttribute("aria-current", "page");
  await expect(page.locator("a.skip-link")).toHaveAttribute("href", "#main");

  const home = page.locator("header").getByTestId("portfolio-link");
  await expect(home).toHaveAttribute("href", PORTFOLIO_URL);
  await expect(home).toHaveAttribute("rel", "noopener noreferrer");
  await expect(home).not.toHaveAttribute("target", /.+/);
  await expect(home).toHaveAccessibleName(/^Mirkan Deniz.*\(externe Seite\)$/);
  await expect(home).toContainText("mirkandeniz.de");
});

test("focus rings are gold-light on navy and gold-deep on light surfaces", async ({ page }) => {
  await page.goto("/projects/netzradar");
  await page.keyboard.press("Tab");
  await expect(page.locator("a.skip-link")).toBeFocused();
  const ring = (target: Locator) =>
    target.evaluate((element) => {
      const style = getComputedStyle(element);
      return { color: style.outlineColor, style: style.outlineStyle, width: style.outlineWidth };
    });
  const onNavy = page.getByTestId("project-repo-link");
  await onNavy.focus();
  expect(await ring(onNavy)).toEqual({ color: "rgb(216, 189, 114)", style: "solid", width: "2px" });
  const onLight = page.getByTestId("section-nav").getByRole("link", { name: "Daten" });
  await onLight.focus();
  expect(await ring(onLight)).toEqual({ color: "rgb(125, 95, 23)", style: "solid", width: "2px" });
});

test("footer and project page link the public repository and mirkandeniz.de", async ({ page }) => {
  await page.goto("/projects/netzradar");
  const portfolio = page.getByTestId("footer-portfolio-link");
  await expect(portfolio).toHaveAttribute("href", PORTFOLIO_URL);
  await expect(portfolio).toHaveAttribute("rel", "noopener noreferrer");
  await expect(portfolio).toHaveAccessibleName("mirkandeniz.de (externe Seite)");
  await expect(page.locator("footer").getByTestId("portfolio-link")).toHaveAttribute("href", PORTFOLIO_URL);
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
  await expect(page.getByRole("main")).toContainText("Diese Seite verlinkt auf die Website mirkandeniz.de und die Projekte DepotDoktor und KontoKlar");
  await expect(page.getByRole("main")).toContainText("guenkaya@mirkandeniz.de");
  await expect(page.getByRole("main")).toContainText("IONOS SE, Elgendorfer Straße 57, 56410 Montabaur");
  await expect(page.getByRole("main")).toContainText("Quellcode-Repositories bei GitHub (GitHub, Inc., USA)");
  await expect(page.getByRole("main")).not.toContainText("JSON-Dateien");
});

for (const legal of LEGAL_PAGES) {
  test(`legal page ${legal.path} renders`, async ({ page }) => {
    await page.goto(legal.path);
    await expect(page.getByRole("heading", { level: 1, name: legal.heading })).toBeVisible();
    await expect(page.getByText(`Stand: ${LEGAL_UPDATED}`, { exact: true })).toBeVisible();
    await expect(page.getByRole("main")).not.toContainText("gmail.com");
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
  await expect(page.getByRole("main").getByRole("link", { name: "Zu NetzRadar" })).toHaveAttribute("href", "/projects/netzradar");
  const website = page.getByRole("main").getByRole("link", { name: "Zur Website mirkandeniz.de (externe Seite)" });
  await expect(website).toHaveAttribute("href", PORTFOLIO_URL);
  await expect(website).toHaveAttribute("rel", "noopener noreferrer");
  await expect(page.getByRole("main").getByRole("link", { name: "Zur Startseite" })).toHaveCount(0);
});

test("pages carry canonical links, link previews and distinct titles", async ({ page, request }) => {
  const titles: Record<string, string> = {};
  for (const path of ["/projects/netzradar", ...LEGAL_PAGES.map((legal) => legal.path)]) {
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
  expect(titles["/projects/netzradar"]).toBe(PROJECT_TITLE);
  expect(titles["/impressum"]).toBe("Impressum · NetzRadar");
});

test("sitemap and robots.txt describe the public pages", async ({ request }) => {
  const sitemap = await request.get("/sitemap.xml");
  expect(sitemap.status()).toBe(200);
  const xml = await sitemap.text();
  for (const path of ["/projects/netzradar", ...LEGAL_PAGES.map((legal) => legal.path)]) {
    expect(xml).toContain(`<loc>${SITE_URL}${path}</loc>`);
  }
  expect(xml).not.toContain(`<loc>${SITE_URL}/</loc>`);
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

  for (const path of ["/projects/netzradar", ...LEGAL_PAGES.map((legal) => legal.path)]) {
    await page.goto(path);
    await page.waitForLoadState("load");
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  }

  expect(violations).toEqual([]);
});
