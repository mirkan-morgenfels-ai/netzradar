import { expect, test } from "@playwright/test";

const LEGAL_PAGES = [
  { path: "/impressum", heading: "Impressum" },
  { path: "/datenschutz", heading: "Datenschutzerklärung" },
  { path: "/nutzungsbedingungen", heading: "Nutzungsbedingungen" },
];
const LEGAL_UPDATED = "06.10.2026";

test("start page lists the three projects", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1, name: "Projekte" })).toBeVisible();
  await expect(page.getByTestId(/^project-/)).toHaveCount(3);

  for (const slug of ["depotdoktor", "kontoklar"]) {
    const link = page.getByTestId(`project-${slug}`).getByRole("link");
    await expect(link).toHaveAttribute("href", /^https:\/\//);
    await expect(link).toHaveAttribute("rel", "noopener noreferrer");
  }

  await expect(page.getByTestId("project-netzradar").getByRole("link")).toHaveAttribute("href", "/projects/netzradar");
});

test("main navigation and skip link are present", async ({ page }) => {
  await page.goto("/");
  const nav = page.getByRole("navigation", { name: "Hauptnavigation" });
  await expect(nav.getByRole("link", { name: "Start" })).toBeVisible();
  await expect(nav.getByRole("link", { name: "NetzRadar" })).toBeVisible();
  await expect(page.locator("a.skip-link")).toHaveAttribute("href", "#main");
});

for (const legal of LEGAL_PAGES) {
  test(`legal page ${legal.path} renders`, async ({ page }) => {
    await page.goto(legal.path);
    await expect(page.getByRole("heading", { level: 1, name: legal.heading })).toBeVisible();
    await expect(page.getByText(`Stand: ${LEGAL_UPDATED}`, { exact: true })).toBeVisible();
  });
}

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
