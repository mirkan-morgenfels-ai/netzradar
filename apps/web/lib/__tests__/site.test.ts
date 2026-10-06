import { afterEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_SITE_URL, LEGAL_LINKS, NAV_LINKS, PROJECTS, siteUrl } from "../site";

describe("PROJECTS", () => {
  it("lists exactly the three portfolio projects in order", () => {
    expect(PROJECTS).toHaveLength(3);
    expect(PROJECTS.map((project) => project.slug)).toEqual(["depotdoktor", "kontoklar", "netzradar"]);
    expect(PROJECTS.map((project) => project.code)).toEqual(["K1", "K2", "K3"]);
  });

  it("links NetzRadar internally", () => {
    const netzradar = PROJECTS.find((project) => project.slug === "netzradar");
    expect(netzradar?.href).toBe("/projects/netzradar");
    expect(netzradar?.external).toBe(false);
  });

  it("uses https for every external link", () => {
    const external = PROJECTS.filter((project) => project.external);
    expect(external.map((project) => project.slug)).toEqual(["depotdoktor", "kontoklar"]);
    for (const project of external) {
      expect(new URL(project.href).protocol).toBe("https:");
    }
  });

  it("keeps internal links root-relative", () => {
    for (const project of PROJECTS.filter((entry) => !entry.external)) {
      expect(project.href.startsWith("/")).toBe(true);
      expect(project.href.startsWith("//")).toBe(false);
    }
  });

  it("gives every project a non-empty title and description", () => {
    for (const project of PROJECTS) {
      expect(project.title.trim().length).toBeGreaterThan(0);
      expect(project.description.trim().length).toBeGreaterThan(0);
    }
  });
});

describe("navigation", () => {
  it("offers start and NetzRadar in the main navigation", () => {
    expect(NAV_LINKS.map((link) => link.href)).toEqual(["/", "/projects/netzradar"]);
  });

  it("offers the three legal pages", () => {
    expect(LEGAL_LINKS.map((link) => link.href)).toEqual(["/impressum", "/datenschutz", "/nutzungsbedingungen"]);
  });
});

describe("siteUrl", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("falls back to localhost for an empty value", () => {
    expect(siteUrl("").href).toBe("http://localhost:3000/");
    expect(siteUrl("   ").href).toBe("http://localhost:3000/");
    expect(DEFAULT_SITE_URL).toBe("http://localhost:3000");
  });

  it("uses a configured value", () => {
    expect(siteUrl("https://netzradar.example").href).toBe("https://netzradar.example/");
  });

  it("reads NEXT_PUBLIC_SITE_URL when no value is passed", () => {
    vi.stubEnv("NEXT_PUBLIC_SITE_URL", "https://example.org");
    expect(siteUrl().href).toBe("https://example.org/");
  });

  it("rejects a malformed value", () => {
    expect(() => siteUrl("kein url")).toThrow();
  });
});
