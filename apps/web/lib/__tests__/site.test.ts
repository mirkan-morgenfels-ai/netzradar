import { afterEach, describe, expect, it, vi } from "vitest";
import {
  DEFAULT_SITE_URL,
  LEGAL_LINKS,
  LICENSE_URL,
  NAV_LINKS,
  PROJECTS,
  README_URL,
  REPO_URL,
  RUNS_URL,
  SITEMAP_PATHS,
  isActivePath,
  isRepoPath,
  repoUrl,
  siteUrl,
} from "../site";

describe("PROJECTS", () => {
  it("lists exactly the three portfolio projects in order", () => {
    expect(PROJECTS).toHaveLength(3);
    expect(PROJECTS.map((project) => project.slug)).toEqual(["depotdoktor", "kontoklar", "netzradar"]);
    expect(PROJECTS.map((project) => project.code)).toEqual(["K1", "K2", "K3"]);
    expect(PROJECTS.map((project) => project.kicker)).toEqual(["Projekt K1", "Projekt K2", "Projekt K3"]);
  });

  it("links NetzRadar internally", () => {
    const netzradar = PROJECTS.find((project) => project.slug === "netzradar");
    expect(netzradar?.href).toBe("/projects/netzradar");
    expect(netzradar?.external).toBe(false);
  });

  it("uses https for every external link", () => {
    const external = PROJECTS.filter((project) => project.external);
    expect(external.map((project) => project.slug)).toEqual(["depotdoktor", "kontoklar"]);
    expect(external.map((project) => project.href)).toEqual([
      "https://depotdoktor.vercel.app/projects/depotdoktor",
      "https://kontoklar-eight.vercel.app/projects/kontoklar",
    ]);
    for (const project of external) {
      expect(new URL(project.href).protocol).toBe("https:");
    }
  });

  it("gives every project a public repository under github.com/mirkan-morgenfels-ai", () => {
    for (const project of PROJECTS) {
      expect(project.repo).toBe(`https://github.com/mirkan-morgenfels-ai/${project.slug}`);
    }
    expect(PROJECTS.find((project) => project.slug === "netzradar")?.repo).toBe(REPO_URL);
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
  it("offers start and the three projects in the main navigation", () => {
    expect(NAV_LINKS.map((link) => link.label)).toEqual(["Start", "DepotDoktor", "KontoKlar", "NetzRadar"]);
    expect(NAV_LINKS.map((link) => link.href)).toEqual([
      "/",
      "https://depotdoktor.vercel.app/projects/depotdoktor",
      "https://kontoklar-eight.vercel.app/projects/kontoklar",
      "/projects/netzradar",
    ]);
    expect(NAV_LINKS.map((link) => link.external)).toEqual([false, true, true, false]);
  });

  it("offers the three legal pages", () => {
    expect(LEGAL_LINKS.map((link) => link.href)).toEqual(["/impressum", "/datenschutz", "/nutzungsbedingungen"]);
    expect(LEGAL_LINKS.every((link) => !link.external)).toBe(true);
  });

  it("marks the start page only on / and a project page also on its subpaths", () => {
    expect(isActivePath("/", "/")).toBe(true);
    expect(isActivePath("/projects/netzradar", "/")).toBe(false);
    expect(isActivePath("/projects/netzradar", "/projects/netzradar")).toBe(true);
    expect(isActivePath("/projects/netzradar/details", "/projects/netzradar")).toBe(true);
    expect(isActivePath("/projects/netzradar-alt", "/projects/netzradar")).toBe(false);
    expect(isActivePath(null, "/")).toBe(false);
  });

  it("lists the start page, the project page and the legal pages for the sitemap", () => {
    expect(SITEMAP_PATHS).toEqual(["/", "/projects/netzradar", "/impressum", "/datenschutz", "/nutzungsbedingungen"]);
  });
});

describe("repository links", () => {
  it("builds links into the public repository", () => {
    expect(REPO_URL).toBe("https://github.com/mirkan-morgenfels-ai/netzradar");
    expect(README_URL).toBe("https://github.com/mirkan-morgenfels-ai/netzradar#readme");
    expect(RUNS_URL).toBe("https://github.com/mirkan-morgenfels-ai/netzradar/tree/main/docs/runs");
    expect(LICENSE_URL).toBe("https://github.com/mirkan-morgenfels-ai/netzradar/blob/main/LICENSE");
    expect(repoUrl("services/k3-train/src/k3_train/synth.py")).toBe(
      "https://github.com/mirkan-morgenfels-ai/netzradar/blob/main/services/k3-train/src/k3_train/synth.py",
    );
    expect(repoUrl("/LICENSE")).toBe(LICENSE_URL);
    expect(repoUrl("")).toBe(REPO_URL);
  });

  it("links only plain relative paths into the repository", () => {
    expect(isRepoPath("services/k3-train/src/k3_train/synth.py")).toBe(true);
    expect(isRepoPath("synth.py")).toBe(false);
    expect(isRepoPath("https://example.org/data.csv")).toBe(false);
    expect(isRepoPath("Kaggle ellipticco/elliptic-data-set")).toBe(false);
    expect(isRepoPath("../secret/file.txt")).toBe(false);
  });
});

describe("siteUrl", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("falls back to the production address for an empty value", () => {
    expect(siteUrl("").href).toBe("https://netzradar.vercel.app/");
    expect(siteUrl("   ").href).toBe("https://netzradar.vercel.app/");
    expect(DEFAULT_SITE_URL).toBe("https://netzradar.vercel.app");
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
