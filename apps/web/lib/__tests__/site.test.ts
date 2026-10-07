import { afterEach, describe, expect, it, vi } from "vitest";
import {
  DEFAULT_SITE_URL,
  GITHUB_PROFILE_URL,
  HOME_DESCRIPTION,
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
    expect(PROJECTS.map((project) => project.kicker)).toEqual(["Projekt 01", "Projekt 02", "Projekt 03"]);
    expect(PROJECTS.map((project) => project.number)).toEqual(["01", "02", "03"]);
    expect(PROJECTS.map((project) => project.topic)).toEqual(["Finanzdaten", "Maschinelles Lernen", "Graph-ML"]);
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

  it("uses the same project descriptions as the sibling sites", () => {
    expect(PROJECTS[0]?.description).toBe(
      "Depot-Steuer- und Performance-Analyzer für Broker-CSV-Exporte. Die Auswertung läuft vollständig im Browser.",
    );
    expect(PROJECTS[1]?.description).toBe("Kategorisiert Bankumsätze aus CSV-Exporten und zeigt, wohin das Geld geht.");
    expect(PROJECTS[2]?.description).toBe(
      "Anomalie-Erkennung in Transaktionsnetzwerken: klassische Baseline gegen Graph Neural Networks, mit zeitlichem Split und PR-AUC.",
    );
  });

  it("describes the start page with the same words as the sibling sites", () => {
    expect(HOME_DESCRIPTION).toBe(
      "Drei Portfolio-Projekte zu Finanzdaten, maschinellem Lernen und Graph-ML: DepotDoktor, KontoKlar und NetzRadar, jeweils mit öffentlichem Quellcode auf GitHub.",
    );
    for (const project of PROJECTS) expect(HOME_DESCRIPTION).toContain(project.title);
  });
});

describe("navigation", () => {
  it("offers start, the three projects and the GitHub profile in the main navigation", () => {
    expect(NAV_LINKS.map((link) => link.label)).toEqual(["Start", "DepotDoktor", "KontoKlar", "NetzRadar", "GitHub"]);
    expect(NAV_LINKS.map((link) => link.href)).toEqual([
      "/",
      "https://depotdoktor.vercel.app/projects/depotdoktor",
      "https://kontoklar-eight.vercel.app/projects/kontoklar",
      "/projects/netzradar",
      "https://github.com/mirkan-morgenfels-ai",
    ]);
    expect(NAV_LINKS.map((link) => link.external)).toEqual([false, true, true, false, true]);
    expect(NAV_LINKS.at(-1)?.href).toBe(GITHUB_PROFILE_URL);
    expect(GITHUB_PROFILE_URL).toBe("https://github.com/mirkan-morgenfels-ai");
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
