import { afterEach, describe, expect, it, vi } from "vitest";
import { transitionDestination } from "../../components/site/transition";
import {
  DEFAULT_SITE_URL,
  GITHUB_PROFILE_URL,
  LEGACY_HOSTS,
  LEGAL_LINKS,
  LICENSE_URL,
  NAV_LINKS,
  PORTFOLIO_HOME_URL,
  PORTFOLIO_LABEL,
  PORTFOLIO_PROJECTS_URL,
  PORTFOLIO_URL,
  PROJECT_PATH,
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
      "https://depotdoktor.mirkandeniz.de/projects/depotdoktor",
      "https://kontoklar.mirkandeniz.de/projects/kontoklar",
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
      "Anomalie-Erkennung in Transaktionsnetzwerken: klassische Baselines und Gradient Boosting gegen Graph Neural Networks, mit zeitlichem Split und PR-AUC.",
    );
  });

});

describe("navigation", () => {
  it("offers the three projects and the project overview on mirkandeniz.de in the main navigation", () => {
    expect(NAV_LINKS.map((link) => link.label)).toEqual(["DepotDoktor", "KontoKlar", "NetzRadar", "Alle Projekte"]);
    expect(NAV_LINKS.map((link) => link.href)).toEqual([
      "https://depotdoktor.mirkandeniz.de/projects/depotdoktor",
      "https://kontoklar.mirkandeniz.de/projects/kontoklar",
      "/projects/netzradar",
      "https://mirkandeniz.de/projekte",
    ]);
    expect(NAV_LINKS.map((link) => link.external)).toEqual([true, true, false, true]);
    expect(PORTFOLIO_PROJECTS_URL).toBe(`${PORTFOLIO_URL}/projekte`);
    expect(PROJECT_PATH).toBe("/projects/netzradar");
    expect(GITHUB_PROFILE_URL).toBe("https://github.com/mirkan-morgenfels-ai");
  });

  it("leads back to the website mirkandeniz.de", () => {
    expect(PORTFOLIO_URL).toBe("https://mirkandeniz.de");
    expect(PORTFOLIO_HOME_URL).toBe("https://mirkandeniz.de/#start");
    expect(PORTFOLIO_LABEL).toBe(new URL(PORTFOLIO_URL).host);
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

  it("lists the project page and the legal pages for the sitemap, without the redirecting start address", () => {
    expect(SITEMAP_PATHS).toEqual(["/projects/netzradar", "/impressum", "/datenschutz", "/nutzungsbedingungen"]);
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
    expect(siteUrl("").href).toBe("https://netzradar.mirkandeniz.de/");
    expect(siteUrl("   ").href).toBe("https://netzradar.mirkandeniz.de/");
    expect(DEFAULT_SITE_URL).toBe("https://netzradar.mirkandeniz.de");
  });

  it("redirects only the former Vercel address", () => {
    expect(LEGACY_HOSTS).toEqual(["netzradar.vercel.app"]);
    expect(LEGACY_HOSTS).not.toContain(siteUrl("").host);
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

describe("transitionDestination", () => {
  it("names the website and the sister projects as destinations of the transition", () => {
    expect(transitionDestination(PORTFOLIO_HOME_URL)).toEqual({ eyebrow: "Zurück zur Website", title: "Mirkan", accent: "Deniz", host: "mirkandeniz.de" });
    expect(transitionDestination(PORTFOLIO_PROJECTS_URL)?.host).toBe(PORTFOLIO_LABEL);
    expect(transitionDestination("https://depotdoktor.mirkandeniz.de/projects/depotdoktor")).toEqual({
      eyebrow: "Weiter zum Projekt",
      title: "DepotDoktor",
      accent: null,
      host: "depotdoktor.mirkandeniz.de",
    });
    expect(transitionDestination(REPO_URL)).toBeNull();
    expect(transitionDestination(GITHUB_PROFILE_URL)).toBeNull();
    expect(transitionDestination("/impressum")).toBeNull();
  });
});
