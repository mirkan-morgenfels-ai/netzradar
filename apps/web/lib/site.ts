export interface Project {
  slug: string;
  number: string;
  topic: string;
  kicker: string;
  title: string;
  href: string;
  external: boolean;
}

export interface NavLink {
  href: string;
  label: string;
  external: boolean;
}

export const DEFAULT_SITE_URL = "https://netzradar.mirkandeniz.de";

export const LEGACY_HOSTS: readonly string[] = ["netzradar.vercel.app"];

export const PROJECT_PATH = "/projects/netzradar";

export const PORTFOLIO_URL = "https://mirkandeniz.de";

export const PORTFOLIO_LABEL = "mirkandeniz.de";

export const PORTFOLIO_PROJECTS_URL = `${PORTFOLIO_URL}/projekte`;

export const PORTFOLIO_HOME_URL = `${PORTFOLIO_URL}/#start`;

export const SITE_NAME = "NetzRadar";

export const OWNER_NAME = "Mirkan Deniz Günkaya";

export const GITHUB_PROFILE_URL = "https://github.com/mirkan-morgenfels-ai";

export const REPO_URL = "https://github.com/mirkan-morgenfels-ai/netzradar";

export const PROJECTS: readonly Project[] = [
  {
    slug: "depotdoktor",
    number: "01",
    topic: "Finanzdaten",
    kicker: "Projekt 01",
    title: "DepotDoktor",
    href: "https://depotdoktor.mirkandeniz.de/projects/depotdoktor",
    external: true,
  },
  {
    slug: "kontoklar",
    number: "02",
    topic: "Maschinelles Lernen",
    kicker: "Projekt 02",
    title: "KontoKlar",
    href: "https://kontoklar.mirkandeniz.de/projects/kontoklar",
    external: true,
  },
  {
    slug: "netzradar",
    number: "03",
    topic: "Graph-ML",
    kicker: "Projekt 03",
    title: "NetzRadar",
    href: PROJECT_PATH,
    external: false,
  },
];

export const NAV_LINKS: readonly NavLink[] = [
  ...PROJECTS.map((project) => ({ href: project.href, label: project.title, external: project.external })),
  { href: PORTFOLIO_PROJECTS_URL, label: "Alle Projekte", external: true },
];

export const LEGAL_LINKS: readonly NavLink[] = [
  { href: "/impressum", label: "Impressum", external: false },
  { href: "/datenschutz", label: "Datenschutz", external: false },
  { href: "/nutzungsbedingungen", label: "Nutzungsbedingungen", external: false },
];

export const SITEMAP_PATHS: readonly string[] = [PROJECT_PATH, ...LEGAL_LINKS.map((link) => link.href)];

export function isActivePath(pathname: string | null, href: string): boolean {
  if (pathname === null) return false;
  return pathname === href || pathname.startsWith(`${href}/`);
}

export function isRepoPath(value: string): boolean {
  return /^[A-Za-z0-9_.-]+(\/[A-Za-z0-9_.-]+)+$/.test(value) && !value.split("/").includes("..");
}

export function repoUrl(path = "", kind: "blob" | "tree" = "blob"): string {
  const trimmed = path.replace(/^\/+/, "");
  return trimmed === "" ? REPO_URL : `${REPO_URL}/${kind}/main/${trimmed}`;
}

export const README_URL = `${REPO_URL}#readme`;

export const RUNS_URL = repoUrl("docs/runs", "tree");

export const LICENSE_URL = repoUrl("LICENSE");

export function siteUrl(value: string | undefined = process.env.NEXT_PUBLIC_SITE_URL): URL {
  const trimmed = value?.trim();
  return new URL(trimmed ? trimmed : DEFAULT_SITE_URL);
}
