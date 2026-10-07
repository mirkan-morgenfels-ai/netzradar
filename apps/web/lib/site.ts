export interface Project {
  slug: string;
  code: string;
  kicker: string;
  title: string;
  description: string;
  href: string;
  external: boolean;
  repo: string;
}

export interface NavLink {
  href: string;
  label: string;
  external: boolean;
}

export const DEFAULT_SITE_URL = "https://netzradar.vercel.app";

export const SITE_NAME = "NetzRadar";

export const REPO_URL = "https://github.com/mirkan-morgenfels-ai/netzradar";

export const PROJECTS: readonly Project[] = [
  {
    slug: "depotdoktor",
    code: "K1",
    kicker: "Projekt K1",
    title: "DepotDoktor",
    description:
      "Depot-Steuer- und Performance-Analyzer für Broker-CSV-Exporte. Die Auswertung läuft vollständig im Browser.",
    href: "https://depotdoktor.vercel.app/projects/depotdoktor",
    external: true,
    repo: "https://github.com/mirkan-morgenfels-ai/depotdoktor",
  },
  {
    slug: "kontoklar",
    code: "K2",
    kicker: "Projekt K2",
    title: "KontoKlar",
    description: "Kategorisiert Bankumsätze aus CSV-Exporten und zeigt, wohin das Geld geht.",
    href: "https://kontoklar-eight.vercel.app/projects/kontoklar",
    external: true,
    repo: "https://github.com/mirkan-morgenfels-ai/kontoklar",
  },
  {
    slug: "netzradar",
    code: "K3",
    kicker: "Projekt K3",
    title: "NetzRadar",
    description:
      "Anomalie-Erkennung in Transaktionsnetzwerken: klassische Baseline gegen Graph Neural Networks, mit zeitlichem Split und PR-AUC.",
    href: "/projects/netzradar",
    external: false,
    repo: REPO_URL,
  },
];

export const NAV_LINKS: readonly NavLink[] = [
  { href: "/", label: "Start", external: false },
  ...PROJECTS.map((project) => ({ href: project.href, label: project.title, external: project.external })),
];

export const LEGAL_LINKS: readonly NavLink[] = [
  { href: "/impressum", label: "Impressum", external: false },
  { href: "/datenschutz", label: "Datenschutz", external: false },
  { href: "/nutzungsbedingungen", label: "Nutzungsbedingungen", external: false },
];

export const SITEMAP_PATHS: readonly string[] = ["/", "/projects/netzradar", ...LEGAL_LINKS.map((link) => link.href)];

export function isActivePath(pathname: string | null, href: string): boolean {
  if (pathname === null) return false;
  if (href === "/") return pathname === "/";
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
