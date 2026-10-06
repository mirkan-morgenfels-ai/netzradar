export interface Project {
  slug: string;
  code: string;
  title: string;
  description: string;
  href: string;
  external: boolean;
}

export interface NavLink {
  href: string;
  label: string;
}

export const DEFAULT_SITE_URL = "http://localhost:3000";

export const PROJECTS: readonly Project[] = [
  {
    slug: "depotdoktor",
    code: "K1",
    title: "DepotDoktor",
    description:
      "Depot-Steuer- und Performance-Analyzer für Broker-CSV-Exporte. Die Auswertung läuft vollständig im Browser.",
    href: "https://ai-project-1-web.vercel.app/projects/depotdoktor",
    external: true,
  },
  {
    slug: "kontoklar",
    code: "K2",
    title: "KontoKlar",
    description: "Kategorisiert Bankumsätze aus CSV-Exporten und zeigt, wohin das Geld geht.",
    href: "https://kontoklar-eight.vercel.app/projects/kontoklar",
    external: true,
  },
  {
    slug: "netzradar",
    code: "K3",
    title: "NetzRadar",
    description:
      "Anomalie-Erkennung in Transaktionsnetzwerken: klassische Baseline gegen Graph Neural Networks, mit zeitlichem Split und PR-AUC.",
    href: "/projects/netzradar",
    external: false,
  },
];

export const NAV_LINKS: readonly NavLink[] = [
  { href: "/", label: "Start" },
  { href: "/projects/netzradar", label: "NetzRadar" },
];

export const LEGAL_LINKS: readonly NavLink[] = [
  { href: "/impressum", label: "Impressum" },
  { href: "/datenschutz", label: "Datenschutz" },
  { href: "/nutzungsbedingungen", label: "Nutzungsbedingungen" },
];

export function siteUrl(value: string | undefined = process.env.NEXT_PUBLIC_SITE_URL): URL {
  const trimmed = value?.trim();
  return new URL(trimmed ? trimmed : DEFAULT_SITE_URL);
}
