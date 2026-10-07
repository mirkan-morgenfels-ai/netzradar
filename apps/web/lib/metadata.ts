import type { Metadata } from "next";
import { SITE_NAME } from "./site";

export const OG_LOCALE = "de_DE";

export const OG_SUBTITLE = "Anomalie-Erkennung in Transaktionsnetzwerken";

export const OG_IMAGE_VERSION = 4;

export const OG_IMAGE_PATH = `/opengraph-image?v=${OG_IMAGE_VERSION}`;

export const OG_IMAGE_SIZE = { width: 1200, height: 630 } as const;

export const OG_IMAGE_ALT = `${SITE_NAME} – ${OG_SUBTITLE}: GCN und GraphSAGE gegen Baselines, zeitlicher Split, PR-AUC`;

export const OG_DESCRIPTION =
  "GCN und GraphSAGE gegen Baselines und eine MLP-Kontrolle auf einem synthetischen Transaktionsnetz: zeitlicher Split, PR-AUC, Einordnung mit Homophilie-Vorbehalt. Alle Ergebnisse sind vorab berechnet.";

export interface PageMetadataInput {
  title: string;
  description: string;
  path: string;
  absoluteTitle?: boolean;
}

export function documentTitle(title: string, absolute = false): string {
  return absolute ? title : `${title} · ${SITE_NAME}`;
}

export function pageMetadata({ title, description, path, absoluteTitle = false }: PageMetadataInput): Metadata {
  return {
    title: absoluteTitle ? { absolute: title } : title,
    description,
    alternates: { canonical: path },
    openGraph: {
      title: documentTitle(title, absoluteTitle),
      description,
      url: path,
      siteName: SITE_NAME,
      locale: OG_LOCALE,
      type: "website",
      images: [{ url: OG_IMAGE_PATH, type: "image/png", ...OG_IMAGE_SIZE, alt: OG_IMAGE_ALT }],
    },
  };
}
