import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { CHART_COLORS } from "@portfolio/charts/theme";
import {
  OG_DESCRIPTION,
  OG_IMAGE_PATH,
  OG_IMAGE_SIZE,
  OG_IMAGE_VERSION,
  OG_SUBTITLE,
  documentTitle,
  pageMetadata,
} from "../metadata";
import { SITE_NAME } from "../site";

const PREVIEW_IMAGE_SOURCES = [
  "../../app/opengraph-image.tsx",
  "../../components/BrandMark.tsx",
  "../../components/site/netMotif.ts",
  "../og-glyphs.ts",
];
const PREVIEW_IMAGE_FINGERPRINT = "da3769f8b2ae6e43";

function previewImageFingerprint(): string {
  const hash = createHash("sha256");
  for (const source of PREVIEW_IMAGE_SOURCES) {
    hash.update(readFileSync(fileURLToPath(new URL(source, import.meta.url)), "utf8").replace(/\r\n/g, "\n"));
  }
  hash.update(JSON.stringify([SITE_NAME, OG_SUBTITLE, OG_IMAGE_SIZE, CHART_COLORS]));
  return hash.digest("hex").slice(0, 16);
}

describe("pageMetadata", () => {
  it("adds canonical, Open Graph url and site name to a page in the title template", () => {
    expect(pageMetadata({ title: "Impressum", description: "Kontakt und Haftung.", path: "/impressum" })).toEqual({
      title: "Impressum",
      description: "Kontakt und Haftung.",
      alternates: { canonical: "/impressum" },
      openGraph: {
        title: "Impressum · NetzRadar",
        description: "Kontakt und Haftung.",
        url: "/impressum",
        siteName: "NetzRadar",
        locale: "de_DE",
        type: "website",
        images: [
          {
            url: "/opengraph-image?v=4",
            type: "image/png",
            width: 1200,
            height: 630,
            alt: "NetzRadar – Anomalie-Erkennung in Transaktionsnetzwerken: GCN und GraphSAGE gegen Baselines, zeitlicher Split, PR-AUC",
          },
        ],
      },
    });
  });

  it("ties the preview image URL to its inputs: bump OG_IMAGE_VERSION and the fingerprint together", () => {
    expect({ version: OG_IMAGE_VERSION, fingerprint: previewImageFingerprint() }).toEqual({
      version: 4,
      fingerprint: PREVIEW_IMAGE_FINGERPRINT,
    });
    expect(OG_IMAGE_PATH).toBe(`/opengraph-image?v=${OG_IMAGE_VERSION}`);
  });

  it("keeps an absolute title without the template", () => {
    const metadata = pageMetadata({
      title: "Projekte · Mirkan Deniz Günkaya",
      absoluteTitle: true,
      description: "Drei Projekte.",
      path: "/",
    });
    expect(metadata.title).toEqual({ absolute: "Projekte · Mirkan Deniz Günkaya" });
    expect(metadata.openGraph?.title).toBe("Projekte · Mirkan Deniz Günkaya");
  });

  it("builds document titles like the layout template", () => {
    expect(documentTitle("Datenschutzerklärung")).toBe("Datenschutzerklärung · NetzRadar");
    expect(documentTitle("NetzRadar – Anomalie-Erkennung in Transaktionsnetzwerken", true)).toBe(
      "NetzRadar – Anomalie-Erkennung in Transaktionsnetzwerken",
    );
  });

  it("describes the link preview without claims about runtime requests", () => {
    expect(OG_DESCRIPTION).toContain("MLP-Kontrolle");
    expect(OG_DESCRIPTION).not.toContain("Laufzeit-Anfragen");
  });
});
