import type { Metadata } from "next";
import type { ReactNode } from "react";
import Link from "next/link";
import { ExternalLink } from "@/components/ExternalLink";
import { NavLinks } from "@/components/NavLinks";
import { OG_DESCRIPTION, OG_LOCALE } from "@/lib/metadata";
import { LEGAL_LINKS, NAV_LINKS, REPO_URL, SITE_NAME, siteUrl } from "@/lib/site";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: siteUrl(),
  title: { default: SITE_NAME, template: `%s · ${SITE_NAME}` },
  description:
    "Methodenstudie zur Anomalie-Erkennung in Transaktionsnetzwerken: klassische Baseline gegen Graph Neural Networks, zeitlicher Split, PR-AUC. Alle Ergebnisse sind vorab berechnet.",
  openGraph: {
    title: SITE_NAME,
    description: OG_DESCRIPTION,
    siteName: SITE_NAME,
    locale: OG_LOCALE,
    type: "website",
  },
  twitter: { card: "summary_large_image" },
  robots: { index: true, follow: true },
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="de">
      <body className="min-h-screen bg-paper text-ink antialiased">
        <a href="#main" className="skip-link">
          Zum Inhalt springen
        </a>
        <header className="border-b border-line bg-surface">
          <nav
            aria-label="Hauptnavigation"
            className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-x-6 gap-y-2 px-6 py-4"
          >
            <Link href="/" className="font-serif text-lg whitespace-nowrap">
              Mirkan Deniz Günkaya
            </Link>
            <NavLinks links={NAV_LINKS} />
          </nav>
        </header>
        <main id="main" className="mx-auto max-w-6xl px-6 py-10">
          {children}
        </main>
        <footer className="mx-auto flex max-w-6xl flex-wrap gap-x-4 gap-y-3 px-6 py-8 text-xs text-stone">
          <span>
            © 2026 Mirkan Deniz Günkaya · Privates, nicht-kommerzielles Projekt ·{" "}
            <ExternalLink
              href={REPO_URL}
              className="underline underline-offset-2 hover:text-gold-deep"
              testId="footer-repo-link"
            >
              Quellcode auf GitHub
            </ExternalLink>{" "}
            (MIT-Lizenz)
          </span>
          <nav aria-label="Rechtliches" className="flex flex-wrap gap-x-4 gap-y-3">
            {LEGAL_LINKS.map((link) => (
              <Link key={link.href} href={link.href} className="hover:text-gold-deep">
                {link.label}
              </Link>
            ))}
          </nav>
        </footer>
      </body>
    </html>
  );
}
