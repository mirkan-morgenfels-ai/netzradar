import type { Metadata } from "next";
import type { ReactNode } from "react";
import Link from "next/link";
import { LEGAL_LINKS, NAV_LINKS, siteUrl } from "@/lib/site";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: siteUrl(),
  title: { default: "NetzRadar", template: "%s · NetzRadar" },
  description:
    "Methodenstudie zur Anomalie-Erkennung in Transaktionsnetzwerken: klassische Baseline gegen Graph Neural Networks, zeitlicher Split, PR-AUC. Alle Ergebnisse sind vorab berechnet.",
  openGraph: {
    title: "NetzRadar",
    description:
      "Anomalie-Erkennung in Transaktionsnetzwerken mit vorab berechneten Ergebnissen. Keine Eingaben, keine Laufzeit-Anfragen.",
    locale: "de_DE",
    type: "website",
  },
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
            <ul className="flex flex-wrap gap-x-5 gap-y-1 text-sm">
              {NAV_LINKS.map((link) => (
                <li key={link.href}>
                  <Link href={link.href} className="hover:text-gold-deep">
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </nav>
        </header>
        <main id="main" className="mx-auto max-w-6xl px-6 py-10">
          {children}
        </main>
        <footer className="mx-auto flex max-w-6xl flex-wrap gap-x-4 gap-y-1 px-6 py-8 text-xs text-stone">
          <span>© 2026 Mirkan Deniz Günkaya · Privates, nicht-kommerzielles Projekt · Quellcode unter MIT-Lizenz</span>
          <nav aria-label="Rechtliches" className="flex flex-wrap gap-x-4 gap-y-1">
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
