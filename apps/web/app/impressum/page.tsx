import type { Metadata } from "next";
import Link from "next/link";
import { OPERATOR } from "@portfolio/legal";
import { LegalPage, LegalSection } from "@/components/LegalPage";

export const metadata: Metadata = { title: "Impressum" };

export default function ImpressumPage() {
  return (
    <LegalPage title="Impressum" updated={OPERATOR.lastUpdated}>
      <LegalSection title="Verantwortlich für diese Seite">
        <p>
          {OPERATOR.name}
          <br />
          {OPERATOR.city}
        </p>
        <p>
          Kontakt:{" "}
          <a href={`mailto:${OPERATOR.email}`} className="text-moss underline">
            {OPERATOR.email}
          </a>
        </p>
        <p className="text-stone">
          Diese Seite ist ein privates, nicht-kommerzielles Portfolio- und Lernprojekt. Es werden keine Waren oder
          Dienstleistungen angeboten, es gibt keine Werbung und keine Bezahlfunktion. Kontaktaufnahme bitte per E-Mail.
        </p>
      </LegalSection>

      <LegalSection title="Haftung für Inhalte">
        <p>
          Die Inhalte dieser Seite wurden mit Sorgfalt erstellt. Für Richtigkeit, Vollständigkeit und Aktualität wird keine
          Gewähr übernommen. NetzRadar ist eine Methodenstudie zur Anomalie-Erkennung auf synthetischen beziehungsweise
          öffentlichen Forschungsdatensätzen. Die gezeigten Ergebnisse sind vorab berechnete Ausgaben statistischer Modelle.
          Sie sind keine Aussage über reale Konten oder Personen und stellen keine Anlage-, Rechts- oder Compliance-Beratung
          dar. Näheres regeln die{" "}
          <Link href="/nutzungsbedingungen" className="text-moss underline">
            Nutzungsbedingungen
          </Link>
          .
        </p>
      </LegalSection>

      <LegalSection title="Haftung für Links">
        <p>
          Diese Seite kann Links auf externe Webseiten enthalten, auf deren Inhalte kein Einfluss besteht. Für diese Inhalte
          ist stets der jeweilige Anbieter verantwortlich. Zum Zeitpunkt der Verlinkung waren keine Rechtsverstöße erkennbar.
          Bei Bekanntwerden von Rechtsverletzungen werden betroffene Links entfernt.
        </p>
      </LegalSection>

      <LegalSection title="Urheberrecht und Lizenz">
        <p>
          Der Quellcode des Projekts steht unter der MIT-Lizenz. Texte und Gestaltung dieser Seite unterliegen dem deutschen
          Urheberrecht. Die verwendeten Datensätze unterliegen ihren eigenen Lizenzen und sind nicht Teil des Quellcodes.
          Genannte Marken und Produktnamen (etwa IBM, Elliptic, Kaggle) gehören ihren jeweiligen Inhabern; es besteht keine
          Verbindung zu diesen Unternehmen.
        </p>
      </LegalSection>
    </LegalPage>
  );
}
