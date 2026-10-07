import type { Metadata } from "next";
import { OPERATOR } from "@portfolio/legal";
import { ExternalLink } from "@/components/ExternalLink";
import { LegalPage, LegalSection } from "@/components/LegalPage";
import { pageMetadata } from "@/lib/metadata";
import { LICENSE_URL, REPO_URL } from "@/lib/site";

export const metadata: Metadata = pageMetadata({
  title: "Nutzungsbedingungen",
  description:
    "Nutzungsbedingungen von NetzRadar: Methodenstudie ohne Beratung, keine Gewähr für Ergebnisse, Lizenzen der Datensätze und des Quellcodes.",
  path: "/nutzungsbedingungen",
});

export default function NutzungsbedingungenPage() {
  return (
    <LegalPage title="Nutzungsbedingungen" updated={OPERATOR.lastUpdated}>
      <LegalSection title="1. Geltungsbereich">
        <p>
          Diese Nutzungsbedingungen gelten für die Nutzung dieser Seite, insbesondere der Projektseite NetzRadar, sowie der
          zugehörigen Texte, Grafiken und Kennzahlen. Betreiber ist {OPERATOR.name} (siehe Impressum). Mit der Nutzung
          erkennen Sie diese Bedingungen an. Die Nutzung ist kostenlos; ein Vertrag über eine entgeltliche Leistung kommt
          nicht zustande.
        </p>
      </LegalSection>

      <LegalSection title="2. Zweck: Methodenstudie">
        <p>
          NetzRadar ist ein privates Portfolio- und Lernprojekt. Es vergleicht Verfahren der Anomalie-Erkennung in
          Transaktionsnetzwerken, eine klassische Baseline und Graph Neural Networks, auf synthetischen beziehungsweise
          öffentlichen Forschungsdatensätzen. Alle gezeigten Ergebnisse wurden vorab berechnet. Die Seite nimmt keine
          Eingaben entgegen und bewertet keine Daten, die Sie bereitstellen.
        </p>
      </LegalSection>

      <LegalSection title="3. Keine Beratung, keine Prüfung realer Konten oder Personen">
        <p>
          Sämtliche Inhalte, Anomalie-Scores, Kennzahlen, Grafiken und Texte stellen keine Anlage-, Rechts- oder
          Compliance-Beratung dar. Ein hoher Anomalie-Score ist die Ausgabe eines statistischen Modells und keine
          Feststellung, dass eine Transaktion oder ein Konto rechtswidrig ist. Es werden keine realen Konten oder Personen
          geprüft. Die Inhalte ersetzen keine geldwäscherechtliche Prüfung und dürfen nicht als Grundlage für Entscheidungen
          über reale Konten, Transaktionen oder Personen verwendet werden.
        </p>
      </LegalSection>

      <LegalSection title="4. Keine Gewähr für Ergebnisse">
        <p>
          Für Richtigkeit, Vollständigkeit und Aktualität der Ergebnisse wird keine Gewähr übernommen. Die Kennzahlen hängen
          vom Datensatz, vom zeitlichen Split, von den Hyperparametern und vom Zufallsstartwert ab und lassen sich nicht ohne
          Weiteres auf andere Daten übertragen. Synthetische Daten bilden reale Transaktionsmuster nur näherungsweise ab. Die
          Seite wird ohne Zusicherung einer bestimmten Verfügbarkeit bereitgestellt und kann jederzeit geändert oder
          eingestellt werden.
        </p>
      </LegalSection>

      <LegalSection title="5. Haftung">
        <p>
          Der Betreiber haftet unbeschränkt für Schäden aus der Verletzung des Lebens, des Körpers oder der Gesundheit sowie
          für Schäden, die auf Vorsatz oder grober Fahrlässigkeit beruhen. Im Übrigen ist die Haftung ausgeschlossen. Da die
          Nutzung unentgeltlich erfolgt, haftet der Betreiber für sonstige Schäden nur, soweit er einen Mangel arglistig
          verschwiegen hat (§§ 521, 599 BGB entsprechend). Insbesondere wird nicht gehaftet für Vermögensschäden oder
          entgangenen Gewinn, die aus der Verwendung der Ergebnisse entstehen.
        </p>
      </LegalSection>

      <LegalSection title="6. Datensätze und Lizenzen">
        <p>
          Verwendet beziehungsweise vorgesehen sind die folgenden Datensätze. Sie unterliegen ihren eigenen Lizenzen und
          sind nicht Teil des Quellcodes. Ausgenommen sind die Exporte des eigenen synthetischen Netzes in data/k3/
          (MIT-Lizenz). Rohdaten werden weder im Repository noch auf dieser Seite veröffentlicht.
        </p>
        <ul className="list-disc space-y-2 pl-6">
          <li>
            Eigenes synthetisches Netz, erzeugt mit einem eigenen Generator (Python, NumPy) im Quellcode des Projekts:
            MIT-Lizenz. Derzeit stammen alle auf dieser Seite gezeigten Ergebnisse aus diesem Netz.
          </li>
          <li>
            IBM Transactions for Anti-Money Laundering (synthetisch): vorgesehen, derzeit nicht verwendet. Daten unter
            CDLA-Sharing-1.0, Bezug über Kaggle und das GitHub-Repository IBM/AML-Data.
          </li>
          <li>
            Elliptic Bitcoin Dataset: CC BY-NC-ND 4.0. Der Datensatz ist nur für einen lokalen Methodenvergleich
            vorgesehen. Derzeit wird daraus nichts veröffentlicht. Rohdaten, bearbeitete Fassungen und Ausschnitte werden
            nie veröffentlicht, künftig höchstens aggregierte Kennzahlen. Bei abweichenden Lizenzangaben in Spiegelungen
            gilt die restriktivere.
          </li>
        </ul>
        <p>
          Wenn Sie einen der Datensätze selbst nutzen möchten, beziehen Sie ihn bitte aus der jeweiligen Originalquelle und
          beachten Sie dessen Lizenz.
        </p>
      </LegalSection>

      <LegalSection title="7. Quellcode und Lizenz">
        <p>
          Der Quellcode ist öffentlich unter{" "}
          <ExternalLink href={REPO_URL}>github.com/mirkan-morgenfels-ai/netzradar</ExternalLink> verfügbar und steht unter
          der <ExternalLink href={LICENSE_URL}>MIT-Lizenz</ExternalLink>. Die Lizenz enthält einen eigenen Haftungs- und
          Gewährleistungsausschluss, der für die Nutzung des Quellcodes gilt.
        </p>
      </LegalSection>

      <LegalSection title="8. Schlussbestimmungen">
        <p>
          Es gilt das Recht der Bundesrepublik Deutschland. Sollten einzelne Bestimmungen unwirksam sein, bleibt die
          Wirksamkeit der übrigen Bestimmungen unberührt. Der Betreiber kann diese Bedingungen mit Wirkung für die Zukunft
          ändern; es gilt die jeweils hier veröffentlichte Fassung.
        </p>
      </LegalSection>
    </LegalPage>
  );
}
