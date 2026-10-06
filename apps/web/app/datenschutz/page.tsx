import type { Metadata } from "next";
import { OPERATOR, PRIVACY_SHORT } from "@portfolio/legal";
import { LegalPage, LegalSection } from "@/components/LegalPage";

export const metadata: Metadata = { title: "Datenschutzerklärung" };

export default function DatenschutzPage() {
  return (
    <LegalPage title="Datenschutzerklärung" updated={OPERATOR.lastUpdated}>
      <LegalSection title="1. Verantwortlicher">
        <p>
          Verantwortlich für die Datenverarbeitung auf dieser Seite im Sinne der Datenschutz-Grundverordnung (DSGVO) ist{" "}
          {OPERATOR.name}, {OPERATOR.city}. Kontakt per E-Mail:{" "}
          <a href={`mailto:${OPERATOR.email}`} className="text-moss underline">
            {OPERATOR.email}
          </a>
          .
        </p>
      </LegalSection>

      <LegalSection title="2. Das Wichtigste in Kürze">
        <p>
          Diese Seite ist ein privates, nicht-kommerzielles Projekt. Sie hat keine Eingabefelder, keine Upload-Funktion,
          keine Anmeldung und keine Kontaktformulare. Sie verwendet keine Cookies, keine Analyse- oder Tracking-Dienste und
          keine Werbung. Die einzige Verarbeitung personenbezogener Daten findet technisch bedingt beim Hosting statt
          (Abschnitt 4).
        </p>
      </LegalSection>

      <LegalSection title="3. Keine Eingaben, nur statische Dateien">
        <p>{PRIVACY_SHORT}</p>
        <p>
          Konkret: NetzRadar zeigt ausschließlich Ergebnisse, die vorab außerhalb dieser Seite berechnet wurden. Beim Aufruf
          lädt Ihr Browser nur eigene statische Dateien dieser Seite (HTML, JavaScript, CSS, Grafiken und die vorberechneten
          JSON-Dateien) von dem Server, über den die Seite ausgeliefert wird (Vercel, Abschnitt 4). Es werden keine Inhalte
          Dritter nachgeladen, kein Modell ausgeführt und keine Programmierschnittstelle aufgerufen. Interaktionen innerhalb
          der Graph-Ansicht und der Diagramme, etwa das Auswählen oder Vergrößern eines Ausschnitts, werden nur in Ihrem
          Browser verarbeitet und weder übertragen noch gespeichert oder protokolliert. Sie können das in den
          Entwicklerwerkzeugen Ihres Browsers (Netzwerk-Tab) nachvollziehen.
        </p>
        <p>
          Die angezeigten Daten stammen aus synthetischen beziehungsweise öffentlichen Forschungsdatensätzen. Sie enthalten
          keine Angaben über Sie.
        </p>
      </LegalSection>

      <LegalSection title="4. Hosting und Server-Logdaten">
        <p>
          Die Seite wird bei Vercel Inc., 440 N Barranca Ave #4133, Covina, CA 91723, USA, gehostet. Beim Aufruf der Seite
          verarbeitet Vercel technisch notwendige Daten, um die Seite auszuliefern, insbesondere IP-Adresse, Datum und Uhrzeit
          der Anfrage, aufgerufene Adresse, übertragene Datenmenge, Browsertyp und Betriebssystem sowie die verweisende Seite.
          Rechtsgrundlage ist Art. 6 Abs. 1 lit. f DSGVO; das berechtigte Interesse liegt im sicheren und stabilen Betrieb der
          Seite. Vercel verarbeitet diese Daten als Auftragsverarbeiter; die Übermittlung in die USA stützt sich auf die
          Standardvertragsklauseln der EU-Kommission und die Zertifizierung von Vercel unter dem EU-US Data Privacy Framework.
          Einzelheiten stehen in der Datenschutzerklärung von Vercel unter https://vercel.com/legal/privacy-policy.
        </p>
        <p>
          Der Betreiber selbst wertet diese Logdaten nicht aus. Vercel Web Analytics und Speed Insights sind nicht aktiviert.
        </p>
      </LegalSection>

      <LegalSection title="5. Kontakt per E-Mail">
        <p>
          Wenn Sie per E-Mail Kontakt aufnehmen, werden die von Ihnen mitgeteilten Daten (E-Mail-Adresse, Inhalt der
          Nachricht) zur Bearbeitung der Anfrage verarbeitet (Art. 6 Abs. 1 lit. f DSGVO, bei vorvertraglichen Anfragen lit.
          b). Die Daten werden gelöscht, sobald die Anfrage erledigt ist und keine gesetzlichen Aufbewahrungspflichten
          entgegenstehen.
        </p>
      </LegalSection>

      <LegalSection title="6. Externe Links">
        <p>
          Die Startseite verlinkt auf die Projekte DepotDoktor und KontoKlar, die unter eigenen Adressen betrieben werden und
          eigene Datenschutzerklärungen haben. Beim Anklicken eines externen Links verlassen Sie diese Seite; für die
          Datenverarbeitung dort gilt die Datenschutzerklärung des jeweiligen Angebots. Beim Aufruf dieser Seite werden keine
          Inhalte Dritter (Schriften, Skripte, Videos, Karten) nachgeladen.
        </p>
      </LegalSection>

      <LegalSection title="7. Ihre Rechte">
        <p>
          Sie haben gegenüber dem Verantwortlichen das Recht auf Auskunft (Art. 15 DSGVO), Berichtigung (Art. 16),
          Löschung (Art. 17), Einschränkung der Verarbeitung (Art. 18), Datenübertragbarkeit (Art. 20) und Widerspruch gegen
          Verarbeitungen auf Grundlage von Art. 6 Abs. 1 lit. f DSGVO (Art. 21). Außerdem können Sie sich bei einer
          Datenschutz-Aufsichtsbehörde beschweren, in Bayern beim Bayerischen Landesamt für Datenschutzaufsicht (BayLDA),
          Promenade 18, 91522 Ansbach.
        </p>
      </LegalSection>

      <LegalSection title="8. Änderungen">
        <p>
          Diese Datenschutzerklärung wird angepasst, wenn sich die Seite oder die Rechtslage ändert. Es gilt die jeweils hier
          veröffentlichte Fassung.
        </p>
      </LegalSection>
    </LegalPage>
  );
}
