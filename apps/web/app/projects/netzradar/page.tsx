import type { Metadata } from "next";
import Link from "next/link";
import type { ReactNode } from "react";
import { PrCurveChart } from "@portfolio/charts";
import { Disclaimer, PRIVACY_SHORT } from "@portfolio/legal";
import { StatTile } from "@portfolio/ui";
import { MetricsTable } from "@/components/netzradar/MetricsTable";
import { NetzRadarExplorer } from "@/components/netzradar/NetzRadarExplorer";
import { NodeSymbol } from "@/components/netzradar/NodeSymbol";
import { curveSeries, prevalenceLevel } from "@/lib/netzradar/curves";
import { netzRadarData } from "@/lib/netzradar/data";
import {
  formatDecimal,
  formatInteger,
  formatIsoDate,
  formatIsoTimeUtc,
  formatPercent,
  formatShare,
  formatStepRange,
  LABEL_TEXT,
  METHOD_TEXT,
} from "@/lib/netzradar/format";
import { countLabels, seedNodes, summarizeComponents } from "@/lib/netzradar/graph";
import {
  beatsRandomRankings,
  bestRun,
  featureNames,
  hitsAtRecall,
  illicitShareAmongLabelled,
  labelledTestCount,
  labelTotal,
  meanDegree,
  numberParameter,
  parameterEntries,
  zeroMadFeatures,
} from "@/lib/netzradar/summary";
import { NODE_LABELS, type EvaluationInfo, type Run } from "@/lib/netzradar/types";

export const metadata: Metadata = {
  title: { absolute: "NetzRadar" },
  description:
    "Fallstudie zur Anomalie-Erkennung in Transaktionsnetzwerken: robuste Z-Scores und Isolation Forest auf einem synthetischen Netz, strikt zeitlicher Split, PR-AUC als Hauptmetrik. Alle Ergebnisse sind vorab berechnet.",
};

const SECTION_TITLE = "font-serif text-2xl";
const PROSE = "max-w-3xl space-y-3 leading-relaxed";
const FEW_HITS_RECALL = 0.1;

function Section({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <section aria-labelledby={`${id}-title`} className="space-y-5 border-t border-line pt-10">
      <h2 id={`${id}-title`} className={SECTION_TITLE}>
        {title}
      </h2>
      {children}
    </section>
  );
}

function Formula({ children }: { children: ReactNode }) {
  return (
    <p className="overflow-x-auto rounded-md border border-line bg-surface px-4 py-3 font-serif text-base whitespace-nowrap">
      {children}
    </p>
  );
}

function ParameterList({ entries }: { entries: Array<[string, string]> }) {
  return (
    <dl className="mt-3 grid gap-x-6 gap-y-1 text-xs sm:grid-cols-2">
      {entries.map(([key, value]) => (
        <div key={key} className="flex min-w-0 gap-3 border-t border-line py-1">
          <dt className="shrink-0 font-mono text-stone">{key}</dt>
          <dd className="min-w-0 break-words tabular-nums">{value}</dd>
        </div>
      ))}
    </dl>
  );
}

function randomSentence(run: Run, evaluation: EvaluationInfo): string {
  const quantile = formatDecimal(evaluation.randomPrAucQ95);
  if (beatsRandomRankings(run.prAuc, evaluation)) {
    return `${run.displayName}: PR-AUC ${formatDecimal(run.prAuc)}, über dem 95-%-Quantil zufälliger Rangfolgen (${quantile}). Der Score ordnet auffällige Knoten also besser als Zufall.`;
  }
  return `${run.displayName}: PR-AUC ${formatDecimal(run.prAuc)}, nicht über dem 95-%-Quantil zufälliger Rangfolgen (${quantile}). Von einer zufälligen Rangfolge (Erwartungswert ${formatDecimal(
    evaluation.randomPrAucExpected,
  )}) ist das nicht zu unterscheiden.`;
}

function flaggedSentence(run: Run): string {
  return `${METHOD_TEXT[run.method]} ${formatInteger(run.accuracyFlagged)}, davon ${formatInteger(
    run.accuracyTruePositives,
  )} auffällig`;
}

function recallSentence(run: Run, positives: number): string {
  if (run.recallAtPrecision50 === 0) return `${METHOD_TEXT[run.method]}: bei keiner Schwelle`;
  const hits = hitsAtRecall(run.recallAtPrecision50, positives);
  return `${METHOD_TEXT[run.method]}: Recall ${formatDecimal(run.recallAtPrecision50)}, also etwa ${formatInteger(hits)} von ${formatInteger(positives)} auffälligen Testknoten`;
}

export default function NetzRadarPage() {
  const { metrics, nodes, edges } = netzRadarData;
  const { dataset, split, evaluation, runs } = metrics;
  const zscore = runs.find((run) => run.method === "zscore");
  const iforest = runs.find((run) => run.method === "iforest");
  const best = bestRun(runs);
  const seeds = seedNodes(nodes.nodes);
  const seedLabels = countLabels(seeds);
  const others = nodes.nodes.filter((node) => node.seedRank === null);
  const components = summarizeComponents(nodes.nodes, edges.edges);
  const labelled = labelledTestCount(evaluation);
  const illicitVisibility = numberParameter(dataset.generator, "illicitVisibility");
  const licitVisibility = numberParameter(dataset.generator, "licitVisibility");
  const localShift = numberParameter(dataset.generator, "localShift");
  const generatorSeed = numberParameter(dataset.generator, "seed");
  const hopNoiseSd = numberParameter(dataset.generator, "hopNoiseSd");
  const zscoreFeatures = zscore ? featureNames(zscore) : [];
  const inactiveFeatures = zscore ? zeroMadFeatures(zscore) : [];
  const { homophily } = dataset;
  const trees = iforest ? numberParameter(iforest.hyperparameters, "nEstimators") : null;
  const contamination = iforest ? numberParameter(iforest.hyperparameters, "contamination") : null;
  const madScale = zscore ? numberParameter(zscore.hyperparameters, "madScale") : null;
  const accuracyThresholds = [...new Set(runs.map((run) => run.accuracyThreshold))];
  const allBelowAllLicit = runs.every((run) => run.accuracy < evaluation.allLicitAccuracy);
  const exportTime = formatIsoTimeUtc(metrics.generatedAt);
  const maxRecallAtPrecision50 = Math.max(...runs.map((run) => run.recallAtPrecision50));
  const hasGnnRun = runs.some((run) => run.method === "gcn" || run.method === "graphsage");
  const explorerDescription = `Netzwerk-Ausschnitt mit ${formatInteger(nodes.nodes.length)} Knoten und ${formatInteger(
    edges.edges.length,
  )} Kanten. Dieselben Startknoten stehen als Text in der Tabelle unter der Grafik.`;

  return (
    <div className="space-y-10">
      <section className="max-w-3xl">
        <p className="text-xs uppercase tracking-widest text-gold-deep">Projekt K3</p>
        <h1 className="mt-2 font-serif text-4xl">NetzRadar</h1>
        <p className="mt-4 text-lg leading-relaxed">
          Fallstudie zur Anomalie-Erkennung in Transaktionsnetzwerken. Die Fragestellung: Wie viel gewinnt ein Modell,
          das die Nachbarschaft einer Transaktion einbezieht, gegenüber einer Baseline auf Einzelmerkmalen, bei gleichem
          Split und gleichen Metriken?
        </p>
        <p className="mt-3 leading-relaxed">
          {hasGnnRun
            ? "Gemessen sind die Baselines und Graph Neural Networks, also Modelle mit Nachbarschaft. Die Einordnung steht weiter unten."
            : "Stand ist Schritt 3 von 5. Gemessen sind bisher nur die beiden Baselines. Die Graph Neural Networks GCN und GraphSAGE, also die Modelle mit Nachbarschaft, folgen in Schritt 4. Die Fragestellung ist deshalb noch nicht beantwortet."}
        </p>
        <p className="mt-3 text-sm text-stone">{PRIVACY_SHORT}</p>
      </section>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4" data-testid="headline-figures">
        {runs.map((run) => (
          <StatTile
            key={run.method}
            label={`PR-AUC ${METHOD_TEXT[run.method]}`}
            value={formatDecimal(run.prAuc)}
            hint={run.featureSet === "local" ? "nur lokale Merkmale" : "lokale Merkmale und Graphmaße"}
          />
        ))}
        <StatTile
          label="Zufällige Rangfolge"
          value={formatDecimal(evaluation.randomPrAucExpected)}
          hint={`PR-AUC im Erwartungswert, 95 % bleiben unter ${formatDecimal(evaluation.randomPrAucQ95)}`}
        />
        <StatTile
          label="Testknoten mit Label"
          value={formatInteger(labelled)}
          hint={`davon ${formatInteger(evaluation.testPositives)} auffällig`}
        />
      </div>

      <Section id="datensatz" title="Datensatz">
        <div className={PROSE}>
          <p>
            Die öffentliche Demo nutzt den Datensatz „{dataset.displayName}“ (Lizenz {dataset.license}), ein eigenes,
            frei veröffentlichbares Netz. Knoten sind Transaktionen, gerichtete Kanten der Geldfluss zwischen ihnen. Der
            Generator (<code className="text-sm break-all">{dataset.source}</code>
            {generatorSeed === null ? "" : `, Seed ${generatorSeed}`}) legt einen gutartigen Hintergrund nach
            Preferential Attachment an, in dem einzelne Knoten viele Kanten haben, und baut darin bekannte
            Geldwäsche-Muster ein: Fan-in, Fan-out, Zyklen und Ketten.
          </p>
        </div>
        <dl className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4" data-testid="dataset-figures">
          {[
            ["Knoten", formatInteger(dataset.nodes)],
            ["Kanten", formatInteger(dataset.edges)],
            ["Zeitschritte", formatInteger(dataset.timeSteps)],
            ["Lokale Merkmale", formatInteger(dataset.features)],
          ].map(([term, value]) => (
            <div key={term} className="rounded-lg border border-line bg-surface p-4">
              <dt className="text-xs uppercase tracking-wide text-stone">{term}</dt>
              <dd className="mt-1 font-serif text-2xl tabular-nums">{value}</dd>
            </div>
          ))}
        </dl>
        <div className="overflow-x-auto rounded-lg border border-line bg-surface">
          <table className="w-full min-w-[24rem] text-sm" data-testid="label-table">
            <caption className="px-4 pt-4 pb-2 text-left text-xs text-stone">Labels im gesamten Netz</caption>
            <thead>
              <tr className="border-b border-line text-left text-xs uppercase tracking-wide text-stone">
                <th scope="col" className="px-4 py-2 font-medium">
                  Label
                </th>
                <th scope="col" className="px-4 py-2 text-right font-medium">
                  Knoten
                </th>
                <th scope="col" className="px-4 py-2 text-right font-medium">
                  Anteil
                </th>
              </tr>
            </thead>
            <tbody>
              {NODE_LABELS.map((label) => (
                <tr key={label} className="border-t border-line tabular-nums">
                  <th scope="row" className="px-4 py-2 text-left font-normal">
                    <span className="inline-flex items-center gap-2">
                      <NodeSymbol label={label} />
                      {LABEL_TEXT[label]}
                    </span>
                  </th>
                  <td className="px-4 py-2 text-right">{formatInteger(dataset.labelCounts[label])}</td>
                  <td className="px-4 py-2 text-right">
                    {formatShare(dataset.labelCounts[label], labelTotal(dataset.labelCounts))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className={PROSE}>
          {illicitVisibility !== null && licitVisibility !== null ? (
            <p>
              Wie bei realen Daten trägt nur ein Teil der Knoten ein Label. Im Generator erhält ein Musterknoten mit
              Wahrscheinlichkeit {formatPercent(illicitVisibility, 0)} das Label auffällig, ein gutartiger Knoten mit
              Wahrscheinlichkeit {formatPercent(licitVisibility, 0)} das Label unauffällig. Alle übrigen bleiben
              unbekannt; unter den unbekannten Knoten stecken also auch Musterknoten.
            </p>
          ) : null}
          <p>
            Die Zahlen gelten nur für dieses Netz. Wie schwer die Aufgabe ist, legen die Annahmen des Generators fest
            {localShift === null
              ? "."
              : `, etwa wie stark sich die meisten lokalen Merkmale der Musterknoten verschieben (localShift = ${formatDecimal(
                  localShift,
                  1,
                )} Standardabweichungen). Einzelne Merkmale weichen stärker ab: Fan-out-Verteiler beim größten Ausgangsanteil, Fan-in-Zubringer bei der geringeren Streuung ihrer Beträge, und runde Beträge sind bei Musterknoten doppelt so häufig.`}
          </p>
          <p>
            Vorgesehen sind zwei weitere Datensätze: IBM Transactions for Anti-Money Laundering (synthetisch, Lizenz
            CDLA-Sharing-1.0) und das Elliptic Bitcoin Dataset (Lizenz CC BY-NC-ND 4.0). Elliptic wird nur lokal für den
            Methodenvergleich ausgewertet. Aus diesem Datensatz werden weder Rohdaten noch bearbeitete Fassungen oder
            Ausschnitte veröffentlicht.
          </p>
        </div>
        {dataset.generator ? (
          <details className="max-w-4xl rounded-lg border border-line bg-surface p-4 text-sm">
            <summary className="cursor-pointer font-medium">
              Alle {parameterEntries(dataset.generator).length} Generator-Parameter
            </summary>
            <ParameterList entries={parameterEntries(dataset.generator)} />
          </details>
        ) : null}
      </Section>

      <Section id="netzwerk" title="Netzwerk-Ausschnitt">
        <div className={PROSE}>
          <p>
            Gezeigt wird die Nachbarschaft bis {nodes.selection.hops} Kanten Abstand um die {formatInteger(seeds.length)}{" "}
            Testknoten mit dem höchsten Isolation-Forest-Score, ungerichtet gezählt und ohne Rücksicht auf das Label
            ausgewählt: {formatInteger(nodes.nodes.length)} Knoten und {formatInteger(edges.edges.length)} Kanten
            {nodes.selection.truncated
              ? `, auf ${formatInteger(nodes.selection.maxNodes)} Knoten gekürzt`
              : ", nicht gekürzt"}
            . Die Positionen hat der Export vorab berechnet (Federlayout, Seed {metrics.seed}); der Browser zeichnet nur.
            {components.onePerTimeStep
              ? ` Der Ausschnitt zerfällt in ${formatInteger(components.count)} getrennte Teile, einen je Zeitschritt: Kanten verbinden hier nur Transaktionen desselben Zeitschritts.`
              : ` Der Ausschnitt besteht aus ${formatInteger(components.count)} zusammenhängenden Teilen.`}
          </p>
          <p>
            Ziehen verschiebt den Ausschnitt, das Mausrad zoomt. Ein Klick auf einen Knoten hebt ihn und seine Nachbarn
            hervor und zeigt die Werte im Detailfeld. Ohne Maus lässt sich jeder Startknoten über die Tabelle auswählen.
          </p>
          <p>
            Von den {formatInteger(seeds.length)} Startknoten tragen {formatInteger(seedLabels.illicit)} das Label
            auffällig, {formatInteger(seedLabels.licit)} das Label unauffällig und {formatInteger(seedLabels.unknown)}{" "}
            kein Label.
          </p>
        </div>
        <NetzRadarExplorer nodes={nodes.nodes} edges={edges.edges} description={explorerDescription} />
      </Section>

      <Section id="metriken" title="Metriken">
        <MetricsTable metrics={metrics} />
        <dl className="grid max-w-4xl gap-x-8 gap-y-3 text-sm sm:grid-cols-2">
          <div>
            <dt className="font-medium">PR-AUC (Average Precision)</dt>
            <dd className="text-stone">Fläche unter der Precision-Recall-Kurve als Treppensumme, Hauptmetrik.</dd>
          </div>
          <div>
            <dt className="font-medium">Precision bei Recall ≥ 0,5</dt>
            <dd className="text-stone">
              Höchste Precision über alle Schwellen, bei denen mindestens die Hälfte der auffälligen Testknoten gefunden
              wird.
            </dd>
          </div>
          <div>
            <dt className="font-medium">Recall bei Precision ≥ 0,5</dt>
            <dd className="text-stone">
              Höchster Recall über alle Schwellen, bei denen mindestens jeder zweite markierte Knoten auffällig ist; 0,
              wenn keine Schwelle das erreicht.
            </dd>
          </div>
          <div>
            <dt className="font-medium">Accuracy (Nebenwert)</dt>
            <dd className="text-stone">
              Schwelle: {accuracyThresholds.join("; ")}. Markierte Knoten: {runs.map(flaggedSentence).join("; ")}.
              Zum Vergleich: Ein Modell, das alle Testknoten unauffällig nennt, erreicht{" "}
              {formatDecimal(evaluation.allLicitAccuracy)}.
            </dd>
          </div>
        </dl>
        <figure className="space-y-3 rounded-lg border border-line bg-surface p-4 sm:p-6" data-testid="pr-curve-figure">
          <h3 className="font-serif text-xl">Precision-Recall-Kurven</h3>
          <PrCurveChart
            series={curveSeries(runs)}
            referenceLevel={prevalenceLevel(evaluation)}
            height={360}
            testId="pr-curve"
          />
          <figcaption className="text-xs text-stone">
            Precision über Recall auf den {formatInteger(labelled)} Testknoten mit Label, als Treppenkurve wie bei der
            Average Precision. Jede Kurve beginnt bei der höchsten Schwelle; den künstlichen Startpunkt von scikit-learn
            (Recall 0, Precision 1) enthält der Export nicht. Für die Darstellung ist jede Kurve auf höchstens 101 Punkte
            ausgedünnt; die Kennzahlen in der Tabelle stammen aus der vollständigen Kurve. Die gestrichelte waagrechte
            Linie ist die Prävalenz, also die Precision, die ein Score ohne Information bei jedem Recall im Mittel
            erreicht.
          </figcaption>
        </figure>
      </Section>

      <Section id="methodik" title="Methodik">
        <div className={PROSE}>
          <h3 className="font-serif text-xl">Zeitlicher Split</h3>
          <p>
            Trainiert wird auf den Zeitschritten {formatStepRange(split.train)}, getestet auf{" "}
            {formatStepRange(split.test)}. Die Zeitschritte {formatStepRange(split.validation)} am Ende des
            Trainingszeitraums sind als Validierungsteil reserviert; in Schritt 4 werden dort die Hyperparameter der
            Graph Neural Networks gewählt, nie am Testteil. Kein Zeitschritt liegt auf beiden Seiten, und zwischen
            Trainings- und Testknoten verläuft keine Kante (crossSplitEdges = {split.crossSplitEdges}). Median, MAD und der
            Isolation Forest werden nur auf Trainingsknoten angepasst, ohne Labels.
          </p>

          <h3 className="pt-3 font-serif text-xl">Robuste Z-Scores</h3>
          <Formula>
            z<sub>ij</sub> = (x<sub>ij</sub> − Median<sub>j</sub>) / ({formatDecimal(madScale ?? 1.4826)} · MAD
            <sub>j</sub>), &nbsp; Score<sub>i</sub> = max<sub>j</sub> |z<sub>ij</sub>|
          </Formula>
          <p>
            Für jedes der {dataset.features} lokalen Merkmale j kommen Median und MAD (Median der absoluten Abweichungen
            vom Median) nur aus den Trainingsknoten. Der Faktor 1,4826 ≈ 1/Φ⁻¹(0,75) macht die MAD bei
            Normalverteilung mit der Standardabweichung vergleichbar. Merkmale mit MAD = 0 tragen 0 bei
            {inactiveFeatures.length > 0
              ? `; auf den Trainingsknoten dieses Netzes trifft das auf ${inactiveFeatures.join(" und ")} zu, wirksam sind also ${formatInteger(
                  zscoreFeatures.length - inactiveFeatures.length,
                )} der ${formatInteger(zscoreFeatures.length)} Merkmale`
              : ""}
            . Median und MAD reagieren kaum auf einzelne Ausreißer, anders als Mittelwert und Standardabweichung.
          </p>

          <h3 className="pt-3 font-serif text-xl">Isolation Forest</h3>
          <p>
            {trees === null ? "Ein Isolation Forest" : `${formatInteger(trees)} Bäume`} (scikit-learn, Seed{" "}
            {metrics.seed}) auf den Trainingsknoten ohne Labels. Zufällige Schnitte isolieren einen Punkt; wird er im
            Mittel nach wenigen Schnitten isoliert, gilt er als auffällig. Der Score ist −score_samples, höher heißt
            auffälliger. Merkmale sind die {dataset.features} lokalen Merkmale und vier Graphmaße
            {contamination === null
              ? "."
              : `. Der Parameter contamination = ${formatDecimal(contamination, 2)} setzt nur die interne Schwelle von scikit-learn und ändert den Score nicht.`}
          </p>

          <h3 className="pt-3 font-serif text-xl">Graphmaße</h3>
          <p>
            Eingangs- und Ausgangsgrad, Betweenness-Zentralität auf dem gerichteten Graphen je Zeitschritt und
            Eigenvektor-Zentralität auf seiner ungerichteten Fassung. Sie beschreiben die Lage eines Knotens im Netz,
            nicht aber die Merkmale seiner Nachbarn. Genau das sollen die Graph Neural Networks in Schritt 4 ergänzen.
          </p>

          <h3 className="pt-3 font-serif text-xl">Warum PR-AUC statt Accuracy</h3>
          <p>
            Im Testzeitraum sind {formatInteger(evaluation.testPositives)} von {formatInteger(labelled)} Knoten mit
            Label auffällig, also {formatPercent(evaluation.prevalence, 2)}. Ein Modell, das jeden Knoten unauffällig
            nennt, erreicht damit eine Accuracy von {formatDecimal(evaluation.allLicitAccuracy)}, ohne einen einzigen
            auffälligen Knoten zu finden. Die PR-AUC bewertet dagegen, wie weit oben die auffälligen Knoten in der
            Rangfolge der Scores stehen:
          </p>
          <Formula>
            AP = Σ<sub>n</sub> (R<sub>n</sub> − R<sub>n−1</sub>) · P<sub>n</sub>
          </Formula>
          <p>
            summiert über alle Schwellen n, mit Recall R<sub>n</sub> und Precision P<sub>n</sub>. Ein konstanter Score,
            der alle Knoten gleich bewertet, erreicht genau die Prävalenz, hier {formatDecimal(evaluation.prevalence)}.
            Eine zufällige Rangfolge liegt im Erwartungswert etwas darüber, bei{" "}
            {formatDecimal(evaluation.randomPrAucExpected)}, weil die ersten Ränge mit großem Gewicht in die Summe
            eingehen. Wie stark der Zufall streut, zeigen {formatInteger(evaluation.randomPermutations)} zufällige
            Rangfolgen der Testknoten (Seed {metrics.seed}): 95 % davon bleiben unter{" "}
            {formatDecimal(evaluation.randomPrAucQ95)}. Gegen diese Werte sind die Zeilen der Tabelle zu lesen.
          </p>
        </div>
        <div className="grid max-w-4xl gap-3">
          {runs.map((run) => (
            <details key={run.method} className="rounded-lg border border-line bg-surface p-4 text-sm">
              <summary className="cursor-pointer font-medium">
                Alle Hyperparameter: {run.displayName} (Lauf vom {formatIsoDate(run.date)}, Seed {run.seed})
              </summary>
              <ParameterList entries={parameterEntries(run.hyperparameters)} />
            </details>
          ))}
        </div>
      </Section>

      <Section id="einordnung" title="Einordnung der Ergebnisse">
        <ul className="max-w-3xl list-disc space-y-3 pl-5 leading-relaxed">
          {runs.map((run) => (
            <li key={run.method}>{randomSentence(run, evaluation)}</li>
          ))}
          {zscore && iforest ? (
            <li>
              {iforest.prAuc < zscore.prAuc
                ? `Der Isolation Forest mit zusätzlichen Graphmaßen liegt um ${formatDecimal(
                    zscore.prAuc - iforest.prAuc,
                  )} unter den robusten Z-Scores auf Einzelmerkmalen.`
                : iforest.prAuc > zscore.prAuc
                  ? `Der Isolation Forest mit zusätzlichen Graphmaßen liegt um ${formatDecimal(
                      iforest.prAuc - zscore.prAuc,
                    )} über den robusten Z-Scores auf Einzelmerkmalen.`
                  : "Der Isolation Forest mit zusätzlichen Graphmaßen und die robusten Z-Scores erreichen dieselbe PR-AUC."}{" "}
              Da sich Verfahren und Merkmalssatz gleichzeitig unterscheiden, lässt sich daraus nicht ablesen, ob die
              Graphmaße helfen.
            </li>
          ) : null}
          <li>
            {maxRecallAtPrecision50 < FEW_HITS_RECALL
              ? "Bei einer Precision von mindestens 0,5 finden die Verfahren kaum auffällige Knoten. "
              : "Recall bei einer Precision von mindestens 0,5: "}
            {runs.map((run) => recallSentence(run, evaluation.testPositives)).join("; ")}.
          </li>
          <li>
            {allBelowAllLicit
              ? `Bei der Accuracy liegen alle gemessenen Verfahren unter den ${formatDecimal(
                  evaluation.allLicitAccuracy,
                )} eines Modells, das alles unauffällig nennt. Deshalb ist Accuracy hier nur Nebenwert.`
              : `Die Accuracy liegt zwischen ${formatDecimal(
                  Math.min(...runs.map((run) => run.accuracy)),
                )} und ${formatDecimal(Math.max(...runs.map((run) => run.accuracy)))}; ein Modell, das alles unauffällig nennt, erreicht ${formatDecimal(
                  evaluation.allLicitAccuracy,
                )}.`}
          </li>
          <li>
            Unter den {formatInteger(seeds.length)} Startknoten mit den höchsten Isolation-Forest-Scores tragen{" "}
            {formatInteger(seedLabels.illicit + seedLabels.licit)} ein Label; davon sind{" "}
            {formatPercent(illicitShareAmongLabelled(seedLabels))} auffällig (Prävalenz im Test:{" "}
            {formatPercent(evaluation.prevalence)}). Die Startknoten haben im Mittel einen Gesamtgrad von{" "}
            {formatDecimal(meanDegree(seeds), 1)}, die übrigen Knoten des Ausschnitts von{" "}
            {formatDecimal(meanDegree(others), 1)}.
            {meanDegree(seeds) > meanDegree(others)
              ? " Der Isolation Forest stuft also vor allem Knoten mit vielen Kanten als auffällig ein, und solche Knoten gibt es auch im gutartigen Hintergrund. Das ist eine Lesart der exportierten Daten, kein eigener Test."
              : ""}
          </li>
          {best && !hasGnnRun ? (
            <li>
              Offen ist, ob ein Modell mit Nachbarschaft besser abschneidet. Das misst Schritt 4 mit demselben Split,
              denselben Testknoten und denselben Metriken. Maßstab ist die bessere Baseline: {best.displayName} mit
              PR-AUC {formatDecimal(best.prAuc)}.
            </li>
          ) : null}
        </ul>
      </Section>

      <Section id="grenzen" title="Grenzen">
        <ul className="max-w-3xl list-disc space-y-3 pl-5 leading-relaxed">
          <li>
            Synthetische Daten: Die Ergebnisse beschreiben das Verhalten auf einem eigenen Generator. Auf reale
            Transaktionsdaten lassen sie sich nicht übertragen.
          </li>
          <li>
            Ein Lauf, keine Streuung: Alle Zahlen stammen aus einem Lauf mit Seed {metrics.seed}. Konfidenzintervalle,
            etwa per Bootstrap über die Testknoten, sind nicht berechnet. Bei {formatInteger(evaluation.testPositives)}{" "}
            auffälligen Testknoten können kleine Unterschiede zufällig sein.
          </li>
          <li>
            Nur Knoten mit Label werden bewertet. Unbekannte Knoten, darunter Musterknoten ohne Label, gehen nicht in die
            Kennzahlen ein.
          </li>
          <li data-testid="homophily-limit">
            Eingebaute Nachbarschaft: Im Generator hängen Musterknoten fast nur an anderen Musterknoten
            {hopNoiseSd === null
              ? ""
              : `, und Kreise und Ketten tragen entlang der Kanten fast denselben Betrag (hopNoiseSd = ${formatDecimal(
                  hopNoiseSd,
                  2,
                )})`}
            . Von den {formatInteger(homophily.labelledEdges)} Kanten zwischen zwei Knoten mit Label verbinden{" "}
            {formatInteger(homophily.illicitIllicitEdges)} zwei auffällige, {formatInteger(homophily.licitLicitEdges)}{" "}
            zwei unauffällige und nur {formatInteger(homophily.illicitLicitEdges)} einen auffälligen mit einem
            unauffälligen Knoten. {formatInteger(homophily.illicitWithIllicitNeighbour)} von{" "}
            {formatInteger(dataset.labelCounts.illicit)} auffälligen Knoten haben einen auffälligen Nachbarn, aber nur{" "}
            {formatInteger(homophily.licitWithIllicitNeighbour)} von {formatInteger(dataset.labelCounts.licit)}{" "}
            unauffälligen. Ein Modell, das die Nachbarschaft einbezieht, bekommt dieses Signal teilweise geschenkt. Ein
            Vorsprung der Graph Neural Networks wäre auf diesem Netz deshalb zum Teil eingebaut und nicht auf andere Daten
            übertragbar.
          </li>
          <li>
            Der Ausschnitt zeigt nur die Nachbarschaft der {formatInteger(seeds.length)} höchsten Isolation-Forest-Scores und
            ist damit bewusst nicht repräsentativ für das ganze Netz.
          </li>
          <li>Accuracy hängt von der gewählten Schwelle ab und ist deshalb nur Nebenwert.</li>
          {hasGnnRun ? null : (
            <li>Die Graph Neural Networks fehlen noch (Schritt 4); bis dahin ist die Fragestellung offen.</li>
          )}
        </ul>
      </Section>

      <Section id="stand" title="Stand">
        <p className="max-w-3xl leading-relaxed" data-testid="data-status">
          Ergebnisse vom {formatIsoDate(metrics.generatedAt)}
          {exportTime ? ` (Export ${exportTime})` : ""}, Seed {metrics.seed}, Datensatz „{dataset.displayName}“. Jeder
          Lauf ist mit Datensatz, Split, Seed, Hyperparametern und Datum protokolliert. Die Seite liest nur die vorab
          exportierten Dateien nodes.json, edges.json und metrics.json. Im Browser läuft kein Modell, und es gehen
          keine Anfragen an Dritte.
        </p>
      </Section>

      <div className="border-t border-line pt-6 text-xs text-stone">
        <Disclaimer variant="long" />
        <p className="mt-2">
          Mit der Nutzung erkennen Sie die{" "}
          <Link href="/nutzungsbedingungen" className="underline hover:text-gold-deep">
            Nutzungsbedingungen
          </Link>{" "}
          an. Einzelheiten zum Datenschutz stehen in der{" "}
          <Link href="/datenschutz" className="underline hover:text-gold-deep">
            Datenschutzerklärung
          </Link>
          .
        </p>
      </div>
    </div>
  );
}
