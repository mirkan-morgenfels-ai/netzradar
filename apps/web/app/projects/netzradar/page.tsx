import type { Metadata } from "next";
import Link from "next/link";
import type { ReactNode } from "react";
import { PrCurveChart } from "@portfolio/charts";
import { Disclaimer, PRIVACY_SHORT } from "@portfolio/legal";
import { StatTile } from "@portfolio/ui";
import { MetricsTable } from "@/components/netzradar/MetricsTable";
import { NetzRadarExplorer } from "@/components/netzradar/NetzRadarExplorer";
import { NodeSymbol } from "@/components/netzradar/NodeSymbol";
import { ScrollRegion } from "@/components/netzradar/ScrollRegion";
import { SearchTable } from "@/components/netzradar/SearchTable";
import {
  accuracySentence,
  baselineSentence,
  caveatConclusion,
  caveatTitle,
  compareItem,
  decompositionSentences,
  graphLeadMeasured,
  homophilyLimitConclusion,
  pairSentence,
  randomSentence,
} from "@/lib/netzradar/assessment";
import { curveSeries, prevalenceLevel } from "@/lib/netzradar/curves";
import { netzRadarData } from "@/lib/netzradar/data";
import {
  FEATURE_SET_TEXT,
  formatDecimal,
  formatInteger,
  formatIsoDate,
  formatIsoTimeUtc,
  formatPercent,
  formatPlainNumber,
  formatShare,
  formatStepRange,
  joinList,
  LABEL_TEXT,
  METHOD_SHORT_TEXT,
  METHOD_TEXT,
} from "@/lib/netzradar/format";
import { countLabels, seedNodes, summarizeComponents } from "@/lib/netzradar/graph";
import { CHAIN_LABELS, chainExample } from "@/lib/netzradar/messagePassing";
import {
  averagedSeparation,
  bestBaseline,
  bestRun,
  comparisonMargin,
  featureNames,
  findRun,
  finalTrainRatioWeight,
  hitsAtRecall,
  illicitShareAmongLabelled,
  isGraphMethod,
  labelledTestCount,
  labelTotal,
  meanDegree,
  numberParameter,
  parameterEntries,
  reachedEpochLimit,
  searchWeight,
  zeroMadFeatures,
} from "@/lib/netzradar/summary";
import { NODE_LABELS, type FeatureSet, type Run, type Training } from "@/lib/netzradar/types";

const DESCRIPTION_METHODS = joinList(netzRadarData.metrics.runs.map((run) => METHOD_SHORT_TEXT[run.method]));

export const metadata: Metadata = {
  title: { absolute: "NetzRadar" },
  description: `Fallstudie zur Anomalie-Erkennung in Transaktionsnetzwerken auf einem synthetischen Netz: ${DESCRIPTION_METHODS} im Vergleich, strikt zeitlicher Split, PR-AUC als Hauptmetrik. Alle Ergebnisse sind vorab berechnet.`,
};

const SECTION_TITLE = "font-serif text-2xl";
const PROSE = "max-w-3xl space-y-3 leading-relaxed";
const SUBTITLE = "pt-3 font-serif text-xl";
const FEW_HITS_RECALL = 0.1;
const EXAMPLE_NEIGHBOURS = 3;
const MIXED_NEIGHBOUR_SHARE = 0.5;

const FEATURE_SET_DATIVE: Record<FeatureSet, string> = {
  local: "nur lokalen Merkmalen",
  "local+graph": "lokalen Merkmalen und Graphmaßen",
};

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

function Formula({ label, children }: { label: string; children: ReactNode }) {
  return (
    <ScrollRegion label={label} className="rounded-md border border-line bg-surface px-4 py-3">
      <p className="font-serif text-base whitespace-nowrap">{children}</p>
    </ScrollRegion>
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

function runHint(run: Run): string {
  const features = FEATURE_SET_TEXT[run.featureSet];
  if (run.training === null) return `${features}, ohne Labels`;
  if (run.training.edges === "none") return `${features}, mit Labels, ohne Kanten`;
  return `${features}, mit Labels und Nachbarschaft`;
}

function statusText(runs: readonly Run[], datasetName: string): string {
  const baselines = runs.filter((run) => run.training === null).map((run) => METHOD_TEXT[run.method]);
  const graphs = runs.filter((run) => isGraphMethod(run.method)).map((run) => METHOD_SHORT_TEXT[run.method]);
  const parts = [
    baselines.length > 0 ? `die Baselines (${joinList(baselines)})` : null,
    graphs.length > 0 ? `die Graph Neural Networks ${joinList(graphs)}` : null,
    findRun(runs, "mlp") ? "als Kontrolle ein MLP ohne Kanten" : null,
  ].filter((part): part is string => part !== null);
  return `Stand: Schritt 4 von 5. Gemessen sind ${joinList(parts)}, alle mit demselben zeitlichen Split und auf denselben Testknoten. Gemessen ist bisher nur auf dem Datensatz „${datasetName}“. Was die Zahlen tragen und was nicht, steht unter „Einordnung“ und „Grenzen“.`;
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

function finalModelText(run: Run, training: Training): string {
  const weight = formatPlainNumber(training.positiveWeight);
  const rule = training.positiveWeightRule === "fixed" ? "fest" : "Verhältnis aus den Labels";
  return `${METHOD_SHORT_TEXT[run.method]} mit ${FEATURE_SET_DATIVE[run.featureSet]}, w = ${weight} (${rule}) und ${formatInteger(
    training.selectedEpoch,
  )} Epochen`;
}

function activationText(value: string): string {
  return value === "relu" ? "ReLU" : value;
}

function optimizerText(value: string): string {
  return value === "adam" ? "Adam" : value;
}

export default function NetzRadarPage() {
  const { metrics, nodes, edges } = netzRadarData;
  const { dataset, split, evaluation, runs } = metrics;
  const zscore = findRun(runs, "zscore");
  const iforest = findRun(runs, "iforest");
  const gcn = findRun(runs, "gcn");
  const graphsage = findRun(runs, "graphsage");
  const mlp = findRun(runs, "mlp");
  const graphRuns = runs.filter((run) => isGraphMethod(run.method));
  const learned = runs.flatMap((run) => (run.training ? [{ run, training: run.training }] : []));
  const graphTraining = graphRuns[0]?.training ?? null;
  const firstTraining = learned[0]?.training ?? null;
  const sameShape =
    graphTraining !== null &&
    learned.every(
      ({ training }) =>
        training.layers === graphTraining.layers &&
        training.hidden === graphTraining.hidden &&
        training.activation === graphTraining.activation &&
        training.dropout === graphTraining.dropout &&
        training.optimizer === graphTraining.optimizer &&
        training.learningRate === graphTraining.learningRate &&
        training.weightDecay === graphTraining.weightDecay,
    );
  const baseline = bestBaseline(runs);
  const best = bestRun(runs);
  const hasGnnRun = graphRuns.length > 0;
  const margin = comparisonMargin(evaluation);
  const leadMeasured = graphLeadMeasured(graphRuns, [baseline, mlp], margin);
  const scoreGnnRun = nodes.scoreGnnMethod === null ? null : findRun(runs, nodes.scoreGnnMethod);
  const otherGraphRuns = graphRuns.filter((run) => run.method !== nodes.scoreGnnMethod);
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
  const sameLabelEdges = homophily.illicitIllicitEdges + homophily.licitLicitEdges;
  const trees = iforest ? numberParameter(iforest.hyperparameters, "nEstimators") : null;
  const contamination = iforest ? numberParameter(iforest.hyperparameters, "contamination") : null;
  const madScale = zscore ? numberParameter(zscore.hyperparameters, "madScale") : null;
  const accuracyThresholds = [...new Set(runs.map((run) => run.accuracyThreshold))];
  const exportTime = formatIsoTimeUtc(metrics.generatedAt);
  const maxRecallAtPrecision50 = Math.max(...runs.map((run) => run.recallAtPrecision50));
  const ratioWeight = searchWeight(runs, "trainRatio");
  const fixedWeight = searchWeight(runs, "fixed");
  const finalRatioWeight = finalTrainRatioWeight(runs);
  const epochLimitRuns = learned.filter(({ training }) => reachedEpochLimit(training));
  const chain = chainExample();
  const chainGcnB = chain.find((row) => row.key === "gcn")?.values[1];
  const chainMeanB = chain.find((row) => row.key === "mean")?.values[1];
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
        <p className="mt-3 leading-relaxed" data-testid="status-text">
          {hasGnnRun
            ? statusText(runs, dataset.displayName)
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
            hint={runHint(run)}
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
        <ScrollRegion labelledBy="label-table-caption" className="rounded-lg border border-line bg-surface">
          <table className="w-full min-w-[24rem] text-sm" data-testid="label-table">
            <caption id="label-table-caption" className="px-4 pt-4 pb-2 text-left text-xs text-stone">
              Labels im gesamten Netz
            </caption>
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
        </ScrollRegion>
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
          {scoreGnnRun?.training ? (
            <p data-testid="score-gnn-text">
              Detailfeld und Tabelle zeigen zusätzlich den Score von {METHOD_SHORT_TEXT[scoreGnnRun.method]}
              {otherGraphRuns.length > 0
                ? `, dem Graph Neural Network mit der höheren PR-AUC auf dem Validierungsteil (${formatDecimal(
                    scoreGnnRun.training.validationPrAuc,
                  )} gegen ${joinList(
                    otherGraphRuns.map(
                      (run) => `${formatDecimal(run.training?.validationPrAuc ?? 0)} bei ${METHOD_SHORT_TEXT[run.method]}`,
                    ),
                  )}); der Testteil entscheidet darüber nicht`
                : ""}
              . Der Ausschnitt selbst ist nach dem Isolation-Forest-Score gewählt, nicht nach dem GNN-Score. Die Knoten
              mit den höchsten GNN-Scores können deshalb fehlen.
            </p>
          ) : null}
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
        <NetzRadarExplorer
          nodes={nodes.nodes}
          edges={edges.edges}
          scoreGnnMethod={nodes.scoreGnnMethod}
          description={explorerDescription}
        />
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
            Trainingszeitraums sind als Validierungsteil reserviert; dort werden Merkmalssatz, Klassengewicht und
            Epochenzahl der gelernten Verfahren gewählt, nie am Testteil. Kein Zeitschritt liegt auf beiden Seiten, und
            zwischen Trainings- und Testknoten verläuft keine Kante (crossSplitEdges = {split.crossSplitEdges}). Median,
            MAD und der Isolation Forest werden nur auf Trainingsknoten angepasst, ohne Labels. Die gelernten Verfahren
            sehen Labels nur aus den Trainingszeitschritten.
          </p>

          <h3 className={SUBTITLE}>Robuste Z-Scores</h3>
          <Formula label="Formel: robuster Z-Score">
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

          <h3 className={SUBTITLE}>Isolation Forest</h3>
          <p>
            {trees === null ? "Ein Isolation Forest" : `${formatInteger(trees)} Bäume`} (scikit-learn, Seed{" "}
            {metrics.seed}) auf den Trainingsknoten ohne Labels. Zufällige Schnitte isolieren einen Punkt; wird er im
            Mittel nach wenigen Schnitten isoliert, gilt er als auffällig. Der Score ist −score_samples, höher heißt
            auffälliger. Merkmale sind die {dataset.features} lokalen Merkmale und vier Graphmaße
            {contamination === null
              ? "."
              : `. Der Parameter contamination = ${formatDecimal(contamination, 2)} setzt nur die interne Schwelle von scikit-learn und ändert den Score nicht.`}
          </p>

          <h3 className={SUBTITLE}>Graphmaße</h3>
          <p>
            Eingangs- und Ausgangsgrad, Betweenness-Zentralität auf dem gerichteten Graphen je Zeitschritt und
            Eigenvektor-Zentralität auf seiner ungerichteten Fassung. Sie beschreiben die Lage eines Knotens im Netz,
            nicht aber die Merkmale seiner Nachbarn.
            {hasGnnRun
              ? " Genau das ergänzen die Graph Neural Networks im nächsten Abschnitt."
              : " Genau das sollen die Graph Neural Networks in Schritt 4 ergänzen."}
          </p>
        </div>

        {graphTraining ? (
          <div className={PROSE} data-testid="method-gnn">
            <h3 className={SUBTITLE}>Graph Neural Networks: Message Passing</h3>
            <p>
              Ein Graph Neural Network rechnet in Schichten. In Schicht k sammelt jeder Knoten v die Darstellungen h
              seiner Nachbarn N(v) und seine eigene aus der vorigen Schicht, fasst sie mit einer Aggregation AGG zusammen,
              etwa einem gewichteten Mittelwert, multipliziert mit gelernten Gewichten W und wendet eine Nichtlinearität σ
              an:
            </p>
            <Formula label="Formel: Message Passing">
              h<sub>v</sub>
              <sup>(k)</sup> = σ(W<sup>(k)</sup> · AGG({"{"}h<sub>u</sub>
              <sup>(k−1)</sup> : u ∈ N(v){"}"} ∪ {"{"}h<sub>v</sub>
              <sup>(k−1)</sup>
              {"}"})), &nbsp; h<sub>v</sub>
              <sup>(0)</sup> = x<sub>v</sub>
            </Formula>
            <p>
              Startwert sind die skalierten Merkmale x<sub>v</sub> des Knotens. Nach {graphTraining.layers} Schichten
              hängt die Ausgabe eines Knotens von allen Knoten ab, die höchstens {graphTraining.layers} Kanten entfernt
              sind. {sameShape ? "Hier haben alle gelernten Verfahren" : `${METHOD_SHORT_TEXT[graphRuns[0]?.method ?? "gcn"]} hat`}{" "}
              {graphTraining.layers} Schichten mit {graphTraining.hidden} versteckten Einheiten, {activationText(graphTraining.activation)} und Dropout{" "}
              {formatPlainNumber(graphTraining.dropout)} nach der ersten Schicht und zwei Ausgaben, je eine für auffällig
              und unauffällig. Trainiert wird mit {optimizerText(graphTraining.optimizer)} (Lernrate{" "}
              {formatPlainNumber(graphTraining.learningRate)}, Weight Decay {formatPlainNumber(graphTraining.weightDecay)})
              auf dem ganzen Graphen auf einmal; eine Epoche ist genau ein Optimierungsschritt.
              {graphTraining.edges === "undirected"
                ? " Die Kanten zählen in beide Richtungen. Die Richtung des Geldflusses steckt weiter in den Merkmalen: Zahl der Ein- und Ausgänge, im Merkmalssatz mit Graphmaßen auch Ein- und Ausgangsgrad."
                : ""}
            </p>
            <p>
              Alle Knoten sind im Graphen, auch die Testknoten und die ohne Label. Das ist kein Leck: Zwischen Trainings-
              und Testknoten verläuft keine Kante (crossSplitEdges = {split.crossSplitEdges}), Nachrichten erreichen
              Trainingsknoten also nur von Trainingsknoten. Damit hängen Ausgaben, Verlust und Gradienten der
              Trainingsknoten nicht von Testmerkmalen ab. Labels sind nie Eingabe, nur Ziel im Verlust, und Testlabels
              gehen nie in den Verlust ein.
            </p>

            <h3 className={SUBTITLE}>GCN und GraphSAGE</h3>
            <p>GCN (Kipf und Welling) bildet eine normierte Summe über die Nachbarn und den Knoten selbst:</p>
            <Formula label="Formel: GCN-Schicht">
              h<sub>v</sub>′ = Σ<sub>u ∈ N(v) ∪ {"{"}v{"}"}</sub> W h<sub>u</sub> / √(d̃<sub>u</sub> d̃<sub>v</sub>),
              &nbsp; d̃ = Grad + 1
            </Formula>
            <p>
              Jeder Beitrag wird durch die Wurzel aus beiden Graden geteilt. Nachbarn mit wenigen Kanten zählen dadurch
              mehr, Knoten mit vielen Kanten weniger. Eigene und fremde Merkmale laufen durch dasselbe Gewicht W.
              GraphSAGE in der Variante mit Mittelwert hält die eigenen Merkmale getrennt:
            </p>
            <Formula label="Formel: GraphSAGE-Schicht">
              h<sub>v</sub>′ = W<sub>1</sub> h<sub>v</sub> + W<sub>2</sub> · (1 / |N(v)|) Σ<sub>u ∈ N(v)</sub> h
              <sub>u</sub>
            </Formula>
            <p>
              Am Beispiel der Kette {CHAIN_LABELS.join(" – ")} mit Startwerten x = (1, 0, 0, 1), allen Gewichten 1, ohne
              Bias und ohne Nichtlinearität ergibt eine Schicht:
            </p>
            <ScrollRegion labelledBy="chain-example-caption" className="rounded-lg border border-line bg-surface">
              <table className="w-full min-w-[28rem] text-sm" data-testid="chain-example">
                <caption id="chain-example-caption" className="sr-only">
                  Eine Schicht Message Passing auf der Kette A bis D
                </caption>
                <thead>
                  <tr className="border-b border-line text-left text-xs uppercase tracking-wide text-stone">
                    <th scope="col" className="px-4 py-2 font-medium">
                      Regel
                    </th>
                    {CHAIN_LABELS.map((label) => (
                      <th key={label} scope="col" className="px-4 py-2 text-right font-medium">
                        {label}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {chain.map((row) => (
                    <tr key={row.key} className="border-t border-line tabular-nums">
                      <th scope="row" className="px-4 py-2 text-left font-normal">
                        {row.label}
                      </th>
                      {row.values.map((value, index) => (
                        <td key={CHAIN_LABELS[index]} className="px-4 py-2 text-right">
                          {formatDecimal(value)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </ScrollRegion>
            <p>
              {chainGcnB !== undefined && chainMeanB !== undefined
                ? `Weil B und C den Wert 0 haben, stammt der Wert von B ganz von A: GCN gibt ${formatDecimal(
                    chainGcnB,
                  )} = 1/√(2 · 3), der einfache Mittelwert ${formatDecimal(chainMeanB)} = 1/3. A hat weniger Nachbarn als B und zählt bei GCN deshalb mehr. `
                : ""}
              GraphSAGE behält bei A und D den eigenen Wert 1 und addiert den Mittelwert der Nachbarn. Das MLP sieht keine
              Nachbarn: B und C bleiben 0.
            </p>

            {mlp?.training ? (
              <>
                <h3 className={SUBTITLE}>Kontrolle: MLP ohne Kanten</h3>
                <Formula label="Formel: MLP-Schicht">
                  h<sub>v</sub>′ = W h<sub>v</sub>
                </Formula>
                <p>
                  Die Kontrolle hat dieselbe Größe ({mlp.training.layers} Schichten, {mlp.training.hidden} Einheiten),
                  dieselben Labels und dasselbe Auswahlverfahren: dieselben Kandidaten für Merkmalssatz und
                  Klassengewicht, Auswahl am selben Validierungsteil. Sie hat aber lineare Schichten statt der Faltungen
                  und damit keine Kanten. Welcher Merkmalssatz, welches Gewicht und wie viele Epochen gewählt werden,
                  entscheidet die Suche für jedes Verfahren getrennt; die Endmodelle stehen unter der Tabelle der Suche.
                  Der Abstand zwischen einem Graph Neural Network und dem MLP misst deshalb, was die Nachbarschaft bei
                  gleicher Überwachung und gleichem Auswahlverfahren bringt. Der Abstand zwischen MLP und Baselines mischt
                  dagegen die Wirkung der Labels mit dem Wechsel des Verfahrens und gegebenenfalls des Merkmalssatzes.
                </p>
              </>
            ) : null}

            <h3 className={SUBTITLE}>Klassengewichtung</h3>
            <p>Auffällige Knoten sind selten. Trainiert wird deshalb mit gewichteter Kreuzentropie:</p>
            <Formula label="Formel: gewichtete Kreuzentropie">
              L = Σ<sub>i ∈ ℒ</sub> w<sub>y(i)</sub> · (−log p<sub>i, y(i)</sub>) / Σ<sub>i ∈ ℒ</sub> w<sub>y(i)</sub>,
              &nbsp; w<sub>unauffällig</sub> = 1, &nbsp; w<sub>auffällig</sub> = w
            </Formula>
            <p>
              Die Summe läuft über die Knoten mit Label in der Verlustmenge ℒ; Knoten ohne Label und Testknoten gehen nie
              in den Verlust ein. Weil durch die Summe der Gewichte geteilt wird, zählt nur das Verhältnis w. Standard ist
              w = Zahl der unauffälligen durch Zahl der auffälligen Knoten mit Label in der Verlustmenge; dann tragen
              beide Klassen dasselbe Gesamtgewicht
              {ratioWeight === null ? "" : `. In der Auswahl ergibt das w = ${formatPlainNumber(ratioWeight)}`}
              {finalRatioWeight === null
                ? ""
                : `, beim Endmodell auf dem ganzen Trainingszeitraum w = ${formatPlainNumber(finalRatioWeight)}`}
              .{fixedWeight === null ? "" : ` Als zweiter Kandidat läuft ein festes Gewicht w = ${formatPlainNumber(fixedWeight)} mit.`}{" "}
              Das Gewicht verschiebt die vorhergesagten Chancen um den Faktor w, ändert im Optimum aber nicht die
              Rangfolge der Knoten, und nur die Rangfolge zählt für die PR-AUC. Es wirkt über den Trainingsverlauf: wie
              stark die seltenen auffälligen Knoten die Gradienten bestimmen. Deshalb ist auch der GNN-Score keine
              Wahrscheinlichkeit, sondern die Differenz der beiden Logits.
            </p>

            {firstTraining ? (
              <>
                <h3 className={SUBTITLE}>Auswahl am Validierungsteil, Endmodell auf dem Trainingszeitraum</h3>
                <p>
                  Die Merkmale werden je Spalte robust skaliert: Median und {formatPlainNumber(firstTraining.scaling.madScale)}{" "}
                  · MAD kommen nur aus den Knoten der jeweiligen Trainingszeitschritte, bei MAD = 0 ersetzt die
                  Standardabweichung die MAD, und Werte jenseits von ±{formatPlainNumber(firstTraining.scaling.clip)}{" "}
                  werden gekappt. Je Verfahren laufen {formatInteger(firstTraining.search.length)} Kandidaten: nur lokale
                  Merkmale oder zusätzlich Graphmaße, Gewicht als Verhältnis oder fest. Jeder trainiert auf den
                  Zeitschritten {formatStepRange(firstTraining.selectionSteps)} und wird nach jeder Epoche auf den Knoten
                  mit Label der Zeitschritte {formatStepRange(firstTraining.validationSteps)} bewertet. Nach{" "}
                  {formatInteger(firstTraining.patience)} Epochen ohne Verbesserung bricht das Training ab, spätestens nach{" "}
                  {formatInteger(firstTraining.maxEpochs)}. Gewählt wird der Kandidat mit der höchsten PR-AUC auf dem
                  Validierungsteil, bei Gleichstand der frühere, als Epochenzahl seine beste Epoche.
                </p>
                <p>
                  Danach wird das Endmodell neu auf den Zeitschritten {formatStepRange(firstTraining.finalFitSteps)}{" "}
                  trainiert, mit Skalierung und Gewicht aus diesen Zeitschritten und genau der gewählten Zahl an Epochen,
                  und einmal auf dem Testteil bewertet, mit denselben Funktionen wie die Baselines. Der Testteil entscheidet
                  weder über Merkmale noch über Gewicht oder Epochen.
                </p>
              </>
            ) : null}
          </div>
        ) : null}

        {learned.length > 0 ? (
          <div className="max-w-4xl space-y-3">
            <SearchTable runs={runs} />
            <p className="max-w-3xl text-sm leading-relaxed" data-testid="final-models">
              Endmodelle: {learned.map(({ run, training }) => finalModelText(run, training)).join("; ")}. Die PR-AUC auf
              dem Validierungsteil dient nur dem Vergleich der Kandidaten untereinander. Der Validierungsteil ist klein
              und umfasst andere Zeitschritte als der Testteil ({formatStepRange(split.validation)} gegen{" "}
              {formatStepRange(split.test)}); Validierungs- und Testwerte sind deshalb nicht direkt vergleichbar.
              {epochLimitRuns.length > 0
                ? ` Bei ${joinList(
                    epochLimitRuns.map(({ run }) => METHOD_SHORT_TEXT[run.method]),
                  )} liegt die beste Epoche auf der Obergrenze von ${formatInteger(
                    epochLimitRuns[0]?.training.maxEpochs ?? 0,
                  )} Epochen; das Early Stopping hat dort nicht gegriffen, mehr Epochen könnten den Wert ändern.`
                : ""}
            </p>
          </div>
        ) : null}

        <div className={PROSE}>
          <h3 className={SUBTITLE}>Warum PR-AUC statt Accuracy</h3>
          <p>
            Im Testzeitraum sind {formatInteger(evaluation.testPositives)} von {formatInteger(labelled)} Knoten mit
            Label auffällig, also {formatPercent(evaluation.prevalence, 2)}. Ein Modell, das jeden Knoten unauffällig
            nennt, erreicht damit eine Accuracy von {formatDecimal(evaluation.allLicitAccuracy)}, ohne einen einzigen
            auffälligen Knoten zu finden. Die PR-AUC bewertet dagegen, wie weit oben die auffälligen Knoten in der
            Rangfolge der Scores stehen:
          </p>
          <Formula label="Formel: Average Precision">
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
        <ul className="max-w-3xl list-disc space-y-3 pl-5 leading-relaxed" data-testid="assessment">
          {runs.map((run) => (
            <li key={run.method}>{randomSentence(run, evaluation)}</li>
          ))}
          <li data-testid="comparison-margin">
            Als Mindestabstand für einen Vergleich zweier Verfahren gilt hier {formatDecimal(margin)}, der Abstand
            zwischen Erwartungswert und 95-%-Quantil zufälliger Rangfolgen. Das ist eine grobe Schwelle, kein Test: Sie
            zeigt, wie stark schon zufällige Rangfolgen auf diesen {formatInteger(labelled)} Testknoten streuen. Kleinere
            Unterschiede werden nicht als Rangfolge gelesen. Alle Werte stammen aus Läufen mit Seed {metrics.seed}.
          </li>
          {zscore && iforest ? <li data-testid="baseline-pair">{baselineSentence(zscore, iforest, margin)}</li> : null}
          {baseline && graphRuns.length > 0 ? (
            <li data-testid="gnn-vs-baseline">
              Gegen die bessere Baseline ({METHOD_TEXT[baseline.method]}, PR-AUC {formatDecimal(baseline.prAuc)}):{" "}
              {graphRuns.map((run) => compareItem(run, baseline, margin)).join("; ")}. Dieser Abstand mischt zwei
              Effekte: Die Graph Neural Networks lernen aus Labels, die Baselines nicht, und sie sehen die Nachbarschaft.
            </li>
          ) : null}
          {gcn && graphsage ? <li data-testid="gcn-vs-graphsage">{pairSentence(gcn, graphsage, margin)}</li> : null}
          {mlp && baseline && graphRuns.length > 0 ? (
            <li data-testid="decomposition">{decompositionSentences(mlp, baseline, graphRuns, margin).join(" ")}</li>
          ) : null}
          <li>
            {maxRecallAtPrecision50 < FEW_HITS_RECALL
              ? "Bei einer Precision von mindestens 0,5 finden die Verfahren kaum auffällige Knoten. "
              : "Recall bei einer Precision von mindestens 0,5: "}
            {runs.map((run) => recallSentence(run, evaluation.testPositives)).join("; ")}.
          </li>
          <li>{accuracySentence(runs, evaluation)}</li>
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
        {hasGnnRun ? (
          <aside
            aria-labelledby="homophily-caveat-title"
            className="max-w-3xl space-y-3 rounded-lg border border-wine bg-wine-soft p-5 leading-relaxed"
            data-testid="homophily-caveat"
          >
            <h3 id="homophily-caveat-title" className="font-serif text-xl">
              {caveatTitle(leadMeasured)}
            </h3>
            <p>
              Das Netz ist stark homophil, das heißt: Knoten hängen vor allem an Knoten mit demselben Label. Von den{" "}
              {formatInteger(homophily.labelledEdges)} Kanten zwischen zwei Knoten mit Label verbinden {formatInteger(sameLabelEdges)} gleiche Labels (Anteil{" "}
              {formatDecimal(homophily.sameLabelShare)}), nur {formatInteger(homophily.illicitLicitEdges)} einen
              auffälligen mit einem unauffälligen Knoten. Im Generator gehört ein auffälliger Knoten immer zu einem
              eingebauten Muster, und die Muster hängen fast nur an anderen Musterknoten. „Meine Nachbarn sehen aus wie
              Musterknoten“ ist hier fast dasselbe wie „ich bin ein Musterknoten“.
            </p>
            {localShift === null ? null : (
              <p>
                Mitteln über Nachbarn derselben Klasse vergrößert außerdem den Abstand der Klassen. Weichen die Merkmale
                um δ = {formatDecimal(localShift, 1)} Standardabweichungen ab (localShift) und hat ein Knoten d Nachbarn,
                davon den Anteil ρ aus der anderen Klasse, ist der standardisierte Abstand nach dem Mitteln über N(v) ∪{" "}
                {"{"}v{"}"} gleich δ · (1 + d(1 − 2ρ)) / √(d + 1), sofern das Rauschen der Knoten unabhängig ist und
                überall dieselbe Varianz hat. Ohne Mitteln ist er δ ={" "}
                {formatDecimal(averagedSeparation(localShift, 0), 2)}. Bei d = {EXAMPLE_NEIGHBOURS} und ρ = 0 wächst er auf{" "}
                {formatDecimal(averagedSeparation(localShift, EXAMPLE_NEIGHBOURS), 2)}, bei ρ ={" "}
                {formatDecimal(MIXED_NEIGHBOUR_SHARE, 1)} fällt er auf{" "}
                {formatDecimal(averagedSeparation(localShift, EXAMPLE_NEIGHBOURS, MIXED_NEIGHBOUR_SHARE), 2)}. Bei
                korrelierten Nachbarn, etwa entlang von Ketten und Kreisen mit fast gleichem Betrag, verkleinert das
                Mitteln das Rauschen kaum, und der Gewinn ist kleiner.
              </p>
            )}
            <p data-testid="homophily-caveat-conclusion">{caveatConclusion(leadMeasured)}</p>
          </aside>
        ) : null}
      </Section>

      <Section id="grenzen" title="Grenzen">
        <ul className="max-w-3xl list-disc space-y-3 pl-5 leading-relaxed">
          <li data-testid="synthetic-limit">
            Synthetische Daten: Alle Ergebnisse{hasGnnRun ? ", auch die der Graph Neural Networks aus Schritt 4," : ""}{" "}
            sind auf dem Datensatz „{dataset.displayName}“ gemessen. Sie beschreiben das Verhalten auf einem eigenen
            Generator und lassen sich nicht auf reale Transaktionsdaten übertragen.
          </li>
          <li>
            Ein Lauf je Verfahren: Alle Zahlen auf dieser Seite stammen aus Läufen mit Seed {metrics.seed}.
            {learned.length > 0
              ? " Für die gelernten Verfahren ist die Streuung über weitere Seeds in den Laufprotokollen festgehalten (README und docs/runs); die Seite zeigt sie nicht."
              : ""}{" "}
            Konfidenzintervalle, etwa per Bootstrap über die Testknoten, sind nicht berechnet. Bei{" "}
            {formatInteger(evaluation.testPositives)} auffälligen Testknoten können kleine Unterschiede zufällig sein.
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
            unauffälligen. Ein Modell, das die Nachbarschaft einbezieht, bekommt dieses Signal teilweise geschenkt.{" "}
            {homophilyLimitConclusion(leadMeasured)}
          </li>
          {learned.length > 0 ? (
            <li>
              Überwacht gegen unüberwacht: Die Baselines sehen keine Labels, die gelernten Verfahren schon.
              {mlp ? " Erst das MLP trennt die Wirkung der Labels von der der Nachbarschaft." : ""} Eine Baseline mit
              gemittelten Nachbarmerkmalen ohne Labels fehlt noch.
            </li>
          ) : null}
          {firstTraining ? (
            <li>
              Die Auswahl der gelernten Verfahren stützt sich auf einen kleinen Validierungsteil (Zeitschritte{" "}
              {formatStepRange(firstTraining.validationSteps)}). Unterschiede von wenigen Hundertsteln zwischen
              Kandidaten sind dort nicht belastbar.
              {epochLimitRuns.length > 0
                ? ` Bei ${joinList(
                    epochLimitRuns.map(({ run }) => METHOD_SHORT_TEXT[run.method]),
                  )} lag die gewählte Epoche auf der Obergrenze; ob mehr Epochen helfen, ist nicht gemessen.`
                : ""}
            </li>
          ) : null}
          {hasGnnRun ? (
            <li>
              Nur {formatInteger(graphRuns.length)} Architekturen mit Nachbarschaft
              {graphTraining?.edges === "undirected" ? " und ungerichteten Kanten" : ""}. GAT, getrennte Ein- und
              Ausgangsnachrichten und zeitliche Modelle sind nicht gemessen.
            </li>
          ) : null}
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
