import type { Metadata } from "next";
import Link from "next/link";
import { Fragment, type ReactNode } from "react";
import { PrCurveChart } from "@portfolio/charts";
import { Disclaimer, PRIVACY_SHORT } from "@portfolio/legal";
import { cx, StatTile } from "@portfolio/ui";
import { ExternalLink, PROSE_LINK_CLASS } from "@/components/ExternalLink";
import { MethodIndex, type MethodIndexItem } from "@/components/netzradar/MethodIndex";
import { MetricsTable } from "@/components/netzradar/MetricsTable";
import { NetzRadarExplorer } from "@/components/netzradar/NetzRadarExplorer";
import { NodeSymbol } from "@/components/netzradar/NodeSymbol";
import { ScrollRegion } from "@/components/netzradar/ScrollRegion";
import { SearchTable } from "@/components/netzradar/SearchTable";
import { BUTTON_GOLD, BUTTON_OUTLINE_LIGHT } from "@/components/site/buttons";
import { ArrowMark, ExternalMark } from "@/components/site/ExternalMark";
import { HeroOrnament } from "@/components/site/motif";
import { ProjectHero, type HeroFact } from "@/components/site/ProjectHero";
import { SectionHeader } from "@/components/site/SectionHeader";
import { Shell } from "@/components/site/Shell";
import { pageMetadata } from "@/lib/metadata";
import {
  accuracySentence,
  baselineGapEffectsSentence,
  baselineSentence,
  caveatConclusion,
  caveatTitle,
  compareItem,
  decompositionSentences,
  graphLeadMeasured,
  homophilyLimitConclusion,
  pairSentence,
  randomComparisonSentence,
  recallSentence,
} from "@/lib/netzradar/assessment";
import { curveSeries, METHOD_CURVE_STYLES, prevalenceLevel } from "@/lib/netzradar/curves";
import { netzRadarData } from "@/lib/netzradar/data";
import {
  FEATURE_SET_TEXT,
  formatDateRange,
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
  illicitShareAmongLabelled,
  isGraphMethod,
  labelledTestCount,
  labelTotal,
  meanDegree,
  numberParameter,
  parameterEntries,
  reachedEpochLimit,
  runDateRange,
  searchWeight,
  zeroMadFeatures,
} from "@/lib/netzradar/summary";
import { NODE_LABELS, type FeatureSet, type Run, type StepRange, type Training } from "@/lib/netzradar/types";
import { isRepoPath, PROJECTS, README_URL, REPO_URL, repoUrl, RUNS_URL } from "@/lib/site";

const DESCRIPTION_METHODS = joinList(netzRadarData.metrics.runs.map((run) => METHOD_SHORT_TEXT[run.method]));
const PROJECT = PROJECTS.find((project) => project.slug === "netzradar");

export const metadata: Metadata = pageMetadata({
  title: "NetzRadar – Anomalie-Erkennung in Transaktionsnetzwerken",
  absoluteTitle: true,
  description: `Fallstudie zur Anomalie-Erkennung in Transaktionsnetzwerken auf einem synthetischen Netz: ${DESCRIPTION_METHODS} im Vergleich, strikt zeitlicher Split, PR-AUC als Hauptmetrik. Alle Ergebnisse sind vorab berechnet.`,
  path: "/projects/netzradar",
});

const PROSE = "max-w-[40rem] space-y-4 text-[15px] leading-[1.75] text-ink sm:text-base";
const BODY = "max-w-[40rem] text-[15px] leading-[1.75] text-ink sm:text-base";
const ANCHOR = "scroll-mt-24 lg:scroll-mt-36";
const SUBTITLE = "display pt-8 text-[1.5rem] leading-tight text-ink first:pt-0 sm:text-[1.625rem]";
const CARD = "rounded-2xl border border-line bg-surface shadow-card";
const EXAMPLE_NEIGHBOURS = 3;
const FINDING =
  "relative border-t border-line py-5 pl-12 first:border-t-0 [counter-increment:finding] before:absolute before:top-[1.35rem] before:left-0 before:font-display before:text-[1.375rem] before:leading-none before:text-gold-deep before:[font-variant-numeric:lining-nums_tabular-nums] before:content-[counter(finding,decimal-leading-zero)] sm:pl-14";
const MIXED_NEIGHBOUR_SHARE = 0.5;

const SECTION_LINKS: ReadonlyArray<{ id: string; label: string }> = [
  { id: "ueberblick", label: "Überblick" },
  { id: "datensatz", label: "Daten" },
  { id: "netzwerk", label: "Netzwerk" },
  { id: "metriken", label: "Ergebnisse" },
  { id: "methodik", label: "Methodik" },
  { id: "einordnung", label: "Einordnung" },
  { id: "grenzen", label: "Grenzen" },
];

const FIVE_RUN_GRID =
  "grid-cols-2 max-sm:[&>*:last-child:nth-child(odd)]:col-span-2 sm:grid-cols-6 sm:[&>*]:col-span-2 sm:[&>*:nth-last-child(-n+2)]:col-span-3 lg:grid-cols-5 lg:[&>*]:col-span-1 lg:[&>*:nth-last-child(-n+2)]:col-span-1";

const RUN_GRID: Record<number, string> = {
  1: "lg:grid-cols-1",
  2: "lg:grid-cols-2",
  3: "lg:grid-cols-3",
  4: "lg:grid-cols-4",
  5: "lg:grid-cols-5",
};

const FEATURE_SET_DATIVE: Record<FeatureSet, string> = {
  local: "nur lokalen Merkmalen",
  "local+graph": "lokalen Merkmalen und Graphmaßen",
};

function Section({
  id,
  eyebrow,
  title,
  lead,
  tone = "ivory",
  size = "lg",
  children,
}: {
  id: string;
  eyebrow: string;
  title: ReactNode;
  lead?: ReactNode;
  tone?: "ivory" | "surface";
  size?: "lg" | "md";
  children?: ReactNode;
}) {
  return (
    <section
      id={id}
      aria-labelledby={`${id}-title`}
      className={cx(
        "scroll-mt-24 border-t border-line lg:scroll-mt-36",
        size === "lg" ? "py-16 sm:py-20 lg:py-24" : "py-14 sm:py-16",
        tone === "surface" ? "bg-surface" : undefined,
      )}
    >
      <Shell>
        <SectionHeader id={`${id}-title`} eyebrow={eyebrow} title={title} lead={lead} size={size} />
        {children ? <div className="mt-10 lg:mt-12">{children}</div> : null}
      </Shell>
    </section>
  );
}

function Formula({ label, children }: { label: string; children: ReactNode }) {
  return (
    <ScrollRegion
      label={label}
      className="rounded-xl border border-line bg-surface px-5 py-4 shadow-[inset_2px_0_0_var(--color-gold)]"
    >
      <p className="display text-[1.25rem] whitespace-nowrap text-ink">{children}</p>
    </ScrollRegion>
  );
}

function ParameterList({ entries }: { entries: Array<[string, string]> }) {
  return (
    <dl className="grid gap-x-8 text-xs sm:grid-cols-2">
      {entries.map(([key, value]) => (
        <div key={key} className="flex min-w-0 gap-3 border-t border-line py-2">
          <dt className="shrink-0 font-mono text-slate">{key}</dt>
          <dd className="num min-w-0 break-words text-ink">{value}</dd>
        </div>
      ))}
    </dl>
  );
}

function Disclosure({ summary, children }: { summary: ReactNode; children: ReactNode }) {
  return (
    <details className={cx("group", CARD, "[&_summary::-webkit-details-marker]:hidden")}>
      <summary className="flex cursor-pointer list-none items-center justify-between gap-4 rounded-2xl px-5 py-4 text-sm font-medium text-ink sm:px-6">
        <span className="text-pretty">{summary}</span>
        <span
          aria-hidden="true"
          className="grid size-7 shrink-0 place-items-center rounded-full border border-line text-base leading-none text-gold-deep transition-transform duration-200 group-open:rotate-45"
        >
          +
        </span>
      </summary>
      <div className="px-5 pb-4 sm:px-6">{children}</div>
    </details>
  );
}

function MethodSection({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <details id={id} className={cx("group", ANCHOR, CARD, "[&_summary::-webkit-details-marker]:hidden")}>
      <summary className="flex cursor-pointer list-none items-center justify-between gap-4 rounded-2xl px-5 py-4 sm:px-6 sm:py-5">
        <h3 className="display text-[1.375rem] leading-tight text-pretty text-ink sm:text-[1.5rem]">{title}</h3>
        <span
          aria-hidden="true"
          className="grid size-8 shrink-0 place-items-center rounded-full border border-line text-base leading-none text-gold-deep transition-[transform,border-color] duration-200 group-hover:border-gold group-open:rotate-45"
        >
          +
        </span>
      </summary>
      <div className="border-t border-line px-5 pt-5 pb-6 sm:px-6 sm:pb-7">
        <div className={PROSE}>{children}</div>
      </div>
    </details>
  );
}

function PrAucMeter({ value, color, reference }: { value: number; color: string; reference: number }) {
  const share = (input: number) => `${Math.max(0, Math.min(1, input)) * 100}%`;
  return (
    <span aria-hidden="true" className="relative mb-3 block h-1.5 rounded-full bg-line/80">
      <span className="absolute inset-y-0 left-0 rounded-full" style={{ width: share(value), backgroundColor: color }} />
      <span className="absolute -top-1 -bottom-1 w-px bg-ink/70" style={{ left: share(reference) }} />
    </span>
  );
}

function SplitRibbon({ steps, train, validation, test }: { steps: number; train: StepRange; validation: StepRange; test: StepRange }) {
  const cells = Array.from({ length: steps }, (_, index) => index + 1);
  const tone = (step: number) => {
    if (step >= validation.from && step <= validation.to) return "bg-gold";
    if (step >= train.from && step <= train.to) return "bg-navy-950";
    if (step >= test.from && step <= test.to) return "bg-sky";
    return "bg-line";
  };
  return (
    <figure>
      <div aria-hidden="true" className="flex gap-[3px]">
        {cells.map((step) => (
          <span key={step} className={cx("h-7 flex-1 rounded-[3px]", tone(step))} />
        ))}
      </div>
      <div aria-hidden="true" className="mt-2 flex justify-between text-[11px] text-slate">
        <span className="num">1</span>
        <span className="num">{steps}</span>
      </div>
      <figcaption className="mt-4 space-y-2 text-[13px] leading-snug text-ink">
        <span className="flex items-center gap-2.5">
          <span aria-hidden="true" className="size-3 shrink-0 rounded-[3px] bg-navy-950" />
          Training, Zeitschritte {formatStepRange(train)}
        </span>
        <span className="flex items-center gap-2.5">
          <span aria-hidden="true" className="size-3 shrink-0 rounded-[3px] bg-gold" />
          davon Validierung, Zeitschritte {formatStepRange(validation)}
        </span>
        <span className="flex items-center gap-2.5">
          <span aria-hidden="true" className="size-3 shrink-0 rounded-[3px] bg-sky" />
          Test, Zeitschritte {formatStepRange(test)}
        </span>
      </figcaption>
    </figure>
  );
}

function BreakablePath({ path }: { path: string }) {
  const parts = path.split("/");
  return (
    <>
      {parts.map((part, index) => (
        <Fragment key={`${part}-${index}`}>
          {index > 0 ? (
            <>
              /<wbr />
            </>
          ) : null}
          {part}
        </Fragment>
      ))}
    </>
  );
}

function runHint(run: Run): string {
  const features = FEATURE_SET_TEXT[run.featureSet];
  if (run.training === null) return `${features}, ohne Labels`;
  if (run.training.edges === "none") return `${features}, mit Labels, ohne Kanten`;
  return `${features}, mit Labels und Nachbarschaft`;
}

function StatusText({ runs, datasetName }: { runs: readonly Run[]; datasetName: string }) {
  const baselines = runs.filter((run) => run.training === null).map((run) => METHOD_SHORT_TEXT[run.method]);
  const graphs = runs.filter((run) => isGraphMethod(run.method)).map((run) => METHOD_SHORT_TEXT[run.method]);
  const parts = [
    baselines.length > 0 ? `die Baselines (${joinList(baselines)})` : null,
    graphs.length > 0 ? `die Graph Neural Networks ${joinList(graphs)}` : null,
    findRun(runs, "mlp") ? "als Kontrolle ein MLP ohne Kanten" : null,
  ].filter((part): part is string => part !== null);
  const range = runDateRange(runs);
  const measured =
    graphs.length > 0
      ? `Gemessen sind ${joinList(parts)}, alle mit demselben zeitlichen Split und auf denselben Testknoten.`
      : `Gemessen sind bisher nur ${joinList(parts)}. Die Graph Neural Networks GCN und GraphSAGE, also die Modelle mit Nachbarschaft, fehlen noch; die Fragestellung ist deshalb noch nicht beantwortet.`;
  return (
    <>
      {range ? `Stand der Ergebnisse: ${formatDateRange(range)}. ` : ""}
      {measured} Alle Messungen stammen aus dem Datensatz „{datasetName}“. Offen: Case-Study, Läufe mit gestörter
      Nachbarschaft, Baseline mit gemittelten Nachbarmerkmalen. Was die Zahlen tragen und was nicht, steht unter{" "}
      <a href="#einordnung" className={PROSE_LINK_CLASS}>
        „Einordnung der Ergebnisse“
      </a>{" "}
      und{" "}
      <a href="#grenzen" className={PROSE_LINK_CLASS}>
        „Grenzen“
      </a>
      .
    </>
  );
}

function flaggedSentence(run: Run): string {
  return `${METHOD_SHORT_TEXT[run.method]} ${formatInteger(run.accuracyFlagged)}, davon ${formatInteger(
    run.accuracyTruePositives,
  )} auffällig`;
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

function countText(count: number, singular: string, plural: string): string {
  return `${formatInteger(count)} ${count === 1 ? singular : plural}`;
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
  const baselineCount = runs.filter((run) => run.training === null).length;
  const methodParts = [
    baselineCount > 0 ? countText(baselineCount, "Baseline", "Baselines") : null,
    graphRuns.length > 0 ? countText(graphRuns.length, "Graph Neural Network", "Graph Neural Networks") : null,
    mlp ? "1 Kontrolle ohne Kanten" : null,
  ].filter((part): part is string => part !== null);
  const facts: readonly HeroFact[] = [
    { value: formatInteger(runs.length), label: "Verfahren", detail: joinList(methodParts) },
    {
      value: formatInteger(dataset.timeSteps),
      label: "Zeitschritte",
      detail: `Training ${formatStepRange(split.train)}, Test ${formatStepRange(split.test)}`,
    },
    { value: "0", label: "Anfragen an Dritte", detail: "Ergebnisse vorab berechnet, kein Modell im Browser" },
  ];
  const methodIndex: MethodIndexItem[] = [
    { id: "methodik-split", label: "Zeitlicher Split" },
    { id: "methodik-zscore", label: "Robuste Z-Scores" },
    { id: "methodik-iforest", label: "Isolation Forest" },
    { id: "methodik-graphmasse", label: "Graphmaße" },
    ...(graphTraining
      ? [
          { id: "methodik-message-passing", label: "Message Passing" },
          { id: "methodik-gcn-graphsage", label: "GCN und GraphSAGE" },
          ...(mlp?.training ? [{ id: "methodik-mlp", label: "Kontrolle: MLP ohne Kanten" }] : []),
          { id: "methodik-gewichtung", label: "Klassengewichtung" },
          ...(firstTraining ? [{ id: "methodik-auswahl", label: "Auswahl und Endmodell" }] : []),
        ]
      : []),
    ...(learned.length > 0 ? [{ id: "methodik-suche", label: "Suche am Validierungsteil" }] : []),
    { id: "methodik-pr-auc", label: "Warum PR-AUC statt Accuracy" },
  ];
  const specRows: Array<[string, string]> = [
    ["Seed", `${metrics.seed}`],
    ["Kanten über den Split", formatInteger(split.crossSplitEdges)],
    ["Testknoten mit Label", formatInteger(labelled)],
    ["Hauptmetrik", "PR-AUC"],
  ];

  return (
    <>
      <ProjectHero
        eyebrow={`${PROJECT?.kicker ?? "Projekt 03"} · ${PROJECT?.topic ?? "Graph-ML"}`}
        title={
          <>
            Netz<em className="text-gold-light">Radar</em>
          </>
        }
        tagline="Anomalie-Erkennung in Transaktionsnetzwerken"
        lead={
          <p>
            Fallstudie mit vorab berechneten Ergebnissen. Die Fragestellung: Wie viel gewinnt ein Modell, das die
            Nachbarschaft einer Transaktion einbezieht, gegenüber einer Baseline auf lokalen Merkmalen, bei gleichem Split
            und gleichen Metriken?
          </p>
        }
        actions={
          <>
            <a href="#netzwerk" className={BUTTON_GOLD}>
              Netzwerk erkunden
              <ArrowMark />
            </a>
            <a href={REPO_URL} rel="noopener noreferrer" className={BUTTON_OUTLINE_LIGHT} data-testid="project-repo-link">
              Quellcode<span className="sr-only"> auf GitHub (externe Seite)</span>
              <ExternalMark className="ml-2 h-2.5 w-2.5 text-gold-light" />
            </a>
          </>
        }
        note={<p className="border-l border-gold/60 pl-4 text-sm leading-relaxed text-navy-300">{PRIVACY_SHORT}</p>}
        facts={facts}
        ornament={<HeroOrnament idPrefix="project-ornament" className="h-auto w-full" />}
      />

      <nav
        aria-label="Abschnitte"
        data-testid="section-nav"
        className="border-b border-line bg-surface lg:sticky lg:top-[4.25rem] lg:z-30 lg:bg-surface/95 lg:backdrop-blur-sm"
      >
        <Shell className="flex flex-col gap-1 py-3 sm:flex-row sm:items-center sm:gap-8 lg:gap-10">
          <p className="eyebrow shrink-0">Auf dieser Seite</p>
          <ol className="-mx-5 flex min-w-0 flex-nowrap gap-x-6 overflow-x-auto pr-12 pl-5 whitespace-nowrap [mask-image:linear-gradient(to_right,#000_calc(100%-3rem),transparent)] sm:mx-0 sm:pl-0 lg:flex-wrap lg:gap-x-7 lg:overflow-visible lg:pr-0 lg:[mask-image:none]">
            {SECTION_LINKS.map((link, index) => (
              <li key={link.id} className="flex shrink-0 items-baseline gap-2">
                <span aria-hidden="true" className="display num text-[0.9375rem] text-gold-deep">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <a
                  href={`#${link.id}`}
                  className="inline-block py-1.5 text-sm text-ink transition-colors duration-150 hover:text-gold-deep"
                >
                  {link.label}
                </a>
              </li>
            ))}
          </ol>
        </Shell>
      </nav>

      <Section
        id="ueberblick"
        eyebrow="Auf einen Blick"
        title={
          <>
            PR-AUC je <em>Verfahren</em>
          </>
        }
      >
        <div data-testid="headline-figures">
          <div
            className={cx(
              "grid gap-3 sm:gap-4",
              runs.length === 5
                ? FIVE_RUN_GRID
                : cx(
                    "grid-cols-2 [&>*:last-child:nth-child(odd)]:col-span-2 lg:[&>*:last-child:nth-child(odd)]:col-span-1",
                    RUN_GRID[runs.length] ?? "lg:grid-cols-5",
                  ),
            )}
          >
            {runs.map((run) => (
              <StatTile key={run.method} label={METHOD_SHORT_TEXT[run.method]} value={formatDecimal(run.prAuc)}>
                <PrAucMeter
                  value={run.prAuc}
                  color={METHOD_CURVE_STYLES[run.method].color}
                  reference={evaluation.randomPrAucExpected}
                />
                {runHint(run)}
              </StatTile>
            ))}
          </div>
          <div className="mt-4 grid gap-px overflow-hidden rounded-2xl border border-line bg-line shadow-card sm:grid-cols-3">
            <StatTile
              variant="ledger"
              label="Zufällige Rangfolge"
              value={formatDecimal(evaluation.randomPrAucExpected)}
              hint={`PR-AUC im Erwartungswert, 95 % bleiben unter ${formatDecimal(evaluation.randomPrAucQ95)}`}
            />
            <StatTile
              variant="ledger"
              label="Konstanter Score"
              value={formatDecimal(evaluation.prevalence)}
              hint="PR-AUC, wenn alle Knoten denselben Score haben (Prävalenz)"
            />
            <StatTile
              variant="ledger"
              label="Testknoten mit Label"
              value={formatInteger(labelled)}
              hint={`davon ${formatInteger(evaluation.testPositives)} auffällig`}
            />
          </div>
          <div className="mt-6 flex flex-col gap-3 text-sm sm:flex-row sm:items-center sm:justify-between">
            <p className="flex items-center gap-3 text-xs text-slate">
              <span aria-hidden="true" className="relative block h-1.5 w-10 shrink-0 rounded-full bg-line/80">
                <span className="absolute inset-y-0 left-0 w-6 rounded-full bg-navy-950" />
                <span className="absolute -top-1 -bottom-1 left-2 w-px bg-ink/70" />
              </span>
              Balken: PR-AUC auf der Skala von 0 bis 1. Strich: Erwartungswert einer zufälligen Rangfolge.
            </p>
            <a
              href="#einordnung"
              className="link inline-flex shrink-0 items-center self-start font-medium whitespace-nowrap sm:self-auto"
            >
              Zur Einordnung der Ergebnisse
              <ArrowMark />
            </a>
          </div>
        </div>
        <p data-testid="status-text" className={cx(BODY, "mt-10 border-l border-gold/60 pl-5 sm:pl-6")}>
          <StatusText runs={runs} datasetName={dataset.displayName} />
        </p>
      </Section>

      <Section
        id="datensatz"
        tone="surface"
        eyebrow="Daten"
        title="Datensatz"
      >
        <div className="grid gap-10 lg:grid-cols-12 lg:gap-14">
          <div className="min-w-0 space-y-8 lg:col-span-7">
            <div className={PROSE}>
              <p>
                Die öffentliche Demo nutzt den Datensatz „{dataset.displayName}“ (Lizenz {dataset.license}), ein eigenes,
                frei veröffentlichbares Netz. Knoten sind Transaktionen, gerichtete Kanten der Geldfluss zwischen ihnen.
                Der Generator (
                {isRepoPath(dataset.source) ? (
                  <ExternalLink href={repoUrl(dataset.source)} testId="generator-source-link">
                    <code className="text-[0.8125rem]">
                      <BreakablePath path={dataset.source} />
                    </code>
                  </ExternalLink>
                ) : (
                  <code className="text-[0.8125rem]">
                    <BreakablePath path={dataset.source} />
                  </code>
                )}
                {generatorSeed === null ? "" : `, Seed ${generatorSeed}`}) legt einen gutartigen Hintergrund nach
                Preferential Attachment an, in dem einzelne Knoten viele Kanten haben, und baut darin bekannte
                Geldwäsche-Muster ein: Fan-in, Fan-out, Kreise und Ketten.
              </p>
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
                    )} Standardabweichungen). Deutlich stärker weicht der größte Ausgangsanteil der Fan-out-Verteiler ab, und runde Beträge sind bei Musterknoten doppelt so häufig. Fan-in-Sammler haben als Summe vieler Zubringer einen höheren Betrag und einen kleineren Variationskoeffizienten der Eingangsbeträge (die Beträge der Zubringer streuen weniger).`}
              </p>
              <p>
                Vorgesehen sind zwei weitere Datensätze: IBM Transactions for Anti-Money Laundering (synthetisch, Lizenz
                CDLA-Sharing-1.0) und das Elliptic Bitcoin Dataset (Lizenz CC BY-NC-ND 4.0). Elliptic wird nur lokal für
                den Methodenvergleich ausgewertet. Aus diesem Datensatz werden weder Rohdaten noch bearbeitete Fassungen
                oder Ausschnitte veröffentlicht.
              </p>
            </div>
            {dataset.generator ? (
              <Disclosure summary={`Alle ${parameterEntries(dataset.generator).length} Generator-Parameter`}>
                <ParameterList entries={parameterEntries(dataset.generator)} />
              </Disclosure>
            ) : null}
          </div>
          <div className="min-w-0 space-y-6 lg:col-span-5">
            <dl
              className="grid grid-cols-2 gap-px overflow-hidden rounded-2xl border border-line bg-line shadow-card"
              data-testid="dataset-figures"
            >
              {[
                ["Knoten", formatInteger(dataset.nodes)],
                ["Kanten", formatInteger(dataset.edges)],
                ["Zeitschritte", formatInteger(dataset.timeSteps)],
                ["Lokale Merkmale", formatInteger(dataset.features)],
              ].map(([term, value]) => (
                <div key={term} className="bg-surface px-5 py-5 sm:px-6">
                  <dt className="text-[0.6875rem] font-medium tracking-[0.14em] text-slate uppercase">{term}</dt>
                  <dd className="display num mt-3 text-[2rem] leading-none text-ink">{value}</dd>
                </div>
              ))}
            </dl>
            <ScrollRegion labelledBy="label-table-caption" className={CARD}>
              <table className="data-table min-w-[16rem]" data-testid="label-table">
                <caption id="label-table-caption" className="eyebrow px-4 pt-5 pb-4 text-left sm:px-6">
                  Labels im gesamten Netz
                </caption>
                <thead>
                  <tr>
                    <th scope="col" className="px-4 sm:px-6">
                      Label
                    </th>
                    <th scope="col" className="px-3 text-right">
                      Knoten
                    </th>
                    <th scope="col" className="px-4 text-right sm:px-6">
                      Anteil
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {NODE_LABELS.map((label) => (
                    <tr key={label}>
                      <th scope="row" className="px-4 py-3 text-left align-baseline font-normal sm:px-6">
                        <span className="inline-flex items-center gap-2.5">
                          <NodeSymbol label={label} />
                          {LABEL_TEXT[label]}
                        </span>
                      </th>
                      <td className="px-3 text-right">{formatInteger(dataset.labelCounts[label])}</td>
                      <td className="px-4 text-right sm:px-6">
                        {formatShare(dataset.labelCounts[label], labelTotal(dataset.labelCounts))}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </ScrollRegion>
          </div>
        </div>
      </Section>

      <Section
        id="netzwerk"
        eyebrow="Graph-Ansicht"
        title={
          <>
            Netzwerk-<em>Ausschnitt</em>
          </>
        }
      >
        <div className="grid gap-8 lg:grid-cols-12 lg:gap-14">
          <div className="min-w-0 space-y-4 lg:col-span-7">
            <p className={BODY}>
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
            <p data-testid="score-gnn-text" className={BODY}>
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
          </div>
          <div className="space-y-5 lg:col-span-5">
            <p className="text-[15px] leading-[1.75] text-ink sm:text-base">
              Von den {formatInteger(seeds.length)} Startknoten tragen {formatInteger(seedLabels.illicit)} das Label
              auffällig, {formatInteger(seedLabels.licit)} das Label unauffällig und {formatInteger(seedLabels.unknown)}{" "}
              kein Label.
            </p>
            <p className="border-l border-gold/60 pl-4 text-[13px] leading-relaxed text-slate" data-testid="graph-hint">
              Ziehen verschiebt den Ausschnitt, das Mausrad zoomt; auf Touch-Geräten mit zwei Fingern zoomen und
              verschieben. Außerhalb des Graphen scrollen Sie die Seite. Ein Klick auf einen Knoten hebt ihn und seine
              Nachbarn hervor und zeigt die Werte im Detailfeld. Ohne Maus lässt sich jeder Startknoten über die Tabelle
              auswählen.
            </p>
          </div>
        </div>
        <div className="mt-10 lg:mt-12">
          <NetzRadarExplorer
            nodes={nodes.nodes}
            edges={edges.edges}
            scoreGnnMethod={nodes.scoreGnnMethod}
            description={explorerDescription}
          />
        </div>
      </Section>

      <Section
        id="metriken"
        tone="surface"
        eyebrow="Ergebnisse"
        title="Metriken"
        lead={
          <p>
            Kennzahlen je Verfahren auf den Testknoten mit Label. Hauptmetrik ist die{" "}
            <span className="whitespace-nowrap">PR-AUC</span>; Accuracy steht nur als Nebenwert daneben.
          </p>
        }
      >
        <MetricsTable metrics={metrics} />
        <dl className="mt-8 grid gap-px overflow-hidden rounded-2xl border border-line bg-line text-sm sm:grid-cols-2">
          <div className="bg-surface p-5 sm:p-6">
            <dt className="font-medium text-ink">PR-AUC (Average Precision)</dt>
            <dd className="mt-1.5 leading-relaxed text-slate">
              Fläche unter der Precision-Recall-Kurve als Treppensumme, Hauptmetrik.
            </dd>
          </div>
          <div className="bg-surface p-5 sm:p-6">
            <dt className="font-medium text-ink">Precision bei Recall ≥ 0,5</dt>
            <dd className="mt-1.5 leading-relaxed text-slate">
              Höchste Precision über alle Schwellen, bei denen mindestens die Hälfte der auffälligen Testknoten gefunden
              wird.
            </dd>
          </div>
          <div className="bg-surface p-5 sm:p-6">
            <dt className="font-medium text-ink">Recall bei Precision ≥ 0,5</dt>
            <dd className="mt-1.5 leading-relaxed text-slate">
              Höchster Recall über alle Schwellen, bei denen mindestens jeder zweite markierte Knoten auffällig ist; 0,
              wenn keine Schwelle das erreicht.
            </dd>
          </div>
          <div className="bg-surface p-5 sm:p-6">
            <dt className="font-medium text-ink">Accuracy (Nebenwert)</dt>
            <dd className="mt-1.5 leading-relaxed text-slate">
              Schwelle: {accuracyThresholds.join("; ")}. Markierte Knoten: {runs.map(flaggedSentence).join("; ")}.
              Zum Vergleich: Ein Modell, das alle Testknoten unauffällig nennt, erreicht{" "}
              {formatDecimal(evaluation.allLicitAccuracy)}.
            </dd>
          </div>
        </dl>
        <figure className={cx("mt-8 p-5 sm:p-8", CARD)} data-testid="pr-curve-figure">
          <p className="eyebrow">Abbildung</p>
          <h3 className="display mt-2 mb-6 text-[1.625rem] leading-tight text-ink sm:text-[1.75rem]">
            Precision-Recall-Kurven
          </h3>
          <PrCurveChart
            series={curveSeries(runs)}
            referenceLevel={prevalenceLevel(evaluation)}
            height={380}
            testId="pr-curve"
          />
          <figcaption className="mt-4 max-w-[90ch] border-t border-line pt-4 text-xs leading-relaxed text-slate">
            Precision über Recall auf den {formatInteger(labelled)} Testknoten mit Label, als Treppenkurve wie bei der
            Average Precision. Jede Kurve beginnt bei der höchsten Schwelle; den künstlichen Startpunkt von scikit-learn
            (Recall 0, Precision 1) enthält der Export nicht. Für die Darstellung ist jede Kurve auf höchstens 101 Punkte
            ausgedünnt; die Kennzahlen in der Tabelle stammen aus der vollständigen Kurve. Die gestrichelte waagrechte
            Linie ist die Prävalenz, also die Precision, die ein Score ohne Information bei jedem Recall im Mittel
            erreicht.
          </figcaption>
        </figure>
      </Section>

      <Section
        id="methodik"
        eyebrow="Vorgehen"
        title="Methodik"
        lead="Split, Verfahren, Auswahl und Metrik im Einzelnen, mit den Werten der Läufe auf dieser Seite."
      >
        <div className="grid gap-12 lg:grid-cols-12 lg:gap-14">
          <div className="min-w-0 lg:col-span-8">
            <div className={PROSE}>
              <h3 id="methodik-split" className={cx(SUBTITLE, ANCHOR)}>
                Zeitlicher Split
              </h3>
              <p>
                Trainiert wird auf den Zeitschritten {formatStepRange(split.train)}, getestet auf{" "}
                {formatStepRange(split.test)}. Die Zeitschritte {formatStepRange(split.validation)} am Ende des
                Trainingszeitraums sind als Validierungsteil reserviert; dort werden Merkmalssatz, Klassengewicht und
                Epochenzahl der gelernten Verfahren gewählt, nie am Testteil. Kein Zeitschritt liegt auf beiden Seiten,
                und zwischen Trainings- und Testknoten verläuft keine Kante (crossSplitEdges = {split.crossSplitEdges}).
                Median, MAD und der Isolation Forest werden nur auf Trainingsknoten angepasst, ohne Labels. Die gelernten
                Verfahren sehen Labels nur aus den Trainingszeitschritten.
              </p>
            </div>
            <div className="mt-10 space-y-3">
              <MethodSection id="methodik-zscore" title="Robuste Z-Scores">
                <Formula label="Formel: robuster Z-Score">
                  z<sub>ij</sub> = (x<sub>ij</sub> − Median<sub>j</sub>) / ({formatDecimal(madScale ?? 1.4826)} · MAD
                  <sub>j</sub>), &nbsp; Score<sub>i</sub> = max<sub>j</sub> |z<sub>ij</sub>|
                </Formula>
                <p>
                  Für jedes der {dataset.features} lokalen Merkmale j kommen Median und MAD (Median der absoluten
                  Abweichungen vom Median) nur aus den Trainingsknoten. Der Faktor 1,4826 ≈ 1/Φ⁻¹(0,75) macht die MAD bei
                  Normalverteilung mit der Standardabweichung vergleichbar. Merkmale mit MAD = 0 tragen 0 bei
                  {inactiveFeatures.length > 0
                    ? `; auf den Trainingsknoten dieses Netzes trifft das auf ${inactiveFeatures.join(" und ")} zu, wirksam sind also ${formatInteger(
                        zscoreFeatures.length - inactiveFeatures.length,
                      )} der ${formatInteger(zscoreFeatures.length)} Merkmale`
                    : ""}
                  . Median und MAD reagieren kaum auf einzelne Ausreißer, anders als Mittelwert und Standardabweichung.
                </p>
              </MethodSection>
              <MethodSection id="methodik-iforest" title="Isolation Forest">
                <p>
                  {trees === null ? "Ein Isolation Forest" : `${formatInteger(trees)} Bäume`} (scikit-learn, Seed{" "}
                  {metrics.seed}) auf den Trainingsknoten ohne Labels. Zufällige Schnitte isolieren einen Punkt; wird er im
                  Mittel nach wenigen Schnitten isoliert, gilt er als auffällig. Der Score ist −score_samples, höher heißt
                  auffälliger. Merkmale sind die {dataset.features} lokalen Merkmale und vier Graphmaße
                  {contamination === null
                    ? "."
                    : `. Der Parameter contamination = ${formatDecimal(contamination, 2)} setzt nur die interne Schwelle von scikit-learn und ändert den Score nicht.`}
                </p>
              </MethodSection>
              <MethodSection id="methodik-graphmasse" title="Graphmaße">
                <p>
                  Eingangs- und Ausgangsgrad, Betweenness-Zentralität auf dem gerichteten Graphen je Zeitschritt und
                  Eigenvektor-Zentralität auf seiner ungerichteten Fassung. Sie beschreiben die Lage eines Knotens im Netz,
                  nicht aber die Merkmale seiner Nachbarn.
                  {hasGnnRun
                    ? " Genau das ergänzen die Graph Neural Networks im nächsten Abschnitt."
                    : " Genau das sollen die Graph Neural Networks ergänzen."}
                </p>
              </MethodSection>
              {graphTraining ? (
                <div className="space-y-3" data-testid="method-gnn">
                  <MethodSection id="methodik-message-passing" title="Graph Neural Networks: Message Passing">
                    <p>
                      Ein Graph Neural Network rechnet in Schichten. In Schicht k fasst jeder Knoten v die Darstellungen h
                      seiner Nachbarn N(v) aus der vorigen Schicht mit einer Aggregation AGG zusammen, etwa einer normierten
                      Summe oder einem Mittelwert. UPDATE verbindet das Ergebnis über gelernte Gewichte mit der eigenen
                      Darstellung des Knotens, danach folgt eine Nichtlinearität σ, in der letzten Schicht ohne σ:
                    </p>
                    <Formula label="Formel: Message Passing">
                      h<sub>v</sub>
                      <sup>(k)</sup> = σ(UPDATE(h<sub>v</sub>
                      <sup>(k−1)</sup>, AGG({"{"}h<sub>u</sub>
                      <sup>(k−1)</sup> : u ∈ N(v){"}"}))), &nbsp; h<sub>v</sub>
                      <sup>(0)</sup> = x<sub>v</sub>
                    </Formula>
                    <p>
                      Startwert sind die skalierten Merkmale x<sub>v</sub> des Knotens. Nach {graphTraining.layers} Schichten
                      hängt die Ausgabe eines Knotens nur von Knoten ab, die höchstens {graphTraining.layers} Kanten entfernt
                      sind.{" "}
                      {sameShape
                        ? "Hier haben alle gelernten Verfahren"
                        : `${METHOD_SHORT_TEXT[graphRuns[0]?.method ?? "gcn"]} hat`}{" "}
                      {graphTraining.layers} Schichten mit {graphTraining.hidden} versteckten Einheiten,{" "}
                      {activationText(graphTraining.activation)} und Dropout {formatPlainNumber(graphTraining.dropout)} nach der
                      ersten Schicht und zwei Ausgaben, je eine für auffällig und unauffällig. Trainiert wird mit{" "}
                      {optimizerText(graphTraining.optimizer)} (Lernrate {formatPlainNumber(graphTraining.learningRate)}, Weight
                      Decay {formatPlainNumber(graphTraining.weightDecay)}) auf dem ganzen Graphen auf einmal; eine Epoche ist
                      genau ein Optimierungsschritt.
                      {graphTraining.edges === "undirected"
                        ? " Die Kanten zählen in beide Richtungen. Die Richtung des Geldflusses steckt weiter in den Merkmalen: Zahl der Ein- und Ausgänge, im Merkmalssatz mit Graphmaßen auch Ein- und Ausgangsgrad."
                        : ""}
                    </p>
                    <p>
                      Alle Knoten sind im Graphen, auch die Testknoten und die ohne Label. Das ist kein Leck: Zwischen
                      Trainings- und Testknoten verläuft keine Kante (crossSplitEdges = {split.crossSplitEdges}), Nachrichten
                      erreichen Trainingsknoten also nur von Trainingsknoten. Damit hängen Ausgaben, Verlust und Gradienten der
                      Trainingsknoten nicht von Testmerkmalen ab. Labels sind nie Eingabe, nur Ziel im Verlust, und Testlabels
                      gehen nie in den Verlust ein.
                    </p>
                  </MethodSection>
                  <MethodSection id="methodik-gcn-graphsage" title="GCN und GraphSAGE">
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
                      Am Beispiel der Kette {CHAIN_LABELS.join(" – ")} mit Startwerten x = (1, 0, 0, 1), allen Gewichten 1,
                      ohne Bias und ohne Nichtlinearität ergibt eine Schicht:
                    </p>
                    <ScrollRegion labelledBy="chain-example-caption" className={CARD}>
                      <table className="data-table min-w-[28rem]" data-testid="chain-example">
                        <caption id="chain-example-caption" className="sr-only">
                          Eine Schicht Message Passing auf der Kette A bis D
                        </caption>
                        <thead>
                          <tr>
                            <th scope="col" className="px-5 pt-5 sm:px-6">
                              Regel
                            </th>
                            {CHAIN_LABELS.map((label) => (
                              <th key={label} scope="col" className="px-4 pt-5 text-right last:pr-5 sm:last:pr-6">
                                {label}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {chain.map((row) => (
                            <tr key={row.key}>
                              <th scope="row" className="px-5 py-3 text-left align-baseline font-normal sm:px-6">
                                {row.label}
                              </th>
                              {row.values.map((value, index) => (
                                <td key={CHAIN_LABELS[index]} className="px-4 text-right last:pr-5 sm:last:pr-6">
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
                      GraphSAGE behält bei A und D den eigenen Wert 1 und addiert den Mittelwert der Nachbarn. Das MLP sieht
                      keine Nachbarn: B und C bleiben 0.
                    </p>
                  </MethodSection>
                  {mlp?.training ? (
                    <MethodSection id="methodik-mlp" title="Kontrolle: MLP ohne Kanten">
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
                        gleicher Überwachung und gleichem Auswahlverfahren bringt. Der Abstand zwischen MLP und Baselines
                        mischt dagegen die Wirkung der Labels mit dem Wechsel des Verfahrens und gegebenenfalls des
                        Merkmalssatzes.
                      </p>
                    </MethodSection>
                  ) : null}
                  <MethodSection id="methodik-gewichtung" title="Klassengewichtung">
                    <p>Auffällige Knoten sind selten. Trainiert wird deshalb mit gewichteter Kreuzentropie:</p>
                    <Formula label="Formel: gewichtete Kreuzentropie">
                      L = Σ<sub>i ∈ ℒ</sub> w<sub>y(i)</sub> · (−log p<sub>i, y(i)</sub>) / Σ<sub>i ∈ ℒ</sub> w
                      <sub>y(i)</sub>, &nbsp; w<sub>unauffällig</sub> = 1, &nbsp; w<sub>auffällig</sub> = w
                    </Formula>
                    <p>
                      Die Summe läuft über die Knoten mit Label in der Verlustmenge ℒ; Knoten ohne Label und Testknoten gehen
                      nie in den Verlust ein. Weil durch die Summe der Gewichte geteilt wird, zählt nur das Verhältnis w.
                      Standard ist w = Zahl der unauffälligen durch Zahl der auffälligen Knoten mit Label in der Verlustmenge;
                      dann tragen beide Klassen dasselbe Gesamtgewicht
                      {ratioWeight === null ? "" : `. In der Auswahl ergibt das w = ${formatPlainNumber(ratioWeight)}`}
                      {finalRatioWeight === null
                        ? ""
                        : `, beim Endmodell auf dem ganzen Trainingszeitraum w = ${formatPlainNumber(finalRatioWeight)}`}
                      .
                      {fixedWeight === null
                        ? ""
                        : ` Als zweiter Kandidat läuft ein festes Gewicht w = ${formatPlainNumber(fixedWeight)} mit.`}{" "}
                      Das Gewicht verschiebt die vorhergesagten Chancen um den Faktor w, ändert im Optimum aber nicht die
                      Rangfolge der Knoten, und nur die Rangfolge zählt für die PR-AUC. Es wirkt über den Trainingsverlauf: wie
                      stark die seltenen auffälligen Knoten die Gradienten bestimmen. Deshalb ist auch der GNN-Score keine
                      Wahrscheinlichkeit, sondern die Differenz der beiden Logits.
                    </p>
                  </MethodSection>
                  {firstTraining ? (
                    <MethodSection id="methodik-auswahl" title="Auswahl am Validierungsteil, Endmodell auf dem Trainingszeitraum">
                      <p>
                        Die Merkmale werden je Spalte robust skaliert: Median und{" "}
                        {formatPlainNumber(firstTraining.scaling.madScale)} · MAD kommen nur aus den Knoten der jeweiligen
                        Trainingszeitschritte, bei MAD = 0 ersetzt die Standardabweichung die MAD, und Werte jenseits von ±
                        {formatPlainNumber(firstTraining.scaling.clip)} werden gekappt. Je Verfahren laufen{" "}
                        {formatInteger(firstTraining.search.length)} Kandidaten: nur lokale Merkmale oder zusätzlich
                        Graphmaße, Gewicht als Verhältnis oder fest. Jeder trainiert auf den Zeitschritten{" "}
                        {formatStepRange(firstTraining.selectionSteps)} und wird nach jeder Epoche auf den Knoten mit Label
                        der Zeitschritte {formatStepRange(firstTraining.validationSteps)} bewertet. Nach{" "}
                        {formatInteger(firstTraining.patience)} Epochen ohne Verbesserung bricht das Training ab, spätestens
                        nach {formatInteger(firstTraining.maxEpochs)}. Gewählt wird der Kandidat mit der höchsten PR-AUC auf
                        dem Validierungsteil, bei Gleichstand der frühere, als Epochenzahl seine beste Epoche.
                      </p>
                      <p>
                        Danach wird das Endmodell neu auf den Zeitschritten {formatStepRange(firstTraining.finalFitSteps)}{" "}
                        trainiert, mit Skalierung und Gewicht aus diesen Zeitschritten und genau der gewählten Zahl an
                        Epochen, und einmal auf dem Testteil bewertet, mit denselben Funktionen wie die Baselines. Der
                        Testteil entscheidet weder über Merkmale noch über Gewicht oder Epochen.
                      </p>
                    </MethodSection>
                  ) : null}
                </div>
              ) : null}
            </div>
          </div>

          <div className="min-w-0 space-y-10 lg:col-span-4">
            <MethodIndex items={methodIndex} className="max-lg:hidden" />
            <div className={cx("p-5 sm:p-6", CARD)}>
              <p className="eyebrow">Versuchsaufbau</p>
              <p className="display mt-2 text-[1.5rem] leading-tight text-ink">Zeitlicher Split</p>
              <div className="mt-5">
                <SplitRibbon steps={dataset.timeSteps} train={split.train} validation={split.validation} test={split.test} />
              </div>
              <dl className="mt-6 border-t border-line pt-2 text-sm">
                {specRows.map(([term, value]) => (
                  <div key={term} className="flex items-baseline justify-between gap-4 border-b border-line py-2.5 last:border-b-0">
                    <dt className="text-slate">{term}</dt>
                    <dd className="num text-right font-medium text-ink">{value}</dd>
                  </div>
                ))}
              </dl>
              <p className="mt-4 text-xs leading-relaxed text-slate">
                Verfahren: {joinList(runs.map((run) => METHOD_SHORT_TEXT[run.method]))}.
              </p>
            </div>
          </div>
        </div>

        {learned.length > 0 ? (
          <div className="mt-14 space-y-4">
            <h3 id="methodik-suche" className={cx("display text-[1.5rem] leading-tight text-ink sm:text-[1.625rem]", ANCHOR)}>
              Suche am Validierungsteil
            </h3>
            <SearchTable runs={runs} />
            <p className="max-w-[62ch] text-sm leading-relaxed text-ink" data-testid="final-models">
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

        <div className="mt-14 grid gap-12 lg:grid-cols-12 lg:gap-14">
          <div className={cx(PROSE, "lg:col-span-8")}>
            <h3 id="methodik-pr-auc" className={cx(SUBTITLE, ANCHOR)}>
              Warum PR-AUC statt Accuracy
            </h3>
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
              {formatDecimal(evaluation.randomPrAucExpected)}, weil die Average Precision die Precision nur an den Rängen
              der auffälligen Knoten mittelt und ein auffälliger Knoten auf Rang k sich selbst mitzählt; der Effekt ist bei
              kleinem k am größten. Wie stark der Zufall streut, zeigen {formatInteger(evaluation.randomPermutations)}{" "}
              zufällige Rangfolgen der Testknoten (Seed {metrics.seed}): 95 % davon bleiben unter{" "}
              {formatDecimal(evaluation.randomPrAucQ95)}. Gegen diese Werte sind die Zeilen der Tabelle zu lesen.
            </p>
          </div>
        </div>

        <div className="mt-12 grid max-w-[46rem] gap-3">
          {runs.map((run) => (
            <Disclosure
              key={run.method}
              summary={`Alle Hyperparameter: ${METHOD_SHORT_TEXT[run.method]} (Lauf vom ${formatIsoDate(run.date)}, Seed ${run.seed})`}
            >
              <ParameterList entries={parameterEntries(run.hyperparameters)} />
            </Disclosure>
          ))}
        </div>
      </Section>

      <Section
        id="einordnung"
        tone="surface"
        eyebrow="Bewertung"
        title={
          <>
            Einordnung der <em>Ergebnisse</em>
          </>
        }
      >
        <div className="grid gap-12 lg:grid-cols-12 lg:gap-14">
          <ol
            className="min-w-0 border-y border-line text-[15px] leading-[1.75] text-ink [counter-reset:finding] sm:text-base lg:col-span-7"
            data-testid="assessment"
          >
            <li data-testid="random-comparison" className={FINDING}>
              {randomComparisonSentence(runs, evaluation)}
            </li>
            <li data-testid="comparison-margin" className={FINDING}>
              Als Mindestabstand für einen Vergleich zweier Verfahren gilt hier {formatDecimal(margin)}, der Abstand
              zwischen Erwartungswert und 95-%-Quantil zufälliger Rangfolgen. Das ist eine grobe Schwelle, kein Test: Sie
              zeigt, wie stark schon zufällige Rangfolgen auf diesen {formatInteger(labelled)} Testknoten streuen.
              Kleinere Unterschiede werden nicht als Rangfolge gelesen. Alle Werte stammen aus Läufen mit Seed{" "}
              {metrics.seed}.
            </li>
            {zscore && iforest ? (
              <li data-testid="baseline-pair" className={FINDING}>
                {baselineSentence(zscore, iforest, margin)}
              </li>
            ) : null}
            {baseline && graphRuns.length > 0 ? (
              <li data-testid="gnn-vs-baseline" className={FINDING}>
                Gegen die bessere Baseline ({METHOD_SHORT_TEXT[baseline.method]}, PR-AUC {formatDecimal(baseline.prAuc)}):{" "}
                {graphRuns.map((run) => compareItem(run, baseline, margin)).join("; ")}.{" "}
                {baselineGapEffectsSentence(graphRuns, baseline)}
              </li>
            ) : null}
            {gcn && graphsage ? (
              <li data-testid="gcn-vs-graphsage" className={FINDING}>
                {pairSentence(gcn, graphsage, margin)}
              </li>
            ) : null}
            {mlp && baseline && graphRuns.length > 0 ? (
              <li data-testid="decomposition" className={FINDING}>
                {decompositionSentences(mlp, baseline, graphRuns, margin).join(" ")}
              </li>
            ) : null}
            <li data-testid="recall-sentence" className={FINDING}>
              {recallSentence(runs, evaluation.testPositives)}
            </li>
            <li className={FINDING}>{accuracySentence(runs, evaluation)}</li>
            <li className={FINDING}>
              Unter den {formatInteger(seeds.length)} Startknoten mit den höchsten Isolation-Forest-Scores tragen{" "}
              {formatInteger(seedLabels.illicit + seedLabels.licit)} ein Label; davon sind{" "}
              {formatPercent(illicitShareAmongLabelled(seedLabels))} auffällig (Prävalenz im Test:{" "}
              {formatPercent(evaluation.prevalence, 2)}). Die Startknoten haben im Mittel einen Gesamtgrad von{" "}
              {formatDecimal(meanDegree(seeds), 1)}, die übrigen Knoten des Ausschnitts von{" "}
              {formatDecimal(meanDegree(others), 1)}.
              {meanDegree(seeds) > meanDegree(others)
                ? " Der Isolation Forest stuft also vor allem Knoten mit vielen Kanten als auffällig ein, und solche Knoten gibt es auch im gutartigen Hintergrund. Das ist eine Lesart der exportierten Daten, kein eigener Test."
                : ""}
            </li>
            {best && !hasGnnRun ? (
              <li className={FINDING}>
                Offen ist, ob ein Modell mit Nachbarschaft besser abschneidet. Gemessen wird das mit demselben Split,
                denselben Testknoten und denselben Metriken. Maßstab ist die bessere Baseline:{" "}
                {METHOD_SHORT_TEXT[best.method]} mit PR-AUC {formatDecimal(best.prAuc)}.
              </li>
            ) : null}
          </ol>
          {hasGnnRun ? (
            <div className="min-w-0 lg:col-span-5">
              <aside
                aria-labelledby="homophily-caveat-title"
                className="space-y-4 rounded-2xl border border-line border-l-[3px] border-l-wine bg-surface p-6 text-[15px] leading-[1.75] text-ink shadow-card sm:p-8"
                data-testid="homophily-caveat"
              >
                <p className="eyebrow text-wine">Vorbehalt</p>
                <h3 id="homophily-caveat-title" className="display text-[1.625rem] leading-tight text-ink">
                  {caveatTitle(leadMeasured)}
                </h3>
                <p>
                  Das Netz ist stark homophil, das heißt: Knoten hängen vor allem an Knoten mit demselben Label. Von den{" "}
                  {formatInteger(homophily.labelledEdges)} Kanten zwischen zwei Knoten mit Label verbinden{" "}
                  {formatInteger(sameLabelEdges)} gleiche Labels (Anteil {formatDecimal(homophily.sameLabelShare)}), nur{" "}
                  {formatInteger(homophily.illicitLicitEdges)} einen auffälligen mit einem unauffälligen Knoten. Im
                  Generator gehört ein auffälliger Knoten immer zu einem eingebauten Muster, und die Muster hängen fast nur
                  an anderen Musterknoten. „Meine Nachbarn sehen aus wie Musterknoten“ ist hier fast dasselbe wie „ich bin
                  ein Musterknoten“.
                </p>
                {localShift === null ? null : (
                  <p>
                    Mitteln über Nachbarn derselben Klasse vergrößert außerdem den Abstand der Klassen. Weichen die
                    Merkmale um δ = {formatDecimal(localShift, 1)} Standardabweichungen ab (localShift) und hat ein Knoten d
                    Nachbarn, davon den Anteil ρ aus der anderen Klasse, ist der standardisierte Abstand nach dem Mitteln
                    über N(v) ∪ {"{"}v{"}"} gleich δ · (1 + d(1 − 2ρ)) / √(d + 1), sofern das Rauschen der Knoten
                    unabhängig ist und überall dieselbe Varianz hat. Ohne Mitteln ist er δ ={" "}
                    {formatDecimal(averagedSeparation(localShift, 0), 2)}. Bei d = {EXAMPLE_NEIGHBOURS} und ρ = 0 wächst er
                    auf {formatDecimal(averagedSeparation(localShift, EXAMPLE_NEIGHBOURS), 2)}, bei ρ ={" "}
                    {formatDecimal(MIXED_NEIGHBOUR_SHARE, 1)} fällt er auf{" "}
                    {formatDecimal(averagedSeparation(localShift, EXAMPLE_NEIGHBOURS, MIXED_NEIGHBOUR_SHARE), 2)}. Bei
                    korrelierten Nachbarn, etwa entlang von Ketten und Kreisen mit fast gleichem Betrag, verkleinert das
                    Mitteln das Rauschen kaum, und der Gewinn ist kleiner.
                  </p>
                )}
                <p data-testid="homophily-caveat-conclusion">{caveatConclusion(leadMeasured)}</p>
              </aside>
            </div>
          ) : null}
        </div>
      </Section>

      <Section id="grenzen" eyebrow="Vorbehalte" title="Grenzen">
        <ul className="bullet-list text-[15px] leading-[1.75] text-ink sm:text-base lg:columns-2 lg:gap-14 [&>li]:mb-6 [&>li]:break-inside-avoid [&>li+li]:mt-0">
          <li data-testid="synthetic-limit">
            Synthetische Daten: Alle Ergebnisse{hasGnnRun ? ", auch die der Graph Neural Networks," : ""} sind auf dem
            Datensatz „{dataset.displayName}“ gemessen. Sie beschreiben das Verhalten auf einem eigenen Generator und
            lassen sich nicht auf reale Transaktionsdaten übertragen.
          </li>
          <li>
            Ein Lauf je Verfahren: Alle Zahlen auf dieser Seite stammen aus Läufen mit Seed {metrics.seed}.
            {learned.length > 0 ? (
              <>
                {" "}
                Für die gelernten Verfahren ist die Streuung über weitere Seeds in den Laufprotokollen festgehalten (
                <ExternalLink href={README_URL} testId="limits-readme-link">
                  README
                </ExternalLink>{" "}
                und{" "}
                <ExternalLink href={RUNS_URL} testId="limits-runs-link">
                  docs/runs
                </ExternalLink>
                ); die Seite zeigt sie nicht.
              </>
            ) : null}{" "}
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
              gemittelten Nachbarmerkmalen ohne Labels fehlt noch. Ein starkes überwachtes Verfahren ohne Kanten (Random
              Forest, Gradient Boosting) fehlt ebenfalls. Auf Elliptic lag bei Weber et al. (2019), gemessen mit dem
              F1-Wert der auffälligen Klasse, ein Random Forest vor dem GCN.
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
            Der Ausschnitt zeigt nur die Nachbarschaft der {formatInteger(seeds.length)} höchsten Isolation-Forest-Scores
            und ist damit bewusst nicht repräsentativ für das ganze Netz.
          </li>
          <li>Accuracy hängt von der gewählten Schwelle ab und ist deshalb nur Nebenwert.</li>
          {hasGnnRun ? null : <li>Die Graph Neural Networks fehlen noch; bis dahin ist die Fragestellung offen.</li>}
        </ul>
      </Section>

      <Section
        id="stand"
        tone="surface"
        size="md"
        eyebrow="Reproduzierbarkeit"
        title="Stand"
        lead={
          <p data-testid="data-status">
            Ergebnisse vom {formatIsoDate(metrics.generatedAt)}
            {exportTime ? ` (Export ${exportTime})` : ""}, Seed {metrics.seed}, Datensatz „{dataset.displayName}“. Jeder
            Lauf ist mit Datensatz, Split, Seed, Hyperparametern und Datum protokolliert (Laufprotokolle in{" "}
            <ExternalLink href={RUNS_URL} testId="status-runs-link">
              docs/runs
            </ExternalLink>
            ). Die Seite bindet beim Erstellen nur die vorab exportierten Dateien nodes.json, edges.json und metrics.json
            ein. Im Browser läuft kein Modell, und es gehen keine Anfragen an Dritte.
          </p>
        }
      />

      <section aria-label="Rechtliche Hinweise" className="border-t border-line py-10">
        <Shell className="text-[13px] leading-relaxed text-slate">
          <div className="max-w-[72ch] space-y-2">
            <Disclaimer variant="long" />
            <p>
              Mit der Nutzung erkennen Sie die{" "}
              <Link href="/nutzungsbedingungen" className="link">
                Nutzungsbedingungen
              </Link>{" "}
              an. Einzelheiten zum Datenschutz stehen in der{" "}
              <Link href="/datenschutz" className="link">
                Datenschutzerklärung
              </Link>
              .
            </p>
          </div>
        </Shell>
      </section>
    </>
  );
}
