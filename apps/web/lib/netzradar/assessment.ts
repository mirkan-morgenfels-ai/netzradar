import { FEATURE_SET_TEXT, formatDecimal, formatInteger, formatSeedList, joinList, METHOD_SHORT_TEXT } from "./format";
import {
  beatsRandomRankings,
  bestAccuracyAtFlagged,
  bestRun,
  compareScores,
  hasSpread,
  hitsAtRecall,
  isGraphMethod,
  neuralTraining,
  seedSpreadOf,
  spreadSeeds,
  type Comparison,
} from "./summary";
import type { EdgelessMethod, EvaluationInfo, NeuralTraining, Run } from "./types";

export const FEW_HITS_RECALL = 0.1;

const EDGELESS_NAME: Record<EdgelessMethod, string> = {
  hgb: "Gradient Boosting",
  mlp: "MLP",
};

const EDGELESS_DETAIL: Record<EdgelessMethod, string> = {
  hgb: "übliches starkes Verfahren für tabellarische Daten, eigene Suche über Merkmalssatz, Klassengewicht und Blattzahl",
  mlp: "gleiche Größe und dieselben Kandidaten für Merkmalssatz und Klassengewicht wie die Graph Neural Networks",
};

export const MLP_CONTRAST_SENTENCE =
  "Gegen das MLP gerechnet käme der größere Teil aus der Nachbarschaft; das MLP ist auf diesem Netz aber der schwächere Vergleich ohne Kanten.";

export function methodChangeSentence(withMlp: boolean): string {
  return `Gegenüber dem Gradient Boosting wechselt neben den Kanten auch das Verfahren, Entscheidungsbäume statt neuronaler Netze; ${
    withMlp
      ? "allein in den Kanten unterscheiden sich die Graph Neural Networks nur vom MLP."
      : "ein neuronales Netz gleicher Größe ohne Kanten ist nicht gemessen."
  }`;
}

function sameNetworkShape(training: NeuralTraining, reference: NeuralTraining): boolean {
  return (
    training.layers === reference.layers &&
    training.hidden === reference.hidden &&
    training.activation === reference.activation &&
    training.dropout === reference.dropout &&
    training.optimizer === reference.optimizer &&
    training.learningRate === reference.learningRate &&
    training.weightDecay === reference.weightDecay
  );
}

export function neuralShapeSubject(runs: readonly Run[]): string {
  const neural = runs.flatMap((run) => {
    const training = neuralTraining(run);
    return training ? [{ run, training }] : [];
  });
  const reference = neural.find(({ run }) => isGraphMethod(run.method)) ?? neural[0];
  if (!reference) return "";
  const shared = neural.length > 1 && neural.every(({ training }) => sameNetworkShape(training, reference.training));
  return shared
    ? `Hier haben die neuronalen Verfahren ${joinList(
        neural.map(({ run }) => METHOD_SHORT_TEXT[run.method]),
      )} dieselbe Form:`
    : `${METHOD_SHORT_TEXT[reference.run.method]} hat`;
}

export function boostingReferenceSentence(hgb: Run, mlp: Run | null, margin: number): string {
  const lead =
    mlp !== null && compareScores(hgb.prAuc, mlp.prAuc, margin) === "higher"
      ? `Auf diesem Netz liegt das Gradient Boosting deutlich über dem MLP (${formatDecimal(hgb.prAuc)} gegen ${formatDecimal(
          mlp.prAuc,
        )}); der Abstand der Graph Neural Networks zu ihm ist deshalb ein strengerer Vergleich als der zum MLP.`
      : "Der Abstand der Graph Neural Networks zum Gradient Boosting ist ein Vergleich mit einem starken Verfahren ohne Kanten.";
  return `${lead} Dieser Abstand mischt aber den Wechsel von Bäumen zu neuronalen Netzen mit der Wirkung der Nachbarschaft; ${
    mlp !== null
      ? "allein in den Kanten unterscheiden sich die Graph Neural Networks nur vom MLP."
      : "ein neuronales Netz gleicher Größe ohne Kanten ist nicht gemessen."
  }`;
}

interface Gap {
  run: Run;
  difference: number;
  kind: Comparison;
}

function edgelessName(run: Run): string {
  const method: EdgelessMethod = run.method === "hgb" ? "hgb" : "mlp";
  return EDGELESS_NAME[method];
}

function edgelessDetail(run: Run): string {
  const method: EdgelessMethod = run.method === "hgb" ? "hgb" : "mlp";
  return EDGELESS_DETAIL[method];
}

function referenceText(reference: Run, count: number): string {
  return count > 1
    ? `dem stärksten gemessenen Verfahren ohne Kanten (${edgelessName(reference)})`
    : `dem Verfahren ohne Kanten (${edgelessName(reference)})`;
}

export function neighbourhoodShareSentence(reference: Run, count: number): string {
  return `Gegenüber ${referenceText(reference, count)} kommt auf diesem Netz der größere Teil des Vorsprungs der Graph Neural Networks aus der Nachbarschaft.`;
}

export function edgelessShareSentence(reference: Run, count: number, baseline: Run, graphRuns: readonly Run[]): string {
  const totals = graphRuns.map(
    (run, index) => `${index === 0 ? "" : "von "}${formatDecimal(run.prAuc - baseline.prAuc)} bei ${METHOD_SHORT_TEXT[run.method]}`,
  );
  return `Die Nachbarschaft bringt gegenüber ${referenceText(reference, count)} einen messbaren Vorsprung. Der größere Teil des Abstands zur besseren Baseline entsteht auf diesem Netz aber schon ohne Kanten: ${formatDecimal(
    reference.prAuc - baseline.prAuc,
  )} von ${joinList(totals)}.`;
}

export function undecidedShareSentence(reference: Run, count: number): string {
  return `Die Nachbarschaft bringt gegenüber ${referenceText(reference, count)} einen messbaren Vorsprung; ob er größer ist als der Gewinn ohne Kanten, lässt sich mit dem Mindestabstand nicht entscheiden.`;
}

export function noNeighbourhoodLeadSentence(reference: Run, count: number): string {
  return `Die Nachbarschaft bringt auf diesem Netz keinen belastbaren Vorsprung gegenüber ${referenceText(reference, count)}.`;
}

export function partialLeadSentence(reference: Run, count: number, ahead: readonly Run[]): string {
  const names = joinList(ahead.map((run) => METHOD_SHORT_TEXT[run.method]));
  return `Nur ${names} ${ahead.length === 1 ? "liegt" : "liegen"} um mehr als den Mindestabstand über ${referenceText(
    reference,
    count,
  )}; ein Vorsprung der Nachbarschaft ist deshalb nicht für jedes Graph Neural Network belastbar.`;
}

function relation(difference: number, kind: Comparison, above: string, below: string, near: string): string {
  const amount = formatDecimal(Math.abs(difference));
  if (kind === "higher") return `um ${amount} ${above}`;
  if (kind === "lower") return `um ${amount} ${below}`;
  return `nur ${amount} ${near}`;
}

function edgelessIntro(edgeless: readonly Run[]): string {
  const values = edgeless.map((run, index) => {
    const name = `${index === 0 ? "Das" : "das"} ${edgelessName(run)} (${edgelessDetail(run)})`;
    return index === 0 ? `${name} erreicht ${formatDecimal(run.prAuc)}` : `${name} ${formatDecimal(run.prAuc)}`;
  });
  const lead =
    edgeless.length === 1
      ? "Ein überwachtes Verfahren ohne Kanten trennt die Effekte: Es lernt aus denselben Labels wie die Graph Neural Networks und wählt am selben Validierungsteil, sieht aber keine Nachbarn."
      : `${edgeless.length === 2 ? "Zwei" : formatInteger(edgeless.length)} überwachte Verfahren ohne Kanten trennen die Effekte: Sie lernen aus denselben Labels wie die Graph Neural Networks und wählen am selben Validierungsteil, sehen aber keine Nachbarn.`;
  return `${lead} ${joinList(values)}.`;
}

function gapsTo(graphRuns: readonly Run[], reference: Run, margin: number): Gap[] {
  return graphRuns.map((run) => ({
    run,
    difference: run.prAuc - reference.prAuc,
    kind: compareScores(run.prAuc, reference.prAuc, margin),
  }));
}

export function neighbourhoodDominates(graphRuns: readonly Run[], reference: Run, baseline: Run, margin: number): boolean {
  const gain = Math.max(reference.prAuc - baseline.prAuc, 0);
  const gaps = gapsTo(graphRuns, reference, margin);
  return gaps.length > 0 && gaps.every(({ kind, difference }) => kind === "higher" && difference - gain > margin);
}

export function randomComparisonSentence(runs: readonly Run[], evaluation: EvaluationInfo): string {
  const quantile = formatDecimal(evaluation.randomPrAucQ95);
  const expected = formatDecimal(evaluation.randomPrAucExpected);
  const below = runs.filter((run) => !beatsRandomRankings(run.prAuc, evaluation));
  if (below.length === 0) {
    return `Alle Verfahren liegen über dem 95-%-Quantil zufälliger Rangfolgen (${quantile}) und ordnen Knoten mit Label illegal damit besser als Zufall.`;
  }
  const values = joinList(below.map((run) => `${METHOD_SHORT_TEXT[run.method]} (${formatDecimal(run.prAuc)})`));
  if (below.length === runs.length) {
    return `Kein Verfahren liegt über dem 95-%-Quantil zufälliger Rangfolgen (${quantile}). Bei ${values} ist die PR-AUC von der einer zufälligen Rangfolge (Erwartungswert ${expected}) nicht zu unterscheiden.`;
  }
  const names = joinList(below.map((run) => METHOD_SHORT_TEXT[run.method]));
  return `Alle Verfahren außer ${names} liegen über dem 95-%-Quantil zufälliger Rangfolgen (${quantile}) und ordnen Knoten mit Label illegal damit besser als Zufall. Bei ${values} ist die PR-AUC von der einer zufälligen Rangfolge (Erwartungswert ${expected}) nicht zu unterscheiden.`;
}

export function recallSentence(runs: readonly Run[], positives: number): string {
  const items = runs.map((run) => {
    const name = METHOD_SHORT_TEXT[run.method];
    if (run.recallAtPrecision50 === 0) return `${name} ${formatDecimal(0)} (keine Schwelle erreicht diese Precision)`;
    const hits = hitsAtRecall(run.recallAtPrecision50, positives);
    return `${name} ${formatDecimal(run.recallAtPrecision50)} (etwa ${formatInteger(hits)} von ${formatInteger(positives)})`;
  });
  const fewHits = runs.every((run) => run.recallAtPrecision50 < FEW_HITS_RECALL);
  return `Recall bei Precision ≥ 0,5 – ${items.join("; ")}.${
    fewHits ? " Bei dieser Precision finden die Verfahren kaum Knoten mit Label illegal." : ""
  }`;
}

export function compareItem(run: Run, reference: Run, margin: number): string {
  const difference = formatDecimal(Math.abs(run.prAuc - reference.prAuc));
  const name = METHOD_SHORT_TEXT[run.method];
  const value = formatDecimal(run.prAuc);
  const kind = compareScores(run.prAuc, reference.prAuc, margin);
  if (kind === "higher") return `${name} ${value}, um ${difference} höher`;
  if (kind === "lower") return `${name} ${value}, um ${difference} niedriger`;
  return `${name} ${value}, Unterschied ${difference}, kleiner als der Mindestabstand`;
}

export function baselineGapEffectsSentence(graphRuns: readonly Run[], baseline: Run): string {
  const differing = graphRuns.filter((run) => run.featureSet !== baseline.featureSet);
  if (differing.length === 0) {
    return "Dieser Abstand mischt zwei Effekte: Die Graph Neural Networks lernen aus Labels, die Baselines nicht, und nur die Graph Neural Networks sehen die Nachbarschaft.";
  }
  const all = differing.length === graphRuns.length;
  const subject = all
    ? "sie nutzen"
    : `${joinList(differing.map((run) => METHOD_SHORT_TEXT[run.method]))} ${differing.length === 1 ? "nutzt" : "nutzen"}`;
  const features =
    baseline.featureSet === "local"
      ? `${subject} zusätzlich die Graphmaße`
      : `${subject} anders als die bessere Baseline keine Graphmaße`;
  return `Dieser Abstand mischt mehrere Effekte: Die Graph Neural Networks lernen aus Labels, die Baselines nicht; ${features}; und nur ${
    all ? "sie" : "die Graph Neural Networks"
  } sehen die Nachbarschaft.`;
}

export function pairSentence(first: Run, second: Run, margin: number): string {
  const kind = compareScores(second.prAuc, first.prAuc, margin);
  const difference = formatDecimal(Math.abs(second.prAuc - first.prAuc));
  const a = METHOD_SHORT_TEXT[first.method];
  const b = METHOD_SHORT_TEXT[second.method];
  if (kind === "higher") return `${b} liegt um ${difference} vor ${a}.`;
  if (kind === "lower") return `${a} liegt um ${difference} vor ${b}.`;
  return `${a} und ${b} liegen nur ${difference} auseinander, weniger als der Mindestabstand; eine Rangfolge lässt sich daraus nicht ablesen.`;
}

export function baselineSentence(zscore: Run, iforest: Run, margin: number): string {
  const kind = compareScores(iforest.prAuc, zscore.prAuc, margin);
  const difference = formatDecimal(Math.abs(iforest.prAuc - zscore.prAuc));
  const lead =
    kind === "lower"
      ? `Der Isolation Forest mit zusätzlichen Graphmaßen liegt um ${difference} unter den robusten Z-Scores auf lokalen Merkmalen.`
      : kind === "higher"
        ? `Der Isolation Forest mit zusätzlichen Graphmaßen liegt um ${difference} über den robusten Z-Scores auf lokalen Merkmalen.`
        : `Der Isolation Forest mit zusätzlichen Graphmaßen und die robusten Z-Scores auf lokalen Merkmalen liegen nur ${difference} auseinander, weniger als der Mindestabstand; eine Rangfolge ist daraus nicht belastbar.`;
  return `${lead} Da sich Verfahren und Merkmalssatz gleichzeitig unterscheiden, lässt sich daraus nicht ablesen, ob die Graphmaße helfen.`;
}

export function graphLeadMeasured(graphRuns: readonly Run[], references: readonly (Run | null)[], margin: number): boolean {
  const measured = references.filter((reference): reference is Run => reference !== null);
  return (
    graphRuns.length > 0 &&
    measured.length > 0 &&
    graphRuns.every((run) => measured.every((reference) => compareScores(run.prAuc, reference.prAuc, margin) === "higher"))
  );
}

export function caveatTitle(leadMeasured: boolean): string {
  return leadMeasured
    ? "Vorbehalt: Der Vorsprung ist zum Teil eingebaut"
    : "Vorbehalt: Ein Vorsprung der Nachbarschaft wäre zum Teil eingebaut";
}

export function caveatConclusion(leadMeasured: boolean): string {
  return leadMeasured
    ? "Der gemessene Vorsprung der Graph Neural Networks ist deshalb zum Teil eine Eigenschaft dieses Generators und nicht auf reale Transaktionsdaten übertragbar. Wie stark er an der Homophilie hängt, zeigen erst Läufe mit absichtlich gestörter Nachbarschaft; sie fehlen noch."
    : "Nicht jedes Graph Neural Network liegt hier um mehr als den Mindestabstand vor den Verfahren ohne Kanten und der besseren Baseline. Ein Vorsprung wäre auf diesem Netz zum Teil eine Eigenschaft des Generators und nicht auf reale Transaktionsdaten übertragbar.";
}

export function homophilyLimitConclusion(leadMeasured: boolean): string {
  return leadMeasured
    ? "Der gemessene Vorsprung der Graph Neural Networks ist auf diesem Netz deshalb zum Teil eingebaut und nicht auf andere Daten übertragbar."
    : "Ein Vorsprung der Graph Neural Networks wäre auf diesem Netz deshalb zum Teil eingebaut und nicht auf andere Daten übertragbar.";
}

export function decompositionSentences(
  edgeless: readonly Run[],
  baseline: Run,
  graphRuns: readonly Run[],
  margin: number,
): string[] {
  const reference = bestRun(edgeless);
  if (reference === null) return [];
  const count = edgeless.length;
  const name = edgelessName(reference);
  const supervision = reference.prAuc - baseline.prAuc;
  const supervisionKind = compareScores(reference.prAuc, baseline.prAuc, margin);
  const baselineText = `${METHOD_SHORT_TEXT[baseline.method]}, ${formatDecimal(baseline.prAuc)}`;
  const featureNote =
    reference.featureSet === baseline.featureSet
      ? ""
      : ` und des Merkmalssatzes (das ${name} nutzt ${FEATURE_SET_TEXT[reference.featureSet]}, die bessere Baseline ${FEATURE_SET_TEXT[baseline.featureSet]})`;
  const sentences = [edgelessIntro(edgeless)];
  if (supervisionKind === "higher") {
    sentences.push(
      `Das ${name} liegt um ${formatDecimal(supervision)} über der besseren Baseline (${baselineText}). Dieser Abstand mischt die Wirkung der Labels mit dem Wechsel des Verfahrens${featureNote}.`,
    );
  } else if (supervisionKind === "lower") {
    sentences.push(
      `Das ${name} liegt um ${formatDecimal(-supervision)} unter der besseren Baseline (${baselineText}); die Labels allein helfen hier also nicht.`,
    );
  } else {
    sentences.push(
      `Das ${name} liegt nahe an der besseren Baseline (${baselineText}); die Labels allein bringen hier keinen belastbaren Vorsprung.`,
    );
  }
  const gaps = gapsTo(graphRuns, reference, margin);
  if (gaps.length === 0) return sentences;
  const mlp = edgeless.find((run) => run.method === "mlp" && run !== reference) ?? null;
  const control = edgeless
    .filter((run) => run !== reference)
    .map(
      (other) =>
        ` Gegenüber dem ${edgelessName(other)}: ${gapsTo(graphRuns, other, margin)
          .map(
            ({ run, difference, kind }) =>
              `${METHOD_SHORT_TEXT[run.method]} ${relation(difference, kind, "darüber", "darunter", "entfernt")}`,
          )
          .join(", ")}.`,
    )
    .join("");
  const chosen = edgeless.some((run) => run.method === "hgb") ? "Epochen- bzw. Baumzahl" : "Epochenzahl";
  sentences.push(
    `Mit Nachbarschaft: ${gaps
      .map(
        ({ run, difference, kind }) =>
          `${METHOD_SHORT_TEXT[run.method]} ${formatDecimal(run.prAuc)} (${relation(
            difference,
            kind,
            `über dem ${name}`,
            `unter dem ${name}`,
            `vom ${name} entfernt`,
          )})`,
      )
      .join(", ")}.${control} Merkmalssatz, Gewicht und ${chosen} wählt dabei jedes Verfahren für sich.`,
  );
  const gain = Math.max(supervision, 0);
  const ahead = gaps.filter(({ kind }) => kind === "higher").map(({ run }) => run);
  const methodChange = reference.method === "hgb" ? [methodChangeSentence(mlp !== null)] : [];
  const contrast = mlp !== null && neighbourhoodDominates(graphRuns, mlp, baseline, margin) ? [MLP_CONTRAST_SENTENCE] : [];
  if (ahead.length === gaps.length) {
    if (gaps.every(({ difference }) => difference - gain > margin)) {
      sentences.push(neighbourhoodShareSentence(reference, count), ...methodChange);
    } else if (gaps.every(({ difference }) => gain - difference > margin)) {
      sentences.push(edgelessShareSentence(reference, count, baseline, graphRuns), ...methodChange, ...contrast);
    } else {
      sentences.push(undecidedShareSentence(reference, count), ...methodChange, ...contrast);
    }
  } else if (ahead.length === 0) {
    sentences.push(noNeighbourhoodLeadSentence(reference, count), ...methodChange);
  } else {
    sentences.push(partialLeadSentence(reference, count, ahead), ...methodChange);
  }
  return sentences;
}

export function spreadNote(runs: readonly Run[]): string {
  const learned = runs.filter((run) => seedSpreadOf(run) !== null);
  if (learned.length === 0) return "";
  const seeds = spreadSeeds(runs);
  const seedText = seeds === null ? "mehrere Seeds" : `die Seeds ${formatSeedList(seeds)}`;
  const constant = learned.filter((run) => {
    const spread = seedSpreadOf(run);
    return spread !== null && !hasSpread(spread);
  });
  const boosting = constant.filter((run) => run.training?.kind === "boosting");
  const other = constant.filter((run) => run.training?.kind !== "boosting");
  const varying = learned.filter((run) => !constant.includes(run));
  const names = (group: readonly Run[]) => joinList(group.map((run) => METHOD_SHORT_TEXT[run.method]));
  const hasBoosting = learned.some((run) => run.training?.kind === "boosting");
  const hasNeural = learned.some((run) => run.training?.kind === "neural");
  const length = hasBoosting && hasNeural ? "Epochen- bzw. Baumzahl" : hasBoosting ? "Baumzahl" : "Epochenzahl";
  const parts = [
    `Spanne: kleinster bis größter Wert der PR-AUC über ${seedText} und ihr Mittel, nur bei den gelernten Verfahren. Mit jedem Seed wird nur das Endmodell neu trainiert; Daten, Merkmalssatz, Gewicht und ${length} bleiben gleich.`,
  ];
  if (boosting.length > 0) {
    parts.push(
      `Beim ${names(boosting)} ist die Spanne 0: Ohne Stichprobe von Zeilen oder Merkmalen und ohne Early Stopping zieht das Verfahren nichts zufällig, jeder Seed liefert dasselbe Modell.`,
    );
  }
  if (other.length > 0) parts.push(`Bei ${names(other)} sind alle Werte gleich.`);
  if (varying.length > 0) {
    parts.push(
      `Bei ${names(varying)} zeigt die Spanne, wie stark Startwerte der Gewichte und Dropout die Kennzahl verschieben.`,
    );
  }
  parts.push("Ein Maß für die Unsicherheit durch die Daten ist die Spanne bei keinem Verfahren.");
  return parts.join(" ");
}

export function accuracySentence(runs: readonly Run[], evaluation: EvaluationInfo): string {
  const allLicit = formatDecimal(evaluation.allLicitAccuracy);
  if (runs.every((run) => run.accuracy < evaluation.allLicitAccuracy)) {
    return `Bei der Accuracy liegen alle gemessenen Verfahren unter den ${allLicit} eines Modells, das keinen Knoten markiert. Deshalb ist Accuracy hier nur Nebenwert.`;
  }
  const range = `Die Accuracy liegt zwischen ${formatDecimal(Math.min(...runs.map((run) => run.accuracy)))} und ${formatDecimal(
    Math.max(...runs.map((run) => run.accuracy)),
  )}; ein Modell, das keinen Knoten markiert, erreicht ${allLicit}.`;
  const groups = new Map<number, Run[]>();
  for (const run of runs) {
    if (run.accuracyTruePositives !== run.accuracyFlagged) continue;
    groups.set(run.accuracyFlagged, [...(groups.get(run.accuracyFlagged) ?? []), run]);
  }
  const perfect = [...groups.entries()].map(([flagged, group]) => {
    const best = bestAccuracyAtFlagged(flagged, evaluation);
    return `${joinList(group.map((run) => METHOD_SHORT_TEXT[run.method]))} ${
      group.length === 1 ? "markiert" : "markieren je"
    } ${formatInteger(flagged)} Knoten, alle mit Label illegal. Mehr als ${formatDecimal(best)} ist bei ${formatInteger(
      flagged,
    )} markierten Knoten nicht möglich, und das sind nur ${formatDecimal(
      best - evaluation.allLicitAccuracy,
    )} mehr als bei „nichts markiert“.`;
  });
  return perfect.length === 0
    ? `${range} Deshalb ist Accuracy hier nur Nebenwert.`
    : `${range} ${perfect.join(" ")} Deshalb ist Accuracy hier nur Nebenwert.`;
}
