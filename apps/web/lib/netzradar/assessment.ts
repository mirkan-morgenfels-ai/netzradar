import { FEATURE_SET_TEXT, formatDecimal, formatInteger, joinList, METHOD_SHORT_TEXT, METHOD_TEXT } from "./format";
import { beatsRandomRankings, bestAccuracyAtFlagged, compareScores } from "./summary";
import type { EvaluationInfo, Run } from "./types";

export const NEIGHBOURHOOD_SHARE_SENTENCE =
  "Auf diesem Netz kommt der größere Teil des Vorsprungs der Graph Neural Networks damit aus der Nachbarschaft, nicht aus den Labels allein.";

export const NO_NEIGHBOURHOOD_LEAD_SENTENCE =
  "Die Nachbarschaft bringt auf diesem Netz keinen belastbaren Vorsprung gegenüber dem MLP.";

export function randomSentence(run: Run, evaluation: EvaluationInfo): string {
  const quantile = formatDecimal(evaluation.randomPrAucQ95);
  if (beatsRandomRankings(run.prAuc, evaluation)) {
    return `${run.displayName}: PR-AUC ${formatDecimal(run.prAuc)}, über dem 95-%-Quantil zufälliger Rangfolgen (${quantile}). Der Score ordnet auffällige Knoten also besser als Zufall.`;
  }
  return `${run.displayName}: PR-AUC ${formatDecimal(run.prAuc)}, nicht über dem 95-%-Quantil zufälliger Rangfolgen (${quantile}). Von einer zufälligen Rangfolge (Erwartungswert ${formatDecimal(
    evaluation.randomPrAucExpected,
  )}) ist das nicht zu unterscheiden.`;
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
      ? `Der Isolation Forest mit zusätzlichen Graphmaßen liegt um ${difference} unter den robusten Z-Scores auf Einzelmerkmalen.`
      : kind === "higher"
        ? `Der Isolation Forest mit zusätzlichen Graphmaßen liegt um ${difference} über den robusten Z-Scores auf Einzelmerkmalen.`
        : `Der Isolation Forest mit zusätzlichen Graphmaßen und die robusten Z-Scores auf Einzelmerkmalen liegen nur ${difference} auseinander, weniger als der Mindestabstand; eine Rangfolge ist daraus nicht belastbar.`;
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
    : "Nicht jedes Graph Neural Network liegt hier um mehr als den Mindestabstand vor dem MLP und der besseren Baseline. Ein Vorsprung wäre auf diesem Netz zum Teil eine Eigenschaft des Generators und nicht auf reale Transaktionsdaten übertragbar.";
}

export function homophilyLimitConclusion(leadMeasured: boolean): string {
  return leadMeasured
    ? "Der gemessene Vorsprung der Graph Neural Networks ist auf diesem Netz deshalb zum Teil eingebaut und nicht auf andere Daten übertragbar."
    : "Ein Vorsprung der Graph Neural Networks wäre auf diesem Netz deshalb zum Teil eingebaut und nicht auf andere Daten übertragbar.";
}

export function decompositionSentences(mlp: Run, baseline: Run, graphRuns: readonly Run[], margin: number): string[] {
  const supervision = mlp.prAuc - baseline.prAuc;
  const supervisionKind = compareScores(mlp.prAuc, baseline.prAuc, margin);
  const featureNote =
    mlp.featureSet === baseline.featureSet
      ? ""
      : ` und des Merkmalssatzes (das MLP nutzt ${FEATURE_SET_TEXT[mlp.featureSet]}, die bessere Baseline ${FEATURE_SET_TEXT[baseline.featureSet]})`;
  const sentences = [
    `Die Kontrolle trennt zwei Effekte. Das MLP ohne Kanten nutzt dieselben Labels und dasselbe Auswahlverfahren wie die Graph Neural Networks (dieselben Kandidaten für Merkmalssatz und Klassengewicht, Auswahl am Validierungsteil), sieht aber keine Nachbarn. Es erreicht ${formatDecimal(mlp.prAuc)}.`,
  ];
  if (supervisionKind === "higher") {
    sentences.push(
      `Das sind ${formatDecimal(supervision)} mehr als die bessere Baseline (${METHOD_TEXT[baseline.method]}, ${formatDecimal(
        baseline.prAuc,
      )}). Dieser Abstand mischt die Wirkung der Labels mit dem Wechsel des Verfahrens${featureNote}.`,
    );
  } else if (supervisionKind === "lower") {
    sentences.push(
      `Das sind ${formatDecimal(-supervision)} weniger als die bessere Baseline (${METHOD_TEXT[baseline.method]}, ${formatDecimal(
        baseline.prAuc,
      )}); die Labels allein helfen hier also nicht.`,
    );
  } else {
    sentences.push(
      `Das liegt nahe an der besseren Baseline (${METHOD_TEXT[baseline.method]}, ${formatDecimal(
        baseline.prAuc,
      )}); die Labels allein bringen hier keinen belastbaren Vorsprung.`,
    );
  }
  const neighbourhood = graphRuns.map((run) => ({
    run,
    difference: run.prAuc - mlp.prAuc,
    kind: compareScores(run.prAuc, mlp.prAuc, margin),
  }));
  sentences.push(
    `Mit Nachbarschaft, bei gleichem Auswahlverfahren: ${neighbourhood
      .map(({ run, difference, kind }) => {
        const amount = formatDecimal(Math.abs(difference));
        const relation =
          kind === "higher"
            ? `um ${amount} über dem MLP`
            : kind === "lower"
              ? `um ${amount} unter dem MLP`
              : `nur ${amount} vom MLP entfernt`;
        return `${METHOD_SHORT_TEXT[run.method]} ${formatDecimal(run.prAuc)} (${relation})`;
      })
      .join(", ")}. Merkmalssatz, Gewicht und Epochenzahl wählt dabei jedes Verfahren für sich.`,
  );
  const supervisionGain = Math.max(supervision, 0);
  if (
    neighbourhood.length > 0 &&
    neighbourhood.every(({ difference, kind }) => kind === "higher" && difference - supervisionGain > margin)
  ) {
    sentences.push(NEIGHBOURHOOD_SHARE_SENTENCE);
  } else if (neighbourhood.length > 0 && neighbourhood.every(({ kind }) => kind !== "higher")) {
    sentences.push(NO_NEIGHBOURHOOD_LEAD_SENTENCE);
  }
  return sentences;
}

export function accuracySentence(runs: readonly Run[], evaluation: EvaluationInfo): string {
  const allLicit = formatDecimal(evaluation.allLicitAccuracy);
  if (runs.every((run) => run.accuracy < evaluation.allLicitAccuracy)) {
    return `Bei der Accuracy liegen alle gemessenen Verfahren unter den ${allLicit} eines Modells, das alles unauffällig nennt. Deshalb ist Accuracy hier nur Nebenwert.`;
  }
  const range = `Die Accuracy liegt zwischen ${formatDecimal(Math.min(...runs.map((run) => run.accuracy)))} und ${formatDecimal(
    Math.max(...runs.map((run) => run.accuracy)),
  )}; ein Modell, das alles unauffällig nennt, erreicht ${allLicit}.`;
  const groups = new Map<number, Run[]>();
  for (const run of runs) {
    if (run.accuracyTruePositives !== run.accuracyFlagged) continue;
    groups.set(run.accuracyFlagged, [...(groups.get(run.accuracyFlagged) ?? []), run]);
  }
  const perfect = [...groups.entries()].map(([flagged, group]) => {
    const best = bestAccuracyAtFlagged(flagged, evaluation);
    return `${joinList(group.map((run) => METHOD_SHORT_TEXT[run.method]))} ${
      group.length === 1 ? "markiert" : "markieren je"
    } ${formatInteger(flagged)} Knoten, alle auffällig. Mehr als ${formatDecimal(best)} ist bei ${formatInteger(
      flagged,
    )} markierten Knoten nicht möglich, und das sind nur ${formatDecimal(
      best - evaluation.allLicitAccuracy,
    )} mehr als bei „alles unauffällig“.`;
  });
  return perfect.length === 0
    ? `${range} Deshalb ist Accuracy hier nur Nebenwert.`
    : `${range} ${perfect.join(" ")} Deshalb ist Accuracy hier nur Nebenwert.`;
}
