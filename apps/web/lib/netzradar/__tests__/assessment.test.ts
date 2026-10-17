import { describe, expect, it } from "vitest";
import {
  MLP_CONTRAST_SENTENCE,
  accuracySentence,
  baselineGapEffectsSentence,
  baselineSentence,
  boostingReferenceSentence,
  caveatConclusion,
  caveatTitle,
  compareItem,
  decompositionSentences,
  edgelessShareSentence,
  graphLeadMeasured,
  homophilyLimitConclusion,
  methodChangeSentence,
  neighbourhoodDominates,
  neighbourhoodShareSentence,
  neuralShapeSubject,
  noNeighbourhoodLeadSentence,
  pairSentence,
  partialLeadSentence,
  randomComparisonSentence,
  recallSentence,
  spreadNote,
  undecidedShareSentence,
} from "../assessment";
import type { Run } from "../types";
import { EVALUATION, boostingRun, learnedRun, run, spread } from "./fixtures";

const MARGIN = 0.0239;
const ZSCORE = run("zscore", 0.1633);
const IFOREST = run("iforest", 0.1281, { featureSet: "local+graph" });
const HGB = boostingRun(0.6087);
const GCN = learnedRun("gcn", 0.7897, 0.7364);
const GRAPHSAGE = learnedRun("graphsage", 0.8941, 0.9349);
const MLP = learnedRun("mlp", 0.3765, 0.497);

const TWO_INTRO =
  "Zwei überwachte Verfahren ohne Kanten trennen die Effekte: Sie lernen aus denselben Labels wie die Graph Neural Networks und wählen am selben Validierungsteil, sehen aber keine Nachbarn.";
const ONE_INTRO =
  "Ein überwachtes Verfahren ohne Kanten trennt die Effekte: Es lernt aus denselben Labels wie die Graph Neural Networks und wählt am selben Validierungsteil, sieht aber keine Nachbarn.";
const HGB_DETAIL =
  "übliches starkes Verfahren für tabellarische Daten, eigene Suche über Merkmalssatz, Klassengewicht und Blattzahl";
const MLP_DETAIL = "gleiche Größe und dieselben Kandidaten für Merkmalssatz und Klassengewicht wie die Graph Neural Networks";
const PER_METHOD_BOTH = "Merkmalssatz, Gewicht und Epochen- bzw. Baumzahl wählt dabei jedes Verfahren für sich.";
const PER_METHOD_EPOCHS = "Merkmalssatz, Gewicht und Epochenzahl wählt dabei jedes Verfahren für sich.";
const STRONGEST = "dem stärksten gemessenen Verfahren ohne Kanten (Gradient Boosting)";

describe("decompositionSentences against the strongest method without edges", () => {
  it("splits the real values against the gradient boosting: 0,6087 − 0,1633 = 0,4454 exceeds 0,8941 − 0,6087 = 0,2854 by 0,1600 > 0,0239", () => {
    expect(decompositionSentences([HGB, MLP], ZSCORE, [GCN, GRAPHSAGE], MARGIN)).toEqual([
      `${TWO_INTRO} Das Gradient Boosting (${HGB_DETAIL}) erreicht 0,6087 und das MLP (${MLP_DETAIL}) 0,3765.`,
      "Das Gradient Boosting liegt um 0,4454 über der besseren Baseline (Z-Scores, 0,1633). Dieser Abstand mischt die Wirkung der Labels mit dem Wechsel des Verfahrens und des Merkmalssatzes (das Gradient Boosting nutzt lokale Merkmale und Graphmaße, die bessere Baseline nur lokale Merkmale).",
      `Mit Nachbarschaft: GCN 0,7897 (um 0,1810 über dem Gradient Boosting), GraphSAGE 0,8941 (um 0,2854 über dem Gradient Boosting). Gegenüber dem MLP: GCN um 0,4132 darüber, GraphSAGE um 0,5176 darüber. ${PER_METHOD_BOTH}`,
      `Die Nachbarschaft bringt gegenüber ${STRONGEST} einen messbaren Vorsprung. Der größere Teil des Abstands zur besseren Baseline entsteht auf diesem Netz aber schon ohne Kanten: 0,4454 von 0,6264 bei GCN und von 0,7308 bei GraphSAGE.`,
      "Gegenüber dem Gradient Boosting wechselt neben den Kanten auch das Verfahren, Entscheidungsbäume statt neuronaler Netze; allein in den Kanten unterscheiden sich die Graph Neural Networks nur vom MLP.",
      MLP_CONTRAST_SENTENCE,
    ]);
  });

  it("never claims the larger part for the neighbourhood with the real values", () => {
    const text = decompositionSentences([HGB, MLP], ZSCORE, [GCN, GRAPHSAGE], MARGIN).join(" ");
    expect(text).not.toContain(neighbourhoodShareSentence(HGB, 2));
    expect(text).not.toContain("kommt auf diesem Netz der größere Teil des Vorsprungs der Graph Neural Networks aus der Nachbarschaft");
  });

  it("keeps the reading against the MLP alone: 0,4132 − 0,2132 = 0,2000 > 0,0239", () => {
    expect(decompositionSentences([MLP], ZSCORE, [GCN, GRAPHSAGE], MARGIN)).toEqual([
      `${ONE_INTRO} Das MLP (${MLP_DETAIL}) erreicht 0,3765.`,
      "Das MLP liegt um 0,2132 über der besseren Baseline (Z-Scores, 0,1633). Dieser Abstand mischt die Wirkung der Labels mit dem Wechsel des Verfahrens und des Merkmalssatzes (das MLP nutzt lokale Merkmale und Graphmaße, die bessere Baseline nur lokale Merkmale).",
      `Mit Nachbarschaft: GCN 0,7897 (um 0,4132 über dem MLP), GraphSAGE 0,8941 (um 0,5176 über dem MLP). ${PER_METHOD_EPOCHS}`,
      "Gegenüber dem Verfahren ohne Kanten (MLP) kommt auf diesem Netz der größere Teil des Vorsprungs der Graph Neural Networks aus der Nachbarschaft.",
    ]);
    expect(neighbourhoodDominates([GCN, GRAPHSAGE], MLP, ZSCORE, MARGIN)).toBe(true);
    expect(neighbourhoodDominates([GCN, GRAPHSAGE], HGB, ZSCORE, MARGIN)).toBe(false);
  });

  it("gives the neighbourhood the larger part against a weak gradient boosting: 0,4 − 0,1 = 0,3 > 0,0239", () => {
    const sentences = decompositionSentences(
      [boostingRun(0.3), learnedRun("mlp", 0.25, 0.5)],
      run("zscore", 0.2),
      [learnedRun("gcn", 0.6, 0.5), learnedRun("graphsage", 0.7, 0.5)],
      MARGIN,
    );
    expect(sentences.slice(3)).toEqual([
      `Gegenüber ${STRONGEST} kommt auf diesem Netz der größere Teil des Vorsprungs der Graph Neural Networks aus der Nachbarschaft.`,
      methodChangeSentence(true),
    ]);
  });

  it("cannot decide when the gain without edges 0,30 and the lead with edges 0,29 and 0,31 differ by less than the margin", () => {
    const sentences = decompositionSentences(
      [boostingRun(0.5), learnedRun("mlp", 0.3, 0.5)],
      run("zscore", 0.2),
      [learnedRun("gcn", 0.79, 0.5), learnedRun("graphsage", 0.81, 0.5)],
      MARGIN,
    );
    expect(sentences.slice(3)).toEqual([
      `Die Nachbarschaft bringt gegenüber ${STRONGEST} einen messbaren Vorsprung; ob er größer ist als der Gewinn ohne Kanten, lässt sich mit dem Mindestabstand nicht entscheiden.`,
      methodChangeSentence(true),
      MLP_CONTRAST_SENTENCE,
    ]);
  });

  it("names the graph method that leads by more than the margin when the other does not: 0,61 − 0,60 = 0,01", () => {
    const sentences = decompositionSentences(
      [boostingRun(0.6)],
      run("zscore", 0.2),
      [learnedRun("gcn", 0.61, 0.5), learnedRun("graphsage", 0.7, 0.5)],
      MARGIN,
    );
    expect(sentences[2]).toBe(
      "Mit Nachbarschaft: GCN 0,6100 (nur 0,0100 vom Gradient Boosting entfernt), GraphSAGE 0,7000 (um 0,1000 über dem Gradient Boosting). Merkmalssatz, Gewicht und Epochen- bzw. Baumzahl wählt dabei jedes Verfahren für sich.",
    );
    expect(sentences.slice(3)).toEqual([
      "Nur GraphSAGE liegt um mehr als den Mindestabstand über dem Verfahren ohne Kanten (Gradient Boosting); ein Vorsprung der Nachbarschaft ist deshalb nicht für jedes Graph Neural Network belastbar.",
      "Gegenüber dem Gradient Boosting wechselt neben den Kanten auch das Verfahren, Entscheidungsbäume statt neuronaler Netze; ein neuronales Netz gleicher Größe ohne Kanten ist nicht gemessen.",
    ]);
  });

  it("reports an edgeless method below the baseline and graph methods within the margin", () => {
    expect(
      decompositionSentences(
        [learnedRun("mlp", 0.3, 0.5)],
        run("zscore", 0.4),
        [learnedRun("gcn", 0.31, 0.5), learnedRun("graphsage", 0.29, 0.5)],
        MARGIN,
      ),
    ).toEqual([
      `${ONE_INTRO} Das MLP (${MLP_DETAIL}) erreicht 0,3000.`,
      "Das MLP liegt um 0,1000 unter der besseren Baseline (Z-Scores, 0,4000); die Labels allein helfen hier also nicht.",
      `Mit Nachbarschaft: GCN 0,3100 (nur 0,0100 vom MLP entfernt), GraphSAGE 0,2900 (nur 0,0100 vom MLP entfernt). ${PER_METHOD_EPOCHS}`,
      noNeighbourhoodLeadSentence(learnedRun("mlp", 0.3, 0.5), 1),
    ]);
  });

  it("calls an edgeless method within the margin of the baseline no reliable lead: 0,32 − 0,30 = 0,02", () => {
    const sentences = decompositionSentences(
      [learnedRun("mlp", 0.32, 0.5)],
      run("zscore", 0.3),
      [learnedRun("gcn", 0.35, 0.5), learnedRun("graphsage", 0.35, 0.5)],
      MARGIN,
    );
    expect(sentences[1]).toBe(
      "Das MLP liegt nahe an der besseren Baseline (Z-Scores, 0,3000); die Labels allein bringen hier keinen belastbaren Vorsprung.",
    );
    expect(sentences[3]).toBe(undecidedShareSentence(learnedRun("mlp", 0.32, 0.5), 1));
  });

  it("takes the MLP as reference when it is stronger than the gradient boosting and keeps the other as comparison", () => {
    const sentences = decompositionSentences(
      [boostingRun(0.3), learnedRun("mlp", 0.4, 0.5)],
      run("zscore", 0.2),
      [learnedRun("gcn", 0.8, 0.5)],
      MARGIN,
    );
    expect(sentences[1]).toMatch(/^Das MLP liegt um 0,2000 über der besseren Baseline/);
    expect(sentences[2]).toBe(
      "Mit Nachbarschaft: GCN 0,8000 (um 0,4000 über dem MLP). Gegenüber dem Gradient Boosting: GCN um 0,5000 darüber. Merkmalssatz, Gewicht und Epochen- bzw. Baumzahl wählt dabei jedes Verfahren für sich.",
    );
    expect(sentences.slice(3)).toEqual([
      "Gegenüber dem stärksten gemessenen Verfahren ohne Kanten (MLP) kommt auf diesem Netz der größere Teil des Vorsprungs der Graph Neural Networks aus der Nachbarschaft.",
    ]);
  });

  it("returns only the intro without graph methods", () => {
    expect(decompositionSentences([HGB, MLP], ZSCORE, [], MARGIN)).toHaveLength(2);
    expect(decompositionSentences([], ZSCORE, [GCN], MARGIN)).toEqual([]);
  });

  it("builds the share sentences with the reference and the totals", () => {
    expect(edgelessShareSentence(HGB, 1, ZSCORE, [GCN])).toBe(
      "Die Nachbarschaft bringt gegenüber dem Verfahren ohne Kanten (Gradient Boosting) einen messbaren Vorsprung. Der größere Teil des Abstands zur besseren Baseline entsteht auf diesem Netz aber schon ohne Kanten: 0,4454 von 0,6264 bei GCN.",
    );
    expect(partialLeadSentence(HGB, 2, [GCN, GRAPHSAGE])).toMatch(/^Nur GCN und GraphSAGE liegen um mehr als den Mindestabstand über/);
  });
});

describe("spreadNote", () => {
  const gcn = learnedRun("gcn", 0.7897, 0.7364, { seedSpread: spread([0.7897, 0.8148, 0.8008, 0.7947, 0.8009]) });
  const graphsage = learnedRun("graphsage", 0.8941, 0.9349, { seedSpread: spread([0.8941, 0.8915, 0.8876, 0.8689, 0.879]) });
  const mlp = learnedRun("mlp", 0.3765, 0.497, { seedSpread: spread([0.3765, 0.3792, 0.3418, 0.401, 0.3935]) });

  it("explains the spread 0 of the gradient boosting and the spread of the neural runs with the real seeds", () => {
    expect(spreadNote([ZSCORE, IFOREST, HGB, gcn, graphsage, mlp])).toBe(
      "Spanne: kleinster bis größter Wert der PR-AUC über die Seeds 42 bis 46 und ihr Mittel, nur bei den gelernten Verfahren. Mit jedem Seed wird nur das Endmodell neu trainiert; Daten, Merkmalssatz, Gewicht und Epochen- bzw. Baumzahl bleiben gleich. Beim Gradient Boosting ist die Spanne 0: Ohne Stichprobe von Zeilen oder Merkmalen und ohne Early Stopping zieht das Verfahren nichts zufällig, jeder Seed liefert dasselbe Modell. Bei GCN, GraphSAGE und MLP (ohne Kanten) zeigt die Spanne, wie stark Startwerte der Gewichte und Dropout die Kennzahl verschieben. Ein Maß für die Unsicherheit durch die Daten ist die Spanne bei keinem Verfahren.",
    );
  });

  it("stays empty without learned runs and names equal neural values", () => {
    expect(spreadNote([ZSCORE, IFOREST])).toBe("");
    const flat = learnedRun("gcn", 0.5, 0.5, { seedSpread: spread([0.5, 0.5, 0.5, 0.5, 0.5]) });
    expect(spreadNote([flat])).toContain("Epochenzahl bleiben gleich. Bei GCN sind alle Werte gleich.");
  });

  it("falls back to several seeds when the runs used different seeds", () => {
    const other = learnedRun("mlp", 0.3, 0.5, {
      seedSpread: spread([0.3, 0.31], { seeds: [1, 2], reportedSeed: 1 }),
    });
    expect(spreadNote([gcn, other])).toContain("über mehrere Seeds und ihr Mittel");
  });

  it("uses only the tree count for a gradient boosting alone", () => {
    expect(spreadNote([HGB])).toContain("Gewicht und Baumzahl bleiben gleich.");
  });

  it("never reads the spread as uncertainty of the data", () => {
    expect(spreadNote([HGB, gcn])).toMatch(/Ein Maß für die Unsicherheit durch die Daten ist die Spanne bei keinem Verfahren\.$/);
  });
});

describe("baselineGapEffectsSentence", () => {
  it("names the graph measures when both graph methods use local+graph and the Z-Scores only local features", () => {
    expect(baselineGapEffectsSentence([GCN, GRAPHSAGE], ZSCORE)).toBe(
      "Dieser Abstand mischt mehrere Effekte: Die Graph Neural Networks lernen aus Labels, die Baselines nicht; sie nutzen zusätzlich die Graphmaße; und nur sie sehen die Nachbarschaft.",
    );
  });

  it("keeps two effects when the better baseline uses the same feature set", () => {
    expect(baselineGapEffectsSentence([GCN, GRAPHSAGE], IFOREST)).toBe(
      "Dieser Abstand mischt zwei Effekte: Die Graph Neural Networks lernen aus Labels, die Baselines nicht, und nur die Graph Neural Networks sehen die Nachbarschaft.",
    );
  });

  it("names only the graph method whose feature set differs", () => {
    const localGcn: Run = { ...GCN, featureSet: "local" };
    expect(baselineGapEffectsSentence([localGcn, GRAPHSAGE], ZSCORE)).toBe(
      "Dieser Abstand mischt mehrere Effekte: Die Graph Neural Networks lernen aus Labels, die Baselines nicht; GraphSAGE nutzt zusätzlich die Graphmaße; und nur die Graph Neural Networks sehen die Nachbarschaft.",
    );
  });

  it("says that the graph methods lack the graph measures of a local+graph baseline", () => {
    const localGcn: Run = { ...GCN, featureSet: "local" };
    const localGraphsage: Run = { ...GRAPHSAGE, featureSet: "local" };
    expect(baselineGapEffectsSentence([localGcn, localGraphsage], IFOREST)).toBe(
      "Dieser Abstand mischt mehrere Effekte: Die Graph Neural Networks lernen aus Labels, die Baselines nicht; sie nutzen anders als die bessere Baseline keine Graphmaße; und nur sie sehen die Nachbarschaft.",
    );
  });
});

describe("pairSentence", () => {
  it("ranks GraphSAGE 0,1044 ahead of GCN with the real values", () => {
    expect(pairSentence(GCN, GRAPHSAGE, MARGIN)).toBe("GraphSAGE liegt um 0,1044 vor GCN.");
    expect(pairSentence(learnedRun("gcn", 0.9, 0.5), learnedRun("graphsage", 0.8, 0.5), MARGIN)).toBe(
      "GCN liegt um 0,1000 vor GraphSAGE.",
    );
  });

  it("refuses a ranking within the margin", () => {
    expect(pairSentence(learnedRun("gcn", 0.8, 0.5), learnedRun("graphsage", 0.81, 0.5), MARGIN)).toBe(
      "GCN und GraphSAGE liegen nur 0,0100 auseinander, weniger als der Mindestabstand; eine Rangfolge lässt sich daraus nicht ablesen.",
    );
  });
});

describe("baselineSentence", () => {
  const tail = "Da sich Verfahren und Merkmalssatz gleichzeitig unterscheiden, lässt sich daraus nicht ablesen, ob die Graphmaße helfen.";

  it("reads 0,1633 − 0,1281 = 0,0352 > 0,0239 as a ranking", () => {
    expect(baselineSentence(ZSCORE, IFOREST, MARGIN)).toBe(
      `Der Isolation Forest mit zusätzlichen Graphmaßen liegt um 0,0352 unter den robusten Z-Scores auf lokalen Merkmalen. ${tail}`,
    );
    expect(baselineSentence(run("zscore", 0.15), run("iforest", 0.2), MARGIN)).toBe(
      `Der Isolation Forest mit zusätzlichen Graphmaßen liegt um 0,0500 über den robusten Z-Scores auf lokalen Merkmalen. ${tail}`,
    );
  });

  it("calls a difference of 0,0100 not reliable", () => {
    expect(baselineSentence(run("zscore", 0.15), run("iforest", 0.16), MARGIN)).toBe(
      `Der Isolation Forest mit zusätzlichen Graphmaßen und die robusten Z-Scores auf lokalen Merkmalen liegen nur 0,0100 auseinander, weniger als der Mindestabstand; eine Rangfolge ist daraus nicht belastbar. ${tail}`,
    );
  });
});

describe("compareItem", () => {
  it("states the difference to the better baseline", () => {
    expect(compareItem(GCN, ZSCORE, MARGIN)).toBe("GCN 0,7897, um 0,6264 höher");
    expect(compareItem(learnedRun("gcn", 0.1, 0.5), run("zscore", 0.2), MARGIN)).toBe("GCN 0,1000, um 0,1000 niedriger");
    expect(compareItem(learnedRun("graphsage", 0.17, 0.5), ZSCORE, MARGIN)).toBe(
      "GraphSAGE 0,1700, Unterschied 0,0067, kleiner als der Mindestabstand",
    );
  });
});

describe("graphLeadMeasured", () => {
  it("needs every graph method above every measured reference by more than the margin", () => {
    expect(graphLeadMeasured([GCN, GRAPHSAGE], [ZSCORE, HGB, MLP], MARGIN)).toBe(true);
    expect(graphLeadMeasured([GCN, GRAPHSAGE], [ZSCORE, MLP], MARGIN)).toBe(true);
    expect(graphLeadMeasured([GCN, learnedRun("graphsage", 0.62, 0.5)], [ZSCORE, HGB, MLP], MARGIN)).toBe(false);
    expect(graphLeadMeasured([GCN, GRAPHSAGE], [ZSCORE, null], MARGIN)).toBe(true);
    expect(graphLeadMeasured([GCN, learnedRun("graphsage", 0.39, 0.5)], [ZSCORE, MLP], MARGIN)).toBe(false);
    expect(graphLeadMeasured([learnedRun("gcn", 0.18, 0.5)], [ZSCORE, null], MARGIN)).toBe(false);
    expect(graphLeadMeasured([], [ZSCORE, MLP], MARGIN)).toBe(false);
    expect(graphLeadMeasured([GCN], [null, null], MARGIN)).toBe(false);
  });

  it("switches the caveat between a measured and a possible lead", () => {
    expect(caveatTitle(true)).toBe("Vorbehalt: Der Vorsprung ist zum Teil eingebaut");
    expect(caveatTitle(false)).toBe("Vorbehalt: Ein Vorsprung der Nachbarschaft wäre zum Teil eingebaut");
    expect(caveatConclusion(true)).toMatch(/^Der gemessene Vorsprung der Graph Neural Networks/);
    expect(caveatConclusion(false)).toMatch(/^Nicht jedes Graph Neural Network liegt hier um mehr als den Mindestabstand/);
    expect(caveatConclusion(false)).not.toContain("gemessene Vorsprung");
    expect(homophilyLimitConclusion(true)).toMatch(/^Der gemessene Vorsprung/);
    expect(homophilyLimitConclusion(false)).toMatch(/^Ein Vorsprung der Graph Neural Networks wäre/);
  });
});

describe("accuracySentence", () => {
  const runs = [
    run("zscore", 0.1633, { accuracy: 0.8744, accuracyFlagged: 22, accuracyTruePositives: 5 }),
    run("iforest", 0.1281, { accuracy: 0.8744, accuracyFlagged: 18, accuracyTruePositives: 3 }),
    run("hgb", 0.6087, { accuracy: 0.9049, accuracyFlagged: 18, accuracyTruePositives: 16 }),
    run("gcn", 0.7897, { accuracy: 0.9096, accuracyFlagged: 18, accuracyTruePositives: 18 }),
    run("graphsage", 0.8941, { accuracy: 0.9096, accuracyFlagged: 18, accuracyTruePositives: 18 }),
    run("mlp", 0.3765, { accuracy: 0.8955, accuracyFlagged: 18, accuracyTruePositives: 12 }),
  ];

  it("names the ceiling 775 / 852 = 0,9096 for 18 flagged nodes, only 0,0211 above flagging nothing (757 / 852 = 0,8885)", () => {
    expect(accuracySentence(runs, EVALUATION)).toBe(
      "Die Accuracy liegt zwischen 0,8744 und 0,9096; ein Modell, das keinen Knoten markiert, erreicht 0,8885. GCN und GraphSAGE markieren je 18 Knoten, alle mit Label illegal. Mehr als 0,9096 ist bei 18 markierten Knoten nicht möglich, und das sind nur 0,0211 mehr als bei „nichts markiert“. Deshalb ist Accuracy hier nur Nebenwert.",
    );
  });

  it("says when every method stays below a model that flags nothing", () => {
    expect(accuracySentence(runs.slice(0, 2), EVALUATION)).toBe(
      "Bei der Accuracy liegen alle gemessenen Verfahren unter den 0,8885 eines Modells, das keinen Knoten markiert. Deshalb ist Accuracy hier nur Nebenwert.",
    );
  });

  it("never names the model output with a label word", () => {
    for (const sentence of [accuracySentence(runs, EVALUATION), accuracySentence(runs.slice(0, 2), EVALUATION)]) {
      expect(sentence).not.toMatch(/\bals (il)?legal\b|alles (il)?legal|einstuft/);
    }
  });
});

describe("randomComparisonSentence", () => {
  it("names only the Isolation Forest with 0,1281 < 0,1420 as indistinguishable from random rankings", () => {
    expect(randomComparisonSentence([ZSCORE, IFOREST, HGB, GCN, GRAPHSAGE, MLP], EVALUATION)).toBe(
      "Alle Verfahren außer Isolation Forest liegen über dem 95-%-Quantil zufälliger Rangfolgen (0,1420) und ordnen Knoten mit Label illegal damit besser als Zufall. Bei Isolation Forest (0,1281) ist die PR-AUC von der einer zufälligen Rangfolge (Erwartungswert 0,1181) nicht zu unterscheiden.",
    );
  });

  it("lists several methods below the quantile", () => {
    expect(randomComparisonSentence([run("zscore", 0.142), IFOREST, GCN], EVALUATION)).toBe(
      "Alle Verfahren außer Z-Scores und Isolation Forest liegen über dem 95-%-Quantil zufälliger Rangfolgen (0,1420) und ordnen Knoten mit Label illegal damit besser als Zufall. Bei Z-Scores (0,1420) und Isolation Forest (0,1281) ist die PR-AUC von der einer zufälligen Rangfolge (Erwartungswert 0,1181) nicht zu unterscheiden.",
    );
  });

  it("handles all methods above and all below the quantile", () => {
    expect(randomComparisonSentence([ZSCORE, GCN], EVALUATION)).toBe(
      "Alle Verfahren liegen über dem 95-%-Quantil zufälliger Rangfolgen (0,1420) und ordnen Knoten mit Label illegal damit besser als Zufall.",
    );
    expect(randomComparisonSentence([IFOREST], EVALUATION)).toBe(
      "Kein Verfahren liegt über dem 95-%-Quantil zufälliger Rangfolgen (0,1420). Bei Isolation Forest (0,1281) ist die PR-AUC von der einer zufälligen Rangfolge (Erwartungswert 0,1181) nicht zu unterscheiden.",
    );
  });
});

describe("recallSentence", () => {
  it("converts recall at precision 0,5 into hits: 0,0211 · 95 ≈ 2, 0,5789 · 95 ≈ 55, 0,8316 · 95 ≈ 79", () => {
    const runs = [
      run("zscore", 0.1633, { recallAtPrecision50: 0.0211 }),
      run("iforest", 0.1281, { recallAtPrecision50: 0 }),
      run("hgb", 0.6087, { recallAtPrecision50: 0.5789 }),
      run("gcn", 0.7897, { recallAtPrecision50: 0.8316 }),
      run("graphsage", 0.8941, { recallAtPrecision50: 0.9474 }),
      run("mlp", 0.3765, { recallAtPrecision50: 0.2211 }),
    ];
    expect(recallSentence(runs, 95)).toBe(
      "Recall bei Precision ≥ 0,5 – Z-Scores 0,0211 (etwa 2 von 95); Isolation Forest 0,0000 (keine Schwelle erreicht diese Precision); Gradient Boosting 0,5789 (etwa 55 von 95); GCN 0,8316 (etwa 79 von 95); GraphSAGE 0,9474 (etwa 90 von 95); MLP (ohne Kanten) 0,2211 (etwa 21 von 95).",
    );
  });

  it("adds a note when every recall stays below 0,1", () => {
    expect(
      recallSentence([run("zscore", 0.1633, { recallAtPrecision50: 0.0211 }), run("iforest", 0.1281)], 95),
    ).toBe(
      "Recall bei Precision ≥ 0,5 – Z-Scores 0,0211 (etwa 2 von 95); Isolation Forest 0,0000 (keine Schwelle erreicht diese Precision). Bei dieser Precision finden die Verfahren kaum Knoten mit Label illegal.",
    );
  });
});

describe("method sentences for the network shape and the gradient boosting", () => {
  const MIXING =
    "Dieser Abstand mischt aber den Wechsel von Bäumen zu neuronalen Netzen mit der Wirkung der Nachbarschaft;";

  it("names only GCN, GraphSAGE and MLP for the shared shape, never the gradient boosting", () => {
    const subject = neuralShapeSubject([ZSCORE, IFOREST, HGB, GCN, GRAPHSAGE, MLP]);
    expect(subject).toBe("Hier haben die neuronalen Verfahren GCN, GraphSAGE und MLP (ohne Kanten) dieselbe Form:");
    expect(subject).not.toContain("Gradient Boosting");
    expect(subject).not.toContain("gelernten");
  });

  it("names only the graph method when the shapes differ or it is the only neural run", () => {
    expect(neuralShapeSubject([HGB, GCN, GRAPHSAGE, learnedRun("mlp", 0.3765, 0.497, { hidden: 128 })])).toBe("GCN hat");
    expect(neuralShapeSubject([HGB, GRAPHSAGE])).toBe("GraphSAGE hat");
    expect(neuralShapeSubject([ZSCORE, IFOREST, HGB])).toBe("");
  });

  it("calls the gradient boosting the stricter reference only when it lies above the MLP by more than the margin", () => {
    expect(boostingReferenceSentence(HGB, MLP, MARGIN)).toBe(
      `Auf diesem Netz liegt das Gradient Boosting deutlich über dem MLP (0,6087 gegen 0,3765); der Abstand der Graph Neural Networks zu ihm ist deshalb ein strengerer Vergleich als der zum MLP. ${MIXING} allein in den Kanten unterscheiden sich die Graph Neural Networks nur vom MLP.`,
    );
    expect(boostingReferenceSentence(HGB, learnedRun("mlp", 0.59, 0.5), MARGIN)).toBe(
      `Der Abstand der Graph Neural Networks zum Gradient Boosting ist ein Vergleich mit einem starken Verfahren ohne Kanten. ${MIXING} allein in den Kanten unterscheiden sich die Graph Neural Networks nur vom MLP.`,
    );
    expect(boostingReferenceSentence(HGB, null, MARGIN)).toBe(
      `Der Abstand der Graph Neural Networks zum Gradient Boosting ist ein Vergleich mit einem starken Verfahren ohne Kanten. ${MIXING} ein neuronales Netz gleicher Größe ohne Kanten ist nicht gemessen.`,
    );
  });
});
