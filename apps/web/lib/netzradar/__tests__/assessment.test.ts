import { describe, expect, it } from "vitest";
import {
  NEIGHBOURHOOD_SHARE_SENTENCE,
  NO_NEIGHBOURHOOD_LEAD_SENTENCE,
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
} from "../assessment";
import type { Run } from "../types";
import { EVALUATION, learnedRun, run } from "./fixtures";

const MARGIN = 0.0239;
const ZSCORE = run("zscore", 0.1633);
const IFOREST = run("iforest", 0.1281, { featureSet: "local+graph" });
const GCN = learnedRun("gcn", 0.7897, 0.7364);
const GRAPHSAGE = learnedRun("graphsage", 0.8941, 0.9349);
const MLP = learnedRun("mlp", 0.3765, 0.497);

const MLP_INTRO =
  "Die Kontrolle trennt zwei Effekte. Das MLP ohne Kanten nutzt dieselben Labels und dasselbe Auswahlverfahren wie die Graph Neural Networks (dieselben Kandidaten für Merkmalssatz und Klassengewicht, Auswahl am Validierungsteil), sieht aber keine Nachbarn.";
const PER_METHOD = "Merkmalssatz, Gewicht und Epochenzahl wählt dabei jedes Verfahren für sich.";

describe("decompositionSentences", () => {
  it("splits the measured gain: 0,3765 − 0,1633 = 0,2132 and 0,4132 − 0,2132 = 0,2000 > 0,0239", () => {
    expect(decompositionSentences(MLP, ZSCORE, [GCN, GRAPHSAGE], MARGIN)).toEqual([
      `${MLP_INTRO} Es erreicht 0,3765.`,
      "Das sind 0,2132 mehr als die bessere Baseline (Z-Scores, 0,1633). Dieser Abstand mischt die Wirkung der Labels mit dem Wechsel des Verfahrens und des Merkmalssatzes (das MLP nutzt lokale Merkmale und Graphmaße, die bessere Baseline nur lokale Merkmale).",
      `Mit Nachbarschaft, bei gleichem Auswahlverfahren: GCN 0,7897 (um 0,4132 über dem MLP), GraphSAGE 0,8941 (um 0,5176 über dem MLP). ${PER_METHOD}`,
      NEIGHBOURHOOD_SHARE_SENTENCE,
    ]);
  });

  it("drops the neighbourhood sentence when GNN − MLP = 0,03 exceeds the supervision gain 0,02 by less than the margin", () => {
    const sentences = decompositionSentences(
      learnedRun("mlp", 0.32, 0.5),
      run("zscore", 0.3),
      [learnedRun("gcn", 0.35, 0.5), learnedRun("graphsage", 0.35, 0.5)],
      MARGIN,
    );
    expect(sentences).toEqual([
      `${MLP_INTRO} Es erreicht 0,3200.`,
      "Das liegt nahe an der besseren Baseline (Z-Scores, 0,3000); die Labels allein bringen hier keinen belastbaren Vorsprung.",
      `Mit Nachbarschaft, bei gleichem Auswahlverfahren: GCN 0,3500 (um 0,0300 über dem MLP), GraphSAGE 0,3500 (um 0,0300 über dem MLP). ${PER_METHOD}`,
    ]);
    expect(sentences).not.toContain(NEIGHBOURHOOD_SHARE_SENTENCE);
    expect(sentences).not.toContain(NO_NEIGHBOURHOOD_LEAD_SENTENCE);
  });

  it("reports an MLP below the baseline and graph methods within the margin of the MLP", () => {
    expect(
      decompositionSentences(
        learnedRun("mlp", 0.3, 0.5),
        run("zscore", 0.4),
        [learnedRun("gcn", 0.31, 0.5), learnedRun("graphsage", 0.29, 0.5)],
        MARGIN,
      ),
    ).toEqual([
      `${MLP_INTRO} Es erreicht 0,3000.`,
      "Das sind 0,1000 weniger als die bessere Baseline (Z-Scores, 0,4000); die Labels allein helfen hier also nicht.",
      `Mit Nachbarschaft, bei gleichem Auswahlverfahren: GCN 0,3100 (nur 0,0100 vom MLP entfernt), GraphSAGE 0,2900 (nur 0,0100 vom MLP entfernt). ${PER_METHOD}`,
      NO_NEIGHBOURHOOD_LEAD_SENTENCE,
    ]);
  });

  it("names a graph method below the MLP", () => {
    const sentences = decompositionSentences(MLP, ZSCORE, [learnedRun("gcn", 0.3, 0.5)], MARGIN);
    expect(sentences[2]).toBe(`Mit Nachbarschaft, bei gleichem Auswahlverfahren: GCN 0,3000 (um 0,0765 unter dem MLP). ${PER_METHOD}`);
    expect(sentences[3]).toBe(NO_NEIGHBOURHOOD_LEAD_SENTENCE);
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
    expect(graphLeadMeasured([GCN, GRAPHSAGE], [ZSCORE, MLP], MARGIN)).toBe(true);
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
    run("gcn", 0.7897, { accuracy: 0.9096, accuracyFlagged: 18, accuracyTruePositives: 18 }),
    run("graphsage", 0.8941, { accuracy: 0.9096, accuracyFlagged: 18, accuracyTruePositives: 18 }),
    run("mlp", 0.3765, { accuracy: 0.8955, accuracyFlagged: 18, accuracyTruePositives: 12 }),
  ];

  it("names the ceiling 775 / 852 = 0,9096 for 18 flagged nodes, only 0,0211 above all licit", () => {
    expect(accuracySentence(runs, EVALUATION)).toBe(
      "Die Accuracy liegt zwischen 0,8744 und 0,9096; ein Modell, das alles unauffällig nennt, erreicht 0,8885. GCN und GraphSAGE markieren je 18 Knoten, alle auffällig. Mehr als 0,9096 ist bei 18 markierten Knoten nicht möglich, und das sind nur 0,0211 mehr als bei „alles unauffällig“. Deshalb ist Accuracy hier nur Nebenwert.",
    );
  });

  it("says when every method stays below all licit", () => {
    expect(accuracySentence(runs.slice(0, 2), EVALUATION)).toBe(
      "Bei der Accuracy liegen alle gemessenen Verfahren unter den 0,8885 eines Modells, das alles unauffällig nennt. Deshalb ist Accuracy hier nur Nebenwert.",
    );
  });
});

describe("randomComparisonSentence", () => {
  it("names only the Isolation Forest with 0,1281 < 0,1420 as indistinguishable from random rankings", () => {
    expect(randomComparisonSentence([ZSCORE, IFOREST, GCN, GRAPHSAGE, MLP], EVALUATION)).toBe(
      "Alle Verfahren außer Isolation Forest liegen über dem 95-%-Quantil zufälliger Rangfolgen (0,1420) und ordnen auffällige Knoten damit besser als Zufall. Bei Isolation Forest (0,1281) ist die PR-AUC von der einer zufälligen Rangfolge (Erwartungswert 0,1181) nicht zu unterscheiden.",
    );
  });

  it("lists several methods below the quantile", () => {
    expect(randomComparisonSentence([run("zscore", 0.142), IFOREST, GCN], EVALUATION)).toBe(
      "Alle Verfahren außer Z-Scores und Isolation Forest liegen über dem 95-%-Quantil zufälliger Rangfolgen (0,1420) und ordnen auffällige Knoten damit besser als Zufall. Bei Z-Scores (0,1420) und Isolation Forest (0,1281) ist die PR-AUC von der einer zufälligen Rangfolge (Erwartungswert 0,1181) nicht zu unterscheiden.",
    );
  });

  it("handles all methods above and all below the quantile", () => {
    expect(randomComparisonSentence([ZSCORE, GCN], EVALUATION)).toBe(
      "Alle Verfahren liegen über dem 95-%-Quantil zufälliger Rangfolgen (0,1420) und ordnen auffällige Knoten damit besser als Zufall.",
    );
    expect(randomComparisonSentence([IFOREST], EVALUATION)).toBe(
      "Kein Verfahren liegt über dem 95-%-Quantil zufälliger Rangfolgen (0,1420). Bei Isolation Forest (0,1281) ist die PR-AUC von der einer zufälligen Rangfolge (Erwartungswert 0,1181) nicht zu unterscheiden.",
    );
  });
});

describe("recallSentence", () => {
  it("converts recall at precision 0,5 into hits: 0,0211 · 95 ≈ 2, 0,8316 · 95 ≈ 79", () => {
    const runs = [
      run("zscore", 0.1633, { recallAtPrecision50: 0.0211 }),
      run("iforest", 0.1281, { recallAtPrecision50: 0 }),
      run("gcn", 0.7897, { recallAtPrecision50: 0.8316 }),
      run("graphsage", 0.8941, { recallAtPrecision50: 0.9474 }),
      run("mlp", 0.3765, { recallAtPrecision50: 0.2211 }),
    ];
    expect(recallSentence(runs, 95)).toBe(
      "Recall bei Precision ≥ 0,5 – Z-Scores 0,0211 (etwa 2 von 95); Isolation Forest 0,0000 (keine Schwelle erreicht diese Precision); GCN 0,8316 (etwa 79 von 95); GraphSAGE 0,9474 (etwa 90 von 95); MLP (ohne Kanten) 0,2211 (etwa 21 von 95).",
    );
  });

  it("adds a note when every recall stays below 0,1", () => {
    expect(
      recallSentence([run("zscore", 0.1633, { recallAtPrecision50: 0.0211 }), run("iforest", 0.1281)], 95),
    ).toBe(
      "Recall bei Precision ≥ 0,5 – Z-Scores 0,0211 (etwa 2 von 95); Isolation Forest 0,0000 (keine Schwelle erreicht diese Precision). Bei dieser Precision finden die Verfahren kaum auffällige Knoten.",
    );
  });
});
