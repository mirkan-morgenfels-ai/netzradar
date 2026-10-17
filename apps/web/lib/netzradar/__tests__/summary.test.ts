import { describe, expect, it } from "vitest";
import {
  averagedSeparation,
  beatsRandomRankings,
  bestAccuracyAtFlagged,
  bestBaseline,
  bestEdgeless,
  bestRun,
  boostingTraining,
  compareScores,
  comparisonMargin,
  edgelessRuns,
  excerptCounts,
  expectedRandomPrAuc,
  expectedScoreGnnMethod,
  featureNames,
  finalTrainRatioWeight,
  hitsAtRecall,
  illicitShareAmongLabelled,
  hasSpread,
  isGraphMethod,
  isLearnedMethod,
  isNeuralMethod,
  labelTotal,
  labelledTestCount,
  meanDegree,
  metricRows,
  neighbourhoodNodes,
  neuralTraining,
  numberParameter,
  parameterEntries,
  reachedEpochLimit,
  reachedIterationLimit,
  runDateRange,
  searchWeight,
  seedSpreadOf,
  selectedCandidate,
  spreadSeeds,
  zeroMadFeatures,
} from "../summary";
import { METHODS, type NetNode } from "../types";
import {
  EVALUATION,
  boostingCandidate,
  boostingRun,
  boostingTraining as boostingFixture,
  candidate,
  learnedRun,
  run,
  spread,
  training,
} from "./fixtures";

describe("metricRows", () => {
  it("lists measured baselines first and marks missing learned runs as pending", () => {
    const rows = metricRows([run("iforest", 0.1281), run("zscore", 0.1633)]);
    expect(rows.map((row) => [row.method, row.kind])).toEqual([
      ["zscore", "measured"],
      ["iforest", "measured"],
      ["hgb", "pending"],
      ["gcn", "pending"],
      ["graphsage", "pending"],
      ["mlp", "pending"],
    ]);
  });

  it("shows a GNN run as soon as it exists", () => {
    const rows = metricRows([run("zscore", 0.2), run("gcn", 0.3)]);
    expect(rows.map((row) => [row.method, row.kind])).toEqual([
      ["zscore", "measured"],
      ["hgb", "pending"],
      ["gcn", "measured"],
      ["graphsage", "pending"],
      ["mlp", "pending"],
    ]);
  });

  it("orders all six measured runs like the contract and leaves nothing pending", () => {
    const rows = metricRows([
      run("mlp", 0.3765),
      run("graphsage", 0.8941),
      run("zscore", 0.1633),
      run("gcn", 0.7897),
      run("hgb", 0.6087),
      run("iforest", 0.1281),
    ]);
    expect(rows.map((row) => [row.method, row.kind])).toEqual([
      ["zscore", "measured"],
      ["iforest", "measured"],
      ["hgb", "measured"],
      ["gcn", "measured"],
      ["graphsage", "measured"],
      ["mlp", "measured"],
    ]);
  });
});

describe("method groups", () => {
  it("separates learned and graph methods", () => {
    expect(METHODS).toEqual(["zscore", "iforest", "hgb", "gcn", "graphsage", "mlp"]);
    expect(METHODS.map(isLearnedMethod)).toEqual([false, false, true, true, true, true]);
    expect(METHODS.map(isNeuralMethod)).toEqual([false, false, false, true, true, true]);
    expect(METHODS.map(isGraphMethod)).toEqual([false, false, false, true, true, false]);
  });
});

describe("expectedScoreGnnMethod", () => {
  it("takes the graph method with the higher validation PR-AUC and never the MLP", () => {
    const runs = [learnedRun("gcn", 0.7897, 0.7364), learnedRun("graphsage", 0.8941, 0.9349), learnedRun("mlp", 0.9, 0.99)];
    expect(expectedScoreGnnMethod(runs)).toBe("graphsage");
  });

  it("prefers GCN on a tie and ignores the test PR-AUC", () => {
    expect(expectedScoreGnnMethod([learnedRun("graphsage", 0.9, 0.8), learnedRun("gcn", 0.1, 0.8)])).toBe("gcn");
    expect(expectedScoreGnnMethod([learnedRun("gcn", 0.9, 0.7), learnedRun("graphsage", 0.1, 0.71)])).toBe("graphsage");
  });

  it("returns null without graph runs", () => {
    expect(expectedScoreGnnMethod([run("zscore", 0.1633), learnedRun("mlp", 0.3765, 0.497)])).toBeNull();
    expect(expectedScoreGnnMethod([learnedRun("graphsage", 0.8941, 0.9349)])).toBe("graphsage");
  });
});

describe("comparisons with the random level", () => {
  it("counts the labelled test nodes", () => {
    expect(labelledTestCount(EVALUATION)).toBe(852);
  });

  it("computes the expected PR-AUC of a random ranking by hand", () => {
    expect(expectedRandomPrAuc(1, 1)).toBe(1);
    expect(expectedRandomPrAuc(1, 2)).toBeCloseTo((1 + 1 / 2) / 2, 10);
    expect(expectedRandomPrAuc(2, 3)).toBeCloseTo((1 + 5 / 6 + 7 / 12) / 3, 10);
    expect(expectedRandomPrAuc(1, 5)).toBeCloseTo(137 / 300, 10);
    expect(expectedRandomPrAuc(95, 852)).toBeCloseTo(0.1181, 4);
    expect(expectedRandomPrAuc(0, 5)).toBeNaN();
    expect(expectedRandomPrAuc(6, 5)).toBeNaN();
  });

  it("compares a PR-AUC with the 95 % quantile of random rankings", () => {
    expect(beatsRandomRankings(0.1633, EVALUATION)).toBe(true);
    expect(beatsRandomRankings(0.1281, EVALUATION)).toBe(false);
    expect(beatsRandomRankings(0.142, EVALUATION)).toBe(false);
  });

  it("picks the run with the highest PR-AUC", () => {
    expect(bestRun([run("iforest", 0.1281), run("zscore", 0.1633)])?.method).toBe("zscore");
    expect(bestRun([])).toBeNull();
  });

  it("picks the better baseline and ignores learned runs", () => {
    const runs = [run("zscore", 0.1633), run("iforest", 0.1281), learnedRun("gcn", 0.7897, 0.7364)];
    expect(bestBaseline(runs)?.method).toBe("zscore");
    expect(bestBaseline([learnedRun("gcn", 0.7897, 0.7364)])).toBeNull();
  });

  it("uses the spread of random rankings as comparison margin", () => {
    expect(comparisonMargin(EVALUATION)).toBeCloseTo(0.142 - 0.1181, 10);
    expect(comparisonMargin(EVALUATION)).toBeCloseTo(0.0239, 10);
  });

  it("only calls a difference a ranking when it exceeds the margin", () => {
    expect(compareScores(0.7897, 0.1633, 0.0239)).toBe("higher");
    expect(compareScores(0.1633, 0.1281, 0.0239)).toBe("higher");
    expect(compareScores(0.1281, 0.1633, 0.0239)).toBe("lower");
    expect(compareScores(0.15, 0.14, 0.0239)).toBe("similar");
    expect(compareScores(0.75, 0.5, 0.25)).toBe("similar");
    expect(compareScores(0.25, 0.5, 0.25)).toBe("similar");
  });
});

describe("accuracy ceiling", () => {
  it("computes the best accuracy for a fixed number of flagged nodes by hand", () => {
    expect(bestAccuracyAtFlagged(18, EVALUATION)).toBeCloseTo(775 / 852, 10);
    expect(bestAccuracyAtFlagged(18, EVALUATION)).toBeCloseTo(0.9096, 4);
    expect(bestAccuracyAtFlagged(95, EVALUATION)).toBe(1);
    expect(bestAccuracyAtFlagged(100, EVALUATION)).toBeCloseTo(847 / 852, 10);
    expect(bestAccuracyAtFlagged(18, EVALUATION) - EVALUATION.allLicitAccuracy).toBeCloseTo(0.0211, 4);
  });
});

describe("averagedSeparation", () => {
  it("follows delta * (1 + d(1 - 2 rho)) / sqrt(d + 1)", () => {
    expect(averagedSeparation(0.5, 3)).toBeCloseTo(1, 10);
    expect(averagedSeparation(0.5, 3, 0.5)).toBeCloseTo(0.25, 10);
    expect(averagedSeparation(0.5, 0)).toBeCloseTo(0.5, 10);
    expect(averagedSeparation(1, 8, 0.25)).toBeCloseTo(5 / 3, 10);
  });
});

describe("training summaries", () => {
  it("finds the selected candidate and the epoch limit", () => {
    const search = [
      candidate({ positiveWeightRule: "trainRatio", positiveWeight: 8.3409, selected: false }),
      candidate({ selected: true, validationPrAuc: 0.7364 }),
    ];
    expect(selectedCandidate(training({ search }))?.validationPrAuc).toBe(0.7364);
    expect(reachedEpochLimit(training({ selectedEpoch: 300, maxEpochs: 300 }))).toBe(true);
    expect(reachedEpochLimit(training({ selectedEpoch: 140, maxEpochs: 300 }))).toBe(false);
  });

  it("reads the weights of the search and of the final model", () => {
    const search = [
      candidate({ positiveWeightRule: "trainRatio", positiveWeight: 8.3409, selected: false }),
      candidate({ positiveWeightRule: "fixed", positiveWeight: 20 }),
    ];
    const runs = [
      run("zscore", 0.1633),
      run("gcn", 0.7897, { training: training({ search }) }),
      run("graphsage", 0.8941, {
        training: training({ search, positiveWeightRule: "trainRatio", positiveWeight: 9.2893 }),
      }),
    ];
    expect(searchWeight(runs, "trainRatio")).toBe(8.3409);
    expect(searchWeight(runs, "fixed")).toBe(20);
    expect(finalTrainRatioWeight(runs)).toBe(9.2893);
    expect(finalTrainRatioWeight([run("gcn", 0.7897, { training: training({ search }) })])).toBeNull();
    expect(searchWeight([run("zscore", 0.1633)], "fixed")).toBeNull();
  });
});

describe("runDateRange", () => {
  it("spans the earliest and latest run date", () => {
    expect(runDateRange([run("zscore", 0.1), run("gcn", 0.2)])).toEqual({ from: "2026-10-06", to: "2026-10-06" });
    expect(
      runDateRange([run("gcn", 0.2, { date: "2026-10-07" }), run("zscore", 0.1, { date: "2026-10-06" })]),
    ).toEqual({ from: "2026-10-06", to: "2026-10-07" });
    expect(runDateRange([])).toBeNull();
  });
});

describe("label totals", () => {
  it("adds all three labels", () => {
    expect(labelTotal({ illicit: 1, licit: 3, unknown: 6 })).toBe(10);
    expect(labelTotal({ illicit: 0, licit: 0, unknown: 0 })).toBe(0);
  });
});

describe("parameterEntries", () => {
  it("formats values for the page", () => {
    expect(
      parameterEntries({
        nEstimators: 200,
        bootstrap: false,
        features: ["f_log_amount", "g_in_degree"],
        maxFeatures: 1.0,
        contamination: 0.02,
        maxSamples: "auto",
        unused: null,
      }),
    ).toEqual([
      ["nEstimators", "200"],
      ["bootstrap", "nein"],
      ["features", "f_log_amount, g_in_degree"],
      ["maxFeatures", "1"],
      ["contamination", "0,02"],
      ["maxSamples", "auto"],
      ["unused", "–"],
    ]);
    expect(parameterEntries(null)).toEqual([]);
  });

  it("flattens nested settings and marks empty lists", () => {
    expect(
      parameterEntries({
        graphMeasures: { exactBetweennessLimit: 5000, eigenvectorTol: "1e-06", sampledSteps: [] },
        zeroMadFeatures: ["f_round_amount", "f_change_output"],
      }),
    ).toEqual([
      ["graphMeasures.exactBetweennessLimit", "5000"],
      ["graphMeasures.eigenvectorTol", "1e-06"],
      ["graphMeasures.sampledSteps", "–"],
      ["zeroMadFeatures", "f_round_amount, f_change_output"],
    ]);
  });

  it("flattens lists of objects such as the search candidates", () => {
    expect(
      parameterEntries({
        search: [
          { featureSet: "local", validationPrAuc: 0.6793, selected: false },
          { featureSet: "local+graph", validationPrAuc: 0.7364, selected: true },
        ],
        fallback: ["std", "one"],
      }),
    ).toEqual([
      ["search[0].featureSet", "local"],
      ["search[0].validationPrAuc", "0,6793"],
      ["search[0].selected", "nein"],
      ["search[1].featureSet", "local+graph"],
      ["search[1].validationPrAuc", "0,7364"],
      ["search[1].selected", "ja"],
      ["fallback", "std, one"],
    ]);
  });
});

describe("feature lists of a run", () => {
  it("reads features and features without spread", () => {
    const zscore = run("zscore", 0.1633, {
      hyperparameters: { features: ["f_a", "f_b", "f_c"], zeroMadFeatures: ["f_b"] },
    });
    expect(featureNames(zscore)).toEqual(["f_a", "f_b", "f_c"]);
    expect(zeroMadFeatures(zscore)).toEqual(["f_b"]);
    expect(zeroMadFeatures(run("iforest", 0.1281))).toEqual([]);
    expect(featureNames(run("iforest", 0.1281))).toEqual([]);
  });
});

describe("figures for the evaluation text", () => {
  it("computes the illicit share among labelled start nodes", () => {
    expect(illicitShareAmongLabelled({ illicit: 2, licit: 10, unknown: 38 })).toBeCloseTo(2 / 12, 10);
    expect(illicitShareAmongLabelled({ illicit: 0, licit: 0, unknown: 5 })).toBe(0);
  });

  it("averages in plus out degree", () => {
    const nodes = [
      { inDegree: 14, outDegree: 9 },
      { inDegree: 1, outDegree: 0 },
      { inDegree: 2, outDegree: 1 },
    ] as NetNode[];
    expect(meanDegree(nodes)).toBeCloseTo(27 / 3, 10);
    expect(meanDegree([])).toBe(0);
  });

  it("converts a recall into found positives", () => {
    expect(hitsAtRecall(0.0211, 95)).toBe(2);
    expect(hitsAtRecall(0, 95)).toBe(0);
    expect(hitsAtRecall(0.5, 95)).toBe(48);
    expect(hitsAtRecall(0.8316, 95)).toBe(79);
    expect(hitsAtRecall(0.9474, 95)).toBe(90);
  });

  it("reads numeric generator parameters only", () => {
    expect(numberParameter({ illicitVisibility: 0.55, name: "x" }, "illicitVisibility")).toBe(0.55);
    expect(numberParameter({ name: "x" }, "name")).toBeNull();
    expect(numberParameter(null, "seed")).toBeNull();
  });
});

describe("gradient boosting and seed spread", () => {
  it("separates neural and boosting training", () => {
    const hgb = boostingRun(0.6087);
    const gcn = learnedRun("gcn", 0.7897, 0.7364);
    expect(boostingTraining(hgb)?.maxLeafNodes).toBe(7);
    expect(neuralTraining(hgb)).toBeNull();
    expect(neuralTraining(gcn)?.layers).toBe(2);
    expect(boostingTraining(gcn)).toBeNull();
    expect(boostingTraining(null)).toBeNull();
  });

  it("finds the selected boosting candidate and the tree limit", () => {
    const search = [
      boostingCandidate({ maxLeafNodes: 31, validationPrAuc: 0.5034, bestIteration: 68, selected: false }),
      boostingCandidate(),
    ];
    expect(selectedCandidate(boostingFixture({ search }))?.bestIteration).toBe(128);
    expect(reachedIterationLimit(boostingFixture({ selectedIteration: 300, maxIter: 300 }))).toBe(true);
    expect(reachedIterationLimit(boostingFixture({ selectedIteration: 128, maxIter: 300 }))).toBe(false);
  });

  it("ignores the boosting weights in the weights of the neural search", () => {
    const runs = [boostingRun(0.6087), learnedRun("gcn", 0.7897, 0.7364)];
    expect(searchWeight(runs, "fixed")).toBe(20);
    expect(finalTrainRatioWeight([boostingRun(0.6087, { positiveWeightRule: "trainRatio", positiveWeight: 9.2893 })])).toBeNull();
  });

  it("collects the methods without edges and picks the stronger one", () => {
    const runs = [run("zscore", 0.1633), boostingRun(0.6087), learnedRun("gcn", 0.7897, 0.7364), learnedRun("mlp", 0.3765, 0.497)];
    expect(edgelessRuns(runs).map((entry) => entry.method)).toEqual(["hgb", "mlp"]);
    expect(bestEdgeless(runs)?.method).toBe("hgb");
    expect(bestEdgeless([run("zscore", 0.1633)])).toBeNull();
  });

  it("reads the seed spread and recognises a spread of 0", () => {
    const hgb = boostingRun(0.6087);
    const gcn = learnedRun("gcn", 0.7897, 0.7364, { seedSpread: spread([0.7897, 0.8148, 0.8008, 0.7947, 0.8009]) });
    expect(seedSpreadOf(run("zscore", 0.1633))).toBeNull();
    expect(seedSpreadOf(gcn)?.mean).toBeCloseTo(4.0009 / 5, 10);
    expect(hasSpread(seedSpreadOf(hgb) ?? spread([0]))).toBe(false);
    expect(hasSpread(seedSpreadOf(gcn) ?? spread([0]))).toBe(true);
    expect(spreadSeeds([run("zscore", 0.1633), hgb, gcn])).toEqual([42, 43, 44, 45, 46]);
    expect(spreadSeeds([hgb, learnedRun("mlp", 0.3, 0.5, { seedSpread: spread([0.3], { seeds: [7], reportedSeed: 7 }) })])).toBeNull();
    expect(spreadSeeds([run("zscore", 0.1633)])).toBeNull();
  });
});

describe("excerpt of two start sets", () => {
  function node(id: string, overrides: Partial<NetNode>): NetNode {
    return {
      id,
      x: 0,
      y: 0,
      label: "unknown",
      timeStep: 22,
      scoreZscore: 1,
      scoreIforest: 0.5,
      scoreGnn: 0,
      seedRank: null,
      seedRankGnn: null,
      hop: null,
      hopGnn: null,
      inDegree: 0,
      outDegree: 0,
      ...overrides,
    };
  }

  const nodes = [
    node("a", { hop: 0, seedRank: 1, hopGnn: 0, seedRankGnn: 1 }),
    node("b", { hop: 0, seedRank: 2 }),
    node("c", { hop: 1, hopGnn: 1 }),
    node("d", { hop: 2 }),
    node("e", { hopGnn: 0, seedRankGnn: 2 }),
    node("f", { hopGnn: 2 }),
  ];

  it("counts both neighbourhoods, their overlap and the shared start nodes by hand", () => {
    expect(excerptCounts(nodes)).toEqual({ total: 6, iforest: 4, gnn: 4, both: 2, sharedSeeds: 1 });
  });

  it("lists the neighbours of a start set without its start nodes", () => {
    expect(neighbourhoodNodes(nodes, "iforest").map((entry) => entry.id)).toEqual(["c", "d"]);
    expect(neighbourhoodNodes(nodes, "gnn").map((entry) => entry.id)).toEqual(["c", "f"]);
  });
});
