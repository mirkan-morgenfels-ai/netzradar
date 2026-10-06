import { describe, expect, it } from "vitest";
import {
  beatsRandomRankings,
  bestRun,
  expectedRandomPrAuc,
  featureNames,
  hitsAtRecall,
  illicitShareAmongLabelled,
  labelTotal,
  labelledTestCount,
  meanDegree,
  metricRows,
  numberParameter,
  parameterEntries,
  zeroMadFeatures,
} from "../summary";
import type { EvaluationInfo, Method, NetNode, Run } from "../types";

const EVALUATION: EvaluationInfo = {
  positiveLabel: "illicit",
  excludedLabel: "unknown",
  testPositives: 95,
  testNegatives: 757,
  prevalence: 0.1115,
  allLicitAccuracy: 0.8885,
  randomPrAucExpected: 0.1181,
  randomPrAucQ95: 0.142,
  randomPermutations: 10000,
};

function run(method: Method, prAuc: number): Run {
  return {
    method,
    displayName: method,
    featureSet: "local",
    seed: 42,
    date: "2026-10-06",
    hyperparameters: {},
    prAuc,
    precisionAtRecall50: 0,
    recallAtPrecision50: 0,
    accuracy: 0,
    accuracyThreshold: "oberste 2 %",
    accuracyFlagged: 1,
    accuracyTruePositives: 0,
    prCurve: [{ recall: 0, precision: 1 }],
  };
}

describe("metricRows", () => {
  it("lists measured baselines first and marks missing GNN runs as pending", () => {
    const rows = metricRows([run("iforest", 0.1281), run("zscore", 0.1633)]);
    expect(rows.map((row) => [row.method, row.kind])).toEqual([
      ["zscore", "measured"],
      ["iforest", "measured"],
      ["gcn", "pending"],
      ["graphsage", "pending"],
    ]);
  });

  it("shows a GNN run as soon as it exists", () => {
    const rows = metricRows([run("zscore", 0.2), run("gcn", 0.3)]);
    expect(rows.map((row) => [row.method, row.kind])).toEqual([
      ["zscore", "measured"],
      ["gcn", "measured"],
      ["graphsage", "pending"],
    ]);
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
});

describe("feature lists of a run", () => {
  it("reads features and features without spread", () => {
    const zscore = {
      ...run("zscore", 0.1633),
      hyperparameters: { features: ["f_a", "f_b", "f_c"], zeroMadFeatures: ["f_b"] },
    };
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
  });

  it("reads numeric generator parameters only", () => {
    expect(numberParameter({ illicitVisibility: 0.55, name: "x" }, "illicitVisibility")).toBe(0.55);
    expect(numberParameter({ name: "x" }, "name")).toBeNull();
    expect(numberParameter(null, "seed")).toBeNull();
  });
});
