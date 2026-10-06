import { describe, expect, it } from "vitest";
import { METHOD_CURVE_STYLES, PREVALENCE_LEVEL_STYLE, curveSeries, prevalenceLevel } from "../curves";
import type { EvaluationInfo, Run } from "../types";

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

const RUNS: Run[] = [
  {
    method: "zscore",
    displayName: "Robuste Z-Scores (Einzelmerkmale)",
    featureSet: "local",
    seed: 42,
    date: "2026-10-06",
    hyperparameters: {},
    prAuc: 0.1633,
    precisionAtRecall50: 0.1324,
    recallAtPrecision50: 0.0211,
    accuracy: 0.8744,
    accuracyThreshold: "oberste 2 %",
    accuracyFlagged: 22,
    accuracyTruePositives: 5,
    prCurve: [
      { recall: 0.0211, precision: 1 },
      { recall: 1, precision: 0.1115 },
    ],
  },
  {
    method: "iforest",
    displayName: "Isolation Forest (Einzelmerkmale + Graphmaße)",
    featureSet: "local+graph",
    seed: 42,
    date: "2026-10-06",
    hyperparameters: {},
    prAuc: 0.1281,
    precisionAtRecall50: 0.134,
    recallAtPrecision50: 0,
    accuracy: 0.8744,
    accuracyThreshold: "oberste 2 %",
    accuracyFlagged: 18,
    accuracyTruePositives: 3,
    prCurve: [{ recall: 1, precision: 0.1115 }],
  },
];

describe("curveSeries", () => {
  it("draws Z-Scores in gold and the Isolation Forest in wine with distinct dashes", () => {
    const series = curveSeries(RUNS);
    expect(series.map((entry) => [entry.name, entry.color, entry.dash])).toEqual([
      ["Robuste Z-Scores", "#b8912f", ""],
      ["Isolation Forest", "#7a1f2b", "8 3"],
    ]);
    expect(series[0]?.points).toBe(RUNS[0]?.prCurve);
  });

  it("gives every method its own dash pattern", () => {
    const dashes = Object.values(METHOD_CURVE_STYLES).map((style) => style.dash);
    expect(new Set([...dashes, PREVALENCE_LEVEL_STYLE.dash]).size).toBe(dashes.length + 1);
  });
});

describe("prevalenceLevel", () => {
  it("draws the prevalence as a dashed stone line", () => {
    expect(prevalenceLevel(EVALUATION)).toEqual({
      name: "Prävalenz (0,1115)",
      value: 0.1115,
      color: "#6b6b66",
      dash: "4 4",
    });
  });
});
