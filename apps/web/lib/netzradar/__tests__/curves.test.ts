import { describe, expect, it } from "vitest";
import { CHART_COLORS } from "@portfolio/charts/theme";
import { METHOD_CURVE_STYLES, PREVALENCE_LEVEL_STYLE, curveSeries, prevalenceLevel } from "../curves";
import { METHODS, type Run } from "../types";
import { EVALUATION, run } from "./fixtures";

const RUNS: Run[] = [
  run("zscore", 0.1633, {
    displayName: "Robuste Z-Scores (Einzelmerkmale)",
    prCurve: [
      { recall: 0.0211, precision: 1 },
      { recall: 1, precision: 0.1115 },
    ],
  }),
  run("iforest", 0.1281, {
    displayName: "Isolation Forest (Einzelmerkmale + Graphmaße)",
    featureSet: "local+graph",
    prCurve: [{ recall: 1, precision: 0.1115 }],
  }),
];

const ALL_RUNS: Run[] = METHODS.map((method, index) => run(method, 0.1 * (index + 1)));

function isBluish(hex: string): boolean {
  const value = Number.parseInt(hex.slice(1), 16);
  const red = (value >> 16) & 0xff;
  const green = (value >> 8) & 0xff;
  const blue = value & 0xff;
  return blue > red && blue >= green;
}

describe("curveSeries", () => {
  it("draws Z-Scores in gold and the Isolation Forest in wine with distinct dashes", () => {
    const series = curveSeries(RUNS);
    expect(series.map((entry) => [entry.name, entry.color, entry.dash])).toEqual([
      ["Robuste Z-Scores", "#b8912f", ""],
      ["Isolation Forest", "#7a1f2b", "8 3"],
    ]);
    expect(series[0]?.points).toBe(RUNS[0]?.prCurve);
  });

  it("draws all five methods with the styles of the contract", () => {
    expect(curveSeries(ALL_RUNS).map((entry) => [entry.name, entry.color, entry.dash])).toEqual([
      ["Robuste Z-Scores", "#b8912f", ""],
      ["Isolation Forest", "#7a1f2b", "8 3"],
      ["GCN (Graph Convolutional Network)", "#2f6b3a", "2 3"],
      ["GraphSAGE", "#111111", "8 3 2 3"],
      ["MLP ohne Kanten (Kontrolle)", "#6b6b66", "12 4"],
    ]);
  });

  it("gives every method its own dash pattern", () => {
    const dashes = Object.values(METHOD_CURVE_STYLES).map((style) => style.dash);
    expect(dashes).toHaveLength(METHODS.length);
    expect(new Set([...dashes, PREVALENCE_LEVEL_STYLE.dash]).size).toBe(dashes.length + 1);
  });

  it("uses only palette colours and no blue", () => {
    const palette = new Set<string>(Object.values(CHART_COLORS));
    for (const style of [...Object.values(METHOD_CURVE_STYLES), PREVALENCE_LEVEL_STYLE]) {
      expect(palette.has(style.color)).toBe(true);
      expect(isBluish(style.color)).toBe(false);
    }
  });

  it("recognises a pure blue channel in the test helper", () => {
    expect(isBluish(`#${(0xff).toString(16).padStart(6, "0")}`)).toBe(true);
    expect(isBluish(CHART_COLORS.moss)).toBe(false);
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
