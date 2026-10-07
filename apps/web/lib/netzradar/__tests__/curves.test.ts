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

describe("curveSeries", () => {
  it("draws both baselines in warm gold tones with distinct dashes", () => {
    const series = curveSeries(RUNS);
    expect(series.map((entry) => [entry.name, entry.color, entry.dash])).toEqual([
      ["Z-Scores", "#a8832a", ""],
      ["Isolation Forest", "#7d5f17", "8 3"],
    ]);
    expect(series[0]?.points).toBe(RUNS[0]?.prCurve);
  });

  it("draws all five methods with the styles of the contract", () => {
    expect(curveSeries(ALL_RUNS).map((entry) => [entry.name, entry.color, entry.dash])).toEqual([
      ["Z-Scores", "#a8832a", ""],
      ["Isolation Forest", "#7d5f17", "8 3"],
      ["GCN", "#3e6a9e", "2 3"],
      ["GraphSAGE", "#1d3a5f", "8 3 2 3"],
      ["MLP (ohne Kanten)", "#5b6474", "12 4"],
    ]);
  });

  it("draws the dotted GCN line thicker and leaves the other widths to the chart", () => {
    expect(curveSeries(ALL_RUNS).map((entry) => entry.width)).toEqual([undefined, undefined, 2.5, undefined, undefined]);
  });

  it("keeps the label colours moss and wine out of the method colours", () => {
    const colours = Object.values(METHOD_CURVE_STYLES).map((style) => style.color);
    expect(colours).not.toContain(CHART_COLORS.moss);
    expect(colours).not.toContain(CHART_COLORS.wine);
  });

  it("gives every method its own dash pattern", () => {
    const dashes = Object.values(METHOD_CURVE_STYLES).map((style) => style.dash);
    expect(dashes).toHaveLength(METHODS.length);
    expect(new Set([...dashes, PREVALENCE_LEVEL_STYLE.dash]).size).toBe(dashes.length + 1);
  });

  it("uses only colours of the palette, one per method, and keeps the prevalence apart", () => {
    const series = new Set<string>(Object.values(CHART_COLORS));
    const colours = Object.values(METHOD_CURVE_STYLES).map((style) => style.color);
    for (const colour of [...colours, PREVALENCE_LEVEL_STYLE.color]) expect(series.has(colour)).toBe(true);
    expect(new Set(colours).size).toBe(METHODS.length);
    expect(colours).not.toContain(PREVALENCE_LEVEL_STYLE.color);
  });
});

describe("prevalenceLevel", () => {
  it("draws the prevalence as a dashed line in line-strong", () => {
    expect(prevalenceLevel(EVALUATION)).toEqual({
      name: "Prävalenz (0,1115)",
      value: 0.1115,
      color: "#858d9b",
      dash: "4 4",
    });
  });
});
