import { CHART_COLORS } from "@portfolio/charts/theme";
import { formatDecimal, METHOD_SHORT_TEXT } from "./format";
import type { EvaluationInfo, Method, PrCurvePoint, Run } from "./types";

export interface CurveStyle {
  color: string;
  dash: string;
  width?: number;
}

export interface CurveSeries extends CurveStyle {
  name: string;
  points: PrCurvePoint[];
}

export interface ReferenceLevel {
  name: string;
  value: number;
  color: string;
  dash: string;
}

export const METHOD_CURVE_STYLES: Record<Method, CurveStyle> = {
  zscore: { color: CHART_COLORS.goldLine, dash: "" },
  iforest: { color: CHART_COLORS.goldDeep, dash: "8 3" },
  gcn: { color: CHART_COLORS.sky, dash: "2 3", width: 2.5 },
  graphsage: { color: CHART_COLORS.navy, dash: "8 3 2 3" },
  mlp: { color: CHART_COLORS.slate, dash: "12 4" },
};

export const PREVALENCE_LEVEL_STYLE: CurveStyle = { color: CHART_COLORS.lineStrong, dash: "4 4" };

export function curveSeries(runs: readonly Run[]): CurveSeries[] {
  return runs.map((run) => ({
    name: METHOD_SHORT_TEXT[run.method],
    points: run.prCurve,
    ...METHOD_CURVE_STYLES[run.method],
  }));
}

export function prevalenceLevel(evaluation: EvaluationInfo): ReferenceLevel {
  return {
    name: `Prävalenz (${formatDecimal(evaluation.prevalence)})`,
    value: evaluation.prevalence,
    ...PREVALENCE_LEVEL_STYLE,
  };
}
