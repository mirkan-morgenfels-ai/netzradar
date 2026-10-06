import { formatPlainNumber } from "./format";
import {
  METHODS,
  type EvaluationInfo,
  type JsonObject,
  type JsonValue,
  type LabelCounts,
  type Method,
  type NetNode,
  type Run,
} from "./types";

export const PENDING_METHODS: readonly Method[] = ["gcn", "graphsage"];

export interface MeasuredRow {
  kind: "measured";
  method: Method;
  run: Run;
}

export interface PendingRow {
  kind: "pending";
  method: Method;
}

export type MetricRow = MeasuredRow | PendingRow;

export function metricRows(runs: readonly Run[]): MetricRow[] {
  return METHODS.flatMap((method): MetricRow[] => {
    const run = runs.find((entry) => entry.method === method);
    if (run) return [{ kind: "measured", method, run }];
    return PENDING_METHODS.includes(method) ? [{ kind: "pending", method }] : [];
  });
}

export function labelledTestCount(evaluation: EvaluationInfo): number {
  return evaluation.testPositives + evaluation.testNegatives;
}

export function expectedRandomPrAuc(positives: number, total: number): number {
  if (!Number.isInteger(positives) || !Number.isInteger(total) || positives < 1 || positives > total) {
    return Number.NaN;
  }
  if (total === 1) return 1;
  let sum = 0;
  for (let rank = 1; rank <= total; rank += 1) {
    sum += (1 + ((rank - 1) * (positives - 1)) / (total - 1)) / rank;
  }
  return sum / total;
}

export function beatsRandomRankings(prAuc: number, evaluation: EvaluationInfo): boolean {
  return prAuc > evaluation.randomPrAucQ95;
}

export function bestRun(runs: readonly Run[]): Run | null {
  return runs.reduce<Run | null>((best, run) => (best === null || run.prAuc > best.prAuc ? run : best), null);
}

export function labelTotal(counts: LabelCounts): number {
  return counts.illicit + counts.licit + counts.unknown;
}

function describeValue(value: JsonValue): string {
  if (Array.isArray(value)) return value.length === 0 ? "–" : value.map(describeValue).join(", ");
  if (value === null) return "–";
  if (typeof value === "number") return formatPlainNumber(value);
  if (typeof value === "boolean") return value ? "ja" : "nein";
  if (typeof value === "object") return JSON.stringify(value);
  return value;
}

function flattenEntry(key: string, value: JsonValue): Array<[string, string]> {
  if (value !== null && typeof value === "object" && !Array.isArray(value)) {
    return Object.entries(value).flatMap(([child, item]) => flattenEntry(`${key}.${child}`, item));
  }
  return [[key, describeValue(value)]];
}

export function parameterEntries(parameters: JsonObject | null): Array<[string, string]> {
  if (parameters === null) return [];
  return Object.entries(parameters).flatMap(([key, value]) => flattenEntry(key, value));
}

export function zeroMadFeatures(run: Run): string[] {
  const value = run.hyperparameters.zeroMadFeatures;
  return Array.isArray(value) ? value.filter((entry): entry is string => typeof entry === "string") : [];
}

export function featureNames(run: Run): string[] {
  const value = run.hyperparameters.features;
  return Array.isArray(value) ? value.filter((entry): entry is string => typeof entry === "string") : [];
}

export function illicitShareAmongLabelled(counts: LabelCounts): number {
  const labelled = counts.illicit + counts.licit;
  return labelled > 0 ? counts.illicit / labelled : 0;
}

export function meanDegree(nodes: readonly NetNode[]): number {
  if (nodes.length === 0) return 0;
  return nodes.reduce((sum, node) => sum + node.inDegree + node.outDegree, 0) / nodes.length;
}

export function hitsAtRecall(recall: number, positives: number): number {
  return Math.round(recall * positives);
}

export function numberParameter(parameters: JsonObject | null, key: string): number | null {
  const value = parameters?.[key];
  return typeof value === "number" ? value : null;
}
