import { formatPlainNumber } from "./format";
import {
  BASELINE_METHODS,
  GRAPH_METHODS,
  LEARNED_METHODS,
  METHODS,
  type EvaluationInfo,
  type GraphMethod,
  type JsonObject,
  type JsonValue,
  type LabelCounts,
  type LearnedMethod,
  type Method,
  type NetNode,
  type Run,
  type SearchCandidate,
  type Training,
} from "./types";

export const PENDING_METHODS: readonly Method[] = LEARNED_METHODS;

export type Comparison = "higher" | "lower" | "similar";

export function isLearnedMethod(method: Method): method is LearnedMethod {
  return (LEARNED_METHODS as readonly Method[]).includes(method);
}

export function isGraphMethod(method: Method): method is GraphMethod {
  return (GRAPH_METHODS as readonly Method[]).includes(method);
}

export function findRun(runs: readonly Run[], method: Method): Run | null {
  return runs.find((run) => run.method === method) ?? null;
}

export function expectedScoreGnnMethod(runs: readonly Run[]): GraphMethod | null {
  let best: GraphMethod | null = null;
  let bestValue = -Infinity;
  for (const method of GRAPH_METHODS) {
    const training = findRun(runs, method)?.training;
    if (!training) continue;
    if (best === null || training.validationPrAuc > bestValue) {
      best = method;
      bestValue = training.validationPrAuc;
    }
  }
  return best;
}

export function runDateRange(runs: readonly Run[]): { from: string; to: string } | null {
  const dates = [...new Set(runs.map((run) => run.date))].sort();
  const from = dates[0];
  const to = dates[dates.length - 1];
  return from === undefined || to === undefined ? null : { from, to };
}

export function bestRunAmong(runs: readonly Run[], methods: readonly Method[]): Run | null {
  return bestRun(runs.filter((run) => methods.includes(run.method)));
}

export function bestBaseline(runs: readonly Run[]): Run | null {
  return bestRunAmong(runs, BASELINE_METHODS);
}

export function comparisonMargin(evaluation: EvaluationInfo): number {
  return evaluation.randomPrAucQ95 - evaluation.randomPrAucExpected;
}

export function compareScores(value: number, reference: number, margin: number): Comparison {
  const difference = value - reference;
  if (difference > margin) return "higher";
  if (difference < -margin) return "lower";
  return "similar";
}

export function bestAccuracyAtFlagged(flagged: number, evaluation: EvaluationInfo): number {
  const labelled = labelledTestCount(evaluation);
  if (labelled === 0) return Number.NaN;
  return (labelled - Math.abs(evaluation.testPositives - flagged)) / labelled;
}

export function averagedSeparation(shift: number, neighbours: number, otherClassShare = 0): number {
  return (shift * (1 + neighbours * (1 - 2 * otherClassShare))) / Math.sqrt(neighbours + 1);
}

export function selectedCandidate(training: Training): SearchCandidate | null {
  return training.search.find((candidate) => candidate.selected) ?? null;
}

export function reachedEpochLimit(training: Training): boolean {
  return training.selectedEpoch >= training.maxEpochs;
}

export function searchWeight(runs: readonly Run[], rule: SearchCandidate["positiveWeightRule"]): number | null {
  for (const run of runs) {
    const candidate = run.training?.search.find((entry) => entry.positiveWeightRule === rule);
    if (candidate) return candidate.positiveWeight;
  }
  return null;
}

export function finalTrainRatioWeight(runs: readonly Run[]): number | null {
  const run = runs.find((entry) => entry.training?.positiveWeightRule === "trainRatio");
  return run?.training?.positiveWeight ?? null;
}

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

function isJsonObject(value: JsonValue): value is JsonObject {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function flattenEntry(key: string, value: JsonValue): Array<[string, string]> {
  if (isJsonObject(value)) {
    return Object.entries(value).flatMap(([child, item]) => flattenEntry(`${key}.${child}`, item));
  }
  if (Array.isArray(value) && value.some(isJsonObject)) {
    return value.flatMap((item, index) => flattenEntry(`${key}[${index}]`, item));
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
