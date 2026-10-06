import edgesJson from "../../../../data/k3/edges.json";
import metricsJson from "../../../../data/k3/metrics.json";
import nodesJson from "../../../../data/k3/nodes.json";
import { expectedRandomPrAuc } from "./summary";
import {
  FEATURE_SETS,
  HOPS,
  METHODS,
  NODE_LABELS,
  SCHEMA_VERSION,
  type DatasetInfo,
  type EdgesFile,
  type EvaluationInfo,
  type Homophily,
  type JsonObject,
  type Metrics,
  type NetEdge,
  type NetNode,
  type NetzRadarData,
  type NodeSelection,
  type NodesFile,
  type PrCurvePoint,
  type Run,
  type SplitInfo,
  type StepRange,
} from "./types";

export const MAX_PR_CURVE_POINTS = 101;
const CONSISTENCY_TOLERANCE = 1e-4;
const MAX_HOP = 2;
const ISO_UTC_PATTERN = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$/;
const DATE_PATTERN = /^\d{4}-\d{2}-\d{2}$/;

export class NetzRadarSchemaError extends Error {
  readonly path: string;

  constructor(path: string, message: string) {
    super(`NetzRadar-Daten ungültig bei ${path}: ${message}`);
    this.name = "NetzRadarSchemaError";
    this.path = path;
  }
}

type RawObject = Record<string, unknown>;

function fail(path: string, message: string): never {
  throw new NetzRadarSchemaError(path, message);
}

function isPlainObject(value: unknown): value is RawObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readObject(value: unknown, path: string, keys: readonly string[]): RawObject {
  if (!isPlainObject(value)) fail(path, "Objekt erwartet");
  for (const key of keys) {
    if (!(key in value)) fail(`${path}.${key}`, "Feld fehlt");
  }
  for (const key of Object.keys(value)) {
    if (!keys.includes(key)) fail(`${path}.${key}`, "unbekanntes Feld");
  }
  return value;
}

function readArray(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) fail(path, "Liste erwartet");
  return value;
}

function readString(value: unknown, path: string): string {
  if (typeof value !== "string" || value.length === 0) fail(path, "nicht leerer Text erwartet");
  return value;
}

function readBoolean(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") fail(path, "Wahrheitswert erwartet");
  return value;
}

function readNumber(value: unknown, path: string, min = -Infinity, max = Infinity): number {
  if (typeof value !== "number" || !Number.isFinite(value)) fail(path, "endliche Zahl erwartet");
  if (value < min || value > max) fail(path, `Wert ${value} liegt nicht in [${min}, ${max}]`);
  return value;
}

function readInteger(value: unknown, path: string, min = 0, max = Number.MAX_SAFE_INTEGER): number {
  const number = readNumber(value, path, min, max);
  if (!Number.isInteger(number)) fail(path, "ganze Zahl erwartet");
  return number;
}

function readUnit(value: unknown, path: string): number {
  return readNumber(value, path, 0, 1);
}

function readLiteral<T extends string | number>(value: unknown, expected: T, path: string): T {
  if (value !== expected) fail(path, `${JSON.stringify(expected)} erwartet`);
  return expected;
}

function readEnum<T extends string | number>(value: unknown, allowed: readonly T[], path: string): T {
  const match = allowed.find((entry) => entry === value);
  if (match === undefined) fail(path, `einer von ${allowed.join(", ")} erwartet`);
  return match;
}

function readJsonObject(value: unknown, path: string): JsonObject {
  if (!isPlainObject(value)) fail(path, "Objekt erwartet");
  return value as JsonObject;
}

function readStepRange(value: unknown, path: string): StepRange {
  const raw = readObject(value, path, ["from", "to"]);
  const from = readInteger(raw.from, `${path}.from`, 1);
  const to = readInteger(raw.to, `${path}.to`, 1);
  if (from > to) fail(path, "from liegt nach to");
  return { from, to };
}

function checkClose(actual: number, expected: number, path: string): void {
  if (!(Math.abs(actual - expected) <= CONSISTENCY_TOLERANCE)) {
    fail(path, `Wert ${actual} passt nicht zu ${expected.toFixed(4)}`);
  }
}

function parseHomophily(value: unknown, path: string): Homophily {
  const raw = readObject(value, path, [
    "labelledEdges",
    "sameLabelShare",
    "illicitIllicitEdges",
    "licitLicitEdges",
    "illicitLicitEdges",
    "illicitWithIllicitNeighbour",
    "licitWithIllicitNeighbour",
  ]);
  const homophily = {
    labelledEdges: readInteger(raw.labelledEdges, `${path}.labelledEdges`),
    sameLabelShare: readUnit(raw.sameLabelShare, `${path}.sameLabelShare`),
    illicitIllicitEdges: readInteger(raw.illicitIllicitEdges, `${path}.illicitIllicitEdges`),
    licitLicitEdges: readInteger(raw.licitLicitEdges, `${path}.licitLicitEdges`),
    illicitLicitEdges: readInteger(raw.illicitLicitEdges, `${path}.illicitLicitEdges`),
    illicitWithIllicitNeighbour: readInteger(raw.illicitWithIllicitNeighbour, `${path}.illicitWithIllicitNeighbour`),
    licitWithIllicitNeighbour: readInteger(raw.licitWithIllicitNeighbour, `${path}.licitWithIllicitNeighbour`),
  };
  const { labelledEdges, illicitIllicitEdges, licitLicitEdges, illicitLicitEdges } = homophily;
  if (illicitIllicitEdges + licitLicitEdges + illicitLicitEdges !== labelledEdges) {
    fail(path, "Kantenzahlen ergeben nicht labelledEdges");
  }
  if (labelledEdges > 0) {
    checkClose(homophily.sameLabelShare, (illicitIllicitEdges + licitLicitEdges) / labelledEdges, `${path}.sameLabelShare`);
  }
  return homophily;
}

function parseDataset(value: unknown, path: string): DatasetInfo {
  const raw = readObject(value, path, [
    "name",
    "displayName",
    "license",
    "source",
    "generator",
    "nodes",
    "edges",
    "timeSteps",
    "features",
    "labelCounts",
    "homophily",
  ]);
  const counts = readObject(raw.labelCounts, `${path}.labelCounts`, NODE_LABELS);
  const labelCounts = {
    illicit: readInteger(counts.illicit, `${path}.labelCounts.illicit`),
    licit: readInteger(counts.licit, `${path}.labelCounts.licit`),
    unknown: readInteger(counts.unknown, `${path}.labelCounts.unknown`),
  };
  const nodes = readInteger(raw.nodes, `${path}.nodes`, 1);
  if (labelCounts.illicit + labelCounts.licit + labelCounts.unknown !== nodes) {
    fail(`${path}.labelCounts`, "Summe der Labels ist ungleich der Knotenzahl");
  }
  const homophily = parseHomophily(raw.homophily, `${path}.homophily`);
  if (homophily.illicitWithIllicitNeighbour > labelCounts.illicit) {
    fail(`${path}.homophily.illicitWithIllicitNeighbour`, "mehr Knoten als auffällige Knoten im Netz");
  }
  if (homophily.licitWithIllicitNeighbour > labelCounts.licit) {
    fail(`${path}.homophily.licitWithIllicitNeighbour`, "mehr Knoten als unauffällige Knoten im Netz");
  }
  return {
    name: readString(raw.name, `${path}.name`),
    displayName: readString(raw.displayName, `${path}.displayName`),
    license: readString(raw.license, `${path}.license`),
    source: readString(raw.source, `${path}.source`),
    generator: raw.generator === null ? null : readJsonObject(raw.generator, `${path}.generator`),
    nodes,
    edges: readInteger(raw.edges, `${path}.edges`),
    timeSteps: readInteger(raw.timeSteps, `${path}.timeSteps`, 1),
    features: readInteger(raw.features, `${path}.features`, 1),
    labelCounts,
    homophily,
  };
}

function parseSplit(value: unknown, path: string): SplitInfo {
  const raw = readObject(value, path, ["kind", "train", "validation", "test", "crossSplitEdges"]);
  const train = readStepRange(raw.train, `${path}.train`);
  const validation = readStepRange(raw.validation, `${path}.validation`);
  const test = readStepRange(raw.test, `${path}.test`);
  if (train.to >= test.from) fail(`${path}.test`, "Testzeitraum beginnt nicht nach dem Trainingszeitraum");
  if (validation.from < train.from || validation.to > train.to) {
    fail(`${path}.validation`, "Validierung liegt nicht im Trainingszeitraum");
  }
  const crossSplitEdges = readInteger(raw.crossSplitEdges, `${path}.crossSplitEdges`);
  if (crossSplitEdges !== 0) fail(`${path}.crossSplitEdges`, "Kanten zwischen Training und Test müssen 0 sein");
  return { kind: readLiteral(raw.kind, "temporal", `${path}.kind`), train, validation, test, crossSplitEdges };
}

function parseEvaluation(value: unknown, path: string): EvaluationInfo {
  const raw = readObject(value, path, [
    "positiveLabel",
    "excludedLabel",
    "testPositives",
    "testNegatives",
    "prevalence",
    "allLicitAccuracy",
    "randomPrAucExpected",
    "randomPrAucQ95",
    "randomPermutations",
  ]);
  const testPositives = readInteger(raw.testPositives, `${path}.testPositives`, 1);
  const testNegatives = readInteger(raw.testNegatives, `${path}.testNegatives`, 1);
  const prevalence = readUnit(raw.prevalence, `${path}.prevalence`);
  const allLicitAccuracy = readUnit(raw.allLicitAccuracy, `${path}.allLicitAccuracy`);
  const randomPrAucExpected = readUnit(raw.randomPrAucExpected, `${path}.randomPrAucExpected`);
  const randomPrAucQ95 = readUnit(raw.randomPrAucQ95, `${path}.randomPrAucQ95`);
  const labelled = testPositives + testNegatives;
  checkClose(prevalence, testPositives / labelled, `${path}.prevalence`);
  checkClose(allLicitAccuracy, testNegatives / labelled, `${path}.allLicitAccuracy`);
  checkClose(randomPrAucExpected, expectedRandomPrAuc(testPositives, labelled), `${path}.randomPrAucExpected`);
  if (randomPrAucQ95 < prevalence) fail(`${path}.randomPrAucQ95`, "liegt unter der Prävalenz");
  return {
    positiveLabel: readLiteral(raw.positiveLabel, "illicit", `${path}.positiveLabel`),
    excludedLabel: readLiteral(raw.excludedLabel, "unknown", `${path}.excludedLabel`),
    testPositives,
    testNegatives,
    prevalence,
    allLicitAccuracy,
    randomPrAucExpected,
    randomPrAucQ95,
    randomPermutations: readInteger(raw.randomPermutations, `${path}.randomPermutations`, 1),
  };
}

function parsePrCurve(value: unknown, path: string): PrCurvePoint[] {
  const raw = readArray(value, path);
  if (raw.length === 0 || raw.length > MAX_PR_CURVE_POINTS) {
    fail(path, `1 bis ${MAX_PR_CURVE_POINTS} Punkte erwartet, gefunden ${raw.length}`);
  }
  const points = raw.map((entry, index) => {
    const point = readObject(entry, `${path}[${index}]`, ["recall", "precision"]);
    return {
      recall: readUnit(point.recall, `${path}[${index}].recall`),
      precision: readUnit(point.precision, `${path}[${index}].precision`),
    };
  });
  points.forEach((point, index) => {
    const previous = points[index - 1];
    if (previous && point.recall < previous.recall) fail(`${path}[${index}]`, "nicht nach recall aufsteigend sortiert");
  });
  return points;
}

function parseRun(value: unknown, path: string): Run {
  const raw = readObject(value, path, [
    "method",
    "displayName",
    "featureSet",
    "seed",
    "date",
    "hyperparameters",
    "prAuc",
    "precisionAtRecall50",
    "recallAtPrecision50",
    "accuracy",
    "accuracyThreshold",
    "accuracyFlagged",
    "accuracyTruePositives",
    "prCurve",
  ]);
  const date = readString(raw.date, `${path}.date`);
  if (!DATE_PATTERN.test(date)) fail(`${path}.date`, "Datum im Format JJJJ-MM-TT erwartet");
  const accuracyFlagged = readInteger(raw.accuracyFlagged, `${path}.accuracyFlagged`, 1);
  const accuracyTruePositives = readInteger(raw.accuracyTruePositives, `${path}.accuracyTruePositives`);
  if (accuracyTruePositives > accuracyFlagged) {
    fail(`${path}.accuracyTruePositives`, "mehr Treffer als markierte Knoten");
  }
  return {
    method: readEnum(raw.method, METHODS, `${path}.method`),
    displayName: readString(raw.displayName, `${path}.displayName`),
    featureSet: readEnum(raw.featureSet, FEATURE_SETS, `${path}.featureSet`),
    seed: readInteger(raw.seed, `${path}.seed`),
    date,
    hyperparameters: readJsonObject(raw.hyperparameters, `${path}.hyperparameters`),
    prAuc: readUnit(raw.prAuc, `${path}.prAuc`),
    precisionAtRecall50: readUnit(raw.precisionAtRecall50, `${path}.precisionAtRecall50`),
    recallAtPrecision50: readUnit(raw.recallAtPrecision50, `${path}.recallAtPrecision50`),
    accuracy: readUnit(raw.accuracy, `${path}.accuracy`),
    accuracyThreshold: readString(raw.accuracyThreshold, `${path}.accuracyThreshold`),
    accuracyFlagged,
    accuracyTruePositives,
    prCurve: parsePrCurve(raw.prCurve, `${path}.prCurve`),
  };
}

export function parseMetrics(value: unknown): Metrics {
  const path = "metrics";
  const raw = readObject(value, path, ["schemaVersion", "generatedAt", "dataset", "split", "evaluation", "seed", "runs"]);
  const generatedAt = readString(raw.generatedAt, `${path}.generatedAt`);
  if (!ISO_UTC_PATTERN.test(generatedAt) || Number.isNaN(Date.parse(generatedAt))) {
    fail(`${path}.generatedAt`, "Zeitpunkt im Format ISO 8601 (UTC) erwartet");
  }
  const runs = readArray(raw.runs, `${path}.runs`).map((entry, index) => parseRun(entry, `${path}.runs[${index}]`));
  if (runs.length === 0) fail(`${path}.runs`, "mindestens ein Lauf erwartet");
  const methods = new Set<string>();
  runs.forEach((run, index) => {
    if (methods.has(run.method)) fail(`${path}.runs[${index}].method`, `Verfahren ${run.method} doppelt`);
    methods.add(run.method);
  });
  const schemaVersion = readLiteral(raw.schemaVersion, SCHEMA_VERSION, `${path}.schemaVersion`);
  const evaluation = parseEvaluation(raw.evaluation, `${path}.evaluation`);
  const labelled = evaluation.testPositives + evaluation.testNegatives;
  runs.forEach((run, index) => {
    if (run.accuracyFlagged > labelled) {
      fail(`${path}.runs[${index}].accuracyFlagged`, "mehr markierte Knoten als Testknoten mit Label");
    }
    if (run.accuracyTruePositives > evaluation.testPositives) {
      fail(`${path}.runs[${index}].accuracyTruePositives`, "mehr Treffer als auffällige Testknoten");
    }
  });
  return {
    schemaVersion,
    generatedAt,
    dataset: parseDataset(raw.dataset, `${path}.dataset`),
    split: parseSplit(raw.split, `${path}.split`),
    evaluation,
    seed: readInteger(raw.seed, `${path}.seed`),
    runs,
  };
}

function parseSelection(value: unknown, path: string): NodeSelection {
  const raw = readObject(value, path, ["scoreField", "pool", "seeds", "hops", "maxNodes", "truncated"]);
  return {
    scoreField: readLiteral(raw.scoreField, "scoreIforest", `${path}.scoreField`),
    pool: readLiteral(raw.pool, "test", `${path}.pool`),
    seeds: readInteger(raw.seeds, `${path}.seeds`, 1),
    hops: readInteger(raw.hops, `${path}.hops`, 0, MAX_HOP),
    maxNodes: readInteger(raw.maxNodes, `${path}.maxNodes`, 1),
    truncated: readBoolean(raw.truncated, `${path}.truncated`),
  };
}

function parseNode(value: unknown, path: string, selection: NodeSelection): NetNode {
  const raw = readObject(value, path, [
    "id",
    "x",
    "y",
    "label",
    "timeStep",
    "scoreZscore",
    "scoreIforest",
    "scoreGnn",
    "seedRank",
    "hop",
    "inDegree",
    "outDegree",
  ]);
  const hop = readEnum(raw.hop, HOPS, `${path}.hop`);
  if (hop > selection.hops) fail(`${path}.hop`, `Abstand größer als ${selection.hops}`);
  const seedRank = raw.seedRank === null ? null : readInteger(raw.seedRank, `${path}.seedRank`, 1, selection.seeds);
  if ((seedRank === null) !== (hop !== 0)) fail(`${path}.seedRank`, "Startknoten und Abstand 0 passen nicht zusammen");
  return {
    id: readString(raw.id, `${path}.id`),
    x: readNumber(raw.x, `${path}.x`, -1, 1),
    y: readNumber(raw.y, `${path}.y`, -1, 1),
    label: readEnum(raw.label, NODE_LABELS, `${path}.label`),
    timeStep: readInteger(raw.timeStep, `${path}.timeStep`, 1),
    scoreZscore: readNumber(raw.scoreZscore, `${path}.scoreZscore`),
    scoreIforest: readNumber(raw.scoreIforest, `${path}.scoreIforest`),
    scoreGnn: raw.scoreGnn === null ? null : readNumber(raw.scoreGnn, `${path}.scoreGnn`),
    seedRank,
    hop,
    inDegree: readInteger(raw.inDegree, `${path}.inDegree`),
    outDegree: readInteger(raw.outDegree, `${path}.outDegree`),
  };
}

export function parseNodes(value: unknown): NodesFile {
  const path = "nodes";
  const raw = readObject(value, path, ["schemaVersion", "selection", "nodes"]);
  const selection = parseSelection(raw.selection, `${path}.selection`);
  const entries = readArray(raw.nodes, `${path}.nodes`);
  if (entries.length === 0 || entries.length > selection.maxNodes) {
    fail(`${path}.nodes`, `1 bis ${selection.maxNodes} Knoten erwartet, gefunden ${entries.length}`);
  }
  const nodes = entries.map((entry, index) => parseNode(entry, `${path}.nodes[${index}]`, selection));
  const ids = new Set<string>();
  const ranks = new Set<number>();
  nodes.forEach((node, index) => {
    if (ids.has(node.id)) fail(`${path}.nodes[${index}].id`, `ID ${node.id} doppelt`);
    ids.add(node.id);
    if (node.seedRank !== null) {
      if (ranks.has(node.seedRank)) fail(`${path}.nodes[${index}].seedRank`, `Rang ${node.seedRank} doppelt`);
      ranks.add(node.seedRank);
    }
  });
  if (ranks.size === 0) fail(`${path}.nodes`, "kein Startknoten");
  if (Math.max(...ranks) !== ranks.size) {
    fail(`${path}.nodes`, `Startknoten-Ränge müssen lückenlos 1 bis ${ranks.size} sein`);
  }
  return { schemaVersion: readLiteral(raw.schemaVersion, SCHEMA_VERSION, `${path}.schemaVersion`), selection, nodes };
}

export function parseEdges(value: unknown, nodeIds: ReadonlySet<string>): EdgesFile {
  const path = "edges";
  const raw = readObject(value, path, ["schemaVersion", "edges"]);
  const edges: NetEdge[] = readArray(raw.edges, `${path}.edges`).map((entry, index) => {
    const edgePath = `${path}.edges[${index}]`;
    const edge = readObject(entry, edgePath, ["source", "target"]);
    const source = readString(edge.source, `${edgePath}.source`);
    const target = readString(edge.target, `${edgePath}.target`);
    if (!nodeIds.has(source)) fail(`${edgePath}.source`, `unbekannter Knoten ${source}`);
    if (!nodeIds.has(target)) fail(`${edgePath}.target`, `unbekannter Knoten ${target}`);
    if (source === target) fail(edgePath, "Selbstschleife");
    return { source, target };
  });
  edges.forEach((edge, index) => {
    const previous = edges[index - 1];
    if (!previous) return;
    if (previous.source === edge.source && previous.target === edge.target) {
      fail(`${path}.edges[${index}]`, "doppelte Kante");
    }
    if (previous.source > edge.source || (previous.source === edge.source && previous.target > edge.target)) {
      fail(`${path}.edges[${index}]`, "Kanten nicht nach source und target sortiert");
    }
  });
  return { schemaVersion: readLiteral(raw.schemaVersion, SCHEMA_VERSION, `${path}.schemaVersion`), edges };
}

export function parseNetzRadarData(input: { metrics: unknown; nodes: unknown; edges: unknown }): NetzRadarData {
  const metrics = parseMetrics(input.metrics);
  const nodes = parseNodes(input.nodes);
  const edges = parseEdges(input.edges, new Set(nodes.nodes.map((node) => node.id)));
  const { test } = metrics.split;
  nodes.nodes.forEach((node, index) => {
    if (node.timeStep < test.from || node.timeStep > test.to) {
      fail(`nodes.nodes[${index}].timeStep`, `Zeitschritt ${node.timeStep} liegt außerhalb des Testzeitraums`);
    }
  });
  return { metrics, nodes, edges };
}

export const netzRadarData: NetzRadarData = parseNetzRadarData({
  metrics: metricsJson,
  nodes: nodesJson,
  edges: edgesJson,
});
