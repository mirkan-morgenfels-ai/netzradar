import edgesJson from "../../../../data/k3/edges.json";
import metricsJson from "../../../../data/k3/metrics.json";
import nodesJson from "../../../../data/k3/nodes.json";
import { expectedRandomPrAuc, expectedScoreGnnMethod, isLearnedMethod, isNeuralMethod } from "./summary";
import {
  EDGE_MODES,
  FEATURE_SETS,
  GRAPH_METHODS,
  HOPS,
  METHODS,
  NODE_LABELS,
  POSITIVE_WEIGHT_RULES,
  SCHEMA_VERSION,
  type BoostingCandidate,
  type BoostingEnvironment,
  type BoostingTraining,
  type DatasetInfo,
  type EdgeMode,
  type EdgesFile,
  type EvaluationInfo,
  type FeatureSet,
  type GraphMethod,
  type Homophily,
  type Hop,
  type JsonObject,
  type Method,
  type Metrics,
  type NetEdge,
  type NetNode,
  type NetzRadarData,
  type NeuralMethod,
  type NeuralTraining,
  type NodeSelection,
  type NodesFile,
  type PositiveWeightRule,
  type PrCurvePoint,
  type Run,
  type Scaling,
  type SearchCandidate,
  type SeedSpread,
  type SplitInfo,
  type StepRange,
  type Training,
  type TrainingEnvironment,
} from "./types";

export const MAX_PR_CURVE_POINTS = 101;
const CONSISTENCY_TOLERANCE = 1e-4;
const ROUNDED_MEAN_TOLERANCE = CONSISTENCY_TOLERANCE + 1e-9;
const MAX_HOP = 2;
const MIN_LEAF_NODES = 2;
const ISO_UTC_PATTERN = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$/;
const DATE_PATTERN = /^\d{4}-\d{2}-\d{2}$/;

export const ARCHITECTURES: Record<NeuralMethod, string> = {
  gcn: "GCNConv",
  graphsage: "SAGEConv(aggr=mean)",
  mlp: "Linear",
};

export const EDGE_MODE_BY_METHOD: Record<NeuralMethod, EdgeMode> = {
  gcn: "undirected",
  graphsage: "undirected",
  mlp: "none",
};

const TRAINING_KEYS = [
  "architecture",
  "layers",
  "hidden",
  "activation",
  "dropout",
  "optimizer",
  "learningRate",
  "weightDecay",
  "loss",
  "score",
  "positiveWeight",
  "positiveWeightRule",
  "edges",
  "dtype",
  "scaling",
  "features",
  "maxEpochs",
  "patience",
  "selectedEpoch",
  "selectionMetric",
  "validationPrAuc",
  "selectionSteps",
  "validationSteps",
  "finalFitSteps",
  "search",
  "seedSpread",
  "environment",
] as const;

const BOOSTING_KEYS = [
  "estimator",
  "loss",
  "score",
  "learningRate",
  "maxIter",
  "maxLeafNodes",
  "minSamplesLeaf",
  "l2Regularization",
  "maxBins",
  "maxFeatures",
  "earlyStopping",
  "randomState",
  "positiveWeight",
  "positiveWeightRule",
  "edges",
  "features",
  "selectedIteration",
  "selectionMetric",
  "validationPrAuc",
  "selectionSteps",
  "validationSteps",
  "finalFitSteps",
  "search",
  "seedSpread",
  "environment",
] as const;

const SCALING_KEYS = [
  "center",
  "scale",
  "madScale",
  "fallback",
  "clip",
  "fitOn",
  "zeroMadFeatures",
  "unitScaleFeatures",
] as const;

const SEARCH_KEYS = [
  "featureSet",
  "positiveWeightRule",
  "positiveWeight",
  "validationPrAuc",
  "bestEpoch",
  "stoppedEpoch",
  "selected",
] as const;

const BOOSTING_SEARCH_KEYS = [
  "featureSet",
  "positiveWeightRule",
  "positiveWeight",
  "maxLeafNodes",
  "validationPrAuc",
  "bestIteration",
  "selected",
] as const;

const SPREAD_KEYS = ["seeds", "reportedSeed", "prAuc", "mean", "min", "max"] as const;

const ENVIRONMENT_KEYS = ["torch", "torchGeometric", "python", "platform", "threads", "deterministicAlgorithms"] as const;

const BOOSTING_ENVIRONMENT_KEYS = ["scikitLearn", "python", "platform", "threads"] as const;

const SELECTION_KEYS = ["scoreField", "scoreFieldGnn", "pool", "seeds", "hops", "maxNodes", "truncated"] as const;

const NODE_KEYS = [
  "id",
  "x",
  "y",
  "label",
  "timeStep",
  "scoreZscore",
  "scoreIforest",
  "scoreGnn",
  "seedRank",
  "seedRankGnn",
  "hop",
  "hopGnn",
  "inDegree",
  "outDegree",
] as const;

export const FIXED_TRAINING = {
  layers: 2,
  hidden: 64,
  activation: "relu",
  dropout: 0.5,
  optimizer: "adam",
  learningRate: 0.01,
  weightDecay: 0.0005,
  loss: "weightedCrossEntropy",
  score: "logitIllicit - logitLicit",
} as const;

export const FIXED_SCALING = {
  center: "median",
  scale: "mad",
  madScale: 1.4826,
  fallback: ["std", "one"],
} as const;

export const FIXED_BOOSTING = {
  estimator: "HistGradientBoostingClassifier",
  loss: "log_loss",
  score: "logOddsIllicit",
  learningRate: 0.1,
  minSamplesLeaf: 20,
  l2Regularization: 0,
  maxBins: 255,
  maxFeatures: 1,
  edges: "none",
} as const;

export const DTYPES = ["float64", "float32"] as const;

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

function readStringList(value: unknown, path: string, minLength = 0): string[] {
  const list = readArray(value, path).map((entry, index) => readString(entry, `${path}[${index}]`));
  if (list.length < minLength) fail(path, `mindestens ${minLength} Einträge erwartet`);
  if (new Set(list).size !== list.length) fail(path, "Einträge doppelt");
  return list;
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

function readPositive(value: unknown, path: string): number {
  const number = readNumber(value, path);
  if (!(number > 0)) fail(path, `Wert ${number} ist nicht positiv`);
  return number;
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

function sameRange(a: StepRange, b: StepRange): boolean {
  return a.from === b.from && a.to === b.to;
}

function checkClose(actual: number, expected: number, path: string, tolerance = CONSISTENCY_TOLERANCE): void {
  if (!(Math.abs(actual - expected) <= tolerance)) {
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
    fail(`${path}.homophily.illicitWithIllicitNeighbour`, "mehr Knoten als Knoten mit Label illegal im Netz");
  }
  if (homophily.licitWithIllicitNeighbour > labelCounts.licit) {
    fail(`${path}.homophily.licitWithIllicitNeighbour`, "mehr Knoten als Knoten mit Label legal im Netz");
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

function parseScaling(value: unknown, path: string, features: readonly string[]): Scaling {
  const raw = readObject(value, path, SCALING_KEYS);
  const zeroMadFeatures = readStringList(raw.zeroMadFeatures, `${path}.zeroMadFeatures`);
  const unitScaleFeatures = readStringList(raw.unitScaleFeatures, `${path}.unitScaleFeatures`);
  for (const [key, names] of [
    ["zeroMadFeatures", zeroMadFeatures],
    ["unitScaleFeatures", unitScaleFeatures],
  ] as const) {
    names.forEach((name, index) => {
      if (!features.includes(name)) fail(`${path}.${key}[${index}]`, `Merkmal ${name} fehlt in features`);
    });
  }
  const fallback = readStringList(raw.fallback, `${path}.fallback`);
  if (fallback.length !== FIXED_SCALING.fallback.length || fallback.some((rule, index) => rule !== FIXED_SCALING.fallback[index])) {
    fail(`${path}.fallback`, `${JSON.stringify(FIXED_SCALING.fallback)} erwartet`);
  }
  return {
    center: readLiteral(raw.center, FIXED_SCALING.center, `${path}.center`),
    scale: readLiteral(raw.scale, FIXED_SCALING.scale, `${path}.scale`),
    madScale: readLiteral(raw.madScale, FIXED_SCALING.madScale, `${path}.madScale`),
    fallback,
    clip: readPositive(raw.clip, `${path}.clip`),
    fitOn: readLiteral(raw.fitOn, "train", `${path}.fitOn`),
    zeroMadFeatures,
    unitScaleFeatures,
  };
}

function parseCandidate(value: unknown, path: string, maxEpochs: number, patience: number): SearchCandidate {
  const raw = readObject(value, path, SEARCH_KEYS);
  const bestEpoch = readInteger(raw.bestEpoch, `${path}.bestEpoch`, 1, maxEpochs);
  const stoppedEpoch = readInteger(raw.stoppedEpoch, `${path}.stoppedEpoch`, bestEpoch, maxEpochs);
  const expectedStop = Math.min(bestEpoch + patience, maxEpochs);
  if (stoppedEpoch !== expectedStop) {
    fail(`${path}.stoppedEpoch`, `Abbruch in Epoche ${expectedStop} erwartet (Geduld ${patience}, höchstens ${maxEpochs})`);
  }
  return {
    featureSet: readEnum(raw.featureSet, FEATURE_SETS, `${path}.featureSet`),
    positiveWeightRule: readEnum(raw.positiveWeightRule, POSITIVE_WEIGHT_RULES, `${path}.positiveWeightRule`),
    positiveWeight: readPositive(raw.positiveWeight, `${path}.positiveWeight`),
    validationPrAuc: readUnit(raw.validationPrAuc, `${path}.validationPrAuc`),
    bestEpoch,
    stoppedEpoch,
    selected: readBoolean(raw.selected, `${path}.selected`),
  };
}

function parseBoostingCandidate(value: unknown, path: string, maxIter: number): BoostingCandidate {
  const raw = readObject(value, path, BOOSTING_SEARCH_KEYS);
  return {
    featureSet: readEnum(raw.featureSet, FEATURE_SETS, `${path}.featureSet`),
    positiveWeightRule: readEnum(raw.positiveWeightRule, POSITIVE_WEIGHT_RULES, `${path}.positiveWeightRule`),
    positiveWeight: readPositive(raw.positiveWeight, `${path}.positiveWeight`),
    maxLeafNodes: readInteger(raw.maxLeafNodes, `${path}.maxLeafNodes`, MIN_LEAF_NODES),
    validationPrAuc: readUnit(raw.validationPrAuc, `${path}.validationPrAuc`),
    bestIteration: readInteger(raw.bestIteration, `${path}.bestIteration`, 1, maxIter),
    selected: readBoolean(raw.selected, `${path}.selected`),
  };
}

function parseEnvironment(value: unknown, path: string): TrainingEnvironment {
  const raw = readObject(value, path, ENVIRONMENT_KEYS);
  return {
    torch: readString(raw.torch, `${path}.torch`),
    torchGeometric: readString(raw.torchGeometric, `${path}.torchGeometric`),
    python: readString(raw.python, `${path}.python`),
    platform: readString(raw.platform, `${path}.platform`),
    threads: readInteger(raw.threads, `${path}.threads`, 1),
    deterministicAlgorithms: readBoolean(raw.deterministicAlgorithms, `${path}.deterministicAlgorithms`),
  };
}

function parseBoostingEnvironment(value: unknown, path: string): BoostingEnvironment {
  const raw = readObject(value, path, BOOSTING_ENVIRONMENT_KEYS);
  return {
    scikitLearn: readString(raw.scikitLearn, `${path}.scikitLearn`),
    python: readString(raw.python, `${path}.python`),
    platform: readString(raw.platform, `${path}.platform`),
    threads: readInteger(raw.threads, `${path}.threads`, 1),
  };
}

function parseSteps(raw: RawObject, path: string) {
  const selectionSteps = readStepRange(raw.selectionSteps, `${path}.selectionSteps`);
  const validationSteps = readStepRange(raw.validationSteps, `${path}.validationSteps`);
  const finalFitSteps = readStepRange(raw.finalFitSteps, `${path}.finalFitSteps`);
  if (selectionSteps.from !== finalFitSteps.from || selectionSteps.to + 1 !== validationSteps.from) {
    fail(`${path}.selectionSteps`, "Auswahlteil muss direkt vor dem Validierungsteil enden");
  }
  if (validationSteps.to !== finalFitSteps.to) {
    fail(`${path}.validationSteps`, "Validierungsteil muss am Ende des Trainingszeitraums liegen");
  }
  return { selectionSteps, validationSteps, finalFitSteps };
}

interface Candidate {
  featureSet: FeatureSet;
  positiveWeightRule: PositiveWeightRule;
  positiveWeight: number;
  validationPrAuc: number;
  selected: boolean;
}

function selectFirstBest<T extends Candidate>(search: readonly T[], path: string, identity: (candidate: T) => string): T {
  if (search.length === 0) fail(path, "mindestens ein Kandidat erwartet");
  const keys = new Set<string>();
  search.forEach((candidate, index) => {
    const key = identity(candidate);
    if (keys.has(key)) fail(`${path}[${index}]`, `Kandidat ${key} doppelt`);
    keys.add(key);
  });
  const selectedIndices = search.flatMap((candidate, index) => (candidate.selected ? [index] : []));
  if (selectedIndices.length !== 1) fail(path, `genau ein gewählter Kandidat erwartet, gefunden ${selectedIndices.length}`);
  const selectedIndex = selectedIndices[0] ?? 0;
  const selected = search[selectedIndex];
  if (!selected) fail(path, "gewählter Kandidat fehlt");
  search.forEach((candidate, index) => {
    if (candidate.validationPrAuc > selected.validationPrAuc) {
      fail(`${path}[${index}].validationPrAuc`, "höher als beim gewählten Kandidaten");
    }
    if (index < selectedIndex && candidate.validationPrAuc === selected.validationPrAuc) {
      fail(`${path}[${index}].validationPrAuc`, "gleich hoch wie beim gewählten Kandidaten; bei Gleichstand gilt der frühere");
    }
  });
  return selected;
}

function checkSelected(
  selected: Candidate,
  path: string,
  featureSet: FeatureSet,
  positiveWeightRule: PositiveWeightRule,
  positiveWeight: number,
  validationPrAuc: number,
): void {
  if (selected.featureSet !== featureSet) fail(`${path}.search`, "gewählter Merkmalssatz passt nicht zu featureSet");
  if (selected.positiveWeightRule !== positiveWeightRule) {
    fail(`${path}.search`, "Gewichtsregel des gewählten Kandidaten passt nicht zu positiveWeightRule");
  }
  checkClose(validationPrAuc, selected.validationPrAuc, `${path}.validationPrAuc`);
  if (positiveWeightRule === "fixed") checkClose(positiveWeight, selected.positiveWeight, `${path}.positiveWeight`);
}

function parseSeedSpread(value: unknown, path: string): SeedSpread {
  const raw = readObject(value, path, SPREAD_KEYS);
  const seeds = readArray(raw.seeds, `${path}.seeds`).map((entry, index) => readInteger(entry, `${path}.seeds[${index}]`));
  if (seeds.length === 0) fail(`${path}.seeds`, "mindestens ein Seed erwartet");
  if (new Set(seeds).size !== seeds.length) fail(`${path}.seeds`, "Seeds doppelt");
  const prAuc = readArray(raw.prAuc, `${path}.prAuc`).map((entry, index) => readUnit(entry, `${path}.prAuc[${index}]`));
  if (prAuc.length !== seeds.length) fail(`${path}.prAuc`, `${seeds.length} Werte erwartet, gefunden ${prAuc.length}`);
  const reportedSeed = readInteger(raw.reportedSeed, `${path}.reportedSeed`);
  if (!seeds.includes(reportedSeed)) fail(`${path}.reportedSeed`, `Seed ${reportedSeed} fehlt in seeds`);
  const mean = readUnit(raw.mean, `${path}.mean`);
  const min = readUnit(raw.min, `${path}.min`);
  const max = readUnit(raw.max, `${path}.max`);
  const smallest = Math.min(...prAuc);
  const largest = Math.max(...prAuc);
  if (min !== smallest) fail(`${path}.min`, `kleinster Wert ${smallest} erwartet`);
  if (max !== largest) fail(`${path}.max`, `größter Wert ${largest} erwartet`);
  if (mean < min || mean > max) fail(`${path}.mean`, "liegt nicht zwischen min und max");
  checkClose(mean, prAuc.reduce((sum, entry) => sum + entry, 0) / prAuc.length, `${path}.mean`, ROUNDED_MEAN_TOLERANCE);
  return { seeds, reportedSeed, prAuc, mean, min, max };
}

function checkSpreadAgainstRun(spread: SeedSpread, path: string, seed: number, prAuc: number): void {
  if (spread.reportedSeed !== seed) fail(`${path}.reportedSeed`, `Seed ${seed} des Runs erwartet`);
  const index = spread.seeds.indexOf(spread.reportedSeed);
  const value = spread.prAuc[index];
  if (value !== prAuc) fail(`${path}.prAuc[${index}]`, `Wert ${value} weicht von prAuc ${prAuc} des Runs ab`);
}

function parseNeuralTraining(value: unknown, path: string, method: NeuralMethod, featureSet: FeatureSet): NeuralTraining {
  const raw = readObject(value, path, TRAINING_KEYS);
  const features = readStringList(raw.features, `${path}.features`, 1);
  const maxEpochs = readInteger(raw.maxEpochs, `${path}.maxEpochs`, 1);
  const patience = readInteger(raw.patience, `${path}.patience`, 1);
  const selectedEpoch = readInteger(raw.selectedEpoch, `${path}.selectedEpoch`, 1, maxEpochs);
  const validationPrAuc = readUnit(raw.validationPrAuc, `${path}.validationPrAuc`);
  const positiveWeight = readPositive(raw.positiveWeight, `${path}.positiveWeight`);
  const positiveWeightRule = readEnum(raw.positiveWeightRule, POSITIVE_WEIGHT_RULES, `${path}.positiveWeightRule`);
  const search = readArray(raw.search, `${path}.search`).map((entry, index) =>
    parseCandidate(entry, `${path}.search[${index}]`, maxEpochs, patience),
  );
  const selected = selectFirstBest(search, `${path}.search`, (candidate) => `${candidate.featureSet} ${candidate.positiveWeightRule}`);
  checkSelected(selected, path, featureSet, positiveWeightRule, positiveWeight, validationPrAuc);
  if (selected.bestEpoch !== selectedEpoch) fail(`${path}.selectedEpoch`, "passt nicht zur besten Epoche des gewählten Kandidaten");
  return {
    kind: "neural",
    architecture: readLiteral(raw.architecture, ARCHITECTURES[method], `${path}.architecture`),
    layers: readLiteral(raw.layers, FIXED_TRAINING.layers, `${path}.layers`),
    hidden: readLiteral(raw.hidden, FIXED_TRAINING.hidden, `${path}.hidden`),
    activation: readLiteral(raw.activation, FIXED_TRAINING.activation, `${path}.activation`),
    dropout: readLiteral(raw.dropout, FIXED_TRAINING.dropout, `${path}.dropout`),
    optimizer: readLiteral(raw.optimizer, FIXED_TRAINING.optimizer, `${path}.optimizer`),
    learningRate: readLiteral(raw.learningRate, FIXED_TRAINING.learningRate, `${path}.learningRate`),
    weightDecay: readLiteral(raw.weightDecay, FIXED_TRAINING.weightDecay, `${path}.weightDecay`),
    loss: readLiteral(raw.loss, FIXED_TRAINING.loss, `${path}.loss`),
    score: readLiteral(raw.score, FIXED_TRAINING.score, `${path}.score`),
    positiveWeight,
    positiveWeightRule,
    edges: readLiteral(readEnum(raw.edges, EDGE_MODES, `${path}.edges`), EDGE_MODE_BY_METHOD[method], `${path}.edges`),
    dtype: readEnum(raw.dtype, DTYPES, `${path}.dtype`),
    scaling: parseScaling(raw.scaling, `${path}.scaling`, features),
    features,
    maxEpochs,
    patience,
    selectedEpoch,
    selectionMetric: readLiteral(raw.selectionMetric, "validationPrAuc", `${path}.selectionMetric`),
    validationPrAuc,
    ...parseSteps(raw, path),
    search,
    seedSpread: parseSeedSpread(raw.seedSpread, `${path}.seedSpread`),
    environment: parseEnvironment(raw.environment, `${path}.environment`),
  };
}

function parseBoostingTraining(value: unknown, path: string, featureSet: FeatureSet, seed: number): BoostingTraining {
  const raw = readObject(value, path, BOOSTING_KEYS);
  const features = readStringList(raw.features, `${path}.features`, 1);
  const maxIter = readInteger(raw.maxIter, `${path}.maxIter`, 1);
  const maxLeafNodes = readInteger(raw.maxLeafNodes, `${path}.maxLeafNodes`, MIN_LEAF_NODES);
  const selectedIteration = readInteger(raw.selectedIteration, `${path}.selectedIteration`, 1, maxIter);
  const validationPrAuc = readUnit(raw.validationPrAuc, `${path}.validationPrAuc`);
  const positiveWeight = readPositive(raw.positiveWeight, `${path}.positiveWeight`);
  const positiveWeightRule = readEnum(raw.positiveWeightRule, POSITIVE_WEIGHT_RULES, `${path}.positiveWeightRule`);
  const search = readArray(raw.search, `${path}.search`).map((entry, index) =>
    parseBoostingCandidate(entry, `${path}.search[${index}]`, maxIter),
  );
  const selected = selectFirstBest(
    search,
    `${path}.search`,
    (candidate) =>
      `${candidate.featureSet} ${candidate.positiveWeightRule} ${String(candidate.positiveWeight)} ${candidate.maxLeafNodes}`,
  );
  checkSelected(selected, path, featureSet, positiveWeightRule, positiveWeight, validationPrAuc);
  if (selected.maxLeafNodes !== maxLeafNodes) fail(`${path}.maxLeafNodes`, "passt nicht zum gewählten Kandidaten");
  if (selected.bestIteration !== selectedIteration) {
    fail(`${path}.selectedIteration`, "passt nicht zur besten Iteration des gewählten Kandidaten");
  }
  if (raw.earlyStopping !== false) fail(`${path}.earlyStopping`, "false erwartet");
  const randomState = readInteger(raw.randomState, `${path}.randomState`);
  if (randomState !== seed) fail(`${path}.randomState`, `Seed ${seed} des Runs erwartet`);
  return {
    kind: "boosting",
    estimator: readLiteral(raw.estimator, FIXED_BOOSTING.estimator, `${path}.estimator`),
    loss: readLiteral(raw.loss, FIXED_BOOSTING.loss, `${path}.loss`),
    score: readLiteral(raw.score, FIXED_BOOSTING.score, `${path}.score`),
    learningRate: readLiteral(raw.learningRate, FIXED_BOOSTING.learningRate, `${path}.learningRate`),
    maxIter,
    maxLeafNodes,
    minSamplesLeaf: readLiteral(raw.minSamplesLeaf, FIXED_BOOSTING.minSamplesLeaf, `${path}.minSamplesLeaf`),
    l2Regularization: readLiteral(raw.l2Regularization, FIXED_BOOSTING.l2Regularization, `${path}.l2Regularization`),
    maxBins: readLiteral(raw.maxBins, FIXED_BOOSTING.maxBins, `${path}.maxBins`),
    maxFeatures: readLiteral(raw.maxFeatures, FIXED_BOOSTING.maxFeatures, `${path}.maxFeatures`),
    earlyStopping: false,
    randomState,
    positiveWeight,
    positiveWeightRule,
    edges: readLiteral(readEnum(raw.edges, EDGE_MODES, `${path}.edges`), FIXED_BOOSTING.edges, `${path}.edges`),
    features,
    selectedIteration,
    selectionMetric: readLiteral(raw.selectionMetric, "validationPrAuc", `${path}.selectionMetric`),
    validationPrAuc,
    ...parseSteps(raw, path),
    search,
    seedSpread: parseSeedSpread(raw.seedSpread, `${path}.seedSpread`),
    environment: parseBoostingEnvironment(raw.environment, `${path}.environment`),
  };
}

function parseTraining(
  hyperparameters: JsonObject,
  path: string,
  method: Method,
  featureSet: FeatureSet,
  seed: number,
): Training | null {
  if (!isLearnedMethod(method)) {
    if ("seedSpread" in hyperparameters) fail(`${path}.seedSpread`, "nur bei gelernten Verfahren erwartet");
    return null;
  }
  return isNeuralMethod(method)
    ? parseNeuralTraining(hyperparameters, path, method, featureSet)
    : parseBoostingTraining(hyperparameters, path, featureSet, seed);
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
  const method = readEnum(raw.method, METHODS, `${path}.method`);
  const featureSet = readEnum(raw.featureSet, FEATURE_SETS, `${path}.featureSet`);
  const date = readString(raw.date, `${path}.date`);
  if (!DATE_PATTERN.test(date)) fail(`${path}.date`, "Datum im Format JJJJ-MM-TT erwartet");
  const accuracyFlagged = readInteger(raw.accuracyFlagged, `${path}.accuracyFlagged`, 1);
  const accuracyTruePositives = readInteger(raw.accuracyTruePositives, `${path}.accuracyTruePositives`);
  if (accuracyTruePositives > accuracyFlagged) {
    fail(`${path}.accuracyTruePositives`, "mehr Treffer als markierte Knoten");
  }
  const hyperparameters = readJsonObject(raw.hyperparameters, `${path}.hyperparameters`);
  const seed = readInteger(raw.seed, `${path}.seed`);
  const prAuc = readUnit(raw.prAuc, `${path}.prAuc`);
  const training = parseTraining(hyperparameters, `${path}.hyperparameters`, method, featureSet, seed);
  if (training !== null) checkSpreadAgainstRun(training.seedSpread, `${path}.hyperparameters.seedSpread`, seed, prAuc);
  return {
    method,
    displayName: readString(raw.displayName, `${path}.displayName`),
    featureSet,
    seed,
    date,
    hyperparameters,
    training,
    prAuc,
    precisionAtRecall50: readUnit(raw.precisionAtRecall50, `${path}.precisionAtRecall50`),
    recallAtPrecision50: readUnit(raw.recallAtPrecision50, `${path}.recallAtPrecision50`),
    accuracy: readUnit(raw.accuracy, `${path}.accuracy`),
    accuracyThreshold: readString(raw.accuracyThreshold, `${path}.accuracyThreshold`),
    accuracyFlagged,
    accuracyTruePositives,
    prCurve: parsePrCurve(raw.prCurve, `${path}.prCurve`),
  };
}

function checkTrainingAgainstData(run: Run, path: string, split: SplitInfo, dataset: DatasetInfo): void {
  const { training } = run;
  if (training === null) return;
  const trainingPath = `${path}.hyperparameters`;
  if (!sameRange(training.finalFitSteps, split.train)) {
    fail(`${trainingPath}.finalFitSteps`, "Endmodell muss auf dem Trainingszeitraum des Splits angepasst sein");
  }
  if (!sameRange(training.validationSteps, split.validation)) {
    fail(`${trainingPath}.validationSteps`, "Auswahl muss auf dem Validierungsteil des Splits liegen");
  }
  const localOnly = run.featureSet === "local";
  if (localOnly ? training.features.length !== dataset.features : training.features.length <= dataset.features) {
    fail(`${trainingPath}.features`, `Merkmalszahl passt nicht zu featureSet ${run.featureSet}`);
  }
}

export function parseMetrics(value: unknown): Metrics {
  const path = "metrics";
  const raw = readObject(value, path, ["schemaVersion", "generatedAt", "dataset", "split", "evaluation", "seed", "runs"]);
  const generatedAt = readString(raw.generatedAt, `${path}.generatedAt`);
  if (!ISO_UTC_PATTERN.test(generatedAt) || Number.isNaN(Date.parse(generatedAt))) {
    fail(`${path}.generatedAt`, "Zeitpunkt im Format ISO 8601 (UTC) erwartet");
  }
  const schemaVersion = readLiteral(raw.schemaVersion, SCHEMA_VERSION, `${path}.schemaVersion`);
  const runs = readArray(raw.runs, `${path}.runs`).map((entry, index) => parseRun(entry, `${path}.runs[${index}]`));
  if (runs.length === 0) fail(`${path}.runs`, "mindestens ein Lauf erwartet");
  const methods = new Set<string>();
  runs.forEach((run, index) => {
    if (methods.has(run.method)) fail(`${path}.runs[${index}].method`, `Verfahren ${run.method} doppelt`);
    methods.add(run.method);
    const previous = runs[index - 1];
    if (previous && METHODS.indexOf(previous.method) > METHODS.indexOf(run.method)) {
      fail(`${path}.runs[${index}].method`, `Reihenfolge ${METHODS.join(", ")} erwartet`);
    }
  });
  const evaluation = parseEvaluation(raw.evaluation, `${path}.evaluation`);
  const labelled = evaluation.testPositives + evaluation.testNegatives;
  runs.forEach((run, index) => {
    if (run.accuracyFlagged > labelled) {
      fail(`${path}.runs[${index}].accuracyFlagged`, "mehr markierte Knoten als Testknoten mit Label");
    }
    if (run.accuracyTruePositives > evaluation.testPositives) {
      fail(`${path}.runs[${index}].accuracyTruePositives`, "mehr Treffer als Testknoten mit Label illegal");
    }
  });
  const dataset = parseDataset(raw.dataset, `${path}.dataset`);
  const split = parseSplit(raw.split, `${path}.split`);
  runs.forEach((run, index) => checkTrainingAgainstData(run, `${path}.runs[${index}]`, split, dataset));
  return {
    schemaVersion,
    generatedAt,
    dataset,
    split,
    evaluation,
    seed: readInteger(raw.seed, `${path}.seed`),
    runs,
  };
}

function parseSelection(value: unknown, path: string, scoreGnnMethod: GraphMethod | null): NodeSelection {
  const raw = readObject(value, path, SELECTION_KEYS);
  if (scoreGnnMethod === null && raw.scoreFieldGnn !== null) {
    fail(`${path}.scoreFieldGnn`, "null erwartet, weil scoreGnnMethod null ist");
  }
  return {
    scoreField: readLiteral(raw.scoreField, "scoreIforest", `${path}.scoreField`),
    scoreFieldGnn: scoreGnnMethod === null ? null : readLiteral(raw.scoreFieldGnn, "scoreGnn", `${path}.scoreFieldGnn`),
    pool: readLiteral(raw.pool, "test", `${path}.pool`),
    seeds: readInteger(raw.seeds, `${path}.seeds`, 1),
    hops: readInteger(raw.hops, `${path}.hops`, 0, MAX_HOP),
    maxNodes: readInteger(raw.maxNodes, `${path}.maxNodes`, 1),
    truncated: readBoolean(raw.truncated, `${path}.truncated`),
  };
}

function readStart(
  raw: RawObject,
  path: string,
  selection: NodeSelection,
  rankKey: "seedRank" | "seedRankGnn",
  hopKey: "hop" | "hopGnn",
): { rank: number | null; hop: Hop | null } {
  const hop = raw[hopKey] === null ? null : readEnum(raw[hopKey], HOPS, `${path}.${hopKey}`);
  if (hop !== null && hop > selection.hops) fail(`${path}.${hopKey}`, `Abstand größer als ${selection.hops}`);
  const rank = raw[rankKey] === null ? null : readInteger(raw[rankKey], `${path}.${rankKey}`, 1, selection.seeds);
  if ((rank === null) !== (hop !== 0)) fail(`${path}.${rankKey}`, "Startknoten und Abstand 0 passen nicht zusammen");
  return { rank, hop };
}

function parseNode(
  value: unknown,
  path: string,
  selection: NodeSelection,
  scoreGnnMethod: GraphMethod | null,
): NetNode {
  const raw = readObject(value, path, NODE_KEYS);
  if (scoreGnnMethod === null) {
    for (const key of ["scoreGnn", "seedRankGnn", "hopGnn"] as const) {
      if (raw[key] !== null) fail(`${path}.${key}`, "null erwartet, weil scoreGnnMethod null ist");
    }
  } else if (raw.scoreGnn === null) {
    fail(`${path}.scoreGnn`, `Zahl erwartet, weil scoreGnnMethod ${scoreGnnMethod} ist`);
  }
  const iforest = readStart(raw, path, selection, "seedRank", "hop");
  const gnn = readStart(raw, path, selection, "seedRankGnn", "hopGnn");
  if (iforest.hop === null && gnn.hop === null) fail(path, "weder hop noch hopGnn gesetzt");
  return {
    id: readString(raw.id, `${path}.id`),
    x: readNumber(raw.x, `${path}.x`, -1, 1),
    y: readNumber(raw.y, `${path}.y`, -1, 1),
    label: readEnum(raw.label, NODE_LABELS, `${path}.label`),
    timeStep: readInteger(raw.timeStep, `${path}.timeStep`, 1),
    scoreZscore: readNumber(raw.scoreZscore, `${path}.scoreZscore`, 0),
    scoreIforest: readNumber(raw.scoreIforest, `${path}.scoreIforest`),
    scoreGnn: raw.scoreGnn === null ? null : readNumber(raw.scoreGnn, `${path}.scoreGnn`),
    seedRank: iforest.rank,
    seedRankGnn: gnn.rank,
    hop: iforest.hop,
    hopGnn: gnn.hop,
    inDegree: readInteger(raw.inDegree, `${path}.inDegree`),
    outDegree: readInteger(raw.outDegree, `${path}.outDegree`),
  };
}

function checkRanks(nodes: readonly NetNode[], path: string, key: "seedRank" | "seedRankGnn", selection: NodeSelection): void {
  const ranks = new Set<number>();
  nodes.forEach((node, index) => {
    const rank = node[key];
    if (rank === null) return;
    if (ranks.has(rank)) fail(`${path}.nodes[${index}].${key}`, `Rang ${rank} doppelt`);
    ranks.add(rank);
  });
  if (ranks.size === 0) fail(`${path}.nodes`, `kein Startknoten mit ${key}`);
  if (Math.max(...ranks) !== ranks.size) {
    fail(`${path}.nodes`, `Startknoten-Ränge ${key} müssen lückenlos 1 bis ${ranks.size} sein`);
  }
  if (!selection.truncated && ranks.size !== selection.seeds) {
    fail(`${path}.nodes`, `${selection.seeds} Startknoten mit ${key} erwartet, gefunden ${ranks.size}`);
  }
}

export function parseNodes(value: unknown): NodesFile {
  const path = "nodes";
  const raw = readObject(value, path, ["schemaVersion", "selection", "scoreGnnMethod", "nodes"]);
  const schemaVersion = readLiteral(raw.schemaVersion, SCHEMA_VERSION, `${path}.schemaVersion`);
  const scoreGnnMethod =
    raw.scoreGnnMethod === null ? null : readEnum(raw.scoreGnnMethod, GRAPH_METHODS, `${path}.scoreGnnMethod`);
  const selection = parseSelection(raw.selection, `${path}.selection`, scoreGnnMethod);
  const entries = readArray(raw.nodes, `${path}.nodes`);
  if (entries.length === 0 || entries.length > selection.maxNodes) {
    fail(`${path}.nodes`, `1 bis ${selection.maxNodes} Knoten erwartet, gefunden ${entries.length}`);
  }
  const nodes = entries.map((entry, index) => parseNode(entry, `${path}.nodes[${index}]`, selection, scoreGnnMethod));
  const ids = new Set<string>();
  nodes.forEach((node, index) => {
    if (ids.has(node.id)) fail(`${path}.nodes[${index}].id`, `ID ${node.id} doppelt`);
    ids.add(node.id);
  });
  checkRanks(nodes, path, "seedRank", selection);
  if (scoreGnnMethod !== null) checkRanks(nodes, path, "seedRankGnn", selection);
  return { schemaVersion, selection, scoreGnnMethod, nodes };
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
  const expectedMethod = expectedScoreGnnMethod(metrics.runs);
  if (nodes.scoreGnnMethod !== expectedMethod) {
    fail(
      "nodes.scoreGnnMethod",
      `${JSON.stringify(expectedMethod)} erwartet: das Graph-Verfahren mit der höheren Validierungs-PR-AUC, bei Gleichstand gcn`,
    );
  }
  return { metrics, nodes, edges };
}

export const netzRadarData: NetzRadarData = parseNetzRadarData({
  metrics: metricsJson,
  nodes: nodesJson,
  edges: edgesJson,
});
