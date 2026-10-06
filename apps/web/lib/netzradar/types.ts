export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };

export type JsonObject = { [key: string]: JsonValue };

export const NODE_LABELS = ["illicit", "licit", "unknown"] as const;
export type NodeLabel = (typeof NODE_LABELS)[number];

export const METHODS = ["zscore", "iforest", "gcn", "graphsage"] as const;
export type Method = (typeof METHODS)[number];

export const FEATURE_SETS = ["local", "local+graph"] as const;
export type FeatureSet = (typeof FEATURE_SETS)[number];

export const HOPS = [0, 1, 2] as const;
export type Hop = (typeof HOPS)[number];

export const SCHEMA_VERSION = 2;
export type SchemaVersion = typeof SCHEMA_VERSION;

export interface StepRange {
  from: number;
  to: number;
}

export interface LabelCounts {
  illicit: number;
  licit: number;
  unknown: number;
}

export interface Homophily {
  labelledEdges: number;
  sameLabelShare: number;
  illicitIllicitEdges: number;
  licitLicitEdges: number;
  illicitLicitEdges: number;
  illicitWithIllicitNeighbour: number;
  licitWithIllicitNeighbour: number;
}

export interface DatasetInfo {
  name: string;
  displayName: string;
  license: string;
  source: string;
  generator: JsonObject | null;
  nodes: number;
  edges: number;
  timeSteps: number;
  features: number;
  labelCounts: LabelCounts;
  homophily: Homophily;
}

export interface SplitInfo {
  kind: "temporal";
  train: StepRange;
  validation: StepRange;
  test: StepRange;
  crossSplitEdges: number;
}

export interface EvaluationInfo {
  positiveLabel: "illicit";
  excludedLabel: "unknown";
  testPositives: number;
  testNegatives: number;
  prevalence: number;
  allLicitAccuracy: number;
  randomPrAucExpected: number;
  randomPrAucQ95: number;
  randomPermutations: number;
}

export interface PrCurvePoint {
  recall: number;
  precision: number;
}

export interface Run {
  method: Method;
  displayName: string;
  featureSet: FeatureSet;
  seed: number;
  date: string;
  hyperparameters: JsonObject;
  prAuc: number;
  precisionAtRecall50: number;
  recallAtPrecision50: number;
  accuracy: number;
  accuracyThreshold: string;
  accuracyFlagged: number;
  accuracyTruePositives: number;
  prCurve: PrCurvePoint[];
}

export interface Metrics {
  schemaVersion: SchemaVersion;
  generatedAt: string;
  dataset: DatasetInfo;
  split: SplitInfo;
  evaluation: EvaluationInfo;
  seed: number;
  runs: Run[];
}

export interface NodeSelection {
  scoreField: "scoreIforest";
  pool: "test";
  seeds: number;
  hops: number;
  maxNodes: number;
  truncated: boolean;
}

export interface NetNode {
  id: string;
  x: number;
  y: number;
  label: NodeLabel;
  timeStep: number;
  scoreZscore: number;
  scoreIforest: number;
  scoreGnn: number | null;
  seedRank: number | null;
  hop: Hop;
  inDegree: number;
  outDegree: number;
}

export interface NodesFile {
  schemaVersion: SchemaVersion;
  selection: NodeSelection;
  nodes: NetNode[];
}

export interface NetEdge {
  source: string;
  target: string;
}

export interface EdgesFile {
  schemaVersion: SchemaVersion;
  edges: NetEdge[];
}

export interface NetzRadarData {
  metrics: Metrics;
  nodes: NodesFile;
  edges: EdgesFile;
}
