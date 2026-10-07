export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };

export type JsonObject = { [key: string]: JsonValue };

export const NODE_LABELS = ["illicit", "licit", "unknown"] as const;
export type NodeLabel = (typeof NODE_LABELS)[number];

export const METHODS = ["zscore", "iforest", "gcn", "graphsage", "mlp"] as const;
export type Method = (typeof METHODS)[number];

export const BASELINE_METHODS = ["zscore", "iforest"] as const;
export type BaselineMethod = (typeof BASELINE_METHODS)[number];

export const LEARNED_METHODS = ["gcn", "graphsage", "mlp"] as const;
export type LearnedMethod = (typeof LEARNED_METHODS)[number];

export const GRAPH_METHODS = ["gcn", "graphsage"] as const;
export type GraphMethod = (typeof GRAPH_METHODS)[number];

export const FEATURE_SETS = ["local", "local+graph"] as const;
export type FeatureSet = (typeof FEATURE_SETS)[number];

export const POSITIVE_WEIGHT_RULES = ["trainRatio", "fixed"] as const;
export type PositiveWeightRule = (typeof POSITIVE_WEIGHT_RULES)[number];

export const EDGE_MODES = ["undirected", "none"] as const;
export type EdgeMode = (typeof EDGE_MODES)[number];

export const HOPS = [0, 1, 2] as const;
export type Hop = (typeof HOPS)[number];

export const SCHEMA_VERSION = 3;
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

export interface Scaling {
  center: string;
  scale: string;
  madScale: number;
  fallback: string[];
  clip: number;
  fitOn: string;
  zeroMadFeatures: string[];
  unitScaleFeatures: string[];
}

export interface SearchCandidate {
  featureSet: FeatureSet;
  positiveWeightRule: PositiveWeightRule;
  positiveWeight: number;
  validationPrAuc: number;
  bestEpoch: number;
  stoppedEpoch: number;
  selected: boolean;
}

export interface TrainingEnvironment {
  torch: string;
  torchGeometric: string;
  python: string;
  platform: string;
  threads: number;
  deterministicAlgorithms: boolean;
}

export interface Training {
  architecture: string;
  layers: number;
  hidden: number;
  activation: string;
  dropout: number;
  optimizer: string;
  learningRate: number;
  weightDecay: number;
  loss: string;
  score: string;
  positiveWeight: number;
  positiveWeightRule: PositiveWeightRule;
  edges: EdgeMode;
  dtype: string;
  scaling: Scaling;
  features: string[];
  maxEpochs: number;
  patience: number;
  selectedEpoch: number;
  selectionMetric: "validationPrAuc";
  validationPrAuc: number;
  selectionSteps: StepRange;
  validationSteps: StepRange;
  finalFitSteps: StepRange;
  search: SearchCandidate[];
  environment: TrainingEnvironment;
}

export interface Run {
  method: Method;
  displayName: string;
  featureSet: FeatureSet;
  seed: number;
  date: string;
  hyperparameters: JsonObject;
  training: Training | null;
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
  scoreGnnMethod: GraphMethod | null;
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
