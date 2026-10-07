import type { EvaluationInfo, Method, Run, SearchCandidate, Training } from "../types";

export const EVALUATION: EvaluationInfo = {
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

export function candidate(overrides: Partial<SearchCandidate> = {}): SearchCandidate {
  return {
    featureSet: "local+graph",
    positiveWeightRule: "fixed",
    positiveWeight: 20,
    validationPrAuc: 0.5,
    bestEpoch: 10,
    stoppedEpoch: 60,
    selected: true,
    ...overrides,
  };
}

export function training(overrides: Partial<Training> = {}): Training {
  return {
    architecture: "GCNConv",
    layers: 2,
    hidden: 64,
    activation: "relu",
    dropout: 0.5,
    optimizer: "adam",
    learningRate: 0.01,
    weightDecay: 0.0005,
    loss: "weightedCrossEntropy",
    score: "logitIllicit - logitLicit",
    positiveWeight: 20,
    positiveWeightRule: "fixed",
    edges: "undirected",
    dtype: "float64",
    scaling: {
      center: "median",
      scale: "mad",
      madScale: 1.4826,
      fallback: ["std", "one"],
      clip: 10,
      fitOn: "train",
      zeroMadFeatures: [],
      unitScaleFeatures: [],
    },
    features: ["f_a", "g_b"],
    maxEpochs: 300,
    patience: 50,
    selectedEpoch: 10,
    selectionMetric: "validationPrAuc",
    validationPrAuc: 0.5,
    selectionSteps: { from: 1, to: 17 },
    validationSteps: { from: 18, to: 21 },
    finalFitSteps: { from: 1, to: 21 },
    search: [candidate()],
    environment: {
      torch: "2.8.0+cpu",
      torchGeometric: "2.7.0",
      python: "3.12.15",
      platform: "Windows",
      threads: 1,
      deterministicAlgorithms: true,
    },
    ...overrides,
  };
}

export function run(method: Method, prAuc: number, overrides: Partial<Run> = {}): Run {
  return {
    method,
    displayName: method,
    featureSet: "local",
    seed: 42,
    date: "2026-10-06",
    hyperparameters: {},
    training: null,
    prAuc,
    precisionAtRecall50: 0,
    recallAtPrecision50: 0,
    accuracy: 0,
    accuracyThreshold: "oberste 2 %",
    accuracyFlagged: 1,
    accuracyTruePositives: 0,
    prCurve: [{ recall: 0, precision: 1 }],
    ...overrides,
  };
}

export function learnedRun(method: Method, prAuc: number, validationPrAuc: number, overrides: Partial<Training> = {}): Run {
  return run(method, prAuc, {
    featureSet: "local+graph",
    training: training({ validationPrAuc, search: [candidate({ validationPrAuc })], ...overrides }),
  });
}
