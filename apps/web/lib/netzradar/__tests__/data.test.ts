import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { NetzRadarSchemaError, netzRadarData, parseMetrics, parseNetzRadarData } from "../data";
import type { JsonObject, JsonValue } from "../types";

interface RawExport {
  metrics: JsonValue;
  nodes: JsonValue;
  edges: JsonValue;
}

type Key = string | number;

function readExport(name: string): JsonValue {
  const file = fileURLToPath(new URL(`../../../../../data/k3/${name}`, import.meta.url));
  return JSON.parse(readFileSync(file, "utf8")) as JsonValue;
}

const REAL: RawExport = {
  metrics: readExport("metrics.json"),
  nodes: readExport("nodes.json"),
  edges: readExport("edges.json"),
};

function fresh(): RawExport {
  return structuredClone(REAL);
}

function walk(root: JsonValue, path: readonly Key[]): JsonValue | undefined {
  let current: JsonValue | undefined = root;
  for (const key of path) {
    if (Array.isArray(current) && typeof key === "number") current = current[key];
    else if (current !== null && typeof current === "object" && !Array.isArray(current)) current = current[String(key)];
    else return undefined;
  }
  return current;
}

function objectAt(root: JsonValue, path: readonly Key[]): JsonObject {
  const value = walk(root, path);
  if (value === null || value === undefined || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`no object at ${path.join(".")}`);
  }
  return value;
}

function arrayAt(root: JsonValue, path: readonly Key[]): JsonValue[] {
  const value = walk(root, path);
  if (!Array.isArray(value)) throw new Error(`no array at ${path.join(".")}`);
  return value;
}

function numberAt(root: JsonValue, path: readonly Key[]): number {
  const value = walk(root, path);
  if (typeof value !== "number") throw new Error(`no number at ${path.join(".")}`);
  return value;
}

function expectSchemaError(input: RawExport, pattern: RegExp): void {
  expect(() => parseNetzRadarData(input)).toThrow(NetzRadarSchemaError);
  expect(() => parseNetzRadarData(input)).toThrow(pattern);
}

function stringAt(root: JsonValue, path: readonly Key[]): string {
  const value = walk(root, path);
  if (typeof value !== "string") throw new Error(`no string at ${path.join(".")}`);
  return value;
}

function runIndex(input: RawExport, method: string): number {
  const index = arrayAt(input.metrics, ["runs"]).findIndex((_, position) => walk(input.metrics, ["runs", position, "method"]) === method);
  if (index < 0) throw new Error(`no run ${method}`);
  return index;
}

function hyperparameters(input: RawExport, method: string): JsonObject {
  return objectAt(input.metrics, ["runs", runIndex(input, method), "hyperparameters"]);
}

function searchOf(input: RawExport, method: string): JsonObject[] {
  return arrayAt(hyperparameters(input, method), ["search"]).map((_, index) =>
    objectAt(hyperparameters(input, method), ["search", index]),
  );
}

const LEARNED = ["gcn", "graphsage", "mlp"];

describe("parseNetzRadarData with the real export", () => {
  it("accepts the files in data/k3", () => {
    const raw = fresh();
    const parsed = parseNetzRadarData(raw);
    expect(parsed.metrics.schemaVersion).toBe(3);
    expect(parsed.nodes.schemaVersion).toBe(3);
    expect(parsed.edges.schemaVersion).toBe(3);
    expect(parsed.nodes.nodes).toHaveLength(arrayAt(raw.nodes, ["nodes"]).length);
    expect(parsed.edges.edges).toHaveLength(arrayAt(raw.edges, ["edges"]).length);
    expect(parsed.metrics.runs.map((run) => run.method)).toEqual(
      arrayAt(raw.metrics, ["runs"]).map((_, index) => stringAt(raw.metrics, ["runs", index, "method"])),
    );
    expect(parsed.metrics.split.crossSplitEdges).toBe(0);
  });

  it("reads the learned runs with their search and the GNN score", () => {
    const raw = fresh();
    const parsed = parseNetzRadarData(raw);
    for (const run of parsed.metrics.runs) {
      if (!LEARNED.includes(run.method)) {
        expect(run.training).toBeNull();
        continue;
      }
      expect(run.training).not.toBeNull();
      expect(run.training?.search.filter((candidate) => candidate.selected)).toHaveLength(1);
      expect(run.training?.finalFitSteps).toEqual(parsed.metrics.split.train);
      expect(run.training?.validationSteps).toEqual(parsed.metrics.split.validation);
    }
    expect(parsed.nodes.scoreGnnMethod).toBe(walk(raw.nodes, ["scoreGnnMethod"]));
    if (parsed.nodes.scoreGnnMethod !== null) {
      expect(parsed.nodes.nodes.every((node) => typeof node.scoreGnn === "number")).toBe(true);
    }
  });

  it("exposes the parsed export as netzRadarData", () => {
    expect(netzRadarData).toEqual(parseNetzRadarData(fresh()));
  });

  it("keeps every start node rank exactly once", () => {
    const ranks = netzRadarData.nodes.nodes
      .map((node) => node.seedRank)
      .filter((rank): rank is number => rank !== null)
      .sort((a, b) => a - b);
    expect(ranks).toEqual(Array.from({ length: ranks.length }, (_, index) => index + 1));
    expect(ranks.length).toBeLessThanOrEqual(netzRadarData.nodes.selection.seeds);
  });
});

describe("parseNetzRadarData rejects broken variants", () => {
  it("rejects a node without timeStep", () => {
    const input = fresh();
    delete objectAt(input.nodes, ["nodes", 0]).timeStep;
    expectSchemaError(input, /nodes\.nodes\[0\]\.timeStep: Feld fehlt/);
  });

  it("rejects a run without prAuc", () => {
    const input = fresh();
    delete objectAt(input.metrics, ["runs", 0]).prAuc;
    expectSchemaError(input, /metrics\.runs\[0\]\.prAuc: Feld fehlt/);
  });

  it("rejects an unknown label", () => {
    const input = fresh();
    objectAt(input.nodes, ["nodes", 1]).label = "suspicious";
    expectSchemaError(input, /nodes\.nodes\[1\]\.label: einer von illicit, licit, unknown erwartet/);
  });

  it("rejects an edge to an unknown node", () => {
    const input = fresh();
    objectAt(input.edges, ["edges", 0]).target = "tx_missing";
    expectSchemaError(input, /edges\.edges\[0\]\.target: unbekannter Knoten tx_missing/);
  });

  it("rejects a self loop", () => {
    const input = fresh();
    const edge = objectAt(input.edges, ["edges", 0]);
    edge.target = edge.source ?? null;
    expectSchemaError(input, /edges\.edges\[0\]: Selbstschleife/);
  });

  it("rejects a duplicate edge", () => {
    const input = fresh();
    arrayAt(input.edges, ["edges"]).splice(1, 0, { ...objectAt(input.edges, ["edges", 0]) });
    expectSchemaError(input, /edges\.edges\[1\]: doppelte Kante/);
  });

  it("rejects unsorted edges", () => {
    const input = fresh();
    arrayAt(input.edges, ["edges"]).reverse();
    expectSchemaError(input, /edges\.edges\[1\]: Kanten nicht nach source und target sortiert/);
  });

  it("rejects an unknown field", () => {
    const input = fresh();
    objectAt(input.nodes, ["nodes", 2]).color = "#7a1f2b";
    expectSchemaError(input, /nodes\.nodes\[2\]\.color: unbekanntes Feld/);
  });

  it("rejects coordinates outside [-1, 1]", () => {
    const input = fresh();
    objectAt(input.nodes, ["nodes", 0]).x = 1.5;
    expectSchemaError(input, /nodes\.nodes\[0\]\.x: Wert 1\.5 liegt nicht in \[-1, 1\]/);
  });

  it("rejects a start rank on a node outside hop 0", () => {
    const input = fresh();
    const index = arrayAt(input.nodes, ["nodes"]).findIndex((_, position) => numberAt(input.nodes, ["nodes", position, "hop"]) === 1);
    objectAt(input.nodes, ["nodes", index]).seedRank = 1;
    expectSchemaError(input, new RegExp(`nodes\\.nodes\\[${index}\\]\\.seedRank`));
  });

  it("rejects a node outside the test period", () => {
    const input = fresh();
    objectAt(input.nodes, ["nodes", 0]).timeStep = numberAt(input.metrics, ["split", "train", "to"]);
    expectSchemaError(input, /nodes\.nodes\[0\]\.timeStep: Zeitschritt \d+ liegt außerhalb des Testzeitraums/);
  });

  it("rejects edges between train and test", () => {
    const input = fresh();
    objectAt(input.metrics, ["split"]).crossSplitEdges = 1;
    expectSchemaError(input, /metrics\.split\.crossSplitEdges/);
  });

  it("rejects an overlapping split", () => {
    const input = fresh();
    objectAt(input.metrics, ["split", "test"]).from = numberAt(input.metrics, ["split", "train", "to"]);
    expectSchemaError(input, /metrics\.split\.test: Testzeitraum beginnt nicht nach dem Trainingszeitraum/);
  });

  it("rejects a prevalence that does not match the test counts", () => {
    const input = fresh();
    objectAt(input.metrics, ["evaluation"]).prevalence = 0.5;
    expectSchemaError(input, /metrics\.evaluation\.prevalence/);
  });

  it("rejects an unknown method", () => {
    const input = fresh();
    objectAt(input.metrics, ["runs", 0]).method = "randomforest";
    expectSchemaError(input, /metrics\.runs\[0\]\.method/);
  });

  it("rejects a PR curve that is not sorted by recall", () => {
    const input = fresh();
    arrayAt(input.metrics, ["runs", 0, "prCurve"]).reverse();
    expectSchemaError(input, /metrics\.runs\[0\]\.prCurve\[1\]: nicht nach recall aufsteigend sortiert/);
  });

  it("rejects a PR curve with more than 101 points", () => {
    const input = fresh();
    objectAt(input.metrics, ["runs", 0]).prCurve = Array.from({ length: 102 }, (_, index) => ({
      recall: index / 101,
      precision: 0.5,
    }));
    expectSchemaError(input, /metrics\.runs\[0\]\.prCurve: 1 bis 101 Punkte erwartet, gefunden 102/);
  });

  it("rejects the previous schema version", () => {
    const input = fresh();
    objectAt(input.metrics, []).schemaVersion = 2;
    expectSchemaError(input, /metrics\.schemaVersion: 3 erwartet/);
    const nodesInput = fresh();
    objectAt(nodesInput.nodes, []).schemaVersion = 2;
    expectSchemaError(nodesInput, /nodes\.schemaVersion: 3 erwartet/);
    const edgesInput = fresh();
    objectAt(edgesInput.edges, []).schemaVersion = 2;
    expectSchemaError(edgesInput, /edges\.schemaVersion: 3 erwartet/);
  });

  it("rejects a gap in the start node ranks", () => {
    const input = fresh();
    const nodes = arrayAt(input.nodes, ["nodes"]);
    const index = nodes.findIndex((_, position) => walk(input.nodes, ["nodes", position, "seedRank"]) === 7);
    nodes.splice(index, 1);
    expectSchemaError(input, /nodes\.nodes: Startknoten-Ränge müssen lückenlos 1 bis 49 sein/);
  });

  it("rejects an expected random PR-AUC that does not match the test counts", () => {
    const input = fresh();
    objectAt(input.metrics, ["evaluation"]).randomPrAucExpected = numberAt(input.metrics, ["evaluation", "prevalence"]);
    expectSchemaError(input, /metrics\.evaluation\.randomPrAucExpected/);
  });

  it("rejects more hits than flagged nodes", () => {
    const input = fresh();
    const run = objectAt(input.metrics, ["runs", 0]);
    run.accuracyTruePositives = numberAt(input.metrics, ["runs", 0, "accuracyFlagged"]) + 1;
    expectSchemaError(input, /metrics\.runs\[0\]\.accuracyTruePositives: mehr Treffer als markierte Knoten/);
  });

  it("rejects homophily counts that do not add up", () => {
    const input = fresh();
    const homophily = objectAt(input.metrics, ["dataset", "homophily"]);
    homophily.illicitLicitEdges = numberAt(input.metrics, ["dataset", "homophily", "illicitLicitEdges"]) + 1;
    expectSchemaError(input, /metrics\.dataset\.homophily: Kantenzahlen ergeben nicht labelledEdges/);
  });

  it("rejects label counts that do not add up to the node count", () => {
    const input = fresh();
    const counts = objectAt(input.metrics, ["dataset", "labelCounts"]);
    counts.unknown = numberAt(input.metrics, ["dataset", "labelCounts", "unknown"]) + 1;
    expectSchemaError(input, /metrics\.dataset\.labelCounts: Summe der Labels ist ungleich der Knotenzahl/);
  });

  it("rejects runs outside the contract order", () => {
    const input = fresh();
    const runs = arrayAt(input.metrics, ["runs"]);
    const first = runIndex(input, "gcn");
    const second = runIndex(input, "graphsage");
    const swapped = runs[first];
    runs[first] = runs[second] ?? null;
    runs[second] = swapped ?? null;
    expectSchemaError(input, new RegExp(`metrics\\.runs\\[${second}\\]\\.method: Reihenfolge`));
  });

  it("rejects a second MLP run", () => {
    const input = fresh();
    const runs = arrayAt(input.metrics, ["runs"]);
    runs.push(structuredClone(runs[runIndex(input, "mlp")] ?? null));
    expectSchemaError(input, /Verfahren mlp doppelt/);
  });
});

describe("parseNetzRadarData rejects broken GNN results", () => {
  it("rejects nodes.json without scoreGnnMethod", () => {
    const input = fresh();
    delete objectAt(input.nodes, []).scoreGnnMethod;
    expectSchemaError(input, /nodes\.scoreGnnMethod: Feld fehlt/);
  });

  it("rejects the MLP as source of scoreGnn", () => {
    const input = fresh();
    objectAt(input.nodes, []).scoreGnnMethod = "mlp";
    expectSchemaError(input, /nodes\.scoreGnnMethod: einer von gcn, graphsage erwartet/);
  });

  it("rejects a missing GNN score while scoreGnnMethod is set", () => {
    const input = fresh();
    objectAt(input.nodes, ["nodes", 0]).scoreGnn = null;
    expectSchemaError(input, /nodes\.nodes\[0\]\.scoreGnn: Zahl erwartet/);
  });

  it("rejects GNN scores while scoreGnnMethod is null", () => {
    const input = fresh();
    objectAt(input.nodes, []).scoreGnnMethod = null;
    expectSchemaError(input, /nodes\.nodes\[0\]\.scoreGnn: null erwartet/);
  });

  it("rejects scoreGnnMethod null while GNN runs exist", () => {
    const input = fresh();
    objectAt(input.nodes, []).scoreGnnMethod = null;
    arrayAt(input.nodes, ["nodes"]).forEach((_, index) => {
      objectAt(input.nodes, ["nodes", index]).scoreGnn = null;
    });
    expectSchemaError(input, /nodes\.scoreGnnMethod: "(gcn|graphsage)" erwartet/);
  });

  it("rejects the graph method with the lower validation PR-AUC", () => {
    const input = fresh();
    const current = stringAt(input.nodes, ["scoreGnnMethod"]);
    objectAt(input.nodes, []).scoreGnnMethod = current === "gcn" ? "graphsage" : "gcn";
    expectSchemaError(input, new RegExp(`nodes\\.scoreGnnMethod: "${current}" erwartet`));
  });

  it("rejects scoreGnnMethod without graph runs", () => {
    const input = fresh();
    const metrics = objectAt(input.metrics, []);
    metrics.runs = arrayAt(input.metrics, ["runs"]).filter(
      (_, index) => !LEARNED.includes(stringAt(input.metrics, ["runs", index, "method"])),
    );
    expectSchemaError(input, /nodes\.scoreGnnMethod: null erwartet/);
  });

  it("rejects a GNN run without search", () => {
    const input = fresh();
    delete hyperparameters(input, "gcn").search;
    expectSchemaError(input, new RegExp(`metrics\\.runs\\[${runIndex(input, "gcn")}\\]\\.hyperparameters\\.search: Feld fehlt`));
  });

  it("rejects the seed spread in metrics.json, where it does not belong", () => {
    const input = fresh();
    hyperparameters(input, "graphsage").seedSpread = { seeds: [42], prAuc: [0.9] };
    expectSchemaError(input, /hyperparameters\.seedSpread: unbekanntes Feld/);
  });

  it("rejects two selected candidates", () => {
    const input = fresh();
    for (const candidate of searchOf(input, "gcn")) candidate.selected = true;
    expectSchemaError(input, /hyperparameters\.search: genau ein gewählter Kandidat erwartet, gefunden \d+/);
  });

  it("rejects a selection that is not the best candidate on the validation part", () => {
    const input = fresh();
    const loser = searchOf(input, "graphsage").find((candidate) => candidate.selected === false);
    if (!loser) throw new Error("no unselected candidate");
    loser.validationPrAuc = 0.9999;
    expectSchemaError(input, /validationPrAuc: höher als beim gewählten Kandidaten/);
  });

  it("rejects a final epoch count other than the best epoch of the selection", () => {
    const input = fresh();
    const params = hyperparameters(input, "mlp");
    params.selectedEpoch = numberAt(params, ["selectedEpoch"]) + 1;
    expectSchemaError(input, /hyperparameters\.selectedEpoch: passt nicht zur besten Epoche/);
  });

  it("rejects a stop epoch that does not follow the patience", () => {
    const input = fresh();
    const candidate = searchOf(input, "gcn").find((entry) => numberAt(entry, ["stoppedEpoch"]) < numberAt(hyperparameters(input, "gcn"), ["maxEpochs"]));
    if (!candidate) throw new Error("no early stopped candidate");
    candidate.stoppedEpoch = numberAt(candidate, ["bestEpoch"]) + 1;
    expectSchemaError(input, /stoppedEpoch: Abbruch in Epoche \d+ erwartet/);
  });

  it("rejects a validation part other than the one of the split", () => {
    const input = fresh();
    const validation = objectAt(input.metrics, ["split", "validation"]);
    validation.from = numberAt(validation, ["from"]) + 1;
    expectSchemaError(input, /hyperparameters\.validationSteps: Auswahl muss auf dem Validierungsteil des Splits liegen/);
  });

  it("rejects a final model on other time steps than the training period", () => {
    const input = fresh();
    const finalFit = objectAt(hyperparameters(input, "gcn"), ["finalFitSteps"]);
    const selection = objectAt(hyperparameters(input, "gcn"), ["selectionSteps"]);
    finalFit.from = numberAt(finalFit, ["from"]) + 1;
    selection.from = numberAt(selection, ["from"]) + 1;
    expectSchemaError(input, /hyperparameters\.finalFitSteps: Endmodell muss auf dem Trainingszeitraum/);
  });

  it("rejects edges in the MLP control", () => {
    const input = fresh();
    hyperparameters(input, "mlp").edges = "undirected";
    expectSchemaError(input, /hyperparameters\.edges: "none" erwartet/);
  });

  it("rejects a wrong architecture", () => {
    const input = fresh();
    hyperparameters(input, "graphsage").architecture = "GCNConv";
    expectSchemaError(input, /hyperparameters\.architecture: "SAGEConv\(aggr=mean\)" erwartet/);
  });

  it("rejects a feature set that differs from the selected candidate", () => {
    const input = fresh();
    const run = objectAt(input.metrics, ["runs", runIndex(input, "gcn")]);
    run.featureSet = run.featureSet === "local" ? "local+graph" : "local";
    expectSchemaError(input, /gewählter Merkmalssatz passt nicht zu featureSet/);
  });

  it("rejects a fixed weight that differs from the selected candidate", () => {
    const input = fresh();
    const method = LEARNED.find((name) => walk(hyperparameters(input, name), ["positiveWeightRule"]) === "fixed");
    if (!method) throw new Error("no run with a fixed weight");
    hyperparameters(input, method).positiveWeight = 19;
    expectSchemaError(input, /hyperparameters\.positiveWeight: Wert 19 passt nicht/);
  });

  it("rejects a scaled feature that is not among the features", () => {
    const input = fresh();
    const scaling = objectAt(hyperparameters(input, "gcn"), ["scaling"]);
    scaling.zeroMadFeatures = ["f_missing"];
    expectSchemaError(input, /scaling\.zeroMadFeatures\[0\]: Merkmal f_missing fehlt in features/);
  });

  it("rejects an incomplete training environment", () => {
    const input = fresh();
    delete objectAt(hyperparameters(input, "gcn"), ["environment"]).threads;
    expectSchemaError(input, /environment\.threads: Feld fehlt/);
  });

  it("rejects a feature count that does not fit the feature set", () => {
    const input = fresh();
    const params = hyperparameters(input, "gcn");
    const features = arrayAt(params, ["features"]);
    const local = numberAt(input.metrics, ["dataset", "features"]);
    params.features = features.slice(0, local);
    objectAt(params, ["scaling"]).zeroMadFeatures = [];
    objectAt(params, ["scaling"]).unitScaleFeatures = [];
    expectSchemaError(input, /hyperparameters\.features: Merkmalszahl passt nicht zu featureSet local\+graph/);
  });
});

describe("parseNetzRadarData rejects settings the page text relies on", () => {
  const fixedTraining: Array<[string, JsonValue, RegExp]> = [
    ["score", "probabilityIllicit", /hyperparameters\.score: "logitIllicit - logitLicit" erwartet/],
    ["loss", "focal", /hyperparameters\.loss: "weightedCrossEntropy" erwartet/],
    ["activation", "tanh", /hyperparameters\.activation: "relu" erwartet/],
    ["optimizer", "sgd", /hyperparameters\.optimizer: "adam" erwartet/],
    ["layers", 3, /hyperparameters\.layers: 2 erwartet/],
    ["hidden", 128, /hyperparameters\.hidden: 64 erwartet/],
    ["dropout", 0.25, /hyperparameters\.dropout: 0\.5 erwartet/],
    ["learningRate", 0.001, /hyperparameters\.learningRate: 0\.01 erwartet/],
    ["weightDecay", 0, /hyperparameters\.weightDecay: 0\.0005 erwartet/],
    ["dtype", "float16", /hyperparameters\.dtype: einer von float64, float32 erwartet/],
  ];

  it.each(fixedTraining)("rejects %s = %j", (key, value, pattern) => {
    const input = fresh();
    hyperparameters(input, "gcn")[key] = value;
    expectSchemaError(input, pattern);
  });

  it("accepts float32 as the second documented dtype", () => {
    const input = fresh();
    hyperparameters(input, "mlp").dtype = "float32";
    expect(parseNetzRadarData(input).metrics.runs.find((run) => run.method === "mlp")?.training?.dtype).toBe("float32");
  });

  const fixedScaling: Array<[string, JsonValue, RegExp]> = [
    ["center", "mean", /scaling\.center: "median" erwartet/],
    ["scale", "std", /scaling\.scale: "mad" erwartet/],
    ["madScale", 1.5, /scaling\.madScale: 1\.4826 erwartet/],
    ["fallback", ["one"], /scaling\.fallback: \["std","one"\] erwartet/],
    ["fallback", ["one", "std"], /scaling\.fallback: \["std","one"\] erwartet/],
  ];

  it.each(fixedScaling)("rejects scaling.%s = %j", (key, value, pattern) => {
    const input = fresh();
    objectAt(hyperparameters(input, "graphsage"), ["scaling"])[key] = value;
    expectSchemaError(input, pattern);
  });

  it("rejects a search candidate listed twice", () => {
    const input = fresh();
    const search = arrayAt(hyperparameters(input, "gcn"), ["search"]);
    const first = objectAt(hyperparameters(input, "gcn"), ["search", 0]);
    if (first.selected === true) throw new Error("first candidate is selected");
    search[1] = structuredClone(first);
    expectSchemaError(input, /hyperparameters\.search\[1\]: Kandidat local trainRatio doppelt/);
  });

  it("rejects a later candidate as selection on a tie", () => {
    const input = fresh();
    const search = searchOf(input, "gcn");
    const selectedIndex = search.findIndex((candidate) => candidate.selected === true);
    if (selectedIndex < 1) throw new Error("selected candidate is the first one");
    const first = search[0];
    if (!first) throw new Error("no first candidate");
    first.validationPrAuc = numberAt(search[selectedIndex] ?? {}, ["validationPrAuc"]);
    expectSchemaError(input, /hyperparameters\.search\[0\]\.validationPrAuc: gleich hoch wie beim gewählten Kandidaten/);
  });

  it("rejects a negative robust z-score", () => {
    const input = fresh();
    objectAt(input.nodes, ["nodes", 0]).scoreZscore = -0.5;
    expectSchemaError(input, /nodes\.nodes\[0\]\.scoreZscore: Wert -0\.5 liegt nicht in \[0, Infinity\]/);
  });
});

describe("parseNetzRadarData reports errors", () => {
  it("reports the failing path on the error", () => {
    const input = fresh();
    objectAt(input.metrics, []).generatedAt = "06.10.2026";
    let caught: unknown = null;
    try {
      parseMetrics(input.metrics);
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(NetzRadarSchemaError);
    expect((caught as NetzRadarSchemaError).path).toBe("metrics.generatedAt");
  });
});
