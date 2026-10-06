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

describe("parseNetzRadarData with the real export", () => {
  it("accepts the files in data/k3", () => {
    const raw = fresh();
    const parsed = parseNetzRadarData(raw);
    expect(parsed.metrics.schemaVersion).toBe(2);
    expect(parsed.nodes.nodes).toHaveLength(arrayAt(raw.nodes, ["nodes"]).length);
    expect(parsed.edges.edges).toHaveLength(arrayAt(raw.edges, ["edges"]).length);
    expect(parsed.metrics.runs.map((run) => run.method)).toEqual(expect.arrayContaining(["zscore", "iforest"]));
    expect(parsed.metrics.split.crossSplitEdges).toBe(0);
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

  it("rejects another schema version", () => {
    const input = fresh();
    objectAt(input.metrics, []).schemaVersion = 1;
    expectSchemaError(input, /metrics\.schemaVersion: 2 erwartet/);
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
