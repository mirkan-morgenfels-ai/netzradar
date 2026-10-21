import { describe, expect, it } from "vitest";
import {
  ARROW_HEAD,
  DIMMED_NODE_COLOR,
  EDGE_ALPHA,
  EDGE_SIZE_PER_THICKNESS,
  EDGE_STYLES,
  MIN_EDGE_SIZE,
  NODE_STYLES,
  buildAdjacency,
  countLabels,
  edgeAttributes,
  edgeDisplay,
  edgeEmphasis,
  edgeSize,
  focusOf,
  hopOf,
  neighborsOf,
  nodeAttributes,
  nodeDisplay,
  nodeEmphasis,
  OUTSIDE_NODE_SIZE,
  orderedZIndex,
  seedAttributes,
  seedNodes,
  seedRankOf,
  summarizeComponents,
  withAlpha,
} from "../graph";
import type { NetEdge, NetNode } from "../types";

const PALETTE = new Set([
  "#1d3a5f",
  "#b8912f",
  "#2f6b3a",
  "#7a1f2b",
  "#3e6a9e",
  "#5b6474",
  "#c9b98f",
  "#0f1b2d",
  "#e4ddcc",
  "#ece6d8",
  "#ffffff",
]);

function channels(hex: string): [number, number, number] {
  const value = Number.parseInt(hex.slice(1), 16);
  return [(value >> 16) & 0xff, (value >> 8) & 0xff, value & 0xff];
}

function node(id: string, overrides: Partial<NetNode> = {}): NetNode {
  return {
    id,
    x: 0,
    y: 0,
    label: "unknown",
    timeStep: 22,
    scoreZscore: 1,
    scoreIforest: 0.5,
    scoreGnn: null,
    seedRank: null,
    seedRankGnn: null,
    hop: 2,
    hopGnn: null,
    inDegree: 0,
    outDegree: 0,
    ...overrides,
  };
}

const NODE_A = node("a", { label: "illicit", hop: 0, seedRank: 2, x: -0.5, y: 0.25 });
const NODE_B = node("b", { label: "licit", hop: 1 });
const NODE_D = node("d", { label: "unknown", hop: 2 });
const NODES: NetNode[] = [
  NODE_A,
  NODE_B,
  node("c", { label: "unknown", hop: 1 }),
  NODE_D,
  node("e", { label: "licit", hop: 0, seedRank: 1 }),
];

const EDGES: NetEdge[] = [
  { source: "a", target: "b" },
  { source: "c", target: "a" },
  { source: "b", target: "d" },
  { source: "d", target: "b" },
  { source: "e", target: "e" },
];

const ADJACENCY = buildAdjacency(
  NODES.map((entry) => entry.id),
  EDGES,
);

describe("buildAdjacency and neighborsOf", () => {
  it("returns the undirected neighbours computed by hand", () => {
    expect(neighborsOf(ADJACENCY, "a")).toEqual(["b", "c"]);
    expect(neighborsOf(ADJACENCY, "b")).toEqual(["a", "d"]);
    expect(neighborsOf(ADJACENCY, "c")).toEqual(["a"]);
    expect(neighborsOf(ADJACENCY, "d")).toEqual(["b"]);
  });

  it("ignores self loops and unknown ids", () => {
    expect(neighborsOf(ADJACENCY, "e")).toEqual([]);
    expect(neighborsOf(ADJACENCY, "zz")).toEqual([]);
  });
});

describe("highlight logic", () => {
  it("prefers the hovered node over the selected one", () => {
    expect(focusOf("b", "a")).toBe("b");
    expect(focusOf(null, "a")).toBe("a");
    expect(focusOf(null, null)).toBeNull();
  });

  it("marks the focus, its neighbours and the rest", () => {
    expect(nodeEmphasis("a", "a", ADJACENCY)).toBe("focus");
    expect(nodeEmphasis("b", "a", ADJACENCY)).toBe("neighbor");
    expect(nodeEmphasis("c", "a", ADJACENCY)).toBe("neighbor");
    expect(nodeEmphasis("d", "a", ADJACENCY)).toBe("dimmed");
    expect(nodeEmphasis("e", "a", ADJACENCY)).toBe("dimmed");
  });

  it("keeps everything normal without a known focus", () => {
    expect(nodeEmphasis("a", null, ADJACENCY)).toBe("normal");
    expect(nodeEmphasis("a", "zz", ADJACENCY)).toBe("normal");
    expect(edgeEmphasis("a", "b", null, ADJACENCY)).toBe("normal");
  });

  it("activates exactly the edges at the focus", () => {
    expect(edgeEmphasis("a", "b", "a", ADJACENCY)).toBe("active");
    expect(edgeEmphasis("c", "a", "a", ADJACENCY)).toBe("active");
    expect(edgeEmphasis("b", "d", "a", ADJACENCY)).toBe("dimmed");
  });
});

describe("node and edge attributes", () => {
  it("encodes illicit as wine with a gold ring and start nodes larger", () => {
    expect(nodeAttributes(NODE_A)).toEqual({
      x: -0.5,
      y: 0.25,
      size: 9,
      color: "#7a1f2b",
      borderColor: "#b8912f",
      borderSize: 0.4,
      label: "a",
      zIndex: 2,
    });
  });

  it("encodes licit as filled moss without ring", () => {
    const attributes = nodeAttributes(NODE_B);
    expect(attributes.color).toBe("#2f6b3a");
    expect(attributes.borderSize).toBe(0);
    expect(attributes.size).toBe(4.5);
    expect(attributes.zIndex).toBe(1);
  });

  it("encodes unknown as hollow surface with a slate outline", () => {
    const attributes = nodeAttributes(NODE_D);
    expect(attributes.color).toBe("#ffffff");
    expect(attributes.borderColor).toBe("#5b6474");
    expect(attributes.borderSize).toBe(0.3);
    expect(attributes.size).toBe(3.5);
  });

  it("enlarges and labels the focus", () => {
    const display = nodeDisplay(nodeAttributes(NODE_A), "focus", false);
    expect(display.size).toBeCloseTo(11.7, 10);
    expect(display.x).toBe(-0.5);
    expect(display.y).toBe(0.25);
    expect(display.label).toBe("a");
    expect(display.labelVisibility).toBe("visible");
    expect(display.backdropVisibility).toBe("visible");
    expect(display.depth).toBe("topNodes");
    expect(display.labelDepth).toBe("topNodes");
    expect(display.zIndex).toBe(4);
  });

  it("keeps neighbours in their colours above the rest", () => {
    const display = nodeDisplay(nodeAttributes(NODE_B), "neighbor", false);
    expect(display.color).toBe("#2f6b3a");
    expect(display.zIndex).toBe(3);
    expect(display.depth).toBe("nodes");
    expect(display.labelVisibility).toBe("hidden");
    expect(display.backdropVisibility).toBe("hidden");
  });

  it("shows no label and no backdrop without a focus", () => {
    const display = nodeDisplay(nodeAttributes(NODE_A), "normal", false);
    expect(display).toEqual({
      ...nodeAttributes(NODE_A),
      depth: "nodes",
      labelDepth: "nodes",
      labelVisibility: "hidden",
      backdropVisibility: "hidden",
    });
  });

  it("orders items by level first and keeps the data order within a level", () => {
    expect(orderedZIndex(0, 0, 5)).toBe(0);
    expect(orderedZIndex(0, 4, 5)).toBe(4);
    expect(orderedZIndex(1, 0, 5)).toBe(5);
    expect(orderedZIndex(4, 2, 5)).toBe(22);
    expect(orderedZIndex(3, 4, 5)).toBeLessThan(orderedZIndex(4, 0, 5));
  });

  it("greys out dimmed nodes unless they are selected", () => {
    const dimmed = nodeDisplay(nodeAttributes(NODE_A), "dimmed", false);
    expect(dimmed.color).toBe("#dfe2e6");
    expect(dimmed.borderColor).toBe("#dfe2e6");
    expect(dimmed.zIndex).toBe(0);
    const selected = nodeDisplay(nodeAttributes(NODE_A), "dimmed", true);
    expect(selected.color).toBe("#7a1f2b");
    expect(selected.labelVisibility).toBe("visible");
    expect(selected.backdropVisibility).toBe("visible");
    expect(selected.depth).toBe("topNodes");
  });

  it("draws active edges in navy and thicker, normal edges in slate with alpha", () => {
    expect(edgeDisplay("active")).toEqual({ color: "#1d3a5f", size: 5, zIndex: 2 });
    expect(edgeDisplay("dimmed")).toEqual({ color: "#e9ebee", size: 2, zIndex: 0 });
    expect(edgeDisplay("normal")).toEqual({ color: "rgba(91, 100, 116, 0.5)", size: 2, zIndex: 1 });
    expect(edgeAttributes()).toEqual({ color: "rgba(91, 100, 116, 0.5)", size: 2, zIndex: 0 });
    expect(EDGE_ALPHA).toBe(0.5);
  });

  it("gives the renderer the full edge width, twice the half-width the thicknesses 1, 2.5 and 1.7 describe", () => {
    expect(EDGE_SIZE_PER_THICKNESS).toBe(2);
    expect(edgeSize(1)).toBe(2);
    expect(edgeSize(2.5)).toBe(5);
    expect(MIN_EDGE_SIZE).toBeCloseTo(3.4, 12);
    expect(EDGE_STYLES.normal.size).toBeLessThan(MIN_EDGE_SIZE);
    expect(EDGE_STYLES.active.size).toBeGreaterThan(MIN_EDGE_SIZE);
  });

  it("keeps the arrow head at 5 and 4 times the half-width of the edge", () => {
    expect(ARROW_HEAD).toEqual({ lengthRatio: 2.5, widthRatio: 2 });
    for (const style of Object.values(EDGE_STYLES)) {
      const halfWidth = style.size / EDGE_SIZE_PER_THICKNESS;
      expect(ARROW_HEAD.lengthRatio * style.size).toBeCloseTo(5 * halfWidth, 12);
      expect(ARROW_HEAD.widthRatio * style.size).toBeCloseTo(4 * halfWidth, 12);
    }
    expect(ARROW_HEAD.lengthRatio * EDGE_STYLES.active.size).toBe(12.5);
    expect(ARROW_HEAD.widthRatio * EDGE_STYLES.active.size).toBe(10);
  });

  it("writes a colour with straight alpha, because the renderer blends it itself", () => {
    expect(withAlpha("#5b6474", 0.5)).toBe("rgba(91, 100, 116, 0.5)");
    expect(withAlpha("#ffffff", 1)).toBe("rgba(255, 255, 255, 1)");
    expect(withAlpha("#0f1b2d", 0)).toBe("rgba(15, 27, 45, 0)");
  });

  it("uses only palette colours for labels and active edges", () => {
    const colours = [
      ...Object.values(NODE_STYLES).flatMap((style) => [style.fill, style.ring]),
      EDGE_STYLES.active.color,
    ];
    for (const colour of colours) expect(PALETTE.has(colour)).toBe(true);
    expect(EDGE_STYLES.normal.color).toBe(withAlpha("#5b6474", EDGE_ALPHA));
  });

  it("dims the context in a light, cool grey that stays behind every label colour", () => {
    for (const colour of [DIMMED_NODE_COLOR, EDGE_STYLES.dimmed.color]) {
      const [red, green, blue] = channels(colour);
      expect(Math.min(red, green, blue)).toBeGreaterThanOrEqual(0xd8);
      expect(blue).toBeGreaterThanOrEqual(red);
      expect(PALETTE.has(colour)).toBe(false);
    }
    expect(new Set(Object.values(NODE_STYLES).map((style) => style.fill)).has(DIMMED_NODE_COLOR)).toBe(false);
  });
});

describe("start nodes and label counts", () => {
  it("orders start nodes by rank", () => {
    expect(seedNodes(NODES).map((entry) => entry.id)).toEqual(["e", "a"]);
  });

  it("orders the GNN start nodes by their own rank", () => {
    const nodes = [
      node("p", { hop: 0, seedRank: 1, hopGnn: 1 }),
      node("q", { hop: null, hopGnn: 0, seedRankGnn: 2 }),
      node("r", { hop: 1, hopGnn: 0, seedRankGnn: 1 }),
    ];
    expect(seedNodes(nodes, "gnn").map((entry) => entry.id)).toEqual(["r", "q"]);
    expect(seedNodes(nodes, "iforest").map((entry) => entry.id)).toEqual(["p"]);
    expect(seedNodes(nodes).map((entry) => entry.id)).toEqual(["p"]);
  });

  it("counts labels by hand", () => {
    expect(countLabels(NODES)).toEqual({ illicit: 1, licit: 2, unknown: 2 });
  });
});

describe("summarizeComponents", () => {
  const steps = [
    node("p", { timeStep: 22 }),
    node("q", { timeStep: 22 }),
    node("r", { timeStep: 22 }),
    node("s", { timeStep: 23 }),
    node("t", { timeStep: 23 }),
  ];

  it("counts components and notices when a time step is split", () => {
    const summary = summarizeComponents(steps, [
      { source: "p", target: "q" },
      { source: "r", target: "q" },
    ]);
    expect(summary).toEqual({ count: 3, timeSteps: 2, onePerTimeStep: false });
  });

  it("recognises one component per time step", () => {
    const summary = summarizeComponents(steps, [
      { source: "p", target: "q" },
      { source: "r", target: "q" },
      { source: "t", target: "s" },
    ]);
    expect(summary).toEqual({ count: 2, timeSteps: 2, onePerTimeStep: true });
  });

  it("rejects the claim when an edge joins two time steps", () => {
    const summary = summarizeComponents(steps, [
      { source: "p", target: "q" },
      { source: "r", target: "q" },
      { source: "q", target: "s" },
      { source: "t", target: "s" },
    ]);
    expect(summary).toEqual({ count: 1, timeSteps: 2, onePerTimeStep: false });
  });
});

describe("start sets in the graph", () => {
  const both = node("both", { label: "illicit", hop: 0, seedRank: 3, hopGnn: 0, seedRankGnn: 1 });
  const gnnOnly = node("gnn-only", { label: "illicit", hop: null, hopGnn: 1 });
  const iforestOnly = node("iforest-only", { hop: 2, hopGnn: null });

  it("reads rank and distance of the chosen start set", () => {
    expect(seedRankOf(both, "iforest")).toBe(3);
    expect(seedRankOf(both, "gnn")).toBe(1);
    expect(hopOf(gnnOnly, "iforest")).toBeNull();
    expect(hopOf(gnnOnly, "gnn")).toBe(1);
  });

  it("draws nodes outside the chosen start set smallest and start nodes on top", () => {
    expect(OUTSIDE_NODE_SIZE).toBeLessThan(3.5);
    expect(seedAttributes(gnnOnly, "iforest")).toEqual({ size: OUTSIDE_NODE_SIZE, zIndex: 1 });
    expect(seedAttributes(gnnOnly, "gnn")).toEqual({ size: 4.5, zIndex: 1 });
    expect(seedAttributes(iforestOnly, "gnn")).toEqual({ size: OUTSIDE_NODE_SIZE, zIndex: 1 });
    expect(seedAttributes(iforestOnly, "iforest")).toEqual({ size: 3.5, zIndex: 1 });
    expect(seedAttributes(both, "gnn")).toEqual({ size: 9, zIndex: 2 });
  });

  it("keeps colour and ring of the label in both start sets", () => {
    const iforest = nodeAttributes(gnnOnly, "iforest");
    const gnn = nodeAttributes(gnnOnly, "gnn");
    expect(iforest.color).toBe(gnn.color);
    expect(iforest.borderColor).toBe("#b8912f");
    expect(iforest.size).toBe(OUTSIDE_NODE_SIZE);
    expect(gnn.size).toBe(4.5);
    expect(nodeAttributes(both)).toEqual(nodeAttributes(both, "iforest"));
  });
});
