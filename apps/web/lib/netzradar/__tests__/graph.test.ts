import { describe, expect, it } from "vitest";
import {
  DIMMED_NODE_COLOR,
  EDGE_ALPHA,
  EDGE_STYLES,
  NODE_STYLES,
  buildAdjacency,
  countLabels,
  edgeAttributes,
  edgeDisplay,
  edgeEmphasis,
  focusOf,
  neighborsOf,
  nodeAttributes,
  nodeDisplay,
  nodeEmphasis,
  premultiplied,
  seedNodes,
  summarizeComponents,
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
    hop: 2,
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
    expect(display.forceLabel).toBe(true);
    expect(display.highlighted).toBe(true);
    expect(display.zIndex).toBe(4);
  });

  it("keeps neighbours in their colours above the rest", () => {
    const display = nodeDisplay(nodeAttributes(NODE_B), "neighbor", false);
    expect(display.color).toBe("#2f6b3a");
    expect(display.zIndex).toBe(3);
    expect(display.highlighted).toBe(false);
  });

  it("greys out dimmed nodes unless they are selected", () => {
    const dimmed = nodeDisplay(nodeAttributes(NODE_A), "dimmed", false);
    expect(dimmed.color).toBe("#dfe2e6");
    expect(dimmed.borderColor).toBe("#dfe2e6");
    expect(dimmed.zIndex).toBe(0);
    const selected = nodeDisplay(nodeAttributes(NODE_A), "dimmed", true);
    expect(selected.color).toBe("#7a1f2b");
    expect(selected.highlighted).toBe(true);
  });

  it("draws active edges in navy and thicker, normal edges in slate with alpha", () => {
    expect(edgeDisplay("active")).toEqual({ color: "#1d3a5f", size: 2.5, zIndex: 2 });
    expect(edgeDisplay("dimmed")).toEqual({ color: "#e9ebee", size: 1, zIndex: 0 });
    expect(edgeDisplay("normal")).toEqual({ color: "rgba(46, 50, 58, 0.5)", size: 1, zIndex: 1 });
    expect(edgeAttributes()).toEqual({ color: "rgba(46, 50, 58, 0.5)", size: 1, zIndex: 0 });
    expect(EDGE_ALPHA).toBe(0.5);
  });

  it("premultiplies a colour by its alpha for the WebGL blending of the graph", () => {
    expect(premultiplied("#5b6474", 0.5)).toBe("rgba(46, 50, 58, 0.5)");
    expect(premultiplied("#ffffff", 1)).toBe("rgba(255, 255, 255, 1)");
    expect(premultiplied("#0f1b2d", 0)).toBe("rgba(0, 0, 0, 0)");
  });

  it("uses only palette colours for labels and active edges", () => {
    const colours = [
      ...Object.values(NODE_STYLES).flatMap((style) => [style.fill, style.ring]),
      EDGE_STYLES.active.color,
    ];
    for (const colour of colours) expect(PALETTE.has(colour)).toBe(true);
    expect(EDGE_STYLES.normal.color).toBe(premultiplied("#5b6474", EDGE_ALPHA));
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
