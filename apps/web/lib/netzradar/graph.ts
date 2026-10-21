import { CHART_COLORS } from "@portfolio/charts/theme";
import type { Hop, LabelCounts, NetEdge, NetNode, NodeLabel, SeedSet } from "./types";

export interface NodeStyle {
  fill: string;
  ring: string;
  ringSize: number;
}

export const NODE_STYLES: Record<NodeLabel, NodeStyle> = {
  illicit: { fill: CHART_COLORS.wine, ring: CHART_COLORS.gold, ringSize: 0.4 },
  licit: { fill: CHART_COLORS.moss, ring: CHART_COLORS.moss, ringSize: 0 },
  unknown: { fill: CHART_COLORS.surface, ring: CHART_COLORS.slate, ringSize: 0.3 },
};

export const NODE_SIZE_BY_HOP: Record<Hop, number> = { 0: 9, 1: 4.5, 2: 3.5 };
export const OUTSIDE_NODE_SIZE = 2.5;
export const FOCUS_SIZE_FACTOR = 1.3;

export function withAlpha(hex: string, alpha: number): string {
  const value = Number.parseInt(hex.slice(1), 16);
  const channel = (shift: number) => (value >> shift) & 0xff;
  return `rgba(${channel(16)}, ${channel(8)}, ${channel(0)}, ${alpha})`;
}

export const EDGE_ALPHA = 0.5;

export const EDGE_SIZE_PER_THICKNESS = 2;

export function edgeSize(thickness: number): number {
  return thickness * EDGE_SIZE_PER_THICKNESS;
}

export interface EdgeStyle {
  color: string;
  size: number;
}

export const EDGE_STYLES: Record<EdgeEmphasis, EdgeStyle> = {
  normal: { color: withAlpha(CHART_COLORS.slate, EDGE_ALPHA), size: edgeSize(1) },
  active: { color: CHART_COLORS.navy, size: edgeSize(2.5) },
  dimmed: { color: "#e9ebee", size: edgeSize(1) },
};

export const MIN_EDGE_SIZE = edgeSize(1.7);

export const ARROW_HEAD = {
  lengthRatio: 5 / EDGE_SIZE_PER_THICKNESS,
  widthRatio: 4 / EDGE_SIZE_PER_THICKNESS,
} as const;

export const DIMMED_NODE_COLOR = "#dfe2e6";

export interface GraphNodeAttributes {
  x: number;
  y: number;
  size: number;
  color: string;
  borderColor: string;
  borderSize: number;
  label: string;
  zIndex: number;
}

export interface GraphEdgeAttributes extends EdgeStyle {
  zIndex: number;
}

export type NodeEmphasis = "focus" | "neighbor" | "dimmed" | "normal";
export type EdgeEmphasis = "active" | "dimmed" | "normal";
export type NodeDepth = "nodes" | "topNodes";
export type Visibility = "visible" | "hidden";

export interface NodeDisplay extends GraphNodeAttributes {
  depth: NodeDepth;
  labelDepth: NodeDepth;
  labelVisibility: Visibility;
  backdropVisibility: Visibility;
}

export type Adjacency = ReadonlyMap<string, ReadonlySet<string>>;

export function seedRankOf(node: NetNode, set: SeedSet): number | null {
  return set === "gnn" ? node.seedRankGnn : node.seedRank;
}

export function hopOf(node: NetNode, set: SeedSet): Hop | null {
  return set === "gnn" ? node.hopGnn : node.hop;
}

export function seedAttributes(node: NetNode, set: SeedSet): Pick<GraphNodeAttributes, "size" | "zIndex"> {
  const hop = hopOf(node, set);
  return {
    size: hop === null ? OUTSIDE_NODE_SIZE : NODE_SIZE_BY_HOP[hop],
    zIndex: seedRankOf(node, set) === null ? 1 : 2,
  };
}

export function nodeAttributes(node: NetNode, set: SeedSet = "iforest"): GraphNodeAttributes {
  const style = NODE_STYLES[node.label];
  return {
    x: node.x,
    y: node.y,
    color: style.fill,
    borderColor: style.ring,
    borderSize: style.ringSize,
    label: node.id,
    ...seedAttributes(node, set),
  };
}

export function edgeAttributes(): GraphEdgeAttributes {
  return { ...EDGE_STYLES.normal, zIndex: 0 };
}

export function buildAdjacency(nodeIds: Iterable<string>, edges: readonly NetEdge[]): Map<string, Set<string>> {
  const adjacency = new Map<string, Set<string>>();
  for (const id of nodeIds) adjacency.set(id, new Set());
  for (const { source, target } of edges) {
    if (source === target) continue;
    adjacency.get(source)?.add(target);
    adjacency.get(target)?.add(source);
  }
  return adjacency;
}

export function neighborsOf(adjacency: Adjacency, id: string): string[] {
  return [...(adjacency.get(id) ?? [])].sort();
}

export function focusOf(hoveredId: string | null, selectedId: string | null): string | null {
  return hoveredId ?? selectedId;
}

export function nodeEmphasis(id: string, focusId: string | null, adjacency: Adjacency): NodeEmphasis {
  if (focusId === null || !adjacency.has(focusId)) return "normal";
  if (id === focusId) return "focus";
  return adjacency.get(focusId)?.has(id) ? "neighbor" : "dimmed";
}

export function edgeEmphasis(source: string, target: string, focusId: string | null, adjacency: Adjacency): EdgeEmphasis {
  if (focusId === null || !adjacency.has(focusId)) return "normal";
  return source === focusId || target === focusId ? "active" : "dimmed";
}

export function nodeDisplay(attributes: GraphNodeAttributes, emphasis: NodeEmphasis, selected: boolean): NodeDisplay {
  const display: NodeDisplay = { ...attributes, depth: "nodes", labelDepth: "nodes", labelVisibility: "hidden", backdropVisibility: "hidden" };
  if (emphasis === "dimmed" && !selected) {
    return { ...display, color: DIMMED_NODE_COLOR, borderColor: DIMMED_NODE_COLOR, zIndex: 0 };
  }
  if (emphasis === "neighbor") return { ...display, zIndex: 3 };
  if (emphasis === "focus" || selected) {
    return {
      ...display,
      size: attributes.size * FOCUS_SIZE_FACTOR,
      zIndex: 4,
      depth: "topNodes",
      labelDepth: "topNodes",
      labelVisibility: "visible",
      backdropVisibility: "visible",
    };
  }
  return display;
}

export function orderedZIndex(level: number, index: number, count: number): number {
  return level * count + index;
}

export function edgeDisplay(emphasis: EdgeEmphasis): GraphEdgeAttributes {
  if (emphasis === "active") return { ...EDGE_STYLES.active, zIndex: 2 };
  if (emphasis === "dimmed") return { ...EDGE_STYLES.dimmed, zIndex: 0 };
  return { ...EDGE_STYLES.normal, zIndex: 1 };
}

export function seedNodes(nodes: readonly NetNode[], set: SeedSet = "iforest"): NetNode[] {
  return nodes
    .filter((node) => seedRankOf(node, set) !== null)
    .sort((a, b) => (seedRankOf(a, set) ?? 0) - (seedRankOf(b, set) ?? 0));
}

export function countLabels(nodes: readonly NetNode[]): LabelCounts {
  const counts: LabelCounts = { illicit: 0, licit: 0, unknown: 0 };
  for (const node of nodes) counts[node.label] += 1;
  return counts;
}

export interface ComponentSummary {
  count: number;
  timeSteps: number;
  onePerTimeStep: boolean;
}

export function summarizeComponents(nodes: readonly NetNode[], edges: readonly NetEdge[]): ComponentSummary {
  const parent = new Map<string, string>(nodes.map((node) => [node.id, node.id]));
  const find = (id: string): string => {
    let current = id;
    for (;;) {
      const next = parent.get(current);
      if (next === undefined || next === current) return current;
      current = next;
    }
  };
  for (const { source, target } of edges) {
    const a = find(source);
    const b = find(target);
    if (a !== b) parent.set(a, b);
  }
  const stepsByRoot = new Map<string, Set<number>>();
  for (const node of nodes) {
    const root = find(node.id);
    const steps = stepsByRoot.get(root) ?? new Set<number>();
    steps.add(node.timeStep);
    stepsByRoot.set(root, steps);
  }
  const timeSteps = new Set(nodes.map((node) => node.timeStep)).size;
  const singleStep = [...stepsByRoot.values()].every((steps) => steps.size === 1);
  return { count: stepsByRoot.size, timeSteps, onePerTimeStep: singleStep && stepsByRoot.size === timeSteps };
}
