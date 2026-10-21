"use client";

import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import type Sigma from "sigma";
import type { CameraState } from "sigma/types";
import { CHART_COLORS } from "@portfolio/charts/theme";
import {
  ARROW_HEAD,
  edgeAttributes,
  edgeDisplay,
  edgeEmphasis,
  focusOf,
  MIN_EDGE_SIZE,
  nodeAttributes,
  nodeDisplay,
  nodeEmphasis,
  orderedZIndex,
  seedAttributes,
  type Adjacency,
  type GraphEdgeAttributes,
  type GraphNodeAttributes,
} from "@/lib/netzradar/graph";
import { formatInteger } from "@/lib/netzradar/format";
import { edgeBodyLayer, ringLayer } from "@/lib/netzradar/layers";
import type { NetEdge, NetNode, SeedSet } from "@/lib/netzradar/types";

type Status = "loading" | "ready" | "unsupported" | "error";
type Renderer = Sigma<GraphNodeAttributes, GraphEdgeAttributes>;

const CAMERA_DURATION = 350;
const ZOOM_STEP = 1.6;
const FOCUS_RATIO = 0.35;
const LABEL_FONT = 'ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif';
const LABEL_SIZE = 12;
const LABEL_WEIGHT = "500";
const LABEL_MARGIN = 3;
const BACKDROP_STYLE = {
  backdropColor: "#ffffff",
  backdropShadowColor: "rgba(0, 0, 0, 0.5)",
  backdropShadowBlur: 7.5,
  backdropPadding: 2,
  backdropLabelPadding: 2,
  backdropCornerRadius: 0,
  backdropArea: "both",
} as const;
const DEPTH_LAYERS = ["edges", "nodes", "topNodes"] as const;
const RESET_STATE: Partial<CameraState> = { x: 0.5, y: 0.5, ratio: 1, angle: 0 };
const CONTROL_BASE =
  "inline-flex h-9 items-center justify-center rounded-full text-ink transition-colors duration-150 hover:bg-navy-950 hover:text-ivory";
const CONTROL_ICON = `${CONTROL_BASE} w-9 text-lg leading-none`;
const CONTROL_TEXT = `${CONTROL_BASE} px-4 text-sm font-medium`;

export interface GraphViewProps {
  nodes: readonly NetNode[];
  edges: readonly NetEdge[];
  adjacency: Adjacency;
  seedSet: SeedSet;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  description: string;
}

function hasWebGl2(): boolean {
  try {
    return document.createElement("canvas").getContext("webgl2") !== null;
  } catch {
    return false;
  }
}

function loadLabelFont(family: string): Promise<unknown> {
  try {
    return document.fonts.load(`${LABEL_WEIGHT} ${LABEL_SIZE}px ${family}`).catch(() => undefined);
  } catch {
    return Promise.resolve();
  }
}

function loadRenderer(family: string) {
  return Promise.all([
    import("graphology"),
    import("sigma"),
    import("sigma/rendering"),
    loadLabelFont(family),
  ]);
}

function subscribeNothing(): () => void {
  return () => undefined;
}

function useIsClient(): boolean {
  return useSyncExternalStore(
    subscribeNothing,
    () => true,
    () => false,
  );
}

function prefersReducedMotion(): boolean {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function moveCamera(renderer: Renderer, state: Partial<CameraState>): void {
  const camera = renderer.getCamera();
  if (prefersReducedMotion()) camera.setState(state);
  else void camera.animate(state, { duration: CAMERA_DURATION });
}

export function GraphView({ nodes, edges, adjacency, seedSet, selectedId, onSelect, description }: GraphViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<Renderer | null>(null);
  const hoveredRef = useRef<string | null>(null);
  const selectedRef = useRef<string | null>(selectedId);
  const seedSetRef = useRef<SeedSet>(seedSet);
  const onSelectRef = useRef(onSelect);
  const [status, setStatus] = useState<Status>("loading");
  const isClient = useIsClient();

  useEffect(() => {
    onSelectRef.current = onSelect;
  }, [onSelect]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    let cancelled = false;
    let renderer: Renderer | null = null;
    const labelFont = getComputedStyle(container).fontFamily || LABEL_FONT;

    (hasWebGl2() ? loadRenderer(labelFont) : Promise.resolve(null))
      .then((modules) => {
        if (cancelled) return;
        if (modules === null) {
          setStatus("unsupported");
          return;
        }
        const [graphModule, sigmaModule, renderingModule] = modules;
        const graph = new graphModule.default<GraphNodeAttributes, GraphEdgeAttributes>({
          type: "directed",
          multi: false,
          allowSelfLoops: false,
        });
        const byId = new Map(nodes.map((node) => [node.id, node]));
        const nodeOrder = new Map(nodes.map((node, index) => [node.id, index]));
        for (const node of nodes) graph.addNode(node.id, nodeAttributes(node, seedSetRef.current));
        for (const edge of edges) graph.addDirectedEdge(edge.source, edge.target, edgeAttributes());
        const edgeOrder = new Map(graph.edges().map((edge, index) => [edge, index]));

        try {
          renderer = new sigmaModule.default<GraphNodeAttributes, GraphEdgeAttributes>(graph, container, {
            primitives: {
              nodes: {
                layers: [ringLayer()],
                label: {
                  font: { family: labelFont, weight: LABEL_WEIGHT, size: LABEL_SIZE },
                  color: CHART_COLORS.ink,
                  margin: LABEL_MARGIN,
                },
              },
              edges: {
                extremities: [renderingModule.extremityArrow(ARROW_HEAD)],
                layers: [edgeBodyLayer()],
                defaultHead: "arrow",
              },
              depthLayers: DEPTH_LAYERS,
            },
            styles: {
              nodes: {
                x: { attribute: "x" },
                y: { attribute: "y" },
                size: { attribute: "size" },
                color: { attribute: "color" },
                label: { attribute: "label" },
                zIndex: { attribute: "zIndex" },
                depth: "nodes",
                cursor: "pointer",
                labelFont: `${LABEL_WEIGHT} ${labelFont}`,
                labelSize: LABEL_SIZE,
                labelColor: CHART_COLORS.ink,
                labelVisibility: "hidden",
                backdropVisibility: "hidden",
                ...BACKDROP_STYLE,
              },
              edges: {
                size: { attribute: "size" },
                color: { attribute: "color" },
                zIndex: { attribute: "zIndex" },
                depth: "edges",
              },
            },
            settings: {
              itemSizesReference: "screen",
              minEdgeThickness: MIN_EDGE_SIZE,
              enableCameraRotation: false,
              enableCameraMouseRotation: false,
              minCameraRatio: 0.05,
              maxCameraRatio: 2,
              stagePadding: 24,
            },
            nodeReducer: (id, _data, attributes) => {
              const focus = focusOf(hoveredRef.current, selectedRef.current);
              const node = byId.get(id);
              const base = node ? { ...attributes, ...seedAttributes(node, seedSetRef.current) } : attributes;
              const display = nodeDisplay(base, nodeEmphasis(id, focus, adjacency), id === selectedRef.current);
              return { ...display, zIndex: orderedZIndex(display.zIndex, nodeOrder.get(id) ?? 0, nodeOrder.size) };
            },
            edgeReducer: (edge) => {
              const focus = focusOf(hoveredRef.current, selectedRef.current);
              const display = edgeDisplay(edgeEmphasis(graph.source(edge), graph.target(edge), focus, adjacency));
              return { ...display, zIndex: orderedZIndex(display.zIndex, edgeOrder.get(edge) ?? 0, edgeOrder.size) };
            },
          });
        } catch {
          container.replaceChildren();
          setStatus("unsupported");
          return;
        }

        const current = renderer;
        current.on("enterNode", ({ node }) => {
          hoveredRef.current = node;
          current.refresh({ skipIndexation: true });
        });
        current.on("leaveNode", () => {
          hoveredRef.current = null;
          current.refresh({ skipIndexation: true });
        });
        current.on("clickNode", ({ node }) => onSelectRef.current(node));
        current.on("clickStage", () => onSelectRef.current(null));
        rendererRef.current = current;
        setStatus("ready");
      })
      .catch(() => {
        if (!cancelled) setStatus("error");
      });

    return () => {
      cancelled = true;
      rendererRef.current = null;
      renderer?.kill();
      renderer = null;
    };
  }, [nodes, edges, adjacency]);

  useEffect(() => {
    selectedRef.current = selectedId;
    const renderer = rendererRef.current;
    if (!renderer || status !== "ready") return;
    renderer.refresh({ skipIndexation: true });
    if (selectedId === null) return;
    const display = renderer.getNodeDisplayData(selectedId);
    if (!display) return;
    const ratio = Math.min(renderer.getCamera().ratio, FOCUS_RATIO);
    moveCamera(renderer, { x: display.x, y: display.y, ratio });
  }, [selectedId, status]);

  useEffect(() => {
    if (seedSetRef.current === seedSet) return;
    seedSetRef.current = seedSet;
    const renderer = rendererRef.current;
    if (renderer && status === "ready") renderer.refresh();
  }, [seedSet, status]);

  function zoom(factor: number): void {
    const renderer = rendererRef.current;
    if (!renderer) return;
    moveCamera(renderer, { ratio: renderer.getCamera().ratio / factor });
  }

  function reset(): void {
    const renderer = rendererRef.current;
    if (renderer) moveCamera(renderer, RESET_STATE);
  }

  const unavailable = status === "unsupported" || status === "error";

  return (
    <div className="relative" data-testid="graph-view" data-state={status} data-seed-set={seedSet}>
      <div className="relative">
        <div
          ref={containerRef}
          role="img"
          aria-label={description}
          data-testid="graph-canvas"
          className="h-[380px] w-full overflow-hidden rounded-2xl border border-line bg-surface bg-[radial-gradient(circle,rgb(91_100_116/0.16)_1px,transparent_1.2px)] [background-size:22px_22px] shadow-card sm:h-[560px]"
        />
        <p
          aria-hidden="true"
          className="pointer-events-none absolute top-4 left-4 hidden items-center gap-2 rounded-full border border-line bg-surface/90 px-3 py-1.5 text-[0.6875rem] font-medium tracking-[0.14em] text-slate uppercase sm:inline-flex"
        >
          <span className="size-1.5 rotate-45 bg-gold" />
          {formatInteger(nodes.length)} Knoten · {formatInteger(edges.length)} Kanten
        </p>
        {isClient && status === "loading" ? (
          <p
            className="pointer-events-none absolute inset-0 flex items-center justify-center gap-3 text-sm text-slate"
            data-testid="graph-loading"
          >
            <span aria-hidden="true" className="size-2 rotate-45 bg-gold" />
            Graph wird geladen …
          </p>
        ) : null}
        <noscript>
          <p
            className="absolute inset-0 flex items-center justify-center gap-3 rounded-2xl p-6 text-center text-sm text-slate"
            data-testid="graph-noscript"
          >
            <span aria-hidden="true" className="size-2 shrink-0 rotate-45 bg-gold" />
            Die Graph-Ansicht braucht JavaScript; Kennzahlen und Tabellen stehen unten.
          </p>
        </noscript>
        {unavailable ? (
          <div
            role="status"
            data-testid="graph-fallback"
            className="absolute inset-0 flex items-center justify-center gap-3 rounded-2xl p-6 text-center text-sm text-slate"
          >
            <span aria-hidden="true" className="size-2 shrink-0 rotate-45 bg-gold" />
            <p className="max-w-md">
              {status === "unsupported"
                ? "Ihr Browser stellt kein WebGL 2 bereit. Die Graph-Ansicht ist deshalb ausgeblendet. Die Tabelle der Startknoten und das Detailfeld funktionieren weiterhin."
                : "Die Graph-Ansicht konnte nicht geladen werden. Die Tabelle der Startknoten und das Detailfeld funktionieren weiterhin."}
            </p>
          </div>
        ) : null}
      </div>
      {status === "ready" ? (
        <div className="mt-3 flex justify-end sm:absolute sm:top-4 sm:right-4 sm:mt-0" data-testid="graph-controls">
          <div className="inline-flex items-center gap-0.5 rounded-full border border-line bg-surface p-1 shadow-card">
            <button type="button" onClick={() => zoom(ZOOM_STEP)} className={CONTROL_ICON}>
              <span aria-hidden="true">+</span>
              <span className="sr-only">Vergrößern</span>
            </button>
            <button type="button" onClick={() => zoom(1 / ZOOM_STEP)} className={CONTROL_ICON}>
              <span aria-hidden="true">−</span>
              <span className="sr-only">Verkleinern</span>
            </button>
            <span aria-hidden="true" className="mx-1 h-5 w-px bg-line" />
            <button type="button" onClick={reset} className={CONTROL_TEXT}>
              Gesamtansicht
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
