"use client";

import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import type Sigma from "sigma";
import type { CameraState } from "sigma/types";
import { CHART_COLORS } from "@portfolio/charts/theme";
import {
  edgeAttributes,
  edgeDisplay,
  edgeEmphasis,
  focusOf,
  nodeAttributes,
  nodeDisplay,
  nodeEmphasis,
  type Adjacency,
  type GraphEdgeAttributes,
  type GraphNodeAttributes,
} from "@/lib/netzradar/graph";
import { formatInteger } from "@/lib/netzradar/format";
import type { NetEdge, NetNode } from "@/lib/netzradar/types";

type Status = "loading" | "ready" | "unsupported" | "error";
type Renderer = Sigma<GraphNodeAttributes, GraphEdgeAttributes>;

const CAMERA_DURATION = 350;
const ZOOM_STEP = 1.6;
const FOCUS_RATIO = 0.35;
const LABEL_FONT = 'ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif';
const RESET_STATE: Partial<CameraState> = { x: 0.5, y: 0.5, ratio: 1, angle: 0 };
const CONTROL_BASE =
  "inline-flex h-9 items-center justify-center rounded-full text-ink transition-colors duration-150 hover:bg-navy-950 hover:text-ivory";
const CONTROL_ICON = `${CONTROL_BASE} w-9 text-lg leading-none`;
const CONTROL_TEXT = `${CONTROL_BASE} px-4 text-sm font-medium`;

export interface GraphViewProps {
  nodes: readonly NetNode[];
  edges: readonly NetEdge[];
  adjacency: Adjacency;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  description: string;
}

function hasWebGl(): boolean {
  try {
    const canvas = document.createElement("canvas");
    return Boolean(canvas.getContext("webgl2") ?? canvas.getContext("webgl"));
  } catch {
    return false;
  }
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

export function GraphView({ nodes, edges, adjacency, selectedId, onSelect, description }: GraphViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<Renderer | null>(null);
  const hoveredRef = useRef<string | null>(null);
  const selectedRef = useRef<string | null>(selectedId);
  const onSelectRef = useRef(onSelect);
  const [status, setStatus] = useState<Status>("loading");
  const isClient = useIsClient();

  useEffect(() => {
    onSelectRef.current = onSelect;
  }, [onSelect]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    if (!hasWebGl()) {
      setStatus("unsupported");
      return;
    }
    let cancelled = false;
    let renderer: Renderer | null = null;

    Promise.all([import("graphology"), import("sigma"), import("@sigma/node-border")])
      .then(([graphModule, sigmaModule, borderModule]) => {
        if (cancelled) return;
        const graph = new graphModule.default<GraphNodeAttributes, GraphEdgeAttributes>({
          type: "directed",
          multi: false,
          allowSelfLoops: false,
        });
        for (const node of nodes) graph.addNode(node.id, nodeAttributes(node));
        for (const edge of edges) graph.addDirectedEdge(edge.source, edge.target, edgeAttributes());

        const BorderedNodeProgram = borderModule.createNodeBorderProgram<GraphNodeAttributes, GraphEdgeAttributes>({
          borders: [
            { size: { attribute: "borderSize", defaultValue: 0 }, color: { attribute: "borderColor" } },
            { size: { fill: true }, color: { attribute: "color" } },
          ],
        });

        try {
          renderer = new sigmaModule.default<GraphNodeAttributes, GraphEdgeAttributes>(graph, container, {
            defaultNodeType: "bordered",
            nodeProgramClasses: { bordered: BorderedNodeProgram },
            defaultEdgeType: "arrow",
            zIndex: true,
            enableCameraRotation: false,
            minCameraRatio: 0.05,
            maxCameraRatio: 2,
            stagePadding: 24,
            labelFont: getComputedStyle(container).fontFamily || LABEL_FONT,
            labelSize: 12,
            labelWeight: "500",
            labelColor: { color: CHART_COLORS.ink },
            labelRenderedSizeThreshold: Number.POSITIVE_INFINITY,
            nodeReducer: (id, data) => {
              const focus = focusOf(hoveredRef.current, selectedRef.current);
              return nodeDisplay(data, nodeEmphasis(id, focus, adjacency), id === selectedRef.current);
            },
            edgeReducer: (edge) => {
              const focus = focusOf(hoveredRef.current, selectedRef.current);
              return edgeDisplay(edgeEmphasis(graph.source(edge), graph.target(edge), focus, adjacency));
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
          container.style.cursor = "pointer";
          current.refresh({ skipIndexation: true });
        });
        current.on("leaveNode", () => {
          hoveredRef.current = null;
          container.style.cursor = "";
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
    <div className="relative" data-testid="graph-view" data-state={status}>
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
                ? "Ihr Browser stellt kein WebGL bereit. Die Graph-Ansicht ist deshalb ausgeblendet. Die Tabelle der Startknoten und das Detailfeld funktionieren weiterhin."
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
