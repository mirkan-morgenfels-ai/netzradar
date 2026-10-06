"use client";

import { useEffect, useRef, useState } from "react";
import type Sigma from "sigma";
import type { CameraState } from "sigma/types";
import { Button } from "@portfolio/ui";
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
import type { NetEdge, NetNode } from "@/lib/netzradar/types";

type Status = "loading" | "ready" | "unsupported" | "error";
type Renderer = Sigma<GraphNodeAttributes, GraphEdgeAttributes>;

const CAMERA_DURATION = 350;
const ZOOM_STEP = 1.6;
const FOCUS_RATIO = 0.35;
const LABEL_FONT = 'ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif';
const RESET_STATE: Partial<CameraState> = { x: 0.5, y: 0.5, ratio: 1, angle: 0 };

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
            labelFont: LABEL_FONT,
            labelSize: 12,
            labelWeight: "600",
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
      <div
        ref={containerRef}
        role="img"
        aria-label={description}
        data-testid="graph-canvas"
        className="h-[360px] w-full overflow-hidden rounded-lg border border-line bg-surface sm:h-[520px]"
      />
      {status === "loading" ? (
        <p className="pointer-events-none absolute inset-0 flex items-center justify-center text-sm text-stone">
          Graph wird geladen …
        </p>
      ) : null}
      {unavailable ? (
        <div
          role="status"
          data-testid="graph-fallback"
          className="absolute inset-0 flex items-center justify-center rounded-lg bg-paper p-6 text-center text-sm text-stone"
        >
          <p className="max-w-md">
            {status === "unsupported"
              ? "Ihr Browser stellt kein WebGL bereit. Die Graph-Ansicht ist deshalb ausgeblendet. Die Tabelle der Startknoten und das Detailfeld funktionieren weiterhin."
              : "Die Graph-Ansicht konnte nicht geladen werden. Die Tabelle der Startknoten und das Detailfeld funktionieren weiterhin."}
          </p>
        </div>
      ) : null}
      {status === "ready" ? (
        <div className="absolute right-3 top-3 flex gap-2">
          <Button type="button" variant="secondary" onClick={() => zoom(ZOOM_STEP)}>
            <span aria-hidden="true">+</span>
            <span className="sr-only">Vergrößern</span>
          </Button>
          <Button type="button" variant="secondary" onClick={() => zoom(1 / ZOOM_STEP)}>
            <span aria-hidden="true">−</span>
            <span className="sr-only">Verkleinern</span>
          </Button>
          <Button type="button" variant="secondary" onClick={reset}>
            Gesamtansicht
          </Button>
        </div>
      ) : null}
    </div>
  );
}
