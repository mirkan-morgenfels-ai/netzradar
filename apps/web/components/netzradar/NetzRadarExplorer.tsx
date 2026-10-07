"use client";

import { useCallback, useMemo, useState } from "react";
import { METHOD_SHORT_TEXT } from "@/lib/netzradar/format";
import { buildAdjacency, neighborsOf, seedNodes } from "@/lib/netzradar/graph";
import type { GraphMethod, NetEdge, NetNode } from "@/lib/netzradar/types";
import { GraphLegend } from "./GraphLegend";
import { GraphView } from "./GraphView";
import { NodeDetail } from "./NodeDetail";
import { TopNodesTable } from "./TopNodesTable";

export interface NetzRadarExplorerProps {
  nodes: NetNode[];
  edges: NetEdge[];
  scoreGnnMethod: GraphMethod | null;
  description: string;
}

export function NetzRadarExplorer({ nodes, edges, scoreGnnMethod, description }: NetzRadarExplorerProps) {
  const gnnLabel = scoreGnnMethod === null ? null : METHOD_SHORT_TEXT[scoreGnnMethod];
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const adjacency = useMemo(() => buildAdjacency(nodes.map((node) => node.id), edges), [nodes, edges]);
  const byId = useMemo(() => new Map(nodes.map((node) => [node.id, node])), [nodes]);
  const seeds = useMemo(() => seedNodes(nodes), [nodes]);
  const select = useCallback((id: string | null) => setSelectedId(id), []);

  const selected = selectedId === null ? null : (byId.get(selectedId) ?? null);
  const neighbors = selected
    ? neighborsOf(adjacency, selected.id).flatMap((id) => {
        const neighbor = byId.get(id);
        return neighbor ? [neighbor] : [];
      })
    : [];

  return (
    <div className="grid gap-6 lg:grid-cols-3" data-testid="netzradar-explorer">
      <div className="min-w-0 space-y-3 lg:col-span-2">
        <GraphView
          nodes={nodes}
          edges={edges}
          adjacency={adjacency}
          selectedId={selectedId}
          onSelect={select}
          description={description}
        />
        <GraphLegend />
      </div>
      <div className="min-w-0 lg:col-start-3 lg:row-span-2 lg:row-start-1">
        <div className="lg:sticky lg:top-6">
          <NodeDetail
            node={selected}
            neighbors={neighbors}
            seedCount={seeds.length}
            gnnLabel={gnnLabel}
            onSelect={select}
          />
        </div>
      </div>
      <div className="min-w-0 lg:col-span-2">
        <TopNodesTable nodes={seeds} selectedId={selectedId} gnnLabel={gnnLabel} onSelect={select} />
      </div>
    </div>
  );
}
