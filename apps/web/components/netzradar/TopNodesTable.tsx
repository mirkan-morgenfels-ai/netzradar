"use client";

import { useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { cx } from "@portfolio/ui";
import { formatDecimal, LABEL_TEXT, seedSetScoreText } from "@/lib/netzradar/format";
import { seedRankOf } from "@/lib/netzradar/graph";
import type { NetNode, SeedSet } from "@/lib/netzradar/types";
import { NodeSymbol } from "./NodeSymbol";

export interface TopNodesTableProps {
  nodes: readonly NetNode[];
  seedSet: SeedSet;
  selectedId: string | null;
  gnnLabel: string | null;
  onSelect: (id: string) => void;
}

interface Cursor {
  id: string;
  selection: string | null;
}

const HEAD_CELL =
  "sticky top-0 z-10 border-b-0 bg-surface px-2.5 pt-4 pb-3 align-bottom tracking-[0.08em] whitespace-normal shadow-[inset_0_-1px_0_var(--color-ink)] first:pl-5 last:pr-5 sm:first:pl-6 sm:last:pr-6";
const SCORE_HEAD_CELL = cx(HEAD_CELL, "text-right");
const CELL = "px-2.5 first:pl-5 last:pr-5 sm:first:pl-6 sm:last:pr-6";

function SortedMark() {
  return (
    <span aria-hidden="true" className="ml-1 text-gold-deep">
      ↓
    </span>
  );
}

function ScoreHead({
  sorted,
  className,
  testId,
  children,
}: {
  sorted: boolean;
  className: string;
  testId?: string;
  children: ReactNode;
}) {
  return (
    <th
      scope="col"
      aria-sort={sorted ? "descending" : undefined}
      className={cx(SCORE_HEAD_CELL, className, sorted ? "text-ink" : undefined)}
      data-testid={testId}
    >
      {children}
      {sorted ? <SortedMark /> : null}
    </th>
  );
}

export function TopNodesTable({ nodes, seedSet, selectedId, gnnLabel, onSelect }: TopNodesTableProps) {
  const [cursor, setCursor] = useState<Cursor | null>(null);
  const buttons = useRef(new Map<string, HTMLButtonElement>());
  const byGnn = seedSet === "gnn" && gnnLabel !== null;
  const scoreText = seedSetScoreText(byGnn ? "gnn" : "iforest", gnnLabel ?? "");
  const selectedInTable = selectedId !== null && nodes.some((node) => node.id === selectedId);
  const cursorInTable = cursor !== null && cursor.selection === selectedId && nodes.some((node) => node.id === cursor.id);
  const tabStop = cursorInTable ? cursor.id : selectedInTable ? selectedId : (nodes[0]?.id ?? null);

  function moveFocus(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    let next = index;
    if (event.key === "ArrowDown") next = Math.min(index + 1, nodes.length - 1);
    else if (event.key === "ArrowUp") next = Math.max(index - 1, 0);
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = nodes.length - 1;
    else return;
    event.preventDefault();
    const target = nodes[next];
    if (!target) return;
    setCursor({ id: target.id, selection: selectedId });
    buttons.current.get(target.id)?.focus();
  }

  return (
    <div className="rounded-2xl border border-line bg-surface shadow-card" data-testid="top-nodes" data-seed-set={seedSet}>
      <div className="px-5 pt-5 sm:px-6 sm:pt-6">
        <p className="eyebrow">Startknoten</p>
        <h3 className="display mt-2 text-[1.625rem] leading-tight text-ink sm:text-[1.75rem]" data-testid="top-nodes-title">
          Die {nodes.length} Testknoten mit dem höchsten {scoreText}
        </h3>
        <p id="top-nodes-note" className="mt-3 max-w-[70ch] text-xs leading-relaxed text-slate">
          Die {nodes.length} Startknoten, sortiert nach dem {scoreText}
          {byGnn
            ? ", einer Logit-Differenz ohne feste Ober- und Untergrenze. Die Scores von Isolation Forest und robusten Z-Scores stehen zum Vergleich daneben."
            : gnnLabel === null
              ? "."
              : `. Die letzte Spalte zeigt zum Vergleich den GNN-Score (${gnnLabel}), eine Logit-Differenz ohne feste Ober- und Untergrenze.`}{" "}
          Ein Klick auf eine Zeile oder Enter auf der ID wählt den Knoten im Graph aus.
        </p>
      </div>
      <div
        role="region"
        aria-labelledby="top-nodes-caption"
        aria-describedby="top-nodes-note"
        tabIndex={0}
        className="relative mt-3 max-h-[30rem] overflow-auto rounded-b-2xl"
      >
        <table
          className={cx("data-table", gnnLabel === null ? "min-w-[36rem]" : "min-w-[46rem]")}
          data-testid="top-nodes-table"
        >
          <caption id="top-nodes-caption" className="sr-only">
            Startknoten des Netzwerk-Ausschnitts nach {scoreText}
          </caption>
          <thead>
            <tr>
              <th scope="col" className={cx(HEAD_CELL, "text-right")}>
                Rang
              </th>
              <th scope="col" className={HEAD_CELL}>
                ID
              </th>
              <th scope="col" className={HEAD_CELL}>
                Label
              </th>
              <th scope="col" className={cx(HEAD_CELL, "text-right")}>
                Zeitschritt
              </th>
              <ScoreHead sorted={!byGnn} className="min-w-[8.25rem]">
                Score Isolation Forest
              </ScoreHead>
              <ScoreHead sorted={false} className="min-w-[7.75rem]">
                Score robuste <span className="whitespace-nowrap">Z-Scores</span>
              </ScoreHead>
              {gnnLabel === null ? null : (
                <ScoreHead sorted={byGnn} className="min-w-[7.5rem]" testId="top-nodes-gnn-header">
                  Score GNN <span className="whitespace-nowrap">({gnnLabel})</span>
                </ScoreHead>
              )}
            </tr>
          </thead>
          <tbody>
            {nodes.map((node, index) => {
              const selected = node.id === selectedId;
              return (
                <tr
                  key={node.id}
                  data-testid="top-node-row"
                  data-node-id={node.id}
                  aria-current={selected ? "true" : undefined}
                  onClick={() => onSelect(node.id)}
                  className={cx(
                    "cursor-pointer",
                    selected ? "bg-gold-soft/70 shadow-[inset_3px_0_0_var(--color-gold)] hover:bg-gold-soft/70" : undefined,
                  )}
                >
                  <td className={cx(CELL, "text-right text-slate")} data-testid="top-node-rank">
                    {seedRankOf(node, byGnn ? "gnn" : "iforest")}
                  </td>
                  <td className={CELL}>
                    <button
                      type="button"
                      ref={(element) => {
                        if (element) buttons.current.set(node.id, element);
                        else buttons.current.delete(node.id);
                      }}
                      tabIndex={node.id === tabStop ? 0 : -1}
                      aria-current={selected ? "true" : undefined}
                      onClick={(event) => {
                        event.stopPropagation();
                        onSelect(node.id);
                      }}
                      onKeyDown={(event) => moveFocus(event, index)}
                      className="rounded-sm font-mono text-[13px] text-ink underline decoration-gold/0 underline-offset-4 transition-colors duration-150 hover:text-gold-deep hover:decoration-gold-deep"
                    >
                      {node.id}
                    </button>
                  </td>
                  <td className={CELL}>
                    <span className="inline-flex items-center gap-2 whitespace-nowrap">
                      <NodeSymbol label={node.label} size={12} />
                      {LABEL_TEXT[node.label]}
                    </span>
                  </td>
                  <td className={cx(CELL, "text-right")}>{node.timeStep}</td>
                  <td className={cx(CELL, "text-right")}>{formatDecimal(node.scoreIforest)}</td>
                  <td className={cx(CELL, "text-right")}>{formatDecimal(node.scoreZscore)}</td>
                  {gnnLabel === null ? null : (
                    <td className={cx(CELL, "text-right")} data-testid="top-node-gnn">
                      {node.scoreGnn === null ? "–" : formatDecimal(node.scoreGnn)}
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
