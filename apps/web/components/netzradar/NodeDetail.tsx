"use client";

import { useEffect, useRef, type ReactNode } from "react";
import { formatDecimal, formatInteger, LABEL_TEXT } from "@/lib/netzradar/format";
import type { NetNode } from "@/lib/netzradar/types";
import { NodeSymbol } from "./NodeSymbol";

export interface NodeDetailProps {
  node: NetNode | null;
  neighbors: readonly NetNode[];
  seedCount: number;
  gnnLabel: string | null;
  onSelect: (id: string | null) => void;
}

function Row({ term, children }: { term: string; children: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-t border-line py-2.5 first:border-t-0 first:pt-0">
      <dt className="text-slate">{term}</dt>
      <dd className="num text-right text-ink">{children}</dd>
    </div>
  );
}

function ScoreRow({ term, children }: { term: string; children: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-t border-line py-3 first:border-t-0 first:pt-0">
      <dt className="text-slate">{term}</dt>
      <dd className="display num text-right text-[1.375rem] leading-none text-ink">{children}</dd>
    </div>
  );
}

export function NodeDetail({ node, neighbors, seedCount, gnnLabel, onSelect }: NodeDetailProps) {
  const titleRef = useRef<HTMLHeadingElement>(null);
  const focusTitle = useRef(false);

  useEffect(() => {
    if (!focusTitle.current) return;
    focusTitle.current = false;
    titleRef.current?.focus();
  }, [node]);

  function selectFromDetail(id: string | null) {
    focusTitle.current = true;
    onSelect(id);
  }

  return (
    <section
      aria-labelledby="node-detail-title"
      data-testid="node-detail"
      className="overflow-hidden rounded-2xl border border-line bg-surface text-sm shadow-card"
    >
      <div className="surface-navy relative px-5 pt-5 pb-6 sm:px-6">
        <div
          aria-hidden="true"
          className="absolute inset-x-6 bottom-0 h-px bg-linear-to-r from-gold/0 via-gold/60 to-gold/0"
        />
        <p className="eyebrow">Detailfeld</p>
        <h3
          id="node-detail-title"
          ref={titleRef}
          tabIndex={-1}
          className="display mt-2 rounded-sm text-[1.625rem] leading-tight text-ivory"
        >
          Ausgewählter Knoten
        </h3>
        <p className="sr-only" aria-live="polite">
          {node ? `Ausgewählt: ${node.id}, ${LABEL_TEXT[node.label]}` : "Kein Knoten ausgewählt"}
        </p>
        {node === null ? (
          <p className="mt-3 text-[13px] leading-relaxed text-navy-300">Noch kein Knoten gewählt.</p>
        ) : (
          <div className="mt-4 flex flex-wrap items-center justify-between gap-x-4 gap-y-3">
            <p className="flex items-baseline gap-2">
              <span className="text-[0.6875rem] font-medium tracking-[0.14em] text-navy-300 uppercase">ID</span>
              <span className="font-mono text-[15px] tracking-wide text-gold-light" data-testid="node-detail-id">
                {node.id}
              </span>
            </p>
            <p className="inline-flex items-center gap-2 rounded-full bg-ivory px-3 py-1 text-xs font-medium text-ink">
              <span className="sr-only">Label im Datensatz: </span>
              <NodeSymbol label={node.label} size={12} />
              {LABEL_TEXT[node.label]}
            </p>
          </div>
        )}
      </div>
      <div className="px-5 py-5 sm:px-6 sm:py-6">
        {node === null ? (
          <p className="leading-relaxed text-slate">
            Wählen Sie einen Knoten in der Graph-Ansicht oder in der Tabelle der Startknoten aus. Angezeigt werden dann
            Label, Zeitschritt, die Scores der Verfahren, Grade und Nachbarn.
          </p>
        ) : (
          <>
            <p className="eyebrow">Scores</p>
            <dl className="mt-3">
              <ScoreRow term="Score Isolation Forest">{formatDecimal(node.scoreIforest)}</ScoreRow>
              <ScoreRow term="Score robuste Z-Scores">{formatDecimal(node.scoreZscore)}</ScoreRow>
              <ScoreRow term={gnnLabel === null ? "Score GNN" : `Score GNN (${gnnLabel})`}>
                {node.scoreGnn === null ? (
                  <span className="font-sans text-sm text-slate">noch nicht berechnet</span>
                ) : (
                  <span data-testid="node-detail-gnn">{formatDecimal(node.scoreGnn)}</span>
                )}
              </ScoreRow>
            </dl>
            {node.scoreGnn === null ? null : (
              <p className="mt-3 text-xs leading-relaxed text-slate">
                Der GNN-Score ist die Differenz der beiden Ausgaben des Netzes (Logit auffällig minus Logit unauffällig),
                keine Wahrscheinlichkeit. Höher heißt auffälliger.
              </p>
            )}
            <p className="eyebrow mt-6">Lage im Netz</p>
            <dl className="mt-3">
              <Row term="Zeitschritt">{node.timeStep}</Row>
              <Row term="Eingangsgrad / Ausgangsgrad">
                {formatInteger(node.inDegree)} / {formatInteger(node.outDegree)}
              </Row>
              <Row term="Rang unter den Startknoten">
                {node.seedRank === null ? (
                  <span className="text-slate">kein Startknoten</span>
                ) : (
                  `${node.seedRank} von ${seedCount}`
                )}
              </Row>
              <Row term="Kanten bis zum nächsten Startknoten">{node.hop}</Row>
            </dl>
            <div className="mt-6 border-t border-line pt-5">
              <h4 className="eyebrow">Nachbarn im Ausschnitt ({formatInteger(neighbors.length)})</h4>
              {neighbors.length === 0 ? (
                <p className="mt-2 text-slate">Keine Nachbarn im Ausschnitt.</p>
              ) : (
                <ul className="mt-3 flex flex-wrap gap-2 p-0.5">
                  {neighbors.map((neighbor) => (
                    <li key={neighbor.id}>
                      <button
                        type="button"
                        onClick={() => selectFromDetail(neighbor.id)}
                        className="inline-flex items-center gap-1.5 rounded-full border border-line bg-surface px-2.5 py-1 font-mono text-xs text-ink transition-colors duration-150 hover:border-gold hover:bg-gold-soft/50"
                      >
                        <NodeSymbol label={neighbor.label} size={10} />
                        {neighbor.id}
                        <span className="sr-only">, {LABEL_TEXT[neighbor.label]}</span>
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
            <button
              type="button"
              onClick={() => selectFromDetail(null)}
              className="mt-5 inline-flex items-center rounded-full border border-line-strong bg-surface px-4 py-2 text-xs font-medium text-ink transition-colors duration-150 hover:border-ink"
            >
              Auswahl aufheben
            </button>
          </>
        )}
      </div>
    </section>
  );
}
