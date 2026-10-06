"use client";

import { useEffect, useRef, type ReactNode } from "react";
import { formatDecimal, formatInteger, LABEL_TEXT } from "@/lib/netzradar/format";
import type { NetNode } from "@/lib/netzradar/types";
import { NodeSymbol } from "./NodeSymbol";

export interface NodeDetailProps {
  node: NetNode | null;
  neighbors: readonly NetNode[];
  seedCount: number;
  onSelect: (id: string | null) => void;
}

function Row({ term, children }: { term: string; children: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-t border-line py-2 first:border-t-0">
      <dt className="text-stone">{term}</dt>
      <dd className="text-right tabular-nums">{children}</dd>
    </div>
  );
}

export function NodeDetail({ node, neighbors, seedCount, onSelect }: NodeDetailProps) {
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
      className="rounded-lg border border-line bg-surface p-5 text-sm"
    >
      <h3 id="node-detail-title" ref={titleRef} tabIndex={-1} className="font-serif text-lg">
        Ausgewählter Knoten
      </h3>
      <p className="sr-only" aria-live="polite">
        {node ? `Ausgewählt: ${node.id}, ${LABEL_TEXT[node.label]}` : "Kein Knoten ausgewählt"}
      </p>
      {node === null ? (
        <p className="mt-3 text-stone">
          Wählen Sie einen Knoten in der Graph-Ansicht oder in der Tabelle der Startknoten aus. Angezeigt werden dann
          Label, Zeitschritt, beide Scores, Grade und Nachbarn.
        </p>
      ) : (
        <>
          <dl className="mt-3">
            <Row term="ID">
              <span className="font-mono" data-testid="node-detail-id">
                {node.id}
              </span>
            </Row>
            <Row term="Label im Datensatz">
              <span className="inline-flex items-center gap-2">
                <NodeSymbol label={node.label} />
                {LABEL_TEXT[node.label]}
              </span>
            </Row>
            <Row term="Zeitschritt">{node.timeStep}</Row>
            <Row term="Score Isolation Forest">{formatDecimal(node.scoreIforest)}</Row>
            <Row term="Score robuste Z-Scores">{formatDecimal(node.scoreZscore)}</Row>
            <Row term="Score GNN">
              {node.scoreGnn === null ? (
                <span className="text-stone">Schritt 4, noch nicht berechnet</span>
              ) : (
                formatDecimal(node.scoreGnn)
              )}
            </Row>
            <Row term="Eingangsgrad / Ausgangsgrad">
              {formatInteger(node.inDegree)} / {formatInteger(node.outDegree)}
            </Row>
            <Row term="Rang unter den Startknoten">
              {node.seedRank === null ? (
                <span className="text-stone">kein Startknoten</span>
              ) : (
                `${node.seedRank} von ${seedCount}`
              )}
            </Row>
            <Row term="Kanten bis zum nächsten Startknoten">{node.hop}</Row>
          </dl>
          <div className="mt-4 border-t border-line pt-3">
            <h4 className="text-xs uppercase tracking-wide text-stone">
              Nachbarn im Ausschnitt ({formatInteger(neighbors.length)})
            </h4>
            {neighbors.length === 0 ? (
              <p className="mt-2 text-stone">Keine Nachbarn im Ausschnitt.</p>
            ) : (
              <ul className="mt-2 flex max-h-40 flex-wrap gap-2 overflow-y-auto">
                {neighbors.map((neighbor) => (
                  <li key={neighbor.id}>
                    <button
                      type="button"
                      onClick={() => selectFromDetail(neighbor.id)}
                      className="inline-flex items-center gap-1.5 rounded-md border border-line px-2 py-1 font-mono text-xs hover:border-gold"
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
            className="mt-4 text-xs text-stone underline underline-offset-4 hover:text-gold-deep"
          >
            Auswahl aufheben
          </button>
        </>
      )}
    </section>
  );
}
