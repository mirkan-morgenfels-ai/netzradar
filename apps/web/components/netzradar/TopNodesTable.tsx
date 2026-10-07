import { cx } from "@portfolio/ui";
import { formatDecimal, LABEL_TEXT } from "@/lib/netzradar/format";
import type { NetNode } from "@/lib/netzradar/types";
import { NodeSymbol } from "./NodeSymbol";

export interface TopNodesTableProps {
  nodes: readonly NetNode[];
  selectedId: string | null;
  gnnLabel: string | null;
  onSelect: (id: string) => void;
}

const HEAD_CELL = "sticky top-0 z-10 bg-surface px-3 py-2 font-medium shadow-[inset_0_-1px_0_var(--color-line)]";

export function TopNodesTable({ nodes, selectedId, gnnLabel, onSelect }: TopNodesTableProps) {
  return (
    <div className="space-y-2">
      <p id="top-nodes-note" className="text-xs text-stone">
        Die {nodes.length} Startknoten, sortiert nach dem Isolation-Forest-Score.
        {gnnLabel === null
          ? ""
          : ` Die letzte Spalte zeigt zum Vergleich den GNN-Score (${gnnLabel}), eine Logit-Differenz ohne feste Obergrenze.`}{" "}
        Ein Klick auf eine Zeile oder Enter auf der ID wählt den Knoten im Graph aus.
      </p>
      <div
        role="region"
        aria-labelledby="top-nodes-caption"
        aria-describedby="top-nodes-note"
        tabIndex={0}
        className="max-h-[30rem] overflow-auto rounded-lg border border-line bg-surface"
      >
        <table
          className={cx("w-full border-collapse text-sm", gnnLabel === null ? "min-w-[36rem]" : "min-w-[44rem]")}
          data-testid="top-nodes-table"
        >
          <caption id="top-nodes-caption" className="sr-only">
            Startknoten des Netzwerk-Ausschnitts
          </caption>
          <thead>
            <tr className="text-left text-xs uppercase tracking-wide text-stone">
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
              <th scope="col" className={cx(HEAD_CELL, "text-right")}>
                Score Isolation Forest
              </th>
              <th scope="col" className={cx(HEAD_CELL, "text-right")}>
                Score robuste Z-Scores
              </th>
              {gnnLabel === null ? null : (
                <th scope="col" className={cx(HEAD_CELL, "text-right")} data-testid="top-nodes-gnn-header">
                  Score GNN ({gnnLabel})
                </th>
              )}
            </tr>
          </thead>
          <tbody>
            {nodes.map((node) => {
              const selected = node.id === selectedId;
              return (
                <tr
                  key={node.id}
                  data-testid="top-node-row"
                  data-node-id={node.id}
                  aria-current={selected ? "true" : undefined}
                  onClick={() => onSelect(node.id)}
                  className={cx(
                    "cursor-pointer border-t border-line tabular-nums",
                    selected ? "bg-gold-soft" : "hover:bg-paper",
                  )}
                >
                  <td className="px-3 py-2 text-right">{node.seedRank}</td>
                  <td className="px-3 py-2">
                    <button
                      type="button"
                      aria-current={selected ? "true" : undefined}
                      onClick={(event) => {
                        event.stopPropagation();
                        onSelect(node.id);
                      }}
                      className="font-mono underline-offset-4 hover:text-gold-deep hover:underline"
                    >
                      {node.id}
                    </button>
                  </td>
                  <td className="px-3 py-2">
                    <span className="inline-flex items-center gap-2">
                      <NodeSymbol label={node.label} size={12} />
                      {LABEL_TEXT[node.label]}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-right">{node.timeStep}</td>
                  <td className="px-3 py-2 text-right">{formatDecimal(node.scoreIforest)}</td>
                  <td className="px-3 py-2 text-right">{formatDecimal(node.scoreZscore)}</td>
                  {gnnLabel === null ? null : (
                    <td className="px-3 py-2 text-right" data-testid="top-node-gnn">
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
