"use client";

import { cx } from "@portfolio/ui";
import { formatInteger, seedSetButtonText, seedSetScoreText } from "@/lib/netzradar/format";
import { seedNodes } from "@/lib/netzradar/graph";
import { excerptCounts } from "@/lib/netzradar/summary";
import { SEED_SETS, type NetNode, type SeedSet } from "@/lib/netzradar/types";

export interface SeedSetSwitchProps {
  nodes: readonly NetNode[];
  active: SeedSet;
  gnnLabel: string;
  hops: number;
  onChange: (set: SeedSet) => void;
}

export function SeedSetSwitch({ nodes, active, gnnLabel, hops, onChange }: SeedSetSwitchProps) {
  const counts = excerptCounts(nodes);
  const seeds = seedNodes(nodes, active).length;
  const other: SeedSet = active === "gnn" ? "iforest" : "gnn";
  const onlyOther = active === "gnn" ? counts.total - counts.gnn : counts.total - counts.iforest;
  return (
    <div data-testid="seed-set-switch" data-active={active}>
      <div className="flex flex-col gap-2.5 sm:flex-row sm:items-center sm:gap-5">
        <p id="seed-set-label" className="eyebrow shrink-0">
          Startknoten nach Score
        </p>
        <div
          role="group"
          aria-labelledby="seed-set-label"
          aria-describedby="seed-set-note"
          className="grid grid-cols-1 gap-1 rounded-2xl border border-line bg-surface p-1 shadow-[0_1px_2px_rgb(11_22_38/0.04)] min-[360px]:inline-flex min-[360px]:w-fit min-[360px]:gap-0.5 min-[360px]:rounded-full"
        >
          {SEED_SETS.map((set) => {
            const pressed = set === active;
            return (
              <button
                key={set}
                type="button"
                aria-pressed={pressed}
                data-testid={`seed-set-${set}`}
                onClick={() => onChange(set)}
                className={cx(
                  "rounded-xl px-4 py-1.5 text-[0.8125rem] whitespace-nowrap transition-colors duration-150 min-[360px]:rounded-full sm:text-sm",
                  pressed
                    ? "bg-navy-950 font-medium text-ivory shadow-[0_2px_8px_rgb(11_22_38/0.18)]"
                    : "text-slate hover:bg-ivory hover:text-ink",
                )}
              >
                {seedSetButtonText(set, gnnLabel)}
              </button>
            );
          })}
        </div>
      </div>
      <p id="seed-set-note" className="mt-2.5 max-w-[72ch] text-xs leading-relaxed text-slate" data-testid="seed-set-note">
        Groß gezeichnet und in der Tabelle: die {formatInteger(seeds)} Testknoten mit dem höchsten{" "}
        {seedSetScoreText(active, gnnLabel)}, kleiner ihre Nachbarn bis {hops} Kanten Abstand.{" "}
        {onlyOther > 0
          ? `${formatInteger(onlyOther)} Knoten liegen nur in der Nachbarschaft der Startknoten nach ${seedSetScoreText(
              other,
              gnnLabel,
            )} und erscheinen am kleinsten.`
          : ""}
        {active === "gnn"
          ? ` Der GNN-Score ist eine Logit-Differenz; 0 ist dabei keine kalibrierte Schwelle, gewählt sind schlicht die ${formatInteger(
              seeds,
            )} höchsten Scores.`
          : ""}
      </p>
      <p className="sr-only" aria-live="polite">
        Tabelle und Graph zeigen die Startknoten nach {seedSetScoreText(active, gnnLabel)}.
      </p>
    </div>
  );
}
