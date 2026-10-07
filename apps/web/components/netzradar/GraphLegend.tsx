import { CHART_COLORS } from "@portfolio/charts/theme";
import { LABEL_TEXT } from "@/lib/netzradar/format";
import type { NodeLabel } from "@/lib/netzradar/types";
import { NodeSymbol } from "./NodeSymbol";

const LABEL_ENTRIES: ReadonlyArray<{ label: NodeLabel; shape: string }> = [
  { label: "illicit", shape: "gefüllt, mit Goldring" },
  { label: "licit", shape: "gefüllt, ohne Ring" },
  { label: "unknown", shape: "hohl, kein Label" },
];

export function GraphLegend() {
  return (
    <div
      className="rounded-2xl border border-line bg-surface px-5 py-4 text-sm shadow-card sm:px-6"
      data-testid="graph-legend"
    >
      <h3 className="eyebrow">Legende: Label im Datensatz</h3>
      <div className="mt-3 flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between lg:gap-8">
        <ul className="flex flex-wrap gap-x-6 gap-y-2.5">
          {LABEL_ENTRIES.map((entry) => (
            <li key={entry.label} className="flex items-center gap-2">
              <NodeSymbol label={entry.label} size={16} />
              <span className="font-medium text-ink">{LABEL_TEXT[entry.label]}</span>
              <span className="text-xs text-slate">({entry.shape})</span>
            </li>
          ))}
        </ul>
        <ul className="flex flex-col gap-2 border-line text-xs text-slate lg:shrink-0 lg:border-l lg:pl-8">
          <li className="flex items-center gap-2.5">
            <svg width="30" height="18" viewBox="0 0 30 18" aria-hidden="true" focusable="false" className="shrink-0">
              <circle cx="5" cy="9" r="3" fill="none" stroke={CHART_COLORS.slate} strokeWidth="1.5" />
              <circle cx="20" cy="9" r="8" fill="none" stroke={CHART_COLORS.slate} strokeWidth="1.5" />
            </svg>
            großer Kreis: Startknoten, klein: Nachbar im Abstand 1 oder 2
          </li>
          <li className="flex items-center gap-2.5">
            <svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true" focusable="false" className="shrink-0">
              <line x1="1" y1="5" x2="22" y2="5" stroke={CHART_COLORS.slate} strokeWidth="1.5" />
              <path d="M22 1 L29 5 L22 9 Z" fill={CHART_COLORS.slate} />
            </svg>
            Pfeil: Richtung des Transaktionsflusses
          </li>
        </ul>
      </div>
    </div>
  );
}
