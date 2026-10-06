import type { ReactNode } from "react";
import { cx } from "./cx";

export interface StatTileProps {
  label: string;
  value: string;
  hint?: string | undefined;
  tone?: "neutral" | "positive" | "negative";
  children?: ReactNode;
}

export function StatTile({ label, value, hint, tone = "neutral", children }: StatTileProps) {
  const valueColor = tone === "positive" ? "text-moss" : tone === "negative" ? "text-wine" : "text-ink";
  return (
    <div className="rounded-lg border border-line bg-surface p-4">
      <div className="text-xs uppercase tracking-wide text-stone">{label}</div>
      <div className={cx("mt-1 font-serif text-2xl tabular-nums", valueColor)}>{value}</div>
      {hint ? <div className="mt-1 text-xs text-stone">{hint}</div> : null}
      {children}
    </div>
  );
}
