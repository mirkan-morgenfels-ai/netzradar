import type { FeatureSet, Method, NodeLabel, SeedSet, StepRange } from "./types";

const LOCALE = "de-DE";
const MINUS_SIGN = "−";
const ISO_PARTS_PATTERN = /^(\d{4})-(\d{2})-(\d{2})(?:T(\d{2}):(\d{2}))?/;

export const LABEL_TEXT: Record<NodeLabel, string> = {
  illicit: "illegal",
  licit: "legal",
  unknown: "ohne Label",
};

export const LABEL_PHRASE: Record<NodeLabel, string> = {
  illicit: "Label illegal",
  licit: "Label legal",
  unknown: "ohne Label",
};

export const METHOD_TEXT: Record<Method, string> = {
  zscore: "Robuste Z-Scores",
  iforest: "Isolation Forest",
  hgb: "Gradient Boosting (ohne Kanten)",
  gcn: "GCN (Graph Convolutional Network)",
  graphsage: "GraphSAGE",
  mlp: "MLP ohne Kanten (Kontrolle)",
};

export const METHOD_SHORT_TEXT: Record<Method, string> = {
  zscore: "Z-Scores",
  iforest: "Isolation Forest",
  hgb: "Gradient Boosting",
  gcn: "GCN",
  graphsage: "GraphSAGE",
  mlp: "MLP (ohne Kanten)",
};

export const METHOD_KIND_TEXT: Record<Method, string> = {
  zscore: "Baseline ohne Labels",
  iforest: "Baseline ohne Labels",
  hgb: "Starkes Verfahren ohne Kanten: Entscheidungsbäume, mit Labels trainiert",
  gcn: "Graph Neural Network, mit Labels trainiert, bezieht die Nachbarschaft ein",
  graphsage: "Graph Neural Network, mit Labels trainiert, bezieht die Nachbarschaft ein",
  mlp: "Kontrolle: gleiche Größe, Labels und Auswahl, aber ohne Kanten",
};

export const FEATURE_SET_TEXT: Record<FeatureSet, string> = {
  local: "nur lokale Merkmale",
  "local+graph": "lokale Merkmale und Graphmaße",
};

export function withMinusSign(text: string): string {
  return text.startsWith("-") ? `${MINUS_SIGN}${text.slice(1)}` : text;
}

export function formatDecimal(value: number, digits = 4): string {
  return withMinusSign(
    value.toLocaleString(LOCALE, {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
      useGrouping: true,
    }),
  );
}

export function formatInteger(value: number): string {
  return Math.round(value).toLocaleString(LOCALE, { maximumFractionDigits: 0, useGrouping: true });
}

export function formatPlainNumber(value: number): string {
  return withMinusSign(value.toLocaleString(LOCALE, { maximumFractionDigits: 6, useGrouping: false }));
}

export function formatPercent(value: number, digits = 1): string {
  return value.toLocaleString(LOCALE, {
    style: "percent",
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function formatShare(count: number, total: number, digits = 1): string {
  if (total <= 0) return formatPercent(0, digits);
  return formatPercent(count / total, digits);
}

export function joinList(items: readonly string[], conjunction: "und" | "oder" = "und"): string {
  if (items.length <= 1) return items.join("");
  return `${items.slice(0, -1).join(", ")} ${conjunction} ${items[items.length - 1]}`;
}

export function formatSeedList(seeds: readonly number[]): string {
  const sorted = [...seeds].sort((a, b) => a - b);
  const first = sorted[0];
  const last = sorted[sorted.length - 1];
  if (first === undefined || last === undefined) return "";
  const consecutive = sorted.length > 2 && sorted.every((seed, index) => seed === first + index);
  return consecutive ? `${first} bis ${last}` : joinList(sorted.map(String));
}

export function seedSetButtonText(set: SeedSet, gnnLabel: string): string {
  return set === "gnn" ? `GNN (${gnnLabel})` : "Isolation Forest";
}

export function seedSetScoreText(set: SeedSet, gnnLabel: string): string {
  return set === "gnn" ? `GNN-Score (${gnnLabel})` : "Isolation-Forest-Score";
}

export function formatStepRange(range: StepRange): string {
  return range.from === range.to ? `${range.from}` : `${range.from}–${range.to}`;
}

export function formatIsoDate(iso: string): string {
  const match = ISO_PARTS_PATTERN.exec(iso);
  if (!match) return iso;
  const [, year, month, day] = match;
  return `${day}.${month}.${year}`;
}

export function formatDateRange(range: { from: string; to: string }): string {
  const from = formatIsoDate(range.from);
  const to = formatIsoDate(range.to);
  return from === to ? from : `${from} bis ${to}`;
}

export function formatIsoTimeUtc(iso: string): string | null {
  const match = ISO_PARTS_PATTERN.exec(iso);
  if (!match || match[4] === undefined || match[5] === undefined) return null;
  return `${match[4]}:${match[5]} UTC`;
}
