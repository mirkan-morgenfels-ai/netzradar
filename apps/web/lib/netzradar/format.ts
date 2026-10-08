import type { FeatureSet, Method, NodeLabel, StepRange } from "./types";

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

const EXPORTED_FLAG_TEXT = "als auffällig";
const FLAG_TEXT = "als markiert";

export function accuracyThresholdText(text: string): string {
  return text.replaceAll(EXPORTED_FLAG_TEXT, FLAG_TEXT);
}

export const METHOD_TEXT: Record<Method, string> = {
  zscore: "Robuste Z-Scores",
  iforest: "Isolation Forest",
  gcn: "GCN (Graph Convolutional Network)",
  graphsage: "GraphSAGE",
  mlp: "MLP ohne Kanten (Kontrolle)",
};

export const METHOD_SHORT_TEXT: Record<Method, string> = {
  zscore: "Z-Scores",
  iforest: "Isolation Forest",
  gcn: "GCN",
  graphsage: "GraphSAGE",
  mlp: "MLP (ohne Kanten)",
};

export const METHOD_KIND_TEXT: Record<Method, string> = {
  zscore: "Baseline ohne Labels",
  iforest: "Baseline ohne Labels",
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

export function joinList(items: readonly string[]): string {
  if (items.length <= 1) return items.join("");
  return `${items.slice(0, -1).join(", ")} und ${items[items.length - 1]}`;
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
