export const CHART_COLORS = {
  navy: "#1d3a5f",
  gold: "#b8912f",
  goldLine: "#a8832a",
  goldDeep: "#7d5f17",
  moss: "#2f6b3a",
  wine: "#7a1f2b",
  sky: "#3e6a9e",
  slate: "#5b6474",
  sand: "#c9b98f",
  ink: "#0f1b2d",
  line: "#e4ddcc",
  lineStrong: "#858d9b",
  grid: "#ece6d8",
  surface: "#ffffff",
} as const;

export const SERIES_COLORS: readonly string[] = [
  CHART_COLORS.navy,
  CHART_COLORS.gold,
  CHART_COLORS.moss,
  CHART_COLORS.wine,
  CHART_COLORS.sky,
  CHART_COLORS.slate,
  CHART_COLORS.sand,
];

export function seriesColor(index: number): string {
  return SERIES_COLORS[index % SERIES_COLORS.length] ?? CHART_COLORS.navy;
}

export const SERIES_DASHES: readonly string[] = ["", "6 3", "2 3", "8 3 2 3", "1 2", "12 4", "4 2 1 2"];

export const AXIS_TICK = { fill: CHART_COLORS.slate, fontSize: 12 } as const;

export const TOOLTIP_CONTENT_STYLE = {
  backgroundColor: CHART_COLORS.surface,
  border: `1px solid ${CHART_COLORS.line}`,
  borderRadius: 12,
  boxShadow: "0 2px 6px rgb(11 22 38 / 0.06), 0 12px 32px rgb(11 22 38 / 0.12)",
  padding: "10px 14px",
  fontSize: 13,
  color: CHART_COLORS.ink,
} as const;

export const TOOLTIP_LABEL_STYLE = {
  color: CHART_COLORS.slate,
  fontSize: 11,
  letterSpacing: "0.08em",
  textTransform: "uppercase",
  marginBottom: 4,
} as const;

export const TOOLTIP_ITEM_STYLE = { color: CHART_COLORS.ink, padding: 0 } as const;
