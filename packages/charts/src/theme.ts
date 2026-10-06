export const CHART_COLORS = {
  ink: "#111111",
  paper: "#fbfaf6",
  surface: "#ffffff",
  gold: "#b8912f",
  goldDeep: "#7d5f17",
  goldSoft: "#f3e9c9",
  moss: "#2f6b3a",
  mossSoft: "#dfeadf",
  wine: "#7a1f2b",
  wineSoft: "#f1dcdf",
  stone: "#6b6b66",
  line: "#e3e0d6",
} as const;

export const SERIES_COLORS = [
  CHART_COLORS.wine,
  CHART_COLORS.gold,
  CHART_COLORS.moss,
  CHART_COLORS.ink,
  CHART_COLORS.stone,
] as const;

export const SERIES_DASHES = ["", "6 3", "2 3", "8 3 2 3", "1 2"] as const;

export const AXIS_TICK = { fill: CHART_COLORS.stone, fontSize: 12 } as const;
