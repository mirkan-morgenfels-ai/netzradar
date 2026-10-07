import { CHART_COLORS } from "@portfolio/charts/theme";

export function BrandMark({ size, rounded = true }: { size: number; rounded?: boolean }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
      <rect width="64" height="64" rx={rounded ? 14 : 0} fill={CHART_COLORS.ink} />
      <path
        d="M18 44L32 20L46 44Z M32 20V36 M18 44L32 36L46 44"
        fill="none"
        stroke={CHART_COLORS.gold}
        strokeWidth="2.5"
        strokeLinejoin="round"
      />
      <circle cx="32" cy="20" r="4.5" fill={CHART_COLORS.gold} />
      <circle cx="18" cy="44" r="4.5" fill={CHART_COLORS.gold} />
      <circle cx="46" cy="44" r="4.5" fill={CHART_COLORS.gold} />
      <circle cx="32" cy="36" r="5" fill={CHART_COLORS.wine} stroke={CHART_COLORS.paper} strokeWidth="1.5" />
    </svg>
  );
}
