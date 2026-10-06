"use client";

import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AXIS_TICK, CHART_COLORS, SERIES_COLORS, SERIES_DASHES } from "./theme";

export interface PrPoint {
  recall: number;
  precision: number;
}

export interface PrSeries {
  name: string;
  points: PrPoint[];
  color?: string;
  dash?: string;
}

export interface PrReferenceLevel {
  name: string;
  value: number;
  color?: string;
  dash?: string;
}

export interface PrCurveChartProps {
  series: PrSeries[];
  referenceLevel?: PrReferenceLevel;
  height?: number;
  formatValue?: (value: number) => string;
  testId?: string;
}

export interface LegendEntry {
  name: string;
  color: string;
  dash: string;
}

const UNIT_TICKS = [0, 0.25, 0.5, 0.75, 1];
const REFERENCE_DASH = "4 4";

function formatUnit(value: number): string {
  return value.toLocaleString("de-DE", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

export function sortByRecall(points: PrPoint[]): PrPoint[] {
  return [...points].sort((a, b) => a.recall - b.recall);
}

export function referencePoints(value: number): PrPoint[] {
  return [
    { recall: 0, precision: value },
    { recall: 1, precision: value },
  ];
}

export function legendEntries(series: PrSeries[], referenceLevel?: PrReferenceLevel): LegendEntry[] {
  const entries = series.map((entry, index) => ({
    name: entry.name,
    color: entry.color ?? SERIES_COLORS[index % SERIES_COLORS.length] ?? CHART_COLORS.ink,
    dash: entry.dash ?? SERIES_DASHES[index % SERIES_DASHES.length] ?? "",
  }));
  if (!referenceLevel) return entries;
  return [
    ...entries,
    {
      name: referenceLevel.name,
      color: referenceLevel.color ?? CHART_COLORS.stone,
      dash: referenceLevel.dash ?? REFERENCE_DASH,
    },
  ];
}

function ChartLegend({ entries }: { entries: LegendEntry[] }) {
  return (
    <ul
      data-testid="pr-curve-legend"
      style={{
        display: "flex",
        flexWrap: "wrap",
        justifyContent: "center",
        gap: "4px 20px",
        listStyle: "none",
        margin: 0,
        padding: "0 0 8px",
        fontSize: 12,
        color: CHART_COLORS.ink,
      }}
    >
      {entries.map((entry) => (
        <li key={entry.name} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <svg width="28" height="10" viewBox="0 0 28 10" aria-hidden="true" focusable="false">
            <line
              x1="1"
              y1="5"
              x2="27"
              y2="5"
              stroke={entry.color}
              strokeWidth="2"
              strokeDasharray={entry.dash === "" ? undefined : entry.dash}
            />
          </svg>
          <span>{entry.name}</span>
        </li>
      ))}
    </ul>
  );
}

export function PrCurveChart({ series, referenceLevel, height = 320, formatValue = formatUnit, testId }: PrCurveChartProps) {
  const visible = series.filter((entry) => entry.points.length > 0);
  if (visible.length === 0) return null;
  const legend = legendEntries(visible, referenceLevel);
  return (
    <div style={{ width: "100%" }} data-testid={testId}>
      <ChartLegend entries={legend} />
      <div style={{ width: "100%", height }}>
        <ResponsiveContainer>
          <LineChart accessibilityLayer={false} margin={{ top: 12, right: 16, bottom: 24, left: 8 }}>
            <CartesianGrid stroke={CHART_COLORS.line} />
            <XAxis
              type="number"
              dataKey="recall"
              domain={[0, 1]}
              ticks={UNIT_TICKS}
              tick={AXIS_TICK}
              tickFormatter={(value: number) => formatValue(value)}
              tickLine={false}
              axisLine={{ stroke: CHART_COLORS.line }}
              label={{ value: "Recall", position: "insideBottom", offset: -16, fill: CHART_COLORS.stone, fontSize: 12 }}
            />
            <YAxis
              type="number"
              domain={[0, 1]}
              ticks={UNIT_TICKS}
              tick={AXIS_TICK}
              tickFormatter={(value: number) => formatValue(value)}
              tickLine={false}
              axisLine={false}
              width={56}
              label={{ value: "Precision", angle: -90, position: "insideLeft", fill: CHART_COLORS.stone, fontSize: 12 }}
            />
            <Tooltip
              cursor={{ stroke: CHART_COLORS.stone, strokeDasharray: "3 3" }}
              contentStyle={{ borderColor: CHART_COLORS.line, borderRadius: 6, fontSize: 12, color: CHART_COLORS.ink }}
              labelStyle={{ color: CHART_COLORS.stone }}
              itemStyle={{ color: CHART_COLORS.ink }}
              labelFormatter={(label) => `Recall ${formatValue(Number(label))}`}
              formatter={(value, name) => [formatValue(Number(value)), String(name)]}
            />
            {visible.map((entry, index) => {
              const style = legend[index];
              return (
                <Line
                  key={entry.name}
                  name={entry.name}
                  data={sortByRecall(entry.points)}
                  dataKey="precision"
                  type="stepBefore"
                  stroke={style?.color}
                  strokeDasharray={style?.dash}
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              );
            })}
            {referenceLevel ? (
              <Line
                key={referenceLevel.name}
                name={referenceLevel.name}
                data={referencePoints(referenceLevel.value)}
                dataKey="precision"
                type="linear"
                stroke={referenceLevel.color ?? CHART_COLORS.stone}
                strokeDasharray={referenceLevel.dash ?? REFERENCE_DASH}
                strokeWidth={1.5}
                dot={false}
                activeDot={false}
                isAnimationActive={false}
              />
            ) : null}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
