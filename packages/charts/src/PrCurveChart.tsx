"use client";

import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, type TooltipContentProps } from "recharts";
import {
  AXIS_TICK,
  CHART_COLORS,
  SERIES_COLORS,
  SERIES_DASHES,
  TOOLTIP_CONTENT_STYLE,
  TOOLTIP_LABEL_STYLE,
} from "./theme";

export interface PrPoint {
  recall: number;
  precision: number;
}

export interface PrSeries {
  name: string;
  points: PrPoint[];
  color?: string;
  dash?: string;
  width?: number;
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
  width: number;
}

const UNIT_TICKS = [0, 0.25, 0.5, 0.75, 1];
const REFERENCE_DASH = "4 4";
const LINE_WIDTH = 2;
const REFERENCE_WIDTH = 1.5;

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
    width: entry.width ?? LINE_WIDTH,
  }));
  if (!referenceLevel) return entries;
  return [
    ...entries,
    {
      name: referenceLevel.name,
      color: referenceLevel.color ?? CHART_COLORS.slate,
      dash: referenceLevel.dash ?? REFERENCE_DASH,
      width: REFERENCE_WIDTH,
    },
  ];
}

export function tooltipOrder(series: PrSeries[]): (name: unknown) => number {
  const order = series.map((entry) => entry.name);
  return (name) => {
    const index = order.indexOf(String(name));
    return index === -1 ? order.length : index;
  };
}

function LineSample({ entry, width, height }: { entry: LegendEntry; width: number; height: number }) {
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-hidden="true" focusable="false">
      <line
        x1="0"
        y1={height / 2}
        x2={width}
        y2={height / 2}
        stroke={entry.color}
        strokeWidth={Math.min(entry.width, height)}
        strokeDasharray={entry.dash === "" ? undefined : entry.dash}
      />
    </svg>
  );
}

function TooltipBody({
  active,
  payload,
  label,
  entries,
  rank,
  formatValue,
}: Pick<TooltipContentProps, "active" | "payload" | "label"> & {
  entries: LegendEntry[];
  rank: (name: unknown) => number;
  formatValue: (value: number) => string;
}) {
  if (!active || payload.length === 0) return null;
  const items = payload
    .filter((item) => entries.some((entry) => entry.name === String(item.name)))
    .sort((a, b) => rank(a.name) - rank(b.name));
  if (items.length === 0) return null;
  return (
    <div style={TOOLTIP_CONTENT_STYLE}>
      <p style={{ ...TOOLTIP_LABEL_STYLE, margin: "0 0 6px" }}>Recall {formatValue(Number(label))}</p>
      <ul style={{ listStyle: "none", margin: 0, padding: 0, display: "grid", gap: 3 }}>
        {items.map((item) => {
          const entry = entries.find((candidate) => candidate.name === String(item.name));
          return (
            <li key={String(item.name)} style={{ display: "flex", alignItems: "center", gap: 8, color: CHART_COLORS.ink }}>
              {entry ? <LineSample entry={entry} width={16} height={2} /> : null}
              <span>
                {String(item.name)}: {formatValue(Number(item.value))}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function ChartLegend({ entries }: { entries: LegendEntry[] }) {
  return (
    <ul
      data-testid="pr-curve-legend"
      style={{
        display: "flex",
        flexWrap: "wrap",
        justifyContent: "flex-start",
        gap: "8px 22px",
        listStyle: "none",
        margin: 0,
        padding: "0 0 16px",
        fontSize: 12.5,
        lineHeight: 1.4,
        color: CHART_COLORS.ink,
      }}
    >
      {entries.map((entry) => (
        <li key={entry.name} style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
          <LineSample entry={entry} width={28} height={10} />
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
  const rank = tooltipOrder(visible);
  const seriesLegend = legend.slice(0, visible.length);
  return (
    <div style={{ width: "100%" }} data-testid={testId}>
      <ChartLegend entries={legend} />
      <div style={{ width: "100%", height }}>
        <ResponsiveContainer>
          <LineChart accessibilityLayer={false} margin={{ top: 12, right: 16, bottom: 24, left: 8 }}>
            <CartesianGrid stroke={CHART_COLORS.grid} vertical={false} />
            <XAxis
              type="number"
              dataKey="recall"
              domain={[0, 1]}
              ticks={UNIT_TICKS}
              tick={AXIS_TICK}
              tickFormatter={(value: number) => formatValue(value)}
              tickLine={false}
              axisLine={{ stroke: CHART_COLORS.line }}
              tickMargin={8}
              label={{ value: "Recall", position: "insideBottom", offset: -16, fill: CHART_COLORS.slate, fontSize: 12 }}
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
              label={{ value: "Precision", angle: -90, position: "insideLeft", fill: CHART_COLORS.slate, fontSize: 12 }}
            />
            <Tooltip
              cursor={{ stroke: CHART_COLORS.gold, strokeWidth: 1, strokeDasharray: "3 4" }}
              itemSorter={(item) => rank(item.name)}
              content={(props) => (
                <TooltipBody
                  active={props.active}
                  payload={props.payload}
                  label={props.label}
                  entries={seriesLegend}
                  rank={rank}
                  formatValue={formatValue}
                />
              )}
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
                  strokeWidth={style?.width ?? LINE_WIDTH}
                  dot={false}
                  activeDot={{ r: 4, stroke: CHART_COLORS.surface, strokeWidth: 2 }}
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
                stroke={referenceLevel.color ?? CHART_COLORS.slate}
                strokeDasharray={referenceLevel.dash ?? REFERENCE_DASH}
                strokeWidth={REFERENCE_WIDTH}
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
