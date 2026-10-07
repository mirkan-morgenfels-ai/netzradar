import {
  NET_EDGES,
  NET_FAN_IN,
  NET_FLAGGED,
  NET_NODES,
  NET_RINGS,
  NET_SWEEP,
  netNeighbours,
  netPoint,
  sweepEdge,
  sweepWedge,
} from "./netMotif";

const GOLD = "#c9a548";
const GOLD_LIGHT = "#d8bd72";
const NAVY = "#0b1626";

const MARK_CENTER: readonly [number, number] = [15.5, 16.5];

const MARK_NODES: ReadonlyArray<readonly [number, number]> = [
  [10.4, 14.6],
  [19.8, 12.8],
  [16.6, 22.6],
];

export function BrandMark({ className }: { className?: string }) {
  const [cx, cy] = MARK_CENTER;
  return (
    <svg viewBox="0 0 32 32" aria-hidden="true" focusable="false" className={className}>
      <path d="M16 1.75 30.25 16 16 30.25 1.75 16Z" fill="none" stroke="currentColor" strokeWidth="1.4" />
      <path
        d={MARK_NODES.map(([x, y]) => `M${cx} ${cy} ${x} ${y}`).join("")}
        fill="none"
        stroke="currentColor"
        strokeWidth="1.1"
        strokeOpacity="0.9"
        strokeLinecap="round"
      />
      {MARK_NODES.map(([x, y]) => (
        <circle key={`${x}-${y}`} cx={x} cy={y} r="1.8" fill={NAVY} stroke="currentColor" strokeWidth="1.1" />
      ))}
      <circle cx={cx} cy={cy} r="2.3" fill="currentColor" />
    </svg>
  );
}

export function HeroOrnament({ idPrefix, className }: { idPrefix: string; className?: string }) {
  const fadeId = `${idPrefix}-fade`;
  const maskId = `${idPrefix}-mask`;
  const sweepId = `${idPrefix}-sweep`;
  const near = netNeighbours();
  const fanIn = new Set(NET_FAN_IN);
  const [fx, fy] = NET_FLAGGED;
  const [sx, sy] = sweepEdge(NET_SWEEP.radius, NET_SWEEP.to);
  return (
    <svg
      viewBox="0 0 640 360"
      preserveAspectRatio="xMidYMid meet"
      aria-hidden="true"
      focusable="false"
      className={className}
    >
      <defs>
        <linearGradient id={fadeId} x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stopColor="#fff" stopOpacity="0" />
          <stop offset="0.18" stopColor="#fff" stopOpacity="1" />
          <stop offset="1" stopColor="#fff" stopOpacity="1" />
        </linearGradient>
        <mask id={maskId}>
          <rect width="640" height="360" fill={`url(#${fadeId})`} />
        </mask>
        <radialGradient id={sweepId} cx={fx} cy={fy} r={NET_SWEEP.radius} gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor={GOLD} stopOpacity="0.22" />
          <stop offset="1" stopColor={GOLD} stopOpacity="0" />
        </radialGradient>
      </defs>
      <g mask={`url(#${maskId})`}>
        {NET_RINGS.map((radius, index) => (
          <circle
            key={radius}
            cx={fx}
            cy={fy}
            r={radius}
            fill="none"
            stroke={GOLD}
            strokeOpacity={0.24 - index * 0.03}
            strokeWidth="1"
            strokeDasharray="2 7"
            vectorEffect="non-scaling-stroke"
          />
        ))}
        <path d={sweepWedge(NET_SWEEP.radius, NET_SWEEP.from, NET_SWEEP.to)} fill={`url(#${sweepId})`} />
        <line
          x1={fx}
          y1={fy}
          x2={sx}
          y2={sy}
          stroke={GOLD_LIGHT}
          strokeOpacity="0.4"
          strokeWidth="1"
          vectorEffect="non-scaling-stroke"
        />
        {NET_EDGES.map(([a, b]) => {
          const [x1, y1] = netPoint(a);
          const [x2, y2] = netPoint(b);
          const close = near.has(a) || near.has(b) || fanIn.has(a) || fanIn.has(b);
          return (
            <line
              key={`${a}-${b}`}
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              stroke={GOLD}
              strokeOpacity={close ? 0.45 : 0.25}
              strokeWidth="1"
              vectorEffect="non-scaling-stroke"
            />
          );
        })}
        {NET_FAN_IN.map((index) => {
          const [x, y] = netPoint(index);
          return (
            <path
              key={index}
              d={`M${x} ${y} L${fx} ${fy}`}
              pathLength={1}
              className="draw-line"
              fill="none"
              stroke={GOLD_LIGHT}
              strokeWidth="1.5"
              strokeLinecap="round"
              vectorEffect="non-scaling-stroke"
            />
          );
        })}
        {NET_NODES.map(([x, y], index) => {
          const hot = fanIn.has(index);
          const close = near.has(index);
          return (
            <circle
              key={`${x}-${y}`}
              cx={x}
              cy={y}
              r={hot ? 3.5 : 3}
              fill={NAVY}
              stroke={hot ? GOLD_LIGHT : GOLD}
              strokeOpacity={hot ? 1 : close ? 0.8 : 0.5}
              strokeWidth="1.25"
              vectorEffect="non-scaling-stroke"
            />
          );
        })}
      </g>
      <circle cx={fx} cy={fy} r="22" fill="none" stroke={GOLD_LIGHT} strokeOpacity="0.16" strokeWidth="1" vectorEffect="non-scaling-stroke" />
      <circle cx={fx} cy={fy} r="12" fill={NAVY} stroke={GOLD_LIGHT} strokeOpacity="0.45" strokeWidth="1" vectorEffect="non-scaling-stroke" />
      <circle cx={fx} cy={fy} r="5" fill={GOLD_LIGHT} />
    </svg>
  );
}
