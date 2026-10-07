const GOLD = "#c9a548";
const GOLD_LIGHT = "#d8bd72";
const NAVY = "#0b1626";

const CENTER: readonly [number, number] = [31, 33];

const NODES: ReadonlyArray<readonly [number, number]> = [
  [19.3, 28.7],
  [40.6, 25],
  [33.4, 46.8],
];

export function BrandMark({ size, diamond = 2.6 }: { size: number; diamond?: number }) {
  const [cx, cy] = CENTER;
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
      <path d="M32 4.5 59.5 32 32 59.5 4.5 32Z" fill="none" stroke={GOLD} strokeWidth={diamond} />
      <path
        d={NODES.map(([x, y]) => `M${cx} ${cy} ${x} ${y}`).join("")}
        fill="none"
        stroke={GOLD_LIGHT}
        strokeWidth="2.6"
        strokeLinecap="round"
      />
      {NODES.map(([x, y]) => (
        <circle key={`${x}-${y}`} cx={x} cy={y} r="4.2" fill={NAVY} stroke={GOLD_LIGHT} strokeWidth="2.6" />
      ))}
      <circle cx={cx} cy={cy} r="5.4" fill={GOLD_LIGHT} />
    </svg>
  );
}
