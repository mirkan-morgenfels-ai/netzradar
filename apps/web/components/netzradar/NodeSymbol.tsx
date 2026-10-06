import { NODE_STYLES } from "@/lib/netzradar/graph";
import type { NodeLabel } from "@/lib/netzradar/types";

const MIN_RING_WIDTH = 1.5;

export function NodeSymbol({ label, size = 14 }: { label: NodeLabel; size?: number }) {
  const style = NODE_STYLES[label];
  const radius = size / 2;
  const ring = style.ringSize > 0 ? Math.max(MIN_RING_WIDTH, radius * style.ringSize) : 0;
  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      aria-hidden="true"
      focusable="false"
      className="inline-block shrink-0 align-[-2px]"
    >
      {ring > 0 ? <circle cx={radius} cy={radius} r={radius} fill={style.ring} /> : null}
      <circle cx={radius} cy={radius} r={radius - ring} fill={style.fill} />
    </svg>
  );
}
