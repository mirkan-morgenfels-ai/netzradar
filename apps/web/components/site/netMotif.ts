export type MotifPoint = readonly [number, number];

export const NET_NODES: readonly MotifPoint[] = [
  [36, 252],
  [88, 204],
  [118, 292],
  [166, 236],
  [146, 142],
  [212, 182],
  [238, 290],
  [266, 108],
  [300, 220],
  [334, 306],
  [352, 156],
  [392, 254],
  [404, 70],
  [430, 204],
  [470, 300],
  [538, 234],
  [552, 92],
  [600, 172],
  [606, 282],
  [616, 40],
  [516, 336],
  [236, 34],
  [62, 118],
  [184, 336],
];

export const NET_FLAGGED: MotifPoint = [488, 150];

export const NET_EDGES: ReadonlyArray<readonly [number, number]> = [
  [0, 1],
  [0, 2],
  [1, 3],
  [1, 4],
  [2, 3],
  [3, 5],
  [3, 6],
  [4, 5],
  [4, 22],
  [5, 7],
  [5, 8],
  [6, 8],
  [6, 9],
  [7, 10],
  [7, 12],
  [8, 10],
  [8, 11],
  [9, 11],
  [10, 13],
  [11, 13],
  [11, 14],
  [12, 16],
  [14, 15],
  [15, 18],
  [17, 18],
  [16, 19],
  [17, 19],
  [14, 20],
  [18, 20],
  [21, 7],
  [21, 4],
  [23, 6],
  [23, 9],
  [22, 1],
];

export const NET_FAN_IN: readonly number[] = [10, 12, 13, 15, 16, 17];

export const NET_RINGS: readonly number[] = [46, 92, 146, 210];

export const NET_SWEEP = { radius: 150, from: -152, to: -112 } as const;

export function netPoint(index: number): MotifPoint {
  return NET_NODES[index] ?? NET_FLAGGED;
}

export function netNeighbours(): Set<number> {
  const fanIn = new Set(NET_FAN_IN);
  const near = new Set<number>();
  for (const [a, b] of NET_EDGES) {
    if (fanIn.has(a) && !fanIn.has(b)) near.add(b);
    if (fanIn.has(b) && !fanIn.has(a)) near.add(a);
  }
  return near;
}

export function sweepWedge(radius: number, from: number, to: number): string {
  const [x, y] = NET_FLAGGED;
  const start = [x + radius * Math.cos((from * Math.PI) / 180), y + radius * Math.sin((from * Math.PI) / 180)];
  const end = [x + radius * Math.cos((to * Math.PI) / 180), y + radius * Math.sin((to * Math.PI) / 180)];
  const round = (value: number | undefined) => Math.round((value ?? 0) * 10) / 10;
  return `M${x} ${y} L${round(start[0])} ${round(start[1])} A${radius} ${radius} 0 0 1 ${round(end[0])} ${round(end[1])} Z`;
}

export function sweepEdge(radius: number, angle: number): MotifPoint {
  const [x, y] = NET_FLAGGED;
  return [
    Math.round((x + radius * Math.cos((angle * Math.PI) / 180)) * 10) / 10,
    Math.round((y + radius * Math.sin((angle * Math.PI) / 180)) * 10) / 10,
  ];
}
