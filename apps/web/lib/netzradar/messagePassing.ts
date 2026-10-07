export type NeighbourList = ReadonlyArray<readonly number[]>;

export interface PropagationRow {
  key: string;
  label: string;
  values: number[];
}

export const CHAIN_LABELS = ["A", "B", "C", "D"] as const;
export const CHAIN_START: readonly number[] = [1, 0, 0, 1];

export function chainNeighbours(length: number): number[][] {
  return Array.from({ length }, (_, index) =>
    [index - 1, index + 1].filter((neighbour) => neighbour >= 0 && neighbour < length),
  );
}

function valueAt(values: readonly number[], index: number): number {
  const value = values[index];
  if (value === undefined) throw new RangeError(`Kein Wert für Knoten ${index}`);
  return value;
}

export function meanWithSelf(neighbours: NeighbourList, x: readonly number[]): number[] {
  return neighbours.map((list, node) => {
    const total = list.reduce((sum, neighbour) => sum + valueAt(x, neighbour), valueAt(x, node));
    return total / (list.length + 1);
  });
}

export function gcnPropagate(neighbours: NeighbourList, x: readonly number[]): number[] {
  const degree = neighbours.map((list) => list.length + 1);
  return neighbours.map((list, node) =>
    [node, ...list].reduce(
      (sum, source) => sum + valueAt(x, source) / Math.sqrt(valueAt(degree, source) * valueAt(degree, node)),
      0,
    ),
  );
}

export function sagePropagate(neighbours: NeighbourList, x: readonly number[]): number[] {
  return neighbours.map((list, node) => {
    const mean = list.length === 0 ? 0 : list.reduce((sum, neighbour) => sum + valueAt(x, neighbour), 0) / list.length;
    return valueAt(x, node) + mean;
  });
}

export function mlpPropagate(x: readonly number[]): number[] {
  return [...x];
}

export function chainExample(): PropagationRow[] {
  const neighbours = chainNeighbours(CHAIN_LABELS.length);
  return [
    { key: "start", label: "Startwert x", values: [...CHAIN_START] },
    { key: "mean", label: "Mittelwert über N(v) ∪ {v}", values: meanWithSelf(neighbours, CHAIN_START) },
    { key: "gcn", label: "GCN", values: gcnPropagate(neighbours, CHAIN_START) },
    { key: "graphsage", label: "GraphSAGE", values: sagePropagate(neighbours, CHAIN_START) },
    { key: "mlp", label: "MLP ohne Kanten", values: mlpPropagate(CHAIN_START) },
  ];
}
