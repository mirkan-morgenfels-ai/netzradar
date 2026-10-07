import { describe, expect, it } from "vitest";
import {
  CHAIN_START,
  chainExample,
  chainNeighbours,
  gcnPropagate,
  meanWithSelf,
  mlpPropagate,
  sagePropagate,
} from "../messagePassing";

const CHAIN = chainNeighbours(4);

function expectValues(actual: readonly number[], expected: readonly number[]): void {
  expect(actual).toHaveLength(expected.length);
  expected.forEach((value, index) => expect(actual[index]).toBeCloseTo(value, 6));
}

describe("chainNeighbours", () => {
  it("builds the undirected chain A - B - C - D", () => {
    expect(CHAIN).toEqual([[1], [0, 2], [1, 3], [2]]);
    expect(chainNeighbours(1)).toEqual([[]]);
  });
});

describe("one round of message passing on the chain", () => {
  it("averages over the node and its neighbours", () => {
    expectValues(meanWithSelf(CHAIN, CHAIN_START), [1 / 2, 1 / 3, 1 / 3, 1 / 2]);
    expectValues(meanWithSelf(CHAIN, meanWithSelf(CHAIN, CHAIN_START)), [5 / 12, 7 / 18, 7 / 18, 5 / 12]);
  });

  it("normalises GCN messages by the square root of both degrees", () => {
    expectValues(gcnPropagate(CHAIN, CHAIN_START), [0.5, 0.408248, 0.408248, 0.5]);
    expectValues(gcnPropagate(CHAIN, CHAIN_START), [1 / 2, 1 / Math.sqrt(6), 1 / Math.sqrt(6), 1 / 2]);
    expectValues(gcnPropagate(CHAIN, gcnPropagate(CHAIN, CHAIN_START)), [0.416667, 0.47629, 0.47629, 0.416667]);
  });

  it("keeps the own features apart in GraphSAGE", () => {
    expectValues(sagePropagate(CHAIN, CHAIN_START), [1, 0.5, 0.5, 1]);
    expectValues(sagePropagate([[]], [3]), [3]);
  });

  it("ignores the neighbours in the MLP", () => {
    expectValues(mlpPropagate(CHAIN_START), [1, 0, 0, 1]);
  });

  it("collects the rows for the page", () => {
    const rows = chainExample();
    expect(rows.map((row) => row.key)).toEqual(["start", "mean", "gcn", "graphsage", "mlp"]);
    expectValues(rows.find((row) => row.key === "gcn")?.values ?? [], [0.5, 0.408248, 0.408248, 0.5]);
  });

  it("rejects a neighbour without a value", () => {
    expect(() => gcnPropagate([[1]], [1])).toThrow(RangeError);
  });
});
