import { describe, expect, it } from "vitest";
import {
  LABEL_TEXT,
  METHOD_KIND_TEXT,
  METHOD_SHORT_TEXT,
  METHOD_TEXT,
  formatDecimal,
  formatInteger,
  formatIsoDate,
  formatIsoTimeUtc,
  formatPercent,
  formatPlainNumber,
  formatShare,
  formatStepRange,
  joinList,
} from "../format";
import { METHODS } from "../types";

const NBSP = " ";

describe("formatDecimal", () => {
  it("uses a decimal comma and four digits by default", () => {
    expect(formatDecimal(0.3812)).toBe("0,3812");
    expect(formatDecimal(0.134)).toBe("0,1340");
    expect(formatDecimal(0)).toBe("0,0000");
  });

  it("groups thousands with a dot", () => {
    expect(formatDecimal(1234.5, 1)).toBe("1.234,5");
  });
});

describe("formatInteger", () => {
  it("groups thousands with a dot", () => {
    expect(formatInteger(12000)).toBe("12.000");
    expect(formatInteger(13757)).toBe("13.757");
    expect(formatInteger(95)).toBe("95");
  });
});

describe("formatPlainNumber", () => {
  it("keeps parameters without grouping", () => {
    expect(formatPlainNumber(0.04)).toBe("0,04");
    expect(formatPlainNumber(200)).toBe("200");
    expect(formatPlainNumber(12000)).toBe("12000");
  });
});

describe("formatPercent and formatShare", () => {
  it("writes German percentages with a non-breaking space", () => {
    expect(formatPercent(0.0243)).toBe(`2,4${NBSP}%`);
    expect(formatPercent(0.2156)).toBe(`21,6${NBSP}%`);
    expect(formatPercent(0.5, 0)).toBe(`50${NBSP}%`);
  });

  it("computes shares by hand", () => {
    expect(formatShare(292, 12000)).toBe(`2,4${NBSP}%`);
    expect(formatShare(2587, 12000)).toBe(`21,6${NBSP}%`);
    expect(formatShare(1, 0)).toBe(`0,0${NBSP}%`);
  });
});

describe("dates and ranges", () => {
  it("formats step ranges with an en dash", () => {
    expect(formatStepRange({ from: 22, to: 30 })).toBe("22–30");
    expect(formatStepRange({ from: 5, to: 5 })).toBe("5");
  });

  it("formats ISO timestamps without time zone shifts", () => {
    expect(formatIsoDate("2026-10-06T17:10:40Z")).toBe("06.10.2026");
    expect(formatIsoDate("2026-10-06")).toBe("06.10.2026");
    expect(formatIsoDate("gestern")).toBe("gestern");
    expect(formatIsoTimeUtc("2026-10-06T17:10:40Z")).toBe("17:10 UTC");
    expect(formatIsoTimeUtc("2026-10-06")).toBeNull();
  });
});

describe("LABEL_TEXT", () => {
  it("names the labels in German", () => {
    expect(LABEL_TEXT).toEqual({ illicit: "auffällig", licit: "unauffällig", unknown: "unbekannt" });
  });
});

describe("method texts", () => {
  it("names every method of the contract", () => {
    for (const texts of [METHOD_TEXT, METHOD_SHORT_TEXT, METHOD_KIND_TEXT]) {
      expect(Object.keys(texts)).toEqual([...METHODS]);
    }
    expect(METHOD_TEXT.mlp).toBe("MLP ohne Kanten (Kontrolle)");
    expect(METHOD_SHORT_TEXT.graphsage).toBe("GraphSAGE");
  });
});

describe("joinList", () => {
  it("joins German lists with commas and und", () => {
    expect(joinList([])).toBe("");
    expect(joinList(["GCN"])).toBe("GCN");
    expect(joinList(["GCN", "GraphSAGE"])).toBe("GCN und GraphSAGE");
    expect(joinList(["Z-Scores", "GCN", "MLP"])).toBe("Z-Scores, GCN und MLP");
  });
});

describe("negative decimals", () => {
  it("keeps the sign of logit differences", () => {
    expect(formatDecimal(-19.9567)).toBe("-19,9567");
    expect(formatDecimal(17.2586)).toBe("17,2586");
  });
});
