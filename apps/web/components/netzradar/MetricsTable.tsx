import { cx } from "@portfolio/ui";
import { FEATURE_SET_TEXT, formatDecimal, formatInteger, METHOD_KIND_TEXT, METHOD_TEXT } from "@/lib/netzradar/format";
import { featureNames, labelledTestCount, metricRows, zeroMadFeatures } from "@/lib/netzradar/summary";
import type { Metrics, Run } from "@/lib/netzradar/types";
import { ScrollRegion } from "./ScrollRegion";

const CELL = "px-4 py-3";
const NUMBER_CELL = cx(CELL, "text-right tabular-nums");

function featureText(run: Run): string {
  const count = featureNames(run).length;
  const inactive = zeroMadFeatures(run).length;
  const base = FEATURE_SET_TEXT[run.featureSet];
  if (count === 0) return base;
  if (inactive === 0) return `${base}, ${count} Merkmale`;
  return `${base}, ${count} Merkmale, davon ${count - inactive} wirksam (MAD = 0 bei ${inactive})`;
}

function trainingText(run: Run): string {
  const { training } = run;
  if (training === null) return "ohne Labels angepasst";
  const neighbourhood =
    training.edges === "none" ? "ohne Kanten" : `Nachbarschaft bis ${training.layers} Kanten Abstand`;
  return `mit Labels trainiert, ${neighbourhood}`;
}

export function MetricsTable({ metrics }: { metrics: Metrics }) {
  const rows = metricRows(metrics.runs);
  const { evaluation, split } = metrics;
  return (
    <div className="space-y-2">
      <p id="metrics-table-note" className="text-xs text-stone">
        Testzeitraum Zeitschritte {split.test.from} bis {split.test.to}: {formatInteger(labelledTestCount(evaluation))}{" "}
        Knoten mit Label, davon {formatInteger(evaluation.testPositives)} auffällig. Knoten ohne Label sind nicht
        bewertet. Seed {metrics.seed}.
      </p>
      <ScrollRegion labelledBy="metrics-table-caption" className="rounded-lg border border-line bg-surface">
        <table
          className="w-full min-w-[46rem] border-collapse text-sm"
          data-testid="metrics-table"
          aria-describedby="metrics-table-note"
        >
          <caption id="metrics-table-caption" className="sr-only">
            Kennzahlen je Verfahren auf den Testknoten mit Label
          </caption>
          <thead>
            <tr className="border-b border-line text-left text-xs uppercase tracking-wide text-stone">
              <th scope="col" className={cx(CELL, "font-medium")}>
                Verfahren
              </th>
              <th scope="col" className={cx(CELL, "text-right font-medium")}>
                PR-AUC
              </th>
              <th scope="col" className={cx(CELL, "text-right font-medium")}>
                Precision bei Recall ≥ 0,5
              </th>
              <th scope="col" className={cx(CELL, "text-right font-medium")}>
                Recall bei Precision ≥ 0,5
              </th>
              <th scope="col" className={cx(CELL, "text-right font-normal normal-case tracking-normal")}>
                Accuracy (Nebenwert)
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) =>
              row.kind === "measured" ? (
                <tr key={row.method} data-testid={`metrics-row-${row.method}`} className="border-t border-line">
                  <th scope="row" className={cx(CELL, "text-left font-normal")}>
                    <span className="font-medium">{row.run.displayName}</span>
                    <span className="block text-xs text-stone">{featureText(row.run)}</span>
                    <span className="block text-xs text-stone">{trainingText(row.run)}</span>
                  </th>
                  <td className={cx(NUMBER_CELL, "font-semibold")} data-testid={`pr-auc-${row.method}`}>
                    {formatDecimal(row.run.prAuc)}
                  </td>
                  <td className={NUMBER_CELL}>{formatDecimal(row.run.precisionAtRecall50)}</td>
                  <td className={NUMBER_CELL}>{formatDecimal(row.run.recallAtPrecision50)}</td>
                  <td className={cx(NUMBER_CELL, "text-xs text-stone")}>{formatDecimal(row.run.accuracy)}</td>
                </tr>
              ) : (
                <tr key={row.method} data-testid={`metrics-row-${row.method}`} className="border-t border-line">
                  <th scope="row" className={cx(CELL, "text-left font-normal")}>
                    <span className="font-medium">{METHOD_TEXT[row.method]}</span>
                    <span className="block text-xs text-stone">{METHOD_KIND_TEXT[row.method]}</span>
                  </th>
                  <td colSpan={4} className={cx(CELL, "text-stone")}>
                    Schritt 4, noch nicht gemessen
                  </td>
                </tr>
              ),
            )}
            <tr data-testid="metrics-row-random" className="border-t border-line bg-paper">
              <th scope="row" className={cx(CELL, "text-left font-normal")}>
                <span className="font-medium">Zufällige Rangfolge</span>
                <span className="block text-xs text-stone">Score ohne Information</span>
              </th>
              <td className={NUMBER_CELL}>{formatDecimal(evaluation.randomPrAucExpected)}</td>
              <td colSpan={3} className={cx(CELL, "text-xs text-stone")}>
                Erwartungswert der PR-AUC; 95 % von {formatInteger(evaluation.randomPermutations)} zufälligen
                Rangfolgen (Seed {metrics.seed}) bleiben unter {formatDecimal(evaluation.randomPrAucQ95)}.
              </td>
            </tr>
            <tr data-testid="metrics-row-prevalence" className="border-t border-line bg-paper">
              <th scope="row" className={cx(CELL, "text-left font-normal")}>
                <span className="font-medium">Konstanter Score</span>
                <span className="block text-xs text-stone">Anteil auffälliger Testknoten (Prävalenz)</span>
              </th>
              <td className={NUMBER_CELL}>{formatDecimal(evaluation.prevalence)}</td>
              <td colSpan={3} className={cx(CELL, "text-xs text-stone")}>
                PR-AUC, wenn alle Knoten denselben Score haben
              </td>
            </tr>
          </tbody>
        </table>
      </ScrollRegion>
    </div>
  );
}
