import { cx } from "@portfolio/ui";
import { METHOD_CURVE_STYLES } from "@/lib/netzradar/curves";
import { FEATURE_SET_TEXT, formatDecimal, formatInteger, METHOD_KIND_TEXT, METHOD_TEXT } from "@/lib/netzradar/format";
import { featureNames, labelledTestCount, metricRows, zeroMadFeatures } from "@/lib/netzradar/summary";
import type { Metrics, Run } from "@/lib/netzradar/types";
import { ScrollRegion } from "./ScrollRegion";

const CELL = "px-4 first:pl-5 last:pr-5 sm:first:pl-6 sm:last:pr-6";
const HEAD = cx(CELL, "pt-5");
const NUMBER_CELL = cx(CELL, "text-right whitespace-nowrap");
const NAME_CELL = "max-sm:sticky max-sm:left-0 max-sm:z-[1] max-sm:shadow-[inset_-1px_0_0_var(--color-line)]";
const REFERENCE_FILL = "bg-[color-mix(in_oklab,var(--color-ivory)_70%,var(--color-surface))]";

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

function ScoreBar({ value, color }: { value: number; color: string }) {
  const width = `${Math.max(0, Math.min(1, value)) * 100}%`;
  return (
    <span aria-hidden="true" className="mt-2.5 ml-auto block h-[3px] w-24 overflow-hidden rounded-full bg-line">
      <span className="block h-full rounded-full" style={{ width, backgroundColor: color }} />
    </span>
  );
}

export function MetricsTable({ metrics }: { metrics: Metrics }) {
  const rows = metricRows(metrics.runs);
  const { evaluation, split } = metrics;
  return (
    <div className="space-y-3">
      <p id="metrics-table-note" className="max-w-[72ch] text-xs leading-relaxed text-slate">
        Testzeitraum Zeitschritte {split.test.from} bis {split.test.to}: {formatInteger(labelledTestCount(evaluation))}{" "}
        Knoten mit Label, davon {formatInteger(evaluation.testPositives)} auffällig. Knoten ohne Label sind nicht
        bewertet. Seed {metrics.seed}.
      </p>
      <ScrollRegion
        labelledBy="metrics-table-caption"
        hint="Tabelle seitlich wischen"
        className="rounded-2xl border border-line bg-surface shadow-card"
      >
        <table className="data-table min-w-[40rem] sm:min-w-[54rem]" data-testid="metrics-table" aria-describedby="metrics-table-note">
          <caption id="metrics-table-caption" className="sr-only">
            Kennzahlen je Verfahren auf den Testknoten mit Label
          </caption>
          <thead>
            <tr>
              <th
                scope="col"
                className={cx(HEAD, NAME_CELL, "w-[10.5rem] min-w-[10.5rem] max-sm:bg-surface sm:w-auto sm:min-w-[16rem]")}
              >
                Verfahren
              </th>
              <th scope="col" className={cx(HEAD, "text-right")}>
                PR-AUC
              </th>
              <th scope="col" className={cx(HEAD, "text-right")}>
                Precision bei Recall&nbsp;≥&nbsp;0,5
              </th>
              <th scope="col" className={cx(HEAD, "text-right")}>
                Recall bei Precision&nbsp;≥&nbsp;0,5
              </th>
              <th scope="col" className={cx(HEAD, "text-right")}>
                Accuracy (Nebenwert)
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) =>
              row.kind === "measured" ? (
                <tr key={row.method} data-testid={`metrics-row-${row.method}`}>
                  <th scope="row" className={cx(CELL, NAME_CELL, "py-4 text-left align-baseline font-normal max-sm:bg-surface")}>
                    <span className="font-medium text-ink">{row.run.displayName}</span>
                    <span className="mt-1 block text-xs text-slate">{featureText(row.run)}</span>
                    <span className="block text-xs text-slate">{trainingText(row.run)}</span>
                  </th>
                  <td className={cx(NUMBER_CELL, "py-4")} data-testid={`pr-auc-${row.method}`}>
                    <span className="display num block text-[1.375rem] leading-none text-ink">
                      {formatDecimal(row.run.prAuc)}
                    </span>
                    <ScoreBar value={row.run.prAuc} color={METHOD_CURVE_STYLES[row.method].color} />
                  </td>
                  <td className={cx(NUMBER_CELL, "py-4")}>{formatDecimal(row.run.precisionAtRecall50)}</td>
                  <td className={cx(NUMBER_CELL, "py-4")}>{formatDecimal(row.run.recallAtPrecision50)}</td>
                  <td className={cx(NUMBER_CELL, "py-4 text-xs text-slate")}>{formatDecimal(row.run.accuracy)}</td>
                </tr>
              ) : (
                <tr key={row.method} data-testid={`metrics-row-${row.method}`}>
                  <th scope="row" className={cx(CELL, NAME_CELL, "py-3 text-left align-baseline font-normal max-sm:bg-surface")}>
                    <span className="font-medium text-ink">{METHOD_TEXT[row.method]}</span>
                    <span className="mt-1 block text-xs text-slate">{METHOD_KIND_TEXT[row.method]}</span>
                  </th>
                  <td colSpan={4} className={cx(CELL, "text-slate")}>
                    noch nicht gemessen
                  </td>
                </tr>
              ),
            )}
            <tr data-testid="metrics-row-random" className={REFERENCE_FILL}>
              <th scope="row" className={cx(CELL, NAME_CELL, "py-3 text-left align-baseline font-normal", REFERENCE_FILL)}>
                <span className="font-medium text-ink">Zufällige Rangfolge</span>
                <span className="mt-1 block text-xs text-slate">Score ohne Information</span>
              </th>
              <td className={cx(NUMBER_CELL, "py-3")}>
                <span className="display num block text-[1.375rem] leading-none text-slate">
                  {formatDecimal(evaluation.randomPrAucExpected)}
                </span>
              </td>
              <td colSpan={3} className={cx(CELL, "text-xs leading-relaxed text-slate")}>
                Erwartungswert der PR-AUC; 95 % von {formatInteger(evaluation.randomPermutations)} zufälligen
                Rangfolgen (Seed {metrics.seed}) bleiben unter {formatDecimal(evaluation.randomPrAucQ95)}.
              </td>
            </tr>
            <tr data-testid="metrics-row-prevalence" className={REFERENCE_FILL}>
              <th scope="row" className={cx(CELL, NAME_CELL, "py-3 text-left align-baseline font-normal", REFERENCE_FILL)}>
                <span className="font-medium text-ink">Konstanter Score</span>
                <span className="mt-1 block text-xs text-slate">Anteil auffälliger Testknoten (Prävalenz)</span>
              </th>
              <td className={cx(NUMBER_CELL, "py-3")}>
                <span className="display num block text-[1.375rem] leading-none text-slate">{formatDecimal(evaluation.prevalence)}</span>
              </td>
              <td colSpan={3} className={cx(CELL, "text-xs leading-relaxed text-slate")}>
                PR-AUC, wenn alle Knoten denselben Score haben
              </td>
            </tr>
          </tbody>
        </table>
      </ScrollRegion>
    </div>
  );
}
