import { cx } from "@portfolio/ui";
import { formatDecimal, formatPlainNumber, formatStepRange, METHOD_SHORT_TEXT } from "@/lib/netzradar/format";
import type { FeatureSet, Run, SearchCandidate, Training } from "@/lib/netzradar/types";
import { ScrollRegion } from "./ScrollRegion";

const FEATURE_SET_SHORT: Record<FeatureSet, string> = {
  local: "lokal",
  "local+graph": "lokal + Graphmaße",
};

const CELL = "px-3 py-2";
const NUMBER_CELL = cx(CELL, "text-right tabular-nums");

function weightText(candidate: SearchCandidate): string {
  const value = formatPlainNumber(candidate.positiveWeight);
  return candidate.positiveWeightRule === "fixed" ? `${value} (fest)` : `${value} (licit / illicit)`;
}

interface LearnedRun {
  run: Run;
  training: Training;
}

export function SearchTable({ runs }: { runs: readonly Run[] }) {
  const learned = runs.flatMap((run): LearnedRun[] => (run.training ? [{ run, training: run.training }] : []));
  const first = learned[0]?.training;
  if (!first) return null;
  return (
    <div className="space-y-2">
      <p id="search-table-note" className="text-xs text-stone">
        Alle Kandidaten der Suche, auch die verworfenen: Training auf den Zeitschritten{" "}
        {formatStepRange(first.selectionSteps)}, PR-AUC auf dem Validierungsteil {formatStepRange(first.validationSteps)}.
        Hervorgehoben ist der gewählte Kandidat je Verfahren. Abbruch ist die letzte trainierte Epoche.
      </p>
      <ScrollRegion labelledBy="search-table-caption" className="rounded-lg border border-line bg-surface">
        <table className="w-full min-w-[44rem] border-collapse text-sm" data-testid="search-table" aria-describedby="search-table-note">
          <caption id="search-table-caption" className="sr-only">
            Suche der gelernten Verfahren am Validierungsteil
          </caption>
          <thead>
            <tr className="border-b border-line text-left text-xs uppercase tracking-wide text-stone">
              <th scope="col" className={cx(CELL, "font-medium")}>
                Verfahren
              </th>
              <th scope="col" className={cx(CELL, "font-medium")}>
                Merkmale
              </th>
              <th scope="col" className={cx(CELL, "text-right font-medium")}>
                Gewicht w
              </th>
              <th scope="col" className={cx(CELL, "text-right font-medium")}>
                PR-AUC Validierung
              </th>
              <th scope="col" className={cx(CELL, "text-right font-medium")}>
                beste Epoche
              </th>
              <th scope="col" className={cx(CELL, "text-right font-medium")}>
                Abbruch
              </th>
              <th scope="col" className={cx(CELL, "font-medium")}>
                Auswahl
              </th>
            </tr>
          </thead>
          {learned.map(({ run, training }) => (
            <tbody key={run.method} data-testid={`search-${run.method}`}>
              {training.search.map((candidate, index) => (
                <tr
                  key={`${candidate.featureSet}-${candidate.positiveWeightRule}`}
                  className={cx("border-t border-line", candidate.selected ? "bg-gold-soft" : undefined)}
                >
                  {index === 0 ? (
                    <th
                      scope="rowgroup"
                      rowSpan={training.search.length}
                      className={cx(CELL, "bg-surface text-left align-top font-medium")}
                    >
                      {METHOD_SHORT_TEXT[run.method]}
                    </th>
                  ) : null}
                  <td className={CELL}>{FEATURE_SET_SHORT[candidate.featureSet]}</td>
                  <td className={NUMBER_CELL}>{weightText(candidate)}</td>
                  <td className={NUMBER_CELL}>{formatDecimal(candidate.validationPrAuc)}</td>
                  <td className={NUMBER_CELL}>{candidate.bestEpoch}</td>
                  <td className={NUMBER_CELL}>{candidate.stoppedEpoch}</td>
                  <td className={cx(CELL, candidate.selected ? "font-semibold" : "text-stone")}>
                    {candidate.selected ? "gewählt" : "–"}
                  </td>
                </tr>
              ))}
            </tbody>
          ))}
        </table>
      </ScrollRegion>
    </div>
  );
}
