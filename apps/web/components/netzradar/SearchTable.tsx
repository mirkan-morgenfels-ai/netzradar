import { cx } from "@portfolio/ui";
import { formatDecimal, formatPlainNumber, formatStepRange, METHOD_SHORT_TEXT } from "@/lib/netzradar/format";
import type { FeatureSet, Run, SearchCandidate, Training } from "@/lib/netzradar/types";
import { ScrollRegion } from "./ScrollRegion";

const FEATURE_SET_SHORT: Record<FeatureSet, string> = {
  local: "lokal",
  "local+graph": "lokal + Graphmaße",
};

const CELL = "px-3 last:pr-5 sm:last:pr-6";
const LEAD_CELL = "pr-3 pl-5 sm:pl-6";
const HEAD = cx(CELL, "pt-5");
const NUMBER_CELL = cx(CELL, "text-right whitespace-nowrap");

function weightText(candidate: SearchCandidate): string {
  const value = formatPlainNumber(candidate.positiveWeight);
  return candidate.positiveWeightRule === "fixed" ? `${value} (fest)` : `${value} (Verhältnis aus den Labels)`;
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
      <p id="search-table-note" className="max-w-[72ch] text-xs leading-relaxed text-slate">
        Alle Kandidaten der Suche, auch die verworfenen: Training auf den Zeitschritten{" "}
        {formatStepRange(first.selectionSteps)}, PR-AUC auf dem Validierungsteil {formatStepRange(first.validationSteps)}.
        Hervorgehoben ist der gewählte Kandidat je Verfahren. Abbruch ist die letzte trainierte Epoche.
      </p>
      <ScrollRegion
        labelledBy="search-table-caption"
        hint="Tabelle seitlich wischen"
        className="rounded-2xl border border-line bg-surface shadow-card"
      >
        <table className="data-table min-w-[44rem]" data-testid="search-table" aria-describedby="search-table-note">
          <caption id="search-table-caption" className="sr-only">
            Suche der gelernten Verfahren am Validierungsteil
          </caption>
          <thead>
            <tr>
              <th scope="col" className={cx(LEAD_CELL, "pt-5")}>
                Verfahren
              </th>
              <th scope="col" className={HEAD}>
                Merkmale
              </th>
              <th scope="col" className={cx(HEAD, "text-left")}>
                Gewicht w
              </th>
              <th scope="col" className={cx(HEAD, "text-right")}>
                PR-AUC Validierung
              </th>
              <th scope="col" className={cx(HEAD, "text-right")}>
                beste Epoche
              </th>
              <th scope="col" className={cx(HEAD, "text-right")}>
                Abbruch
              </th>
              <th scope="col" className={HEAD}>
                Auswahl
              </th>
            </tr>
          </thead>
          {learned.map(({ run, training }) => (
            <tbody key={run.method} data-testid={`search-${run.method}`} className="[&>tr:first-child]:border-ink/20">
              {training.search.map((candidate, index) => (
                <tr
                  key={`${candidate.featureSet}-${candidate.positiveWeightRule}`}
                  className={candidate.selected ? "bg-gold-soft/70 hover:bg-gold-soft/70" : undefined}
                >
                  {index === 0 ? (
                    <th
                      scope="rowgroup"
                      rowSpan={training.search.length}
                      className={cx(LEAD_CELL, "bg-surface py-3 text-left align-top font-medium text-ink")}
                    >
                      {METHOD_SHORT_TEXT[run.method]}
                    </th>
                  ) : null}
                  <td className={CELL}>{FEATURE_SET_SHORT[candidate.featureSet]}</td>
                  <td className={cx(CELL, "text-left whitespace-nowrap")}>{weightText(candidate)}</td>
                  <td className={NUMBER_CELL}>{formatDecimal(candidate.validationPrAuc)}</td>
                  <td className={NUMBER_CELL}>{candidate.bestEpoch}</td>
                  <td className={NUMBER_CELL}>{candidate.stoppedEpoch}</td>
                  <td className={cx(CELL, candidate.selected ? undefined : "text-slate")}>
                    {candidate.selected ? (
                      <span className="inline-flex rounded-full border border-gold/60 bg-surface px-2.5 py-0.5 text-xs font-medium text-gold-deep">
                        gewählt
                      </span>
                    ) : (
                      "–"
                    )}
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
