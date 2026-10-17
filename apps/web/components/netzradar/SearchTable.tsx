import { cx } from "@portfolio/ui";
import { formatDecimal, formatPlainNumber, formatStepRange, joinList, METHOD_SHORT_TEXT } from "@/lib/netzradar/format";
import type { BoostingCandidate, FeatureSet, Run, SearchCandidate, Training } from "@/lib/netzradar/types";
import { ScrollRegion } from "./ScrollRegion";

const FEATURE_SET_SHORT: Record<FeatureSet, string> = {
  local: "lokal",
  "local+graph": "lokal + Graphmaße",
};

const CELL = "px-3 last:pr-5 sm:last:pr-6";
const LEAD_CELL = "pr-3 pl-5 sm:pl-6";
const HEAD = cx(CELL, "pt-5");
const NUMBER_CELL = cx(CELL, "text-right whitespace-nowrap");

type Candidate = SearchCandidate | BoostingCandidate;

interface CandidateRow {
  key: string;
  candidate: Candidate;
  leaves: number | null;
  best: number;
  stopped: number | null;
}

interface LearnedRun {
  run: Run;
  training: Training;
}

function weightText(candidate: Candidate): string {
  const value = formatPlainNumber(candidate.positiveWeight);
  return candidate.positiveWeightRule === "fixed" ? `${value} (fest)` : `${value} (Verhältnis aus den Labels)`;
}

function candidateRows(training: Training): CandidateRow[] {
  if (training.kind === "boosting") {
    return training.search.map((candidate) => ({
      key: `${candidate.featureSet}-${candidate.positiveWeightRule}-${candidate.positiveWeight}-${candidate.maxLeafNodes}`,
      candidate,
      leaves: candidate.maxLeafNodes,
      best: candidate.bestIteration,
      stopped: null,
    }));
  }
  return training.search.map((candidate) => ({
    key: `${candidate.featureSet}-${candidate.positiveWeightRule}`,
    candidate,
    leaves: null,
    best: candidate.bestEpoch,
    stopped: candidate.stoppedEpoch,
  }));
}

function noteText(learned: readonly LearnedRun[]): string {
  const neural = learned.filter(({ training }) => training.kind === "neural");
  const boosting = learned.flatMap(({ run, training }) => (training.kind === "boosting" ? [{ run, training }] : []));
  const parts: string[] = [];
  if (neural.length > 0) {
    parts.push(
      `Bei ${joinList(neural.map(({ run }) => METHOD_SHORT_TEXT[run.method]))} ist die beste Stelle die Epoche mit der höchsten PR-AUC auf dem Validierungsteil und Abbruch die letzte trainierte Epoche.`,
    );
  }
  for (const { run, training } of boosting) {
    parts.push(
      `Beim ${METHOD_SHORT_TEXT[run.method]} nennt die Spalte Blätter die Höchstzahl der Blätter je Baum; die beste Stelle ist die Zahl der Bäume mit der höchsten PR-AUC auf dem Validierungsteil (Kurve bis ${training.maxIter} Bäume, ohne Abbruch).`,
    );
  }
  return parts.join(" ");
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
        Hervorgehoben ist der gewählte Kandidat je Verfahren. {noteText(learned)}
      </p>
      <ScrollRegion
        labelledBy="search-table-caption"
        hint="Tabelle seitlich wischen"
        className="rounded-2xl border border-line bg-surface shadow-card"
      >
        <table className="data-table min-w-[50rem]" data-testid="search-table" aria-describedby="search-table-note">
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
                Blätter
              </th>
              <th scope="col" className={cx(HEAD, "text-right")}>
                PR-AUC Validierung
              </th>
              <th scope="col" className={cx(HEAD, "text-right")}>
                beste Epoche bzw. Bäume
              </th>
              <th scope="col" className={cx(HEAD, "text-right")}>
                Abbruch
              </th>
              <th scope="col" className={HEAD}>
                Auswahl
              </th>
            </tr>
          </thead>
          {learned.map(({ run, training }) => {
            const rows = candidateRows(training);
            return (
              <tbody key={run.method} data-testid={`search-${run.method}`} className="[&>tr:first-child]:border-ink/20">
                {rows.map(({ key, candidate, leaves, best, stopped }, index) => (
                  <tr key={key} className={candidate.selected ? "bg-gold-soft/70 hover:bg-gold-soft/70" : undefined}>
                    {index === 0 ? (
                      <th
                        scope="rowgroup"
                        rowSpan={rows.length}
                        className={cx(LEAD_CELL, "bg-surface py-3 text-left align-top font-medium text-ink")}
                      >
                        {METHOD_SHORT_TEXT[run.method]}
                      </th>
                    ) : null}
                    <td className={CELL}>{FEATURE_SET_SHORT[candidate.featureSet]}</td>
                    <td className={cx(CELL, "text-left whitespace-nowrap")}>{weightText(candidate)}</td>
                    <td className={cx(NUMBER_CELL, leaves === null ? "text-slate" : undefined)}>{leaves ?? "–"}</td>
                    <td className={NUMBER_CELL}>{formatDecimal(candidate.validationPrAuc)}</td>
                    <td className={NUMBER_CELL}>{best}</td>
                    <td className={cx(NUMBER_CELL, stopped === null ? "text-slate" : undefined)}>{stopped ?? "–"}</td>
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
            );
          })}
        </table>
      </ScrollRegion>
    </div>
  );
}
