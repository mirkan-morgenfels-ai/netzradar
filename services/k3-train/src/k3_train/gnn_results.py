from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from k3_train.jsonio import FLOAT_DIGITS, read_json, write_json
from k3_train.pipeline import run_log

GNN_METHODS = ("gcn", "graphsage", "mlp")
GRAPH_METHODS = ("gcn", "graphsage")
DISPLAY_NAMES = {
    "gcn": "GCN (2 Schichten)",
    "graphsage": "GraphSAGE (2 Schichten)",
    "mlp": "MLP ohne Kanten (2 Schichten)",
}
SUMMARY_FILE = "gnn.json"
SCORES_FILE = "gnn_scores.csv"
SPREAD_KEY = "seedSpread"
ENVIRONMENT_KEY = "environment"


def score_column(method: str) -> str:
    return f"score_{method}"


@dataclass
class GnnOutput:
    dataset: dict[str, Any]
    split: dict[str, Any]
    evaluation: dict[str, Any]
    seed: int
    runs: list[dict[str, Any]]
    seed_spread: dict[str, dict[str, Any]]
    scores: pd.DataFrame

    def summary(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "evaluation": self.evaluation,
            "seed": self.seed,
            "runs": self.runs,
            SPREAD_KEY: self.seed_spread,
        }

    def run_logs(self) -> list[dict[str, Any]]:
        logs = []
        for run in self.runs:
            spread = self.seed_spread.get(run["method"])
            hyperparameters = (
                run["hyperparameters"]
                if spread is None
                else with_seed_spread(run["hyperparameters"], spread)
            )
            logs.append(run_log(self, {**run, "hyperparameters": hyperparameters}))
        return logs


def with_seed_spread(hyperparameters: dict[str, Any], spread: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in hyperparameters.items():
        if key == ENVIRONMENT_KEY:
            result[SPREAD_KEY] = spread
        result[key] = value
    if SPREAD_KEY not in result:
        result[SPREAD_KEY] = spread
    return result


def _published_validation(run: dict[str, Any]) -> float:
    return round(float(run["hyperparameters"]["validationPrAuc"]), FLOAT_DIGITS)


def score_gnn_method(runs: Sequence[dict[str, Any]]) -> str | None:
    candidates = [run for run in runs if run["method"] in GRAPH_METHODS]
    if not candidates:
        return None
    candidates.sort(key=lambda run: GRAPH_METHODS.index(run["method"]))
    best = candidates[0]
    for run in candidates[1:]:
        if _published_validation(run) > _published_validation(best):
            best = run
    return str(best["method"])


def check_compatible(baseline: Any, gnn: GnnOutput) -> None:
    for key in ("dataset", "split", "evaluation", "seed"):
        if getattr(baseline, key) != getattr(gnn, key):
            raise ValueError(
                f"Die GNN-Ergebnisse passen nicht zur Baseline ({key} verschieden). "
                "Bitte k3-train baseline und k3-train gnn auf denselben Daten ausführen."
            )


def save_gnn(output: GnnOutput, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    output.scores.to_csv(out_dir / SCORES_FILE, index=False, lineterminator="\n")
    write_json(out_dir / SUMMARY_FILE, output.summary())


def has_gnn(out_dir: Path) -> bool:
    return (out_dir / SUMMARY_FILE).is_file()


def load_gnn(out_dir: Path) -> GnnOutput:
    summary_path = out_dir / SUMMARY_FILE
    if not summary_path.is_file():
        raise FileNotFoundError(f"Keine GNN-Ergebnisse in {out_dir}")
    summary = read_json(summary_path)
    scores = pd.read_csv(
        out_dir / SCORES_FILE, dtype={"node_id": str}, float_precision="round_trip"
    )
    expected = [score_column(run["method"]) for run in summary["runs"]]
    if any(column not in scores.columns for column in expected):
        raise ValueError("GNN-Dateien sind unvollstaendig")
    return GnnOutput(
        dataset=summary["dataset"],
        split=summary["split"],
        evaluation=summary["evaluation"],
        seed=summary["seed"],
        runs=summary["runs"],
        seed_spread=summary[SPREAD_KEY],
        scores=scores,
    )
