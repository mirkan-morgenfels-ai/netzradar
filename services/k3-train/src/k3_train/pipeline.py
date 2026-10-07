from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from k3_train.baseline import (
    IFOREST,
    ZSCORE,
    iforest_for_split,
    iforest_hyperparameters,
    zscore_for_split,
    zscore_hyperparameters,
)
from k3_train.datasets import dataset_info
from k3_train.features import feature_matrix
from k3_train.graph import GRAPH_FEATURES, graph_measures_with_report, label_homophily
from k3_train.jsonio import read_json, write_json
from k3_train.load import Dataset
from k3_train.metrics import evaluate_scores, evaluation_summary, evaluation_target
from k3_train.split import check_split, split_for, split_masks

DEFAULT_SEED = 42
SCHEMA_VERSION = 3
METHODS = ("zscore", "iforest", "gcn", "graphsage", "mlp")
SCORE_COLUMNS = ("score_zscore", "score_iforest")


class RunContext(Protocol):
    dataset: dict[str, Any]
    split: dict[str, Any]
    evaluation: dict[str, Any]


@dataclass
class BaselineOutput:
    dataset: dict[str, Any]
    split: dict[str, Any]
    evaluation: dict[str, Any]
    seed: int
    runs: list[dict[str, Any]]
    scores: pd.DataFrame
    measures: pd.DataFrame

    def summary(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "evaluation": self.evaluation,
            "seed": self.seed,
            "runs": self.runs,
        }

    def run_logs(self) -> list[dict[str, Any]]:
        return [run_log(self, run) for run in self.runs]


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso_timestamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def iso_date(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%d")


def dataset_block(dataset: Dataset) -> dict[str, Any]:
    info = dataset_info(dataset.name)
    return {
        "name": info.name,
        "displayName": info.display_name,
        "license": info.license,
        "source": info.source,
        "generator": dataset.generator,
        "nodes": len(dataset.nodes),
        "edges": len(dataset.edges),
        "timeSteps": dataset.time_steps,
        "features": len(dataset.feature_columns),
        "labelCounts": dataset.label_counts(),
        "homophily": label_homophily(dataset.nodes, dataset.edges),
    }


def run_baseline(dataset: Dataset, run_date: str, seed: int = DEFAULT_SEED) -> BaselineOutput:
    nodes = dataset.nodes
    split = split_for(dataset.name, dataset.time_steps)
    crossing = check_split(nodes, dataset.edges, split)
    masks = split_masks(nodes, split)
    train = masks["train"].to_numpy()

    measures, graph_report = graph_measures_with_report(nodes, dataset.edges, seed=seed)
    local_values, local_names = feature_matrix(nodes, None, "local")
    full_values, full_names = feature_matrix(nodes, measures, "local+graph")
    zscores, robust_stats = zscore_for_split(local_values, train)
    iforest_scores = iforest_for_split(full_values, train, seed=seed)

    evaluated, y = evaluation_target(nodes["label"], masks["test"])
    runs = []
    for spec, scores, hyperparameters in (
        (ZSCORE, zscores, zscore_hyperparameters(local_names, robust_stats)),
        (
            IFOREST,
            iforest_scores,
            iforest_hyperparameters(full_names, seed, graph_report.as_dict()),
        ),
    ):
        runs.append(
            {
                "method": spec.method,
                "displayName": spec.display_name,
                "featureSet": spec.feature_set,
                "seed": seed,
                "date": run_date,
                "hyperparameters": hyperparameters,
                **evaluate_scores(y, scores[evaluated]),
            }
        )
    score_frame = pd.DataFrame(
        {
            "node_id": nodes["node_id"].to_numpy(),
            "score_zscore": zscores,
            "score_iforest": iforest_scores,
        }
    )
    return BaselineOutput(
        dataset=dataset_block(dataset),
        split={**split.as_dict(), "crossSplitEdges": crossing},
        evaluation=evaluation_summary(y, seed=seed),
        seed=seed,
        runs=runs,
        scores=score_frame,
        measures=measures,
    )


def ordered_runs(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = [run["method"] for run in runs]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unbekannte Verfahren: {unknown}")
    if len(set(methods)) != len(methods):
        raise ValueError(f"Verfahren doppelt: {methods}")
    return sorted(runs, key=lambda run: METHODS.index(run["method"]))


def metrics_payload(
    output: BaselineOutput,
    generated_at: str,
    extra_runs: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "generatedAt": generated_at,
        "dataset": output.dataset,
        "split": output.split,
        "evaluation": output.evaluation,
        "seed": output.seed,
        "runs": ordered_runs([*output.runs, *extra_runs]),
    }


def run_log(output: RunContext, run: dict[str, Any]) -> dict[str, Any]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "method": run["method"],
        "displayName": run["displayName"],
        "featureSet": run["featureSet"],
        "date": run["date"],
        "seed": run["seed"],
        "dataset": output.dataset,
        "split": output.split,
        "evaluation": output.evaluation,
        "hyperparameters": run["hyperparameters"],
        "prAuc": run["prAuc"],
        "precisionAtRecall50": run["precisionAtRecall50"],
        "recallAtPrecision50": run["recallAtPrecision50"],
        "accuracy": run["accuracy"],
        "accuracyThreshold": run["accuracyThreshold"],
        "accuracyFlagged": run["accuracyFlagged"],
        "accuracyTruePositives": run["accuracyTruePositives"],
        "prCurve": run["prCurve"],
    }


def save_baseline(output: BaselineOutput, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    output.scores.to_csv(out_dir / "scores.csv", index=False, lineterminator="\n")
    output.measures.to_csv(out_dir / "measures.csv", index=False, lineterminator="\n")
    write_json(out_dir / "baseline.json", output.summary())


def load_baseline(out_dir: Path) -> BaselineOutput:
    summary_path = out_dir / "baseline.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Keine Baseline-Ergebnisse in {out_dir}")
    summary = read_json(summary_path)
    scores = pd.read_csv(
        out_dir / "scores.csv", dtype={"node_id": str}, float_precision="round_trip"
    )
    measures = pd.read_csv(
        out_dir / "measures.csv", dtype={"node_id": str}, float_precision="round_trip"
    )
    missing = [column for column in GRAPH_FEATURES if column not in measures.columns]
    if missing or any(column not in scores.columns for column in SCORE_COLUMNS):
        raise ValueError("Baseline-Dateien sind unvollstaendig")
    return BaselineOutput(
        dataset=summary["dataset"],
        split=summary["split"],
        evaluation=summary["evaluation"],
        seed=summary["seed"],
        runs=summary["runs"],
        scores=scores,
        measures=measures,
    )
