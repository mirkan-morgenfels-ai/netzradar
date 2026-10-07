from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from k3_train.features import FEATURE_SETS, feature_matrix
from k3_train.gnn import (
    ARCHITECTURES,
    LAYER_TYPES,
    GnnConfig,
    GraphTensors,
    environment,
    graph_tensors,
    logit_scores,
    train_model,
    undirected_edge_index,
)
from k3_train.gnn_results import DISPLAY_NAMES, GnnOutput, score_column
from k3_train.load import Dataset
from k3_train.metrics import evaluate_scores, evaluation_summary, evaluation_target, pr_auc
from k3_train.pipeline import DEFAULT_SEED, dataset_block
from k3_train.split import TemporalSplit, check_split, split_for, split_masks
from k3_train.training import (
    SEARCH_SPACE,
    Candidate,
    RobustScaler,
    check_fit_validation_edges,
    fit_scaler,
    loss_targets,
    scaler_hyperparameters,
    select_candidate,
    training_parts,
    transform,
)

SELECTION_METRIC = "validationPrAuc"
SCORE_TEXT = "logitIllicit - logitLicit"


@dataclass(frozen=True)
class SearchResult:
    candidate: Candidate
    positive_weight: float
    validation_pr_auc: float
    best_epoch: int
    stopped_epoch: int

    def as_dict(self, selected: bool) -> dict[str, Any]:
        return {
            "featureSet": self.candidate.feature_set,
            "positiveWeightRule": self.candidate.weight_rule,
            "positiveWeight": self.positive_weight,
            "validationPrAuc": self.validation_pr_auc,
            "bestEpoch": self.best_epoch,
            "stoppedEpoch": self.stopped_epoch,
            "selected": selected,
        }


@dataclass(frozen=True)
class ScaledGraph:
    graph: GraphTensors
    features: list[str]
    scaler: RobustScaler


@dataclass(frozen=True)
class Prepared:
    split: TemporalSplit
    crossing: int
    selection_graphs: dict[str, ScaledGraph]
    final_graphs: dict[str, ScaledGraph]
    fit: tuple[np.ndarray, np.ndarray]
    validation: tuple[np.ndarray, np.ndarray]
    train: tuple[np.ndarray, np.ndarray]
    evaluated: np.ndarray
    y: np.ndarray


def _scaled(
    values: np.ndarray, names: list[str], mask: np.ndarray, edge_index: Any, dtype: str
) -> ScaledGraph:
    scaler = fit_scaler(values, mask)
    graph = graph_tensors(transform(values, scaler), edge_index, dtype)
    return ScaledGraph(graph=graph, features=names, scaler=scaler)


def prepare(dataset: Dataset, measures: pd.DataFrame, config: GnnConfig) -> Prepared:
    nodes = dataset.nodes
    split = split_for(dataset.name, dataset.time_steps)
    crossing = check_split(nodes, dataset.edges, split)
    parts = training_parts(nodes, split)
    check_fit_validation_edges(nodes, dataset.edges, parts)
    edge_index = undirected_edge_index(nodes["node_id"].tolist(), dataset.edges)
    selection_graphs = {}
    final_graphs = {}
    for feature_set in FEATURE_SETS:
        values, names = feature_matrix(nodes, measures, feature_set)
        selection_graphs[feature_set] = _scaled(
            values, names, parts["fit"], edge_index, config.dtype
        )
        final_graphs[feature_set] = _scaled(values, names, parts["train"], edge_index, config.dtype)
    labels = nodes["label"].to_numpy()
    evaluated, y = evaluation_target(nodes["label"], split_masks(nodes, split)["test"])
    return Prepared(
        split=split,
        crossing=crossing,
        selection_graphs=selection_graphs,
        final_graphs=final_graphs,
        fit=loss_targets(labels, parts["fit"]),
        validation=loss_targets(labels, parts["validation"]),
        train=loss_targets(labels, parts["train"]),
        evaluated=evaluated,
        y=y,
    )


def search_candidates(
    architecture: str, prepared: Prepared, config: GnnConfig, seed: int
) -> list[SearchResult]:
    results = []
    for candidate in SEARCH_SPACE:
        weight = candidate.weight(prepared.fit[1])
        trained = train_model(
            architecture,
            prepared.selection_graphs[candidate.feature_set].graph,
            prepared.fit[0],
            prepared.fit[1],
            weight,
            config,
            seed,
            validation=prepared.validation,
        )
        if trained.validation_pr_auc is None:
            raise ValueError("Auswahllauf ohne Validierungswert")
        results.append(
            SearchResult(
                candidate=candidate,
                positive_weight=weight,
                validation_pr_auc=trained.validation_pr_auc,
                best_epoch=trained.best_epoch,
                stopped_epoch=trained.stopped_epoch,
            )
        )
    return results


def final_scores(
    architecture: str,
    scaled: ScaledGraph,
    prepared: Prepared,
    weight: float,
    epochs: int,
    config: GnnConfig,
    seed: int,
) -> np.ndarray:
    trained = train_model(
        architecture,
        scaled.graph,
        prepared.train[0],
        prepared.train[1],
        weight,
        config,
        seed,
        epochs=epochs,
    )
    return logit_scores(trained.model, scaled.graph)


def seed_spread(
    architecture: str,
    scaled: ScaledGraph,
    prepared: Prepared,
    weight: float,
    epochs: int,
    config: GnnConfig,
    seed: int,
    reported: float,
) -> dict[str, Any]:
    values = []
    for other in config.spread_seeds:
        if other == seed:
            values.append(reported)
            continue
        scores = final_scores(architecture, scaled, prepared, weight, epochs, config, other)
        values.append(pr_auc(prepared.y, scores[prepared.evaluated]))
    return {
        "seeds": list(config.spread_seeds),
        "reportedSeed": seed,
        "prAuc": values,
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def gnn_hyperparameters(
    architecture: str,
    config: GnnConfig,
    search: list[SearchResult],
    chosen: int,
    scaled: ScaledGraph,
    weight: float,
    split: TemporalSplit,
) -> dict[str, Any]:
    selected = search[chosen]
    return {
        "architecture": LAYER_TYPES[architecture],
        "layers": config.layers,
        "hidden": config.hidden,
        "activation": "relu",
        "dropout": config.dropout,
        "optimizer": "adam",
        "learningRate": config.learning_rate,
        "weightDecay": config.weight_decay,
        "loss": "weightedCrossEntropy",
        "score": SCORE_TEXT,
        "positiveWeight": weight,
        "positiveWeightRule": selected.candidate.weight_rule,
        "edges": "none" if architecture == "mlp" else "undirected",
        "dtype": config.dtype,
        "scaling": scaler_hyperparameters(scaled.features, scaled.scaler, "train"),
        "features": scaled.features,
        "maxEpochs": config.max_epochs,
        "patience": config.patience,
        "selectedEpoch": selected.best_epoch,
        "selectionMetric": SELECTION_METRIC,
        "validationPrAuc": selected.validation_pr_auc,
        "selectionSteps": {"from": split.train.start, "to": split.validation.start - 1},
        "validationSteps": split.validation.as_dict(),
        "finalFitSteps": split.train.as_dict(),
        "search": [result.as_dict(index == chosen) for index, result in enumerate(search)],
        "environment": environment(),
    }


def run_gnn(
    dataset: Dataset,
    measures: pd.DataFrame,
    run_date: str,
    seed: int = DEFAULT_SEED,
    config: GnnConfig | None = None,
    architectures: tuple[str, ...] = ARCHITECTURES,
) -> GnnOutput:
    config = config or GnnConfig()
    prepared = prepare(dataset, measures, config)
    runs = []
    spreads = {}
    score_frame = pd.DataFrame({"node_id": dataset.nodes["node_id"].to_numpy()})
    for architecture in architectures:
        search = search_candidates(architecture, prepared, config, seed)
        chosen = select_candidate([result.validation_pr_auc for result in search])
        selected = search[chosen]
        scaled = prepared.final_graphs[selected.candidate.feature_set]
        weight = selected.candidate.weight(prepared.train[1])
        scores = final_scores(
            architecture, scaled, prepared, weight, selected.best_epoch, config, seed
        )
        metrics = evaluate_scores(prepared.y, scores[prepared.evaluated])
        spreads[architecture] = seed_spread(
            architecture,
            scaled,
            prepared,
            weight,
            selected.best_epoch,
            config,
            seed,
            metrics["prAuc"],
        )
        runs.append(
            {
                "method": architecture,
                "displayName": DISPLAY_NAMES[architecture],
                "featureSet": selected.candidate.feature_set,
                "seed": seed,
                "date": run_date,
                "hyperparameters": gnn_hyperparameters(
                    architecture, config, search, chosen, scaled, weight, prepared.split
                ),
                **metrics,
            }
        )
        score_frame[score_column(architecture)] = scores
    return GnnOutput(
        dataset=dataset_block(dataset),
        split={**prepared.split.as_dict(), "crossSplitEdges": prepared.crossing},
        evaluation=evaluation_summary(prepared.y, seed=seed),
        seed=seed,
        runs=runs,
        seed_spread=spreads,
        scores=score_frame,
    )
