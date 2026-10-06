from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from k3_train.datasets import dataset_info
from k3_train.jsonio import read_json, write_json
from k3_train.synth import SynthConfig, generate, pattern_summary

LABELS = ("illicit", "licit", "unknown")
ELLIPTIC_FILES = {
    "features": "elliptic_txs_features.csv",
    "classes": "elliptic_txs_classes.csv",
    "edges": "elliptic_txs_edgelist.csv",
}
ELLIPTIC_CLASS_MAP = {"1": "illicit", "2": "licit", "unknown": "unknown"}
ELLIPTIC_FEATURE_COUNT = 165
ELLIPTIC_AGGREGATED_COUNT = 72
NODE_KEY_COLUMNS = ("node_id", "time_step", "label")
LOCAL_PREFIX = "f_"
AGGREGATED_PREFIX = "a_"


@dataclass(frozen=True)
class CleaningReport:
    input_edges: int
    removed_missing_endpoint: int
    removed_self_loops: int
    removed_duplicates: int

    @property
    def removed_total(self) -> int:
        return self.removed_missing_endpoint + self.removed_self_loops + self.removed_duplicates

    def as_dict(self) -> dict[str, int]:
        return {
            "inputEdges": self.input_edges,
            "removedMissingEndpoint": self.removed_missing_endpoint,
            "removedSelfLoops": self.removed_self_loops,
            "removedDuplicates": self.removed_duplicates,
        }


@dataclass
class Dataset:
    name: str
    nodes: pd.DataFrame
    edges: pd.DataFrame
    feature_columns: list[str]
    cleaning: CleaningReport
    generator: dict[str, Any] | None = None
    truth: pd.DataFrame | None = None
    notes: dict[str, Any] = field(default_factory=dict)
    aggregated_columns: list[str] = field(default_factory=list)

    @property
    def time_steps(self) -> int:
        return int(self.nodes["time_step"].nunique())

    def label_counts(self) -> dict[str, int]:
        counts = self.nodes["label"].value_counts()
        return {label: int(counts.get(label, 0)) for label in LABELS}


def feature_columns_of(nodes: pd.DataFrame) -> list[str]:
    return [column for column in nodes.columns if column.startswith(LOCAL_PREFIX)]


def aggregated_columns_of(nodes: pd.DataFrame) -> list[str]:
    return [column for column in nodes.columns if column.startswith(AGGREGATED_PREFIX)]


def validate_nodes(nodes: pd.DataFrame) -> None:
    missing = [column for column in NODE_KEY_COLUMNS if column not in nodes.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten: {missing}")
    if nodes["node_id"].duplicated().any():
        raise ValueError("Knoten-IDs sind nicht eindeutig")
    unknown_labels = set(nodes["label"].unique()) - set(LABELS)
    if unknown_labels:
        raise ValueError(f"Unbekannte Labels: {sorted(unknown_labels)}")
    if (nodes["time_step"] < 1).any():
        raise ValueError("Zeitschritte muessen bei 1 beginnen")
    features = feature_columns_of(nodes)
    if not features:
        raise ValueError("Keine Merkmalsspalten f_* gefunden")
    if nodes[[*features, *aggregated_columns_of(nodes)]].isna().to_numpy().any():
        raise ValueError("Merkmale enthalten fehlende Werte")


def clean_edges(nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    known = set(nodes["node_id"])
    input_edges = len(edges)
    present = edges["source"].isin(known) & edges["target"].isin(known)
    edges = edges[present]
    loops = edges["source"] == edges["target"]
    edges = edges[~loops]
    duplicates = edges.duplicated(["source", "target"])
    edges = edges[~duplicates]
    report = CleaningReport(
        input_edges=input_edges,
        removed_missing_endpoint=int((~present).sum()),
        removed_self_loops=int(loops.sum()),
        removed_duplicates=int(duplicates.sum()),
    )
    cleaned = edges.sort_values(["source", "target"], ignore_index=True)[["source", "target"]]
    return cleaned, report


def _normalise_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    features = [*feature_columns_of(nodes), *aggregated_columns_of(nodes)]
    nodes = nodes[[*NODE_KEY_COLUMNS, *features]].copy()
    nodes["node_id"] = nodes["node_id"].astype(str)
    nodes["time_step"] = nodes["time_step"].astype(np.int64)
    nodes["label"] = nodes["label"].astype(str)
    return nodes.sort_values(["time_step", "node_id"], ignore_index=True)


def _build(
    name: str,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    generator: dict[str, Any] | None = None,
    truth: pd.DataFrame | None = None,
    notes: dict[str, Any] | None = None,
) -> Dataset:
    dataset_info(name)
    nodes = _normalise_nodes(nodes)
    validate_nodes(nodes)
    edges = edges.astype({"source": str, "target": str})
    edges, report = clean_edges(nodes, edges)
    return Dataset(
        name=name,
        nodes=nodes,
        edges=edges,
        feature_columns=feature_columns_of(nodes),
        cleaning=report,
        generator=generator,
        truth=truth,
        notes=notes or {},
        aggregated_columns=aggregated_columns_of(nodes),
    )


def load_synthetic(config: SynthConfig | None = None) -> Dataset:
    result = generate(config)
    return _build(
        "synthetic",
        result.nodes,
        result.edges,
        generator=result.config.as_dict(),
        truth=result.truth,
        notes=pattern_summary(result.truth, result.nodes),
    )


def missing_elliptic_files(raw_dir: Path) -> list[str]:
    return [name for name in ELLIPTIC_FILES.values() if not (raw_dir / name).is_file()]


def elliptic_feature_names(feature_count: int, aggregated_count: int) -> list[str]:
    if not 0 <= aggregated_count < feature_count:
        raise ValueError(
            f"{aggregated_count} aggregierte Merkmale passen nicht zu {feature_count} Merkmalen"
        )
    local_count = feature_count - aggregated_count
    width = max(3, len(str(feature_count)))
    return [
        *(f"{LOCAL_PREFIX}{index:0{width}d}" for index in range(1, local_count + 1)),
        *(f"{AGGREGATED_PREFIX}{index:0{width}d}" for index in range(1, aggregated_count + 1)),
    ]


def load_elliptic(raw_dir: Path, aggregated_count: int | None = None) -> Dataset:
    missing = missing_elliptic_files(raw_dir)
    if missing:
        raise FileNotFoundError(f"Fehlende Elliptic-Dateien in {raw_dir}: {', '.join(missing)}")
    features = pd.read_csv(
        raw_dir / ELLIPTIC_FILES["features"],
        header=None,
        dtype={0: str},
        float_precision="round_trip",
    )
    feature_count = features.shape[1] - 2
    if feature_count < 1:
        raise ValueError("Die Merkmalsdatei enthaelt keine Merkmalsspalten")
    if aggregated_count is None:
        if feature_count != ELLIPTIC_FEATURE_COUNT:
            raise ValueError(
                f"Die Elliptic-Merkmalsdatei hat {feature_count} Merkmale nach dem Zeitschritt, "
                f"erwartet sind {ELLIPTIC_FEATURE_COUNT} "
                f"({ELLIPTIC_FEATURE_COUNT - ELLIPTIC_AGGREGATED_COUNT} lokale und "
                f"{ELLIPTIC_AGGREGATED_COUNT} über Nachbarn aggregierte). Ohne geprüftes Layout "
                "lassen sich lokale und aggregierte Merkmale nicht trennen."
            )
        aggregated_count = ELLIPTIC_AGGREGATED_COUNT
    features.columns = [
        "node_id",
        "time_step",
        *elliptic_feature_names(feature_count, aggregated_count),
    ]
    classes = pd.read_csv(raw_dir / ELLIPTIC_FILES["classes"], dtype=str)
    classes.columns = ["node_id", "class"]
    unmapped = set(classes["class"]) - set(ELLIPTIC_CLASS_MAP)
    if unmapped:
        raise ValueError(f"Unbekannte Elliptic-Klassen: {sorted(unmapped)}")
    classes["label"] = classes["class"].map(ELLIPTIC_CLASS_MAP)
    nodes = features.merge(classes[["node_id", "label"]], on="node_id", how="left")
    if nodes["label"].isna().any():
        raise ValueError("Nicht jede Transaktion hat einen Eintrag in der Klassendatei")
    edges = pd.read_csv(raw_dir / ELLIPTIC_FILES["edges"], dtype=str)
    edges.columns = ["source", "target"]
    notes = {
        "featureCount": feature_count,
        "localFeatures": feature_count - aggregated_count,
        "aggregatedFeatures": aggregated_count,
    }
    return _build("elliptic", nodes, edges, notes=notes)


def write_processed(dataset: Dataset, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset.nodes.to_csv(out_dir / "nodes.csv", index=False, lineterminator="\n")
    dataset.edges.to_csv(out_dir / "edges.csv", index=False, lineterminator="\n")
    if dataset.truth is not None:
        dataset.truth.to_csv(out_dir / "truth.csv", index=False, lineterminator="\n")
    write_json(out_dir / "meta.json", processed_meta(dataset))


def processed_meta(dataset: Dataset) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "dataset": dataset.name,
        "generator": dataset.generator,
        "nodes": len(dataset.nodes),
        "edges": len(dataset.edges),
        "timeSteps": dataset.time_steps,
        "features": len(dataset.feature_columns),
        "featureColumns": dataset.feature_columns,
        "aggregatedColumns": dataset.aggregated_columns,
        "labelCounts": dataset.label_counts(),
        "cleaning": dataset.cleaning.as_dict(),
        "notes": dataset.notes,
    }


def read_processed(out_dir: Path) -> Dataset:
    meta_path = out_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Keine aufbereiteten Daten in {out_dir}")
    meta = read_json(meta_path)
    nodes = pd.read_csv(
        out_dir / "nodes.csv",
        dtype={"node_id": str, "label": str},
        float_precision="round_trip",
    )
    edges = pd.read_csv(out_dir / "edges.csv", dtype=str)
    truth_path = out_dir / "truth.csv"
    truth = pd.read_csv(truth_path, dtype={"node_id": str}) if truth_path.is_file() else None
    cleaning = meta["cleaning"]
    nodes = _normalise_nodes(nodes)
    validate_nodes(nodes)
    return Dataset(
        name=meta["dataset"],
        nodes=nodes,
        edges=edges[["source", "target"]],
        feature_columns=list(meta["featureColumns"]),
        cleaning=CleaningReport(
            input_edges=cleaning["inputEdges"],
            removed_missing_endpoint=cleaning["removedMissingEndpoint"],
            removed_self_loops=cleaning["removedSelfLoops"],
            removed_duplicates=cleaning["removedDuplicates"],
        ),
        generator=meta["generator"],
        truth=truth,
        notes=meta.get("notes", {}),
        aggregated_columns=list(meta.get("aggregatedColumns", [])),
    )
