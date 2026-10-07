import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from k3_train.export import build_exports
from k3_train.gnn_results import GNN_METHODS, GnnOutput
from k3_train.jsonio import read_json, round_floats
from k3_train.load import load_synthetic
from k3_train.pipeline import DEFAULT_SEED, run_baseline
from k3_train.runs import latest_run_log
from k3_train.synth import SynthConfig

IGNORED_KEYS = frozenset({"generatedAt", "date", "environment"})
TOLERANCE = 1e-4
FLOAT_SLACK = 1e-9
MAX_REPORTED = 20
VERIFIED_DATASET = "synthetic"
GNN_MODULES = ("torch", "torch_geometric")
MISSING_GNN_MESSAGE = (
    "metrics.json enthält GNN-Runs, zum Nachrechnen fehlen torch und torch_geometric. "
    "Bitte das Extra installieren: uv sync --extra gnn, dann uv run --extra gnn k3-train verify "
    "(mit make: make verify)."
)
REQUIRE_GNN_ENV = "K3_REQUIRE_GNN"
MISSING_GNN_RUNS_MESSAGE = (
    f"{REQUIRE_GNN_ENV}=1, aber metrics.json enthält keine GNN-Runs (gcn, graphsage, mlp). "
    "Bitte k3-train gnn und danach k3-train export ausführen (mit make: make gnn, make export)."
)


@dataclass(frozen=True)
class Recomputation:
    metrics: dict[str, Any]
    nodes: dict[str, Any]
    edges: dict[str, Any]
    run_logs: dict[str, dict[str, Any]]


def compare(expected: Any, actual: Any, tolerance: float = TOLERANCE, path: str = "$") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        expected_keys = [key for key in expected if key not in IGNORED_KEYS]
        actual_keys = [key for key in actual if key not in IGNORED_KEYS]
        if expected_keys != actual_keys:
            return [f"{path}: Schlüssel {expected_keys} erwartet, {actual_keys} gefunden"]
        differences: list[str] = []
        for key in expected_keys:
            differences.extend(compare(expected[key], actual[key], tolerance, f"{path}.{key}"))
        return differences
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: Länge {len(expected)} erwartet, {len(actual)} gefunden"]
        differences = []
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            differences.extend(compare(left, right, tolerance, f"{path}[{index}]"))
        return differences
    if isinstance(expected, bool) or isinstance(actual, bool):
        return [] if expected is actual else [f"{path}: {expected!r} erwartet, {actual!r} gefunden"]
    if isinstance(expected, int) and isinstance(actual, int):
        return [] if expected == actual else [f"{path}: {expected} erwartet, {actual} gefunden"]
    if isinstance(expected, int | float) and isinstance(actual, int | float):
        if abs(float(expected) - float(actual)) <= tolerance + FLOAT_SLACK:
            return []
        return [f"{path}: {expected} erwartet, {actual} gefunden"]
    if expected == actual:
        return []
    return [f"{path}: {expected!r} erwartet, {actual!r} gefunden"]


def gnn_available() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in GNN_MODULES)


def has_gnn_runs(metrics: dict[str, Any]) -> bool:
    runs = metrics.get("runs", [])
    return isinstance(runs, list) and any(
        isinstance(run, dict) and run.get("method") in GNN_METHODS for run in runs
    )


def _recompute_gnn(dataset: Any, measures: Any, seed: int) -> GnnOutput:
    from k3_train.gnn_pipeline import run_gnn

    return run_gnn(dataset, measures, run_date="", seed=seed)


def recompute_synthetic(include_gnn: bool = False, seed: int = DEFAULT_SEED) -> Recomputation:
    dataset = load_synthetic(SynthConfig())
    output = run_baseline(dataset, run_date="", seed=seed)
    gnn = _recompute_gnn(dataset, output.measures, seed) if include_gnn else None
    bundle = build_exports(dataset, output, generated_at="", gnn=gnn)
    logs = [*output.run_logs(), *([] if gnn is None else gnn.run_logs())]
    return Recomputation(
        metrics=bundle.metrics,
        nodes=bundle.nodes,
        edges=bundle.edges,
        run_logs={log["method"]: round_floats(log) for log in logs},
    )


def _report(path: Path, differences: list[str]) -> None:
    print(
        f"Abweichung zwischen {path} und Neuberechnung "
        f"({len(differences)} Stellen, Toleranz {TOLERANCE}):",
        file=sys.stderr,
    )
    for line in differences[:MAX_REPORTED]:
        print(f"  {line}", file=sys.stderr)


def verify(export_dir: Path, runs_dir: Path) -> int:
    metrics_path = export_dir / "metrics.json"
    if not metrics_path.is_file():
        print(f"Keine Kennzahlen gefunden: {metrics_path}", file=sys.stderr)
        return 1
    committed = read_json(metrics_path)
    name = committed.get("dataset", {}).get("name")
    if name != VERIFIED_DATASET:
        print(
            f"Verifikation übersprungen: metrics.json beschreibt den Datensatz {name!r}; "
            "nachgerechnet wird nur das synthetische Netz."
        )
        return 0
    include_gnn = has_gnn_runs(committed)
    if not include_gnn and os.environ.get(REQUIRE_GNN_ENV) == "1":
        print(MISSING_GNN_RUNS_MESSAGE, file=sys.stderr)
        return 1
    if include_gnn and not gnn_available():
        print(MISSING_GNN_MESSAGE, file=sys.stderr)
        return 1
    recomputed = recompute_synthetic(include_gnn=include_gnn)
    targets: list[tuple[Path, Any]] = [
        (metrics_path, recomputed.metrics),
        (export_dir / "nodes.json", recomputed.nodes),
        (export_dir / "edges.json", recomputed.edges),
    ]
    failed = False
    for method, log in recomputed.run_logs.items():
        path = latest_run_log(runs_dir, VERIFIED_DATASET, method)
        if path is None:
            print(f"Kein Laufprotokoll für {method} in {runs_dir}", file=sys.stderr)
            failed = True
        else:
            targets.append((path, log))
    for path, fresh in targets:
        if not path.is_file():
            print(f"Datei fehlt: {path}", file=sys.stderr)
            failed = True
            continue
        differences = compare(read_json(path), fresh)
        if differences:
            _report(path, differences)
            failed = True
    if failed:
        return 1
    names = ", ".join(path.name for path, _ in targets)
    print(
        f"Reproduzierbar: {names} stimmen mit der Neuberechnung überein "
        f"(seed {DEFAULT_SEED}, Toleranz {TOLERANCE})."
    )
    return 0
