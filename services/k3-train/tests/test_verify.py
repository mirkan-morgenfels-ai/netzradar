import copy
from pathlib import Path
from typing import Any

import pytest

from k3_train import verify as verify_module
from k3_train.jsonio import write_json
from k3_train.verify import Recomputation, compare, verify

PAYLOAD: dict[str, Any] = {
    "schemaVersion": 2,
    "generatedAt": "2026-01-01T00:00:00Z",
    "dataset": {"name": "synthetic", "nodes": 10},
    "runs": [{"method": "zscore", "date": "2026-01-01", "prAuc": 0.5}],
}
NODES: dict[str, Any] = {"schemaVersion": 2, "nodes": [{"id": "tx000001", "x": 0.25}]}
EDGES: dict[str, Any] = {"schemaVersion": 2, "edges": [{"source": "tx000001", "target": "tx2"}]}
RUN_LOG: dict[str, Any] = {"schemaVersion": 2, "method": "zscore", "date": "", "prAuc": 0.5}
RECOMPUTED = Recomputation(metrics=PAYLOAD, nodes=NODES, edges=EDGES, run_logs={"zscore": RUN_LOG})


def test_identical_payloads_have_no_difference() -> None:
    assert compare(PAYLOAD, copy.deepcopy(PAYLOAD)) == []


def test_timestamps_and_dates_are_ignored() -> None:
    other = copy.deepcopy(PAYLOAD)
    other["generatedAt"] = "2030-12-31T23:59:59Z"
    other["runs"][0]["date"] = "2030-12-31"
    assert compare(PAYLOAD, other) == []


def test_floats_are_compared_with_tolerance() -> None:
    close = copy.deepcopy(PAYLOAD)
    close["runs"][0]["prAuc"] = 0.50005
    far = copy.deepcopy(PAYLOAD)
    far["runs"][0]["prAuc"] = 0.5002
    assert compare(PAYLOAD, close) == []
    assert compare(PAYLOAD, far) == ["$.runs[0].prAuc: 0.5 erwartet, 0.5002 gefunden"]


def test_integers_strings_keys_and_lengths_must_match() -> None:
    changed = copy.deepcopy(PAYLOAD)
    changed["dataset"]["nodes"] = 11
    changed["runs"][0]["method"] = "iforest"
    assert len(compare(PAYLOAD, changed)) == 2
    missing = copy.deepcopy(PAYLOAD)
    del missing["dataset"]["nodes"]
    assert compare(PAYLOAD, missing)
    longer = copy.deepcopy(PAYLOAD)
    longer["runs"].append({"method": "iforest"})
    assert compare(PAYLOAD, longer) == ["$.runs: Länge 1 erwartet, 2 gefunden"]


def write_committed(export_dir: Path, runs_dir: Path) -> None:
    write_json(export_dir / "metrics.json", PAYLOAD)
    write_json(export_dir / "nodes.json", NODES)
    write_json(export_dir / "edges.json", EDGES)
    write_json(runs_dir / "2026-01-01_synthetic_zscore.json", {**RUN_LOG, "date": "2026-01-01"})


@pytest.fixture
def committed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    monkeypatch.setattr(verify_module, "recompute_synthetic", lambda: RECOMPUTED)
    export_dir = tmp_path / "k3"
    runs_dir = tmp_path / "runs"
    write_committed(export_dir, runs_dir)
    return export_dir, runs_dir


def test_verify_passes_for_matching_files(committed: tuple[Path, Path]) -> None:
    assert verify(*committed) == 0


def test_verify_detects_tampered_metrics(committed: tuple[Path, Path]) -> None:
    export_dir, runs_dir = committed
    tampered = copy.deepcopy(PAYLOAD)
    tampered["runs"][0]["prAuc"] = 0.9
    write_json(export_dir / "metrics.json", tampered)
    assert verify(export_dir, runs_dir) == 1


def test_verify_detects_tampered_nodes_and_edges(committed: tuple[Path, Path]) -> None:
    export_dir, runs_dir = committed
    moved = copy.deepcopy(NODES)
    moved["nodes"][0]["x"] = 0.5
    write_json(export_dir / "nodes.json", moved)
    assert verify(export_dir, runs_dir) == 1
    write_json(export_dir / "nodes.json", NODES)
    write_json(export_dir / "edges.json", {"schemaVersion": 2, "edges": []})
    assert verify(export_dir, runs_dir) == 1


def test_verify_checks_the_latest_run_log(committed: tuple[Path, Path]) -> None:
    export_dir, runs_dir = committed
    write_json(runs_dir / "2026-02-01_synthetic_zscore.json", {**RUN_LOG, "prAuc": 0.7})
    assert verify(export_dir, runs_dir) == 1
    (runs_dir / "2026-02-01_synthetic_zscore.json").unlink()
    (runs_dir / "2026-01-01_synthetic_zscore.json").unlink()
    assert verify(export_dir, runs_dir) == 1


def test_verify_skips_other_datasets(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_json(tmp_path / "metrics.json", {"dataset": {"name": "elliptic"}})
    assert verify(tmp_path, tmp_path) == 0
    assert "übersprungen" in capsys.readouterr().out


def test_verify_fails_without_metrics(tmp_path: Path) -> None:
    assert verify(tmp_path, tmp_path) == 1
