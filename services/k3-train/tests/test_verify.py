import copy
from pathlib import Path
from typing import Any

import pytest

from k3_train import verify as verify_module
from k3_train.jsonio import write_json
from k3_train.verify import REQUIRE_GNN_ENV, Recomputation, compare, verify

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


def test_environment_is_ignored() -> None:
    left = {"hyperparameters": {"selectedEpoch": 3, "environment": {"torch": "2.8.0+cpu"}}}
    right = {"hyperparameters": {"selectedEpoch": 3, "environment": {"torch": "2.8.0"}}}
    assert compare(left, right) == []
    right["hyperparameters"]["selectedEpoch"] = 4
    assert compare(left, right) == ["$.hyperparameters.selectedEpoch: 3 erwartet, 4 gefunden"]


def test_floats_are_compared_with_tolerance() -> None:
    close = copy.deepcopy(PAYLOAD)
    close["runs"][0]["prAuc"] = 0.50005
    far = copy.deepcopy(PAYLOAD)
    far["runs"][0]["prAuc"] = 0.5002
    assert compare(PAYLOAD, close) == []
    assert compare(PAYLOAD, far) == ["$.runs[0].prAuc: 0.5 erwartet, 0.5002 gefunden"]


def test_one_rounding_step_of_four_digit_values_is_within_tolerance() -> None:
    assert abs(0.1234 - 0.1235) > 1e-4
    assert compare(0.1234, 0.1235) == []
    assert compare(5.0001, 5.0002) == []
    assert compare(12.3456, 12.3457) == []
    assert compare(-19.9567, -19.9568) == []
    assert compare(0.1234, 0.1236) == ["$: 0.1234 erwartet, 0.1236 gefunden"]
    assert compare(-19.9567, -19.9569) == ["$: -19.9567 erwartet, -19.9569 gefunden"]


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
    monkeypatch.delenv(REQUIRE_GNN_ENV, raising=False)
    monkeypatch.setattr(verify_module, "recompute_synthetic", lambda include_gnn=False: RECOMPUTED)
    export_dir = tmp_path / "k3"
    runs_dir = tmp_path / "runs"
    write_committed(export_dir, runs_dir)
    return export_dir, runs_dir


def test_gnn_runs_are_recomputed_when_present(
    committed: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    export_dir, runs_dir = committed
    requested: list[bool] = []

    def recompute(include_gnn: bool = False) -> Recomputation:
        requested.append(include_gnn)
        return RECOMPUTED

    monkeypatch.setattr(verify_module, "recompute_synthetic", recompute)
    monkeypatch.setattr(verify_module, "gnn_available", lambda: True)
    assert verify(export_dir, runs_dir) == 0
    with_gnn = copy.deepcopy(PAYLOAD)
    with_gnn["runs"].append({"method": "gcn", "date": "2026-01-01", "prAuc": 0.5})
    write_json(export_dir / "metrics.json", with_gnn)
    assert verify(export_dir, runs_dir) == 1
    assert requested == [False, True]


def test_missing_torch_with_gnn_runs_fails_with_a_hint(
    committed: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    export_dir, runs_dir = committed
    monkeypatch.setattr(verify_module, "gnn_available", lambda: False)
    with_gnn = copy.deepcopy(PAYLOAD)
    with_gnn["runs"].append({"method": "mlp", "date": "2026-01-01", "prAuc": 0.5})
    write_json(export_dir / "metrics.json", with_gnn)
    assert verify(export_dir, runs_dir) == 1
    assert "uv sync --extra gnn" in capsys.readouterr().err
    write_json(export_dir / "metrics.json", PAYLOAD)
    assert verify(export_dir, runs_dir) == 0


def test_required_gnn_runs_must_be_committed(
    committed: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    export_dir, runs_dir = committed
    requested: list[bool] = []

    def recompute(include_gnn: bool = False) -> Recomputation:
        requested.append(include_gnn)
        return RECOMPUTED

    monkeypatch.setattr(verify_module, "recompute_synthetic", recompute)
    monkeypatch.setattr(verify_module, "gnn_available", lambda: True)
    monkeypatch.setenv(REQUIRE_GNN_ENV, "1")
    assert verify(export_dir, runs_dir) == 1
    assert "keine GNN-Runs" in capsys.readouterr().err
    assert requested == []
    monkeypatch.setenv(REQUIRE_GNN_ENV, "0")
    assert verify(export_dir, runs_dir) == 0
    with_gnn = copy.deepcopy(PAYLOAD)
    with_gnn["runs"].append({"method": "graphsage", "date": "2026-01-01", "prAuc": 0.5})
    write_json(export_dir / "metrics.json", with_gnn)
    monkeypatch.setenv(REQUIRE_GNN_ENV, "1")
    assert verify(export_dir, runs_dir) == 1
    assert requested == [False, True]


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
