import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from k3_train import cli, paths
from k3_train.cli import MISSING_GNN_MESSAGE, main
from k3_train.paths import DATA_DIR_ENV, DEFAULT_DATA_DIR, data_root, processed_dir, raw_dir
from k3_train.synth import SMALL_CONFIG

TORCH_FREE_MODULES = (
    "k3_train.cli",
    "k3_train.export",
    "k3_train.verify",
    "k3_train.runs",
    "k3_train.training",
    "k3_train.gnn_results",
    "k3_train.pipeline",
    "k3_train.boosting",
    "k3_train.ibm_aml",
)
TORCH_PROBE = "\n".join(
    (
        "import importlib, json, sys",
        "for name in sys.argv[1:]:",
        "    importlib.import_module(name)",
        "roots = ('torch', 'torch_geometric')",
        "loaded = sorted(name for name in sys.modules if name.split('.')[0] in roots)",
        "sys.modules['torch'] = None",
        "sys.modules['torch_geometric'] = None",
        "from k3_train.cli import main",
        "status = main(['gnn'])",
        "print(json.dumps({'loaded': loaded, 'status': status}))",
    )
)


def test_core_modules_run_without_torch() -> None:
    result = subprocess.run(
        [sys.executable, "-c", TORCH_PROBE, *TORCH_FREE_MODULES],
        capture_output=True,
        check=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report == {"loaded": [], "status": 2}
    assert MISSING_GNN_MESSAGE in result.stderr


def test_data_without_dataset_explains_options(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["data"]) == 2
    message = capsys.readouterr().err
    assert "SYNTH=1" in message
    assert "DATASET=elliptic" in message
    assert "DATASET=ibm-aml" in message
    assert DATA_DIR_ENV in message


def test_data_with_missing_elliptic_files_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["data", "--dataset", "elliptic", "--raw-dir", str(tmp_path)]) == 2
    assert "elliptic_txs_features.csv" in capsys.readouterr().err


def test_data_with_missing_ibm_aml_file_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["data", "--dataset", "ibm-aml", "--raw-dir", str(tmp_path)]) == 2
    message = capsys.readouterr().err
    assert "HI-Small_Trans.csv" in message
    assert "Kaggle ealtman2019/ibm-transactions-for-anti-money-laundering-aml" in message
    assert "CDLA-Sharing-1.0" in message


def test_ibm_aml_stays_in_the_data_folder_and_is_never_exported(
    ibm_aml_mini_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    public_runs = tmp_path / "public-runs"
    monkeypatch.setattr(cli, "RUNS_DIR", public_runs)
    monkeypatch.delenv(DATA_DIR_ENV, raising=False)
    data_dir = str(tmp_path / "data")
    raw = ["--raw-dir", str(ibm_aml_mini_dir)]
    assert main(["data", "--dataset", "ibm-aml", *raw, "--data-dir", data_dir]) == 0
    output = capsys.readouterr().out
    assert "Datensatz: ibm-aml" in output
    assert "Knoten: 26, Kanten: 4, Zeitschritte: 10" in output
    assert "Fenster 360 min im selben Tag" in output
    assert "davon mit Is Laundering 1: 2" in output
    assert (
        "Anteile nach Transaktionen: Training 1-6 ohne Validierung 18 (69.23%), "
        "Validierung 7-8 4 (15.38%), Test 9-10 4 (15.38%)"
    ) in output
    assert main(["baseline", "--dataset", "ibm-aml", "--data-dir", data_dir]) == 0
    private_runs = tmp_path / "data" / "processed" / "ibm-aml" / "runs"
    assert sorted(path.name.split("_", 1)[1] for path in private_runs.iterdir()) == [
        "ibm-aml_hgb.json",
        "ibm-aml_iforest.json",
        "ibm-aml_zscore.json",
    ]
    assert not public_runs.exists()
    capsys.readouterr()
    assert main(["export", "--dataset", "ibm-aml", "--data-dir", data_dir]) == 2
    message = capsys.readouterr().err
    assert "nicht veröffentlicht" in message
    assert "CDLA-Sharing-1.0" in message
    assert "synthetische Netz" in message


def test_export_of_elliptic_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["export", "--dataset", "elliptic"]) == 2
    assert "nicht veröffentlicht" in capsys.readouterr().err


def test_unknown_dataset_is_rejected_by_parser() -> None:
    with pytest.raises(SystemExit):
        main(["baseline", "--dataset", "ibm"])
    with pytest.raises(SystemExit):
        main(["gnn", "--dataset", "ibm"])


def test_gnn_command_is_registered() -> None:
    args = cli.build_parser().parse_args(["gnn", "--max-epochs", "7"])
    assert args.handler is cli.command_gnn
    assert args.max_epochs == 7
    assert args.dataset == "synthetic"


def test_runs_command_is_registered() -> None:
    args = cli.build_parser().parse_args(["runs", "--data-dir", "x"])
    assert args.handler is cli.command_runs
    assert args.dataset == "synthetic"
    assert args.data_dir == "x"


def test_data_root_defaults_to_the_service_folder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DATA_DIR_ENV, raising=False)
    assert data_root() == DEFAULT_DATA_DIR
    assert raw_dir(data_root(), "elliptic") == DEFAULT_DATA_DIR / "raw" / "elliptic"


def test_data_root_follows_environment_and_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / "from-env"))
    assert data_root() == (tmp_path / "from-env").resolve()
    assert data_root(tmp_path / "from-option") == (tmp_path / "from-option").resolve()
    assert processed_dir(data_root(), "elliptic") == (
        (tmp_path / "from-env").resolve() / "processed" / "elliptic"
    )


def test_data_dir_option_keeps_the_service_folder_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = tmp_path / "service-data"
    monkeypatch.setattr(paths, "DEFAULT_DATA_DIR", sentinel)
    monkeypatch.setattr(cli, "SynthConfig", lambda: SMALL_CONFIG)
    monkeypatch.delenv(DATA_DIR_ENV, raising=False)
    outside = tmp_path / "outside"
    assert main(["data", "--synth", "--data-dir", str(outside)]) == 0
    assert (outside / "processed" / "synthetic" / "meta.json").is_file()
    assert not sentinel.exists()


def test_commands_refuse_to_run_outside_the_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(paths, "REPO_MARKERS", (tmp_path / "pnpm-workspace.yaml",))
    monkeypatch.delenv(DATA_DIR_ENV, raising=False)
    assert main(["verify"]) == 2
    assert "Projektordner nicht gefunden" in capsys.readouterr().err
    assert main(["export"]) == 2
    assert main(["baseline"]) == 2
    assert main(["runs"]) == 2
    assert main(["data", "--synth"]) == 2
