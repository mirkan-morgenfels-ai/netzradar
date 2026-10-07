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
    assert DATA_DIR_ENV in message


def test_data_with_missing_elliptic_files_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["data", "--dataset", "elliptic", "--raw-dir", str(tmp_path)]) == 2
    assert "elliptic_txs_features.csv" in capsys.readouterr().err


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
    assert main(["data", "--synth"]) == 2
