import importlib.util
import os
from pathlib import Path

import pytest

from k3_train.load import Dataset, load_synthetic
from k3_train.pipeline import BaselineOutput, run_baseline
from k3_train.synth import SMALL_CONFIG, SynthConfig
from k3_train.verify import REQUIRE_GNN_ENV

FIXTURES = Path(__file__).parent / "fixtures"
RUN_DATE = "2026-01-01"
GNN_MODULES = ("torch", "torch_geometric")
GNN_SKIP_REASON = "Extra gnn fehlt (torch, torch_geometric): uv sync --extra gnn"


def gnn_installed() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in GNN_MODULES)


def pytest_configure(config: pytest.Config) -> None:
    if os.environ.get(REQUIRE_GNN_ENV) == "1" and not gnn_installed():
        raise pytest.UsageError(
            f"{REQUIRE_GNN_ENV}=1, aber torch oder torch_geometric fehlt. "
            "GNN-Tests dürfen hier nicht übersprungen werden: uv sync --extra gnn"
        )


@pytest.fixture(scope="session")
def elliptic_mini_dir() -> Path:
    return FIXTURES / "elliptic_mini"


@pytest.fixture(scope="session")
def small_dataset() -> Dataset:
    return load_synthetic(SMALL_CONFIG)


@pytest.fixture(scope="session")
def default_dataset() -> Dataset:
    return load_synthetic(SynthConfig())


@pytest.fixture(scope="session")
def small_output(small_dataset: Dataset) -> BaselineOutput:
    return run_baseline(small_dataset, run_date=RUN_DATE)
