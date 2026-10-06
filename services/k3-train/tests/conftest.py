from pathlib import Path

import pytest

from k3_train.load import Dataset, load_synthetic
from k3_train.pipeline import BaselineOutput, run_baseline
from k3_train.synth import SMALL_CONFIG, SynthConfig

FIXTURES = Path(__file__).parent / "fixtures"
RUN_DATE = "2026-01-01"


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
