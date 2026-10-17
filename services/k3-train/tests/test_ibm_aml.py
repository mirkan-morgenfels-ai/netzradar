import math
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from k3_train.datasets import dataset_info
from k3_train.features import feature_matrix
from k3_train.ibm_aml import (
    BASE_FEATURES,
    IBM_AML_HEADER,
    IbmAmlConfig,
    category_slug,
    header_problems,
    load_ibm_aml,
    missing_ibm_aml_files,
    transaction_edges,
)
from k3_train.load import Dataset, read_processed, write_processed
from k3_train.pipeline import run_baseline

FILE = "HI-Small_Trans.csv"
HEADER = ",".join(IBM_AML_HEADER)
CATEGORY_COLUMNS = [
    "f_format_ach",
    "f_format_bitcoin",
    "f_format_cash",
    "f_format_cheque",
    "f_format_credit_card",
    "f_format_reinvestment",
    "f_format_wire",
    "f_paid_bitcoin",
    "f_paid_euro",
    "f_paid_us_dollar",
    "f_paid_yuan",
    "f_received_bitcoin",
    "f_received_euro",
    "f_received_us_dollar",
    "f_received_yuan",
]
FIXTURE_EDGES = [
    ("t0000001", "t0000002"),
    ("t0000001", "t0000003"),
    ("t0000008", "t0000009"),
    ("t0000009", "t0000008"),
]
Rows = list[list[str]]


def fixture_rows(fixture_dir: Path) -> Rows:
    lines = (fixture_dir / FILE).read_text(encoding="utf-8").splitlines()
    return [line.split(",") for line in lines[1:]]


def write_file(target: Path, rows: Rows, header: str = HEADER) -> Path:
    lines = [header, *(",".join(row) for row in rows)]
    (target / FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def edge_pairs(dataset: Dataset) -> list[tuple[str, str]]:
    return list(zip(dataset.edges["source"], dataset.edges["target"], strict=True))


@pytest.fixture(scope="module")
def mini(ibm_aml_mini_dir: Path) -> Dataset:
    return load_ibm_aml(ibm_aml_mini_dir)


def test_ibm_aml_is_registered_as_local_only() -> None:
    info = dataset_info("ibm-aml")
    assert info.license == "CDLA-Sharing-1.0"
    assert info.source == "Kaggle ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
    assert info.publishable is False
    assert "synthetische Netz" in info.withheld_reason


def test_fixture_is_loaded_into_common_format(mini: Dataset) -> None:
    assert mini.name == "ibm-aml"
    assert len(mini.nodes) == 26
    assert mini.time_steps == 10
    assert mini.nodes.columns.tolist()[:3] == ["node_id", "time_step", "label"]
    assert mini.nodes["node_id"].tolist() == [f"t{row:07d}" for row in range(1, 27)]
    assert Counter(mini.nodes["time_step"]) == {1: 7, 2: 3, **dict.fromkeys(range(3, 11), 2)}
    assert mini.label_counts() == {"illicit": 6, "licit": 20, "unknown": 0}
    labels = mini.nodes.set_index("node_id")["label"]
    assert labels["t0000002"] == "illicit"
    assert labels["t0000001"] == "licit"
    assert mini.feature_columns == [*BASE_FEATURES, *CATEGORY_COLUMNS]
    assert mini.aggregated_columns == []
    assert mini.generator is None


def test_notes_record_construction_and_cleaning(mini: Dataset) -> None:
    assert mini.notes == {
        "file": FILE,
        "documentedDays": 10,
        "timeStep": "calendarDay",
        "windowMinutes": 360,
        "maxSuccessors": 10,
        "maxPredecessors": 10,
        "sameTimeStepOnly": True,
        "inputRows": 28,
        "firstDay": "2022-09-01",
        "lastDay": "2022-09-10",
        "removedAfterRange": 2,
        "removedAfterRangeLaundering": 2,
        "accounts": 39,
        "windowPairs": 4,
        "windowPairsAcrossTimeSteps": 1,
        "removedByDegreeLimit": 0,
        "paymentFormats": [
            "ACH",
            "Bitcoin",
            "Cash",
            "Cheque",
            "Credit Card",
            "Reinvestment",
            "Wire",
        ],
        "paymentCurrencies": ["Bitcoin", "Euro", "US Dollar", "Yuan"],
        "receivingCurrencies": ["Bitcoin", "Euro", "US Dollar", "Yuan"],
    }


def test_edges_follow_the_receiving_account_within_the_window(mini: Dataset) -> None:
    pairs = edge_pairs(mini)
    assert pairs == FIXTURE_EDGES
    assert ("t0000001", "t0000004") not in pairs
    assert ("t0000004", "t0000006") not in pairs
    assert ("t0000006", "t0000007") not in pairs
    assert ("t0000005", "t0000008") not in pairs
    assert "t0000010" not in set(mini.edges["source"]) | set(mini.edges["target"])
    assert mini.cleaning.removed_total == 0


def test_no_edge_crosses_a_time_step_even_with_the_widest_window(ibm_aml_mini_dir: Path) -> None:
    wide = load_ibm_aml(ibm_aml_mini_dir, IbmAmlConfig(window_minutes=1439))
    assert edge_pairs(wide) == [
        ("t0000001", "t0000002"),
        ("t0000001", "t0000003"),
        ("t0000001", "t0000004"),
        ("t0000002", "t0000005"),
        ("t0000008", "t0000009"),
        ("t0000009", "t0000008"),
    ]
    assert wide.notes["windowPairs"] == 6
    assert wide.notes["windowPairsAcrossTimeSteps"] == 3
    steps = wide.nodes.set_index("node_id")["time_step"]
    assert (steps[wide.edges["source"]].to_numpy() == steps[wide.edges["target"]].to_numpy()).all()


def test_degree_limits_on_the_fixture(ibm_aml_mini_dir: Path) -> None:
    config = IbmAmlConfig(window_minutes=1439, max_successors=1, max_predecessors=1)
    limited = load_ibm_aml(ibm_aml_mini_dir, config)
    assert edge_pairs(limited) == [
        ("t0000001", "t0000002"),
        ("t0000002", "t0000005"),
        ("t0000008", "t0000009"),
        ("t0000009", "t0000008"),
    ]
    assert limited.notes["removedByDegreeLimit"] == 2


def test_window_boundary_is_inclusive() -> None:
    steps, minutes = [1, 1, 1], [0, 360, 361]
    senders, receivers = [0, 1, 1], [1, 2, 3]
    exact = transaction_edges(steps, minutes, senders, receivers, IbmAmlConfig(window_minutes=360))
    assert list(zip(exact.source, exact.target, strict=True)) == [(0, 1)]
    assert (exact.window_pairs, exact.cross_step_pairs, exact.removed_by_limit) == (1, 0, 0)
    wider = transaction_edges(steps, minutes, senders, receivers, IbmAmlConfig(window_minutes=361))
    assert list(zip(wider.source, wider.target, strict=True)) == [(0, 1), (0, 2)]


def test_only_later_or_same_minute_transactions_are_successors() -> None:
    edges = transaction_edges(
        steps=[1, 1, 1],
        minutes=[100, 99, 100],
        senders=[0, 1, 1],
        receivers=[1, 2, 0],
        config=IbmAmlConfig(),
    )
    assert list(zip(edges.source, edges.target, strict=True)) == [(0, 2), (2, 0)]


def test_successor_limit_keeps_the_earliest() -> None:
    edges = transaction_edges(
        steps=[1, 1, 1, 1, 1, 1],
        minutes=[0, 40, 10, 30, 10, 20],
        senders=[0, 1, 1, 1, 1, 1],
        receivers=[1, 2, 3, 4, 5, 6],
        config=IbmAmlConfig(max_successors=2),
    )
    assert list(zip(edges.source, edges.target, strict=True)) == [(0, 2), (0, 4)]
    assert (edges.window_pairs, edges.removed_by_limit) == (5, 3)


def test_predecessor_limit_keeps_the_latest() -> None:
    edges = transaction_edges(
        steps=[1, 1, 1, 1, 1],
        minutes=[0, 30, 10, 30, 40],
        senders=[0, 1, 2, 3, 9],
        receivers=[9, 9, 9, 9, 8],
        config=IbmAmlConfig(max_predecessors=2),
    )
    assert list(zip(edges.source, edges.target, strict=True)) == [(1, 4), (3, 4)]
    assert edges.removed_by_limit == 2


def test_self_transfer_is_not_its_own_successor() -> None:
    edges = transaction_edges([1, 1], [0, 5], [0, 0], [0, 1], IbmAmlConfig())
    assert list(zip(edges.source, edges.target, strict=True)) == [(0, 1)]
    assert edges.window_pairs == 1


def brute_force(
    steps: np.ndarray, minutes: np.ndarray, senders: np.ndarray, receivers: np.ndarray, window: int
) -> tuple[set[tuple[int, int]], int]:
    same_step, across = set(), 0
    for first in range(len(steps)):
        for second in range(len(steps)):
            gap = minutes[second] - minutes[first]
            if first == second or receivers[first] != senders[second] or not 0 <= gap <= window:
                continue
            if steps[first] == steps[second]:
                same_step.add((first, second))
            else:
                across += 1
    return same_step, across


def test_edges_match_a_brute_force_search_on_random_transactions() -> None:
    rng = np.random.default_rng(7)
    count = 300
    steps = rng.integers(1, 4, count)
    minutes = (steps - 1) * 1440 + rng.integers(0, 1440, count)
    senders = rng.integers(0, 15, count)
    receivers = rng.integers(0, 15, count)
    expected, across = brute_force(steps, minutes, senders, receivers, window=240)
    assert expected
    assert across > 0
    full = transaction_edges(
        steps,
        minutes,
        senders,
        receivers,
        IbmAmlConfig(window_minutes=240, max_successors=count, max_predecessors=count),
    )
    assert set(zip(full.source.tolist(), full.target.tolist(), strict=True)) == expected
    assert len(full.source) == len(expected)
    assert (full.window_pairs, full.cross_step_pairs) == (len(expected), across)
    limited = transaction_edges(
        steps,
        minutes,
        senders,
        receivers,
        IbmAmlConfig(window_minutes=240, max_successors=2, max_predecessors=3),
    )
    kept = set(zip(limited.source.tolist(), limited.target.tolist(), strict=True))
    assert kept <= expected
    assert max(Counter(limited.source.tolist()).values()) <= 2
    assert max(Counter(limited.target.tolist()).values()) <= 3
    assert limited.removed_by_limit == len(expected) - len(kept) > 0
    assert (steps[limited.source] == steps[limited.target]).all()


def test_local_features_by_hand(mini: Dataset) -> None:
    nodes = mini.nodes.set_index("node_id")
    first = nodes.loc["t0000001"]
    assert first["f_log_amount_paid"] == round(math.log(101), 6) == 4.615121
    assert (first["f_time_sin"], first["f_time_cos"]) == (0.866025, -0.5)
    assert (first["f_same_bank"], first["f_self_transfer"], first["f_whole_amount_paid"]) == (
        1,
        0,
        1,
    )
    assert (first["f_format_ach"], first["f_paid_us_dollar"], first["f_received_us_dollar"]) == (
        1,
        1,
        1,
    )
    second = nodes.loc["t0000002"]
    assert second["f_log_amount_paid"] == round(math.log(96.5), 6)
    assert (second["f_whole_amount_paid"], second["f_same_bank"], second["f_format_wire"]) == (
        0,
        0,
        1,
    )
    late = nodes.loc["t0000005"]
    assert late["f_cross_currency"] == 1
    assert late["f_log_amount_paid"] == round(math.log(81), 6) == 4.394449
    assert late["f_log_amount_received"] == round(math.log(71), 6) == 4.26268
    assert late["f_time_sin"] == round(-math.sin(math.pi / 24), 6) == -0.130526
    assert late["f_time_cos"] == round(math.cos(math.pi / 24), 6) == 0.991445
    assert (late["f_paid_us_dollar"], late["f_received_euro"]) == (1, 1)
    noon = nodes.loc["t0000010"]
    assert (noon["f_self_transfer"], noon["f_same_bank"], noon["f_format_reinvestment"]) == (
        1,
        1,
        1,
    )
    assert (noon["f_time_sin"], noon["f_time_cos"]) == (0.0, -1.0)
    assert math.copysign(1.0, noon["f_time_sin"]) == 1.0
    for prefix in ("f_format_", "f_paid_", "f_received_"):
        group = [column for column in mini.feature_columns if column.startswith(prefix)]
        assert (mini.nodes[group].sum(axis=1) == 1).all()
    values, names = feature_matrix(mini.nodes, None, "local")
    assert names == mini.feature_columns
    assert values.shape == (26, 23)
    assert np.isfinite(values).all()


def test_category_names_become_column_names(ibm_aml_mini_dir: Path, tmp_path: Path) -> None:
    assert category_slug("Credit Card") == "credit_card"
    assert category_slug("US Dollar") == "us_dollar"
    assert category_slug(" Saudi Riyal ") == "saudi_riyal"
    with pytest.raises(ValueError, match="verwertbaren Namen"):
        category_slug("--")
    rows = fixture_rows(ibm_aml_mini_dir)
    rows[1][9] = "Credit-Card"
    write_file(tmp_path, rows)
    with pytest.raises(
        ValueError, match=r"gleichem Spaltennamen in payment_format: \['credit_card'\]"
    ):
        load_ibm_aml(tmp_path)


def test_features_and_edges_ignore_labels(ibm_aml_mini_dir: Path, tmp_path: Path) -> None:
    rows = fixture_rows(ibm_aml_mini_dir)
    flipped = [[*row[:-1], "1" if row[-1] == "0" else "0"] for row in rows]
    original = load_ibm_aml(ibm_aml_mini_dir)
    changed = load_ibm_aml(write_file(tmp_path, flipped))
    columns = original.feature_columns
    pd.testing.assert_frame_equal(changed.nodes[columns], original.nodes[columns])
    pd.testing.assert_frame_equal(changed.edges, original.edges)
    assert (changed.nodes["label"] != original.nodes["label"]).all()


def test_earlier_days_ignore_later_days(ibm_aml_mini_dir: Path, tmp_path: Path) -> None:
    rows = [row for row in fixture_rows(ibm_aml_mini_dir) if row[0] < "2022/09/09"]
    original = load_ibm_aml(ibm_aml_mini_dir)
    shortened = load_ibm_aml(write_file(tmp_path, rows))
    assert shortened.time_steps == 8
    assert shortened.feature_columns == original.feature_columns
    early = original.nodes[original.nodes["time_step"] <= 8].reset_index(drop=True)
    pd.testing.assert_frame_equal(shortened.nodes, early)
    pd.testing.assert_frame_equal(shortened.edges, original.edges)


def test_processed_files_round_trip(mini: Dataset, tmp_path: Path) -> None:
    write_processed(mini, tmp_path)
    restored = read_processed(tmp_path)
    assert restored.name == "ibm-aml"
    pd.testing.assert_frame_equal(restored.nodes, mini.nodes, check_dtype=False)
    pd.testing.assert_frame_equal(restored.edges, mini.edges, check_dtype=False)
    assert restored.feature_columns == mini.feature_columns
    assert restored.notes == mini.notes


def test_baseline_pipeline_accepts_the_fixture(mini: Dataset) -> None:
    output = run_baseline(mini, run_date="2026-01-01")
    assert output.split == {
        "kind": "temporal",
        "train": {"from": 1, "to": 8},
        "validation": {"from": 7, "to": 8},
        "test": {"from": 9, "to": 10},
        "crossSplitEdges": 0,
    }
    assert (output.evaluation["testPositives"], output.evaluation["testNegatives"]) == (1, 3)
    assert [run["method"] for run in output.runs] == ["zscore", "iforest", "hgb"]
    assert output.dataset["name"] == "ibm-aml"
    assert output.dataset["license"] == "CDLA-Sharing-1.0"
    assert output.dataset["generator"] is None
    assert output.dataset["features"] == 23


@pytest.mark.parametrize(
    ("header", "problems"),
    [
        (list(IBM_AML_HEADER), []),
        ([*IBM_AML_HEADER[:4], "Account.1", *IBM_AML_HEADER[5:]], []),
        (
            [name for name in IBM_AML_HEADER if name != "Payment Format"],
            ["fehlende Spalten: Payment Format"],
        ),
        (
            [*IBM_AML_HEADER[:4], *IBM_AML_HEADER[5:]],
            ["fehlende Spalten: Account"],
        ),
        (
            [*IBM_AML_HEADER[:-1], "Label"],
            ["fehlende Spalten: Is Laundering", "unerwartete Spalten: Label"],
        ),
        (
            [*IBM_AML_HEADER, "Note"],
            ["unerwartete Spalten: Note"],
        ),
        (
            [IBM_AML_HEADER[1], IBM_AML_HEADER[0], *IBM_AML_HEADER[2:]],
            ["Reihenfolge der Spalten weicht ab"],
        ),
    ],
)
def test_header_problems_are_named(header: list[str], problems: list[str]) -> None:
    assert header_problems(header) == problems


def test_missing_file_is_named(tmp_path: Path) -> None:
    assert missing_ibm_aml_files(tmp_path) == [FILE]
    with pytest.raises(FileNotFoundError, match=FILE):
        load_ibm_aml(tmp_path)


def test_missing_column_is_rejected_with_the_expected_header(
    ibm_aml_mini_dir: Path, tmp_path: Path
) -> None:
    rows = [row[:9] + row[10:] for row in fixture_rows(ibm_aml_mini_dir)]
    header = ",".join(name for name in IBM_AML_HEADER if name != "Payment Format")
    write_file(tmp_path, rows, header=header)
    with pytest.raises(ValueError, match="fehlende Spalten: Payment Format") as error:
        load_ibm_aml(tmp_path)
    assert HEADER.replace(",", ", ") in str(error.value)


def change_row(index: int, column: int, value: str) -> Callable[[Rows], Rows]:
    def change(rows: Rows) -> Rows:
        changed = [list(row) for row in rows]
        changed[index][column] = value
        return changed

    return change


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (change_row(0, 0, "2022-09-01 08:00"), "Zeitstempel passen nicht"),
        (change_row(0, 7, "-1.00"), "Amount Paid: 1 Werte"),
        (change_row(0, 5, "abc"), "Amount Received: 1 Werte"),
        (change_row(0, 10, "2"), "Is Laundering enthält andere Werte"),
        (change_row(0, 2, ""), r"Leere Werte in den Spalten: Account \(Spalte 3\)"),
        (lambda rows: [row for row in rows if not row[0].startswith("2022/09/05")], r"\[5\]"),
        (lambda rows: [], "keine Transaktionen"),
    ],
)
def test_broken_values_are_rejected(
    ibm_aml_mini_dir: Path, tmp_path: Path, change: Callable[[Rows], Rows], message: str
) -> None:
    write_file(tmp_path, change(fixture_rows(ibm_aml_mini_dir)))
    with pytest.raises(ValueError, match=message):
        load_ibm_aml(tmp_path)


def test_empty_file_and_byte_order_mark(ibm_aml_mini_dir: Path, tmp_path: Path) -> None:
    (tmp_path / FILE).write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="leer"):
        load_ibm_aml(tmp_path)
    text = (ibm_aml_mini_dir / FILE).read_text(encoding="utf-8")
    (tmp_path / FILE).write_text("﻿" + text, encoding="utf-8")
    pd.testing.assert_frame_equal(
        load_ibm_aml(tmp_path).nodes, load_ibm_aml(ibm_aml_mini_dir).nodes
    )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (IbmAmlConfig(window_minutes=1440), "Zeitfenster"),
        (IbmAmlConfig(window_minutes=-1), "Zeitfenster"),
        (IbmAmlConfig(max_successors=0), "Kantengrenzen"),
        (IbmAmlConfig(max_predecessors=0), "Kantengrenzen"),
        (IbmAmlConfig(documented_days=0), "dokumentierten Tage"),
    ],
)
def test_invalid_configuration_is_rejected(
    ibm_aml_mini_dir: Path, config: IbmAmlConfig, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_ibm_aml(ibm_aml_mini_dir, config)
