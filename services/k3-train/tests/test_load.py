import shutil
from pathlib import Path

import pandas as pd
import pytest

from k3_train.features import feature_matrix
from k3_train.load import (
    ELLIPTIC_FILES,
    Dataset,
    clean_edges,
    elliptic_feature_names,
    load_elliptic,
    missing_elliptic_files,
    read_processed,
    write_processed,
)


def test_elliptic_fixture_is_loaded_into_common_format(elliptic_mini_dir: Path) -> None:
    dataset = load_elliptic(elliptic_mini_dir, aggregated_count=1)
    assert dataset.name == "elliptic"
    assert len(dataset.nodes) == 8
    assert dataset.feature_columns == ["f_001", "f_002", "f_003"]
    assert dataset.aggregated_columns == ["a_001"]
    assert dataset.notes == {"featureCount": 4, "localFeatures": 3, "aggregatedFeatures": 1}
    assert dataset.nodes.columns.tolist()[:3] == ["node_id", "time_step", "label"]
    labels = dataset.nodes.set_index("node_id")["label"]
    assert labels["9000003"] == "illicit"
    assert labels["9000002"] == "licit"
    assert labels["9000001"] == "unknown"
    assert dataset.label_counts() == {"illicit": 2, "licit": 3, "unknown": 3}
    assert dataset.nodes["time_step"].tolist() == [33, 33, 34, 34, 35, 35, 36, 36]


def test_elliptic_aggregated_features_stay_out_of_the_local_set(elliptic_mini_dir: Path) -> None:
    dataset = load_elliptic(elliptic_mini_dir, aggregated_count=1)
    values, names = feature_matrix(dataset.nodes, None, "local")
    assert names == ["f_001", "f_002", "f_003"]
    assert values.shape == (8, 3)
    first = dataset.nodes.set_index("node_id").loc["9000001"]
    assert first["a_001"] == 2


def test_elliptic_feature_names_split_local_and_aggregated() -> None:
    names = elliptic_feature_names(165, 72)
    assert len(names) == 165
    assert names[:2] == ["f_001", "f_002"]
    assert names[92] == "f_093"
    assert names[93] == "a_001"
    assert names[-1] == "a_072"
    with pytest.raises(ValueError, match="aggregierte"):
        elliptic_feature_names(4, 4)


def test_elliptic_without_checked_layout_is_rejected(elliptic_mini_dir: Path) -> None:
    with pytest.raises(ValueError, match="165"):
        load_elliptic(elliptic_mini_dir)


def test_elliptic_edges_are_cleaned_and_reported(elliptic_mini_dir: Path) -> None:
    dataset = load_elliptic(elliptic_mini_dir, aggregated_count=1)
    report = dataset.cleaning
    assert report.input_edges == 8
    assert report.removed_missing_endpoint == 1
    assert report.removed_self_loops == 1
    assert report.removed_duplicates == 1
    assert report.removed_total == 3
    assert len(dataset.edges) == 5
    pairs = list(zip(dataset.edges["source"], dataset.edges["target"], strict=True))
    assert ("9000005", "9000006") in pairs
    assert ("9000006", "9000005") in pairs


def test_missing_elliptic_files_are_named(tmp_path: Path) -> None:
    assert missing_elliptic_files(tmp_path) == list(ELLIPTIC_FILES.values())
    with pytest.raises(FileNotFoundError, match=r"elliptic_txs_features\.csv"):
        load_elliptic(tmp_path)


def test_unknown_elliptic_class_is_rejected(elliptic_mini_dir: Path, tmp_path: Path) -> None:
    for name in ELLIPTIC_FILES.values():
        shutil.copy(elliptic_mini_dir / name, tmp_path / name)
    classes = tmp_path / ELLIPTIC_FILES["classes"]
    classes.write_text(
        classes.read_text(encoding="utf-8").replace("9000002,2", "9000002,3"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Klassen"):
        load_elliptic(tmp_path, aggregated_count=1)


def test_clean_edges_hand_example() -> None:
    nodes = pd.DataFrame({"node_id": ["a", "b", "c"]})
    edges = pd.DataFrame(
        {
            "source": ["b", "a", "a", "b", "c"],
            "target": ["c", "b", "b", "b", "x"],
        }
    )
    cleaned, report = clean_edges(nodes, edges)
    assert list(zip(cleaned["source"], cleaned["target"], strict=True)) == [("a", "b"), ("b", "c")]
    assert (report.removed_missing_endpoint, report.removed_self_loops) == (1, 1)
    assert report.removed_duplicates == 1


def test_synthetic_data_needs_no_cleaning(small_dataset: Dataset) -> None:
    assert small_dataset.cleaning.removed_total == 0
    assert small_dataset.cleaning.input_edges == len(small_dataset.edges)


def test_elliptic_processed_files_keep_aggregated_columns(
    elliptic_mini_dir: Path, tmp_path: Path
) -> None:
    dataset = load_elliptic(elliptic_mini_dir, aggregated_count=1)
    write_processed(dataset, tmp_path)
    restored = read_processed(tmp_path)
    assert restored.feature_columns == ["f_001", "f_002", "f_003"]
    assert restored.aggregated_columns == ["a_001"]
    pd.testing.assert_frame_equal(restored.nodes, dataset.nodes, check_dtype=False)


def test_processed_files_round_trip_exactly(small_dataset: Dataset, tmp_path: Path) -> None:
    write_processed(small_dataset, tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {
        "nodes.csv",
        "edges.csv",
        "truth.csv",
        "meta.json",
    }
    restored = read_processed(tmp_path)
    pd.testing.assert_frame_equal(restored.nodes, small_dataset.nodes, check_dtype=False)
    pd.testing.assert_frame_equal(restored.edges, small_dataset.edges, check_dtype=False)
    assert restored.feature_columns == small_dataset.feature_columns
    assert restored.generator == small_dataset.generator
    assert restored.cleaning == small_dataset.cleaning
