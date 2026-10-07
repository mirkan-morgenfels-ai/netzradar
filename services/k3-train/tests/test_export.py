import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from contract import validate_edges, validate_metrics, validate_nodes
from k3_train.export import (
    ExportBundle,
    build_exports,
    induced_edges,
    normalise_positions,
    select_neighbourhood,
    write_exports,
)
from k3_train.jsonio import dumps
from k3_train.load import Dataset, load_synthetic
from k3_train.pipeline import BaselineOutput, run_baseline
from k3_train.split import split_for, split_masks
from k3_train.synth import SMALL_CONFIG

GENERATED_AT = "2026-01-01T00:00:00Z"
LAYOUT_PROBE = "\n".join(
    [
        "import json",
        "import pandas as pd",
        "from k3_train.export import layout_positions",
        "names = [f'n{index:03d}' for index in range(60)]",
        "pairs = [(names[i], names[i + 1]) for i in range(59) if i % 20 != 19]",
        "edges = pd.DataFrame(pairs, columns=['source', 'target'])",
        "print(json.dumps(layout_positions(names, edges, seed=42), sort_keys=True))",
    ]
)


@pytest.fixture(scope="module")
def bundle(small_dataset: Dataset, small_output: BaselineOutput) -> ExportBundle:
    return build_exports(small_dataset, small_output, generated_at=GENERATED_AT)


def node_ids(bundle: ExportBundle) -> set[str]:
    return {node["id"] for node in bundle.nodes["nodes"]}


def test_metrics_follow_the_contract(bundle: ExportBundle) -> None:
    assert validate_metrics(bundle.metrics) == []
    assert [run["method"] for run in bundle.metrics["runs"]] == ["zscore", "iforest"]
    assert [run["featureSet"] for run in bundle.metrics["runs"]] == ["local", "local+graph"]
    assert bundle.metrics["dataset"]["name"] == "synthetic"
    assert bundle.metrics["dataset"]["generator"]["seed"] == 42
    assert bundle.metrics["dataset"]["features"] == 12
    assert bundle.metrics["seed"] == 42


def test_evaluation_numbers_are_consistent(bundle: ExportBundle) -> None:
    evaluation = bundle.metrics["evaluation"]
    total = evaluation["testPositives"] + evaluation["testNegatives"]
    assert evaluation["prevalence"] == round(evaluation["testPositives"] / total, 4)
    assert evaluation["allLicitAccuracy"] == round(evaluation["testNegatives"] / total, 4)


def test_nodes_follow_the_contract(bundle: ExportBundle) -> None:
    assert validate_nodes(bundle.nodes) == []
    assert list(bundle.nodes) == ["schemaVersion", "selection", "scoreGnnMethod", "nodes"]
    assert bundle.nodes["scoreGnnMethod"] is None
    assert all(node["scoreGnn"] is None for node in bundle.nodes["nodes"])
    assert bundle.nodes["selection"] == {
        "scoreField": "scoreIforest",
        "pool": "test",
        "seeds": 50,
        "hops": 2,
        "maxNodes": 2000,
        "truncated": False,
    }


def test_edges_follow_the_contract(bundle: ExportBundle) -> None:
    assert validate_edges(bundle.edges, node_ids(bundle)) == []


def test_seeds_are_top_test_nodes_at_hop_zero(
    bundle: ExportBundle, small_dataset: Dataset, small_output: BaselineOutput
) -> None:
    split = split_for("synthetic", small_dataset.time_steps)
    seeds = [node for node in bundle.nodes["nodes"] if node["seedRank"] is not None]
    assert len(seeds) == 50
    assert sorted(node["seedRank"] for node in seeds) == list(range(1, 51))
    assert all(node["hop"] == 0 for node in seeds)
    assert all(split.test.contains(node["timeStep"]) for node in bundle.nodes["nodes"])
    test_ids = set(
        small_dataset.nodes.loc[split_masks(small_dataset.nodes, split)["test"], "node_id"]
    )
    ranking = small_output.scores[small_output.scores["node_id"].isin(test_ids)].sort_values(
        ["score_iforest", "node_id"], ascending=[False, True]
    )
    expected = ranking["node_id"].head(50).tolist()
    by_rank = [node["id"] for node in sorted(seeds, key=lambda node: node["seedRank"])]
    assert by_rank == expected


def test_hops_are_undirected_distances_to_nearest_seed(bundle: ExportBundle) -> None:
    nodes = {node["id"]: node for node in bundle.nodes["nodes"]}
    neighbours: dict[str, set[str]] = {}
    for edge in bundle.edges["edges"]:
        neighbours.setdefault(edge["source"], set()).add(edge["target"])
        neighbours.setdefault(edge["target"], set()).add(edge["source"])
    for node in nodes.values():
        if node["hop"] > 0:
            closer = [nodes[other]["hop"] for other in neighbours.get(node["id"], ())]
            assert node["hop"] - 1 in closer


def test_coordinates_are_normalised(bundle: ExportBundle) -> None:
    xs = [node["x"] for node in bundle.nodes["nodes"]]
    ys = [node["y"] for node in bundle.nodes["nodes"]]
    assert min(xs + ys) >= -1.0
    assert max(xs + ys) <= 1.0
    assert max(max(xs), -min(xs), max(ys), -min(ys)) == pytest.approx(1.0, abs=1e-4)


def test_normalise_positions_by_hand() -> None:
    result = normalise_positions(np.array([[0.0, 0.0], [4.0, 2.0], [2.0, 1.0]]))
    assert result.tolist() == [[-1.0, -0.5], [1.0, 0.5], [0.0, 0.0]]


def test_truncation_keeps_closest_nodes(
    small_dataset: Dataset, small_output: BaselineOutput
) -> None:
    truncated = build_exports(small_dataset, small_output, GENERATED_AT, max_nodes=60)
    nodes = truncated.nodes["nodes"]
    assert truncated.nodes["selection"]["truncated"] is True
    assert truncated.nodes["selection"]["maxNodes"] == 60
    assert len(nodes) == 60
    hops = [node["hop"] for node in nodes]
    assert hops == sorted(hops)
    assert sum(node["seedRank"] is not None for node in nodes) == 50
    assert validate_edges(truncated.edges, {node["id"] for node in nodes}) == []


def test_select_neighbourhood_hand_example() -> None:
    edges = pd.DataFrame({"source": ["a", "c", "d", "x"], "target": ["b", "b", "c", "a"]})
    scores = {"a": 0.9, "b": 0.1, "c": 0.2, "d": 0.3, "x": 0.95}
    selection = select_neighbourhood(["a", "b", "c", "d"], scores, edges, seed_count=1, hops=2)
    frame = selection.nodes.set_index("node_id")
    assert frame["hop"].to_dict() == {"a": 0, "b": 1, "c": 2}
    assert frame.loc["a", "seed_rank"] == 1
    assert selection.truncated is False


def test_induced_edges_drop_outside_nodes() -> None:
    edges = pd.DataFrame({"source": ["b", "a", "a"], "target": ["c", "b", "z"]})
    subset = induced_edges(edges, {"a", "b", "c"})
    assert list(zip(subset["source"], subset["target"], strict=True)) == [("a", "b"), ("b", "c")]


def test_export_files_are_formatted_and_byte_identical(
    bundle: ExportBundle, tmp_path: Path
) -> None:
    first = write_exports(bundle, tmp_path / "first")
    second = write_exports(bundle, tmp_path / "second")
    for left, right in zip(first, second, strict=True):
        content = left.read_bytes()
        assert content == right.read_bytes()
        assert b"\r" not in content
        assert content.endswith(b"}\n")
        assert content.startswith(b'{\n  "schemaVersion": 3,')


def test_layout_does_not_depend_on_the_hash_seed() -> None:
    outputs = set()
    for hash_seed in ("1", "2"):
        result = subprocess.run(
            [sys.executable, "-c", LAYOUT_PROBE],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.add(result.stdout)
    assert len(outputs) == 1


def test_full_pipeline_is_deterministic() -> None:
    def run_once() -> ExportBundle:
        dataset = load_synthetic(SMALL_CONFIG)
        output = run_baseline(dataset, run_date="2026-01-01")
        return build_exports(dataset, output, generated_at=GENERATED_AT)

    first = run_once()
    second = run_once()
    assert dumps(first.metrics) == dumps(second.metrics)
    assert dumps(first.nodes) == dumps(second.nodes)
    assert dumps(first.edges) == dumps(second.edges)
