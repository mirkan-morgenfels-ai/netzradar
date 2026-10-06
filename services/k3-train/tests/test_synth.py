import networkx as nx
import pandas as pd
import pytest

from k3_train.load import Dataset
from k3_train.synth import (
    LOCAL_FEATURES,
    SMALL_CONFIG,
    SynthConfig,
    build_pattern,
    generate,
)


def test_same_seed_gives_identical_network() -> None:
    first = generate(SMALL_CONFIG)
    second = generate(SMALL_CONFIG)
    pd.testing.assert_frame_equal(first.nodes, second.nodes)
    pd.testing.assert_frame_equal(first.edges, second.edges)
    pd.testing.assert_frame_equal(first.truth, second.truth)


def test_other_seed_changes_network() -> None:
    first = generate(SMALL_CONFIG)
    other = generate(SynthConfig(time_steps=12, nodes_per_step=80, seed=7))
    assert not first.edges.equals(other.edges)


def test_node_counts_per_time_step() -> None:
    result = generate(SMALL_CONFIG)
    assert len(result.nodes) == 12 * 80
    assert result.nodes["node_id"].is_unique
    counts = result.nodes["time_step"].value_counts().sort_index()
    assert counts.index.tolist() == list(range(1, 13))
    assert (counts == 80).all()


def test_edges_stay_within_their_time_step() -> None:
    result = generate(SMALL_CONFIG)
    steps = result.nodes.set_index("node_id")["time_step"]
    source_steps = steps.reindex(result.edges["source"]).to_numpy()
    target_steps = steps.reindex(result.edges["target"]).to_numpy()
    assert (source_steps == target_steps).all()


@pytest.mark.parametrize("fixture_name", ["small_dataset", "default_dataset"])
def test_every_time_step_is_one_weak_component(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    dataset: Dataset = request.getfixturevalue(fixture_name)
    for _, step_nodes in dataset.nodes.groupby("time_step"):
        ids = set(step_nodes["node_id"])
        graph = nx.DiGraph()
        graph.add_nodes_from(ids)
        inside = dataset.edges["source"].isin(ids)
        graph.add_edges_from(
            zip(
                dataset.edges.loc[inside, "source"],
                dataset.edges.loc[inside, "target"],
                strict=True,
            )
        )
        assert nx.number_weakly_connected_components(graph) == 1


def test_no_self_loops_and_no_duplicate_edges() -> None:
    edges = generate(SMALL_CONFIG).edges
    assert not (edges["source"] == edges["target"]).any()
    assert not edges.duplicated().any()


def test_local_features_are_complete() -> None:
    nodes = generate(SMALL_CONFIG).nodes
    features = [column for column in nodes.columns if column.startswith("f_")]
    assert features == list(LOCAL_FEATURES)
    assert len(features) == 12
    assert not nodes[features].isna().to_numpy().any()


def test_visible_labels_follow_hidden_patterns() -> None:
    result = generate(SMALL_CONFIG)
    pattern_nodes = set(result.truth["node_id"])
    labels = result.nodes.set_index("node_id")["label"]
    assert set(labels[labels == "illicit"].index) <= pattern_nodes
    assert not set(labels[labels == "licit"].index) & pattern_nodes
    assert set(labels.unique()) == {"illicit", "licit", "unknown"}


def test_illicit_neighbourhoods_are_homophilous(default_dataset: Dataset) -> None:
    assert default_dataset.truth is not None
    pattern_nodes = set(default_dataset.truth["node_id"])
    graph = nx.Graph()
    graph.add_edges_from(
        zip(default_dataset.edges["source"], default_dataset.edges["target"], strict=True)
    )

    def illicit_neighbour_share(members: set[str]) -> float:
        hits = total = 0
        for node in members:
            for neighbour in graph.neighbors(node):
                total += 1
                hits += neighbour in pattern_nodes
        return hits / total

    benign = set(graph.nodes) - pattern_nodes
    assert illicit_neighbour_share(pattern_nodes) > 10 * illicit_neighbour_share(benign)


def test_label_shares_are_measured_not_fixed(default_dataset: Dataset) -> None:
    counts = default_dataset.label_counts()
    assert sum(counts.values()) == 30 * 400
    assert 0 < counts["illicit"] < counts["licit"] < counts["unknown"]


def test_fan_in_pattern_shape() -> None:
    pattern = build_pattern("fan_in", 3, start=10)
    assert pattern.members == (10, 11, 12, 13)
    assert pattern.edges == ((10, 13), (11, 13), (12, 13))
    assert pattern.roles == ("smurf", "smurf", "smurf", "collector")
    assert pattern.exit == 13


def test_fan_out_pattern_shape() -> None:
    pattern = build_pattern("fan_out", 2, start=0)
    assert pattern.edges == ((0, 1), (0, 2))
    assert pattern.entry == 0


def test_cycle_pattern_closes() -> None:
    pattern = build_pattern("cycle", 3, start=5)
    assert pattern.edges == ((5, 6), (6, 7), (7, 5))


def test_chain_pattern_is_open() -> None:
    pattern = build_pattern("chain", 4, start=0)
    assert pattern.edges == ((0, 1), (1, 2), (2, 3))
    assert (pattern.entry, pattern.exit) == (0, 3)


def test_generator_parameters_include_seed_in_camel_case() -> None:
    parameters = SynthConfig().as_dict()
    assert parameters["seed"] == 42
    assert parameters["timeSteps"] == 30
    assert parameters["nodesPerStep"] == 400
    assert all("_" not in key for key in parameters)


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="nodes_per_step"):
        generate(SynthConfig(nodes_per_step=10))
    with pytest.raises(ValueError, match="illicit_share"):
        generate(SynthConfig(illicit_share=1.5))
