import networkx as nx
import pandas as pd
import pytest

from k3_train.graph import (
    GRAPH_FEATURES,
    EigenvectorConvergenceWarning,
    build_graph,
    count_cross_step_edges,
    eigenvector_or_zero,
    graph_measures,
    graph_measures_with_report,
    label_homophily,
)
from k3_train.load import Dataset


def path_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.DataFrame({"node_id": ["a", "b", "c"], "time_step": [1, 1, 1]})
    edges = pd.DataFrame({"source": ["a", "b"], "target": ["b", "c"]})
    return nodes, edges


def test_path_degrees_and_betweenness_by_hand() -> None:
    nodes, edges = path_frames()
    measures = graph_measures(nodes, edges, seed=42).set_index("node_id")
    assert measures.loc["a", "g_in_degree"] == 0
    assert measures.loc["a", "g_out_degree"] == 1
    assert measures.loc["b", "g_in_degree"] == 1
    assert measures.loc["b", "g_out_degree"] == 1
    assert measures.loc["c", "g_in_degree"] == 1
    assert measures.loc["c", "g_out_degree"] == 0
    assert measures.loc["b", "g_betweenness"] == pytest.approx(0.5)
    assert measures.loc["a", "g_betweenness"] == 0.0
    assert measures.loc["c", "g_betweenness"] == 0.0


def test_path_eigenvector_by_hand() -> None:
    nodes, edges = path_frames()
    measures = graph_measures(nodes, edges, seed=42).set_index("node_id")
    assert measures.loc["a", "g_eigenvector"] == pytest.approx(0.5, abs=1e-4)
    assert measures.loc["b", "g_eigenvector"] == pytest.approx(0.7071, abs=1e-4)
    assert measures.loc["c", "g_eigenvector"] == pytest.approx(0.5, abs=1e-4)


def test_measures_are_computed_per_time_step() -> None:
    nodes = pd.DataFrame({"node_id": ["a", "b", "c", "d", "e"], "time_step": [1, 1, 1, 2, 2]})
    edges = pd.DataFrame({"source": ["a", "b", "d"], "target": ["b", "c", "e"]})
    measures = graph_measures(nodes, edges, seed=42).set_index("node_id")
    assert measures.loc["b", "g_betweenness"] == pytest.approx(0.5)
    assert measures.loc["d", "g_eigenvector"] == pytest.approx(0.7071, abs=1e-4)
    assert measures.columns.tolist() == list(GRAPH_FEATURES)


def test_sampled_betweenness_branch_matches_exact_when_all_sources_are_used() -> None:
    nodes, edges = path_frames()
    sampled = graph_measures(nodes, edges, seed=42, exact_limit=2, samples=3)
    exact = graph_measures(nodes, edges, seed=42)
    pd.testing.assert_frame_equal(sampled, exact)


def test_report_names_settings_and_exact_steps() -> None:
    nodes, edges = path_frames()
    _, report = graph_measures_with_report(nodes, edges, seed=42)
    assert report.as_dict() == {
        "exactBetweennessLimit": 5000,
        "betweennessSamples": 1000,
        "betweennessSeed": 42,
        "eigenvectorMaxIter": 1000,
        "eigenvectorTol": "1e-06",
        "sampledSteps": [],
        "eigenvectorFallbackSteps": [],
    }


def test_report_lists_sampled_and_fallback_steps() -> None:
    nodes = pd.DataFrame({"node_id": ["a", "b", "c", "d", "e"], "time_step": [1, 1, 1, 2, 2]})
    edges = pd.DataFrame({"source": ["a", "b", "d"], "target": ["b", "c", "e"]})
    _, sampled = graph_measures_with_report(nodes, edges, seed=42, exact_limit=2, samples=3)
    assert sampled.sampled_steps == (1,)
    with pytest.warns(EigenvectorConvergenceWarning):
        measures, fallback = graph_measures_with_report(nodes, edges, seed=42, eigen_max_iter=1)
    assert fallback.eigenvector_fallback_steps == (1, 2)
    assert (measures["g_eigenvector"] == 0.0).all()


def test_label_homophily_by_hand() -> None:
    nodes = pd.DataFrame(
        {
            "node_id": ["a", "b", "c", "d", "e", "f"],
            "label": ["illicit", "illicit", "licit", "licit", "unknown", "licit"],
        }
    )
    edges = pd.DataFrame(
        {
            "source": ["a", "b", "c", "d", "e", "f"],
            "target": ["b", "c", "d", "e", "a", "c"],
        }
    )
    assert label_homophily(nodes, edges) == {
        "labelledEdges": 4,
        "sameLabelShare": pytest.approx(0.75),
        "illicitIllicitEdges": 1,
        "licitLicitEdges": 2,
        "illicitLicitEdges": 1,
        "illicitWithIllicitNeighbour": 2,
        "licitWithIllicitNeighbour": 1,
    }


def test_graph_counts_and_no_self_loops(small_dataset: Dataset) -> None:
    graph = build_graph(small_dataset.nodes, small_dataset.edges)
    assert graph.number_of_nodes() == len(small_dataset.nodes)
    assert graph.number_of_edges() == len(small_dataset.edges)
    assert nx.number_of_selfloops(graph) == 0
    assert count_cross_step_edges(small_dataset.nodes, small_dataset.edges) == 0


def test_self_loop_is_rejected() -> None:
    nodes = pd.DataFrame({"node_id": ["a", "b"], "time_step": [1, 1]})
    edges = pd.DataFrame({"source": ["a", "b"], "target": ["a", "a"]})
    with pytest.raises(ValueError, match="Selbstschleifen"):
        build_graph(nodes, edges)


def test_unknown_endpoint_is_rejected() -> None:
    nodes = pd.DataFrame({"node_id": ["a"], "time_step": [1]})
    edges = pd.DataFrame({"source": ["a"], "target": ["z"]})
    with pytest.raises(ValueError, match="unbekannte"):
        build_graph(nodes, edges)


def test_eigenvector_without_convergence_is_zero_with_warning() -> None:
    graph = nx.path_graph(["a", "b", "c"])
    with pytest.warns(EigenvectorConvergenceWarning):
        values = eigenvector_or_zero(graph, max_iter=1)
    assert values == {"a": 0.0, "b": 0.0, "c": 0.0}


def test_measures_cover_every_node_in_input_order(small_dataset: Dataset) -> None:
    measures = graph_measures(small_dataset.nodes, small_dataset.edges, seed=42)
    assert measures["node_id"].tolist() == small_dataset.nodes["node_id"].tolist()
    assert measures["g_in_degree"].sum() == len(small_dataset.edges)
    assert measures["g_out_degree"].sum() == len(small_dataset.edges)
    assert not measures[list(GRAPH_FEATURES)].isna().to_numpy().any()
