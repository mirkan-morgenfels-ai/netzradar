import warnings
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

GRAPH_FEATURES = ("g_in_degree", "g_out_degree", "g_betweenness", "g_eigenvector")
EXACT_BETWEENNESS_LIMIT = 5000
BETWEENNESS_SAMPLES = 1000
EIGENVECTOR_MAX_ITER = 1000
EIGENVECTOR_TOL = 1e-6


class EigenvectorConvergenceWarning(UserWarning):
    pass


@dataclass(frozen=True)
class GraphMeasureReport:
    exact_limit: int
    samples: int
    seed: int
    eigen_max_iter: int
    eigen_tol: float
    sampled_steps: tuple[int, ...]
    eigenvector_fallback_steps: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "exactBetweennessLimit": self.exact_limit,
            "betweennessSamples": self.samples,
            "betweennessSeed": self.seed,
            "eigenvectorMaxIter": self.eigen_max_iter,
            "eigenvectorTol": format(self.eigen_tol, "g"),
            "sampledSteps": list(self.sampled_steps),
            "eigenvectorFallbackSteps": list(self.eigenvector_fallback_steps),
        }


def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(
        (node_id, {"time_step": int(step)})
        for node_id, step in zip(nodes["node_id"], nodes["time_step"], strict=True)
    )
    unknown = set(edges["source"]).union(edges["target"]) - set(graph.nodes)
    if unknown:
        raise ValueError(f"{len(unknown)} Kantenenden verweisen auf unbekannte Knoten")
    graph.add_edges_from(zip(edges["source"], edges["target"], strict=True))
    if nx.number_of_selfloops(graph):
        raise ValueError("Der Graph enthaelt Selbstschleifen")
    return graph


def eigenvector_with_status(
    graph: nx.Graph, max_iter: int = EIGENVECTOR_MAX_ITER, tol: float = EIGENVECTOR_TOL
) -> tuple[dict[str, float], bool]:
    if graph.number_of_nodes() == 0:
        return {}, True
    try:
        return nx.eigenvector_centrality(graph, max_iter=max_iter, tol=tol), True
    except nx.PowerIterationFailedConvergence:
        warnings.warn(
            f"Eigenvektor-Zentralitaet konvergiert nicht nach {max_iter} Iterationen "
            f"({graph.number_of_nodes()} Knoten); Wert 0 gesetzt",
            EigenvectorConvergenceWarning,
            stacklevel=2,
        )
        return dict.fromkeys(graph.nodes, 0.0), False


def eigenvector_or_zero(
    graph: nx.Graph, max_iter: int = EIGENVECTOR_MAX_ITER, tol: float = EIGENVECTOR_TOL
) -> dict[str, float]:
    return eigenvector_with_status(graph, max_iter=max_iter, tol=tol)[0]


def step_measures(
    graph: nx.DiGraph,
    seed: int,
    exact_limit: int = EXACT_BETWEENNESS_LIMIT,
    samples: int = BETWEENNESS_SAMPLES,
    eigen_max_iter: int = EIGENVECTOR_MAX_ITER,
) -> tuple[pd.DataFrame, bool]:
    node_count = graph.number_of_nodes()
    sample = None if node_count <= exact_limit else min(samples, node_count)
    betweenness = nx.betweenness_centrality(graph, k=sample, normalized=True, seed=seed)
    eigenvector, converged = eigenvector_with_status(graph.to_undirected(), max_iter=eigen_max_iter)
    order = list(graph.nodes)
    frame = pd.DataFrame(
        {
            "node_id": order,
            "g_in_degree": [graph.in_degree(node) for node in order],
            "g_out_degree": [graph.out_degree(node) for node in order],
            "g_betweenness": [betweenness[node] for node in order],
            "g_eigenvector": [eigenvector[node] for node in order],
        }
    )
    return frame, converged


def step_subgraph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(nodes["node_id"]))
    member = set(graph.nodes)
    inside = edges["source"].isin(member) & edges["target"].isin(member)
    pairs = sorted(zip(edges.loc[inside, "source"], edges.loc[inside, "target"], strict=True))
    graph.add_edges_from(pairs)
    return graph


def count_cross_step_edges(nodes: pd.DataFrame, edges: pd.DataFrame) -> int:
    steps = nodes.set_index("node_id")["time_step"]
    source_steps = steps.reindex(edges["source"]).to_numpy()
    target_steps = steps.reindex(edges["target"]).to_numpy()
    return int((source_steps != target_steps).sum())


def graph_measures_with_report(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    seed: int,
    exact_limit: int = EXACT_BETWEENNESS_LIMIT,
    samples: int = BETWEENNESS_SAMPLES,
    eigen_max_iter: int = EIGENVECTOR_MAX_ITER,
) -> tuple[pd.DataFrame, GraphMeasureReport]:
    steps = nodes.set_index("node_id")["time_step"]
    edge_steps = steps.reindex(edges["source"]).to_numpy()
    frames = []
    sampled: list[int] = []
    fallback: list[int] = []
    for step, step_nodes in nodes.groupby("time_step", sort=True):
        step_edges = edges[edge_steps == step]
        graph = step_subgraph(step_nodes, step_edges)
        frame, converged = step_measures(
            graph,
            seed=seed,
            exact_limit=exact_limit,
            samples=samples,
            eigen_max_iter=eigen_max_iter,
        )
        frames.append(frame)
        if graph.number_of_nodes() > exact_limit:
            sampled.append(int(step))
        if not converged:
            fallback.append(int(step))
    measures = pd.concat(frames, ignore_index=True).set_index("node_id")
    measures = measures.reindex(nodes["node_id"]).reset_index()
    measures["g_in_degree"] = measures["g_in_degree"].astype(np.int64)
    measures["g_out_degree"] = measures["g_out_degree"].astype(np.int64)
    report = GraphMeasureReport(
        exact_limit=exact_limit,
        samples=samples,
        seed=seed,
        eigen_max_iter=eigen_max_iter,
        eigen_tol=EIGENVECTOR_TOL,
        sampled_steps=tuple(sampled),
        eigenvector_fallback_steps=tuple(fallback),
    )
    return measures, report


def graph_measures(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    seed: int,
    exact_limit: int = EXACT_BETWEENNESS_LIMIT,
    samples: int = BETWEENNESS_SAMPLES,
    eigen_max_iter: int = EIGENVECTOR_MAX_ITER,
) -> pd.DataFrame:
    return graph_measures_with_report(
        nodes,
        edges,
        seed=seed,
        exact_limit=exact_limit,
        samples=samples,
        eigen_max_iter=eigen_max_iter,
    )[0]


def label_homophily(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict[str, Any]:
    labels = nodes.set_index("node_id")["label"]
    source = labels.reindex(edges["source"]).to_numpy()
    target = labels.reindex(edges["target"]).to_numpy()
    labelled = (source != "unknown") & (target != "unknown")
    both_illicit = labelled & (source == "illicit") & (target == "illicit")
    both_licit = labelled & (source == "licit") & (target == "licit")
    mixed = labelled & (source != target)
    illicit_ends = set(edges.loc[target == "illicit", "source"]) | set(
        edges.loc[source == "illicit", "target"]
    )
    with_illicit_neighbour = labels[labels.index.isin(illicit_ends)]
    labelled_count = int(labelled.sum())
    same_label = int((both_illicit | both_licit).sum())
    return {
        "labelledEdges": labelled_count,
        "sameLabelShare": same_label / labelled_count if labelled_count else 0.0,
        "illicitIllicitEdges": int(both_illicit.sum()),
        "licitLicitEdges": int(both_licit.sum()),
        "illicitLicitEdges": int(mixed.sum()),
        "illicitWithIllicitNeighbour": int((with_illicit_neighbour == "illicit").sum()),
        "licitWithIllicitNeighbour": int((with_illicit_neighbour == "licit").sum()),
    }
