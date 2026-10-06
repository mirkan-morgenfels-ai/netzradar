import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from k3_train.jsonio import FLOAT_DIGITS, round_floats, write_json
from k3_train.load import Dataset
from k3_train.pipeline import SCHEMA_VERSION, BaselineOutput, metrics_payload
from k3_train.split import TemporalSplit, split_for, split_masks

SEED_COUNT = 50
HOPS = 2
MAX_NODES = 2000
SCORE_FIELD = "scoreIforest"
POOL = "test"
LAYOUT_ITERATIONS = 100
COMPONENT_FILL = 0.9
EXPORT_FILES = ("nodes.json", "edges.json", "metrics.json")


@dataclass(frozen=True)
class Selection:
    nodes: pd.DataFrame
    truncated: bool


@dataclass(frozen=True)
class ExportBundle:
    metrics: dict[str, Any]
    nodes: dict[str, Any]
    edges: dict[str, Any]


def _undirected_neighbours(edges: pd.DataFrame, allowed: set[str]) -> dict[str, set[str]]:
    neighbours: dict[str, set[str]] = {}
    for source, target in zip(edges["source"], edges["target"], strict=True):
        if source in allowed and target in allowed:
            neighbours.setdefault(source, set()).add(target)
            neighbours.setdefault(target, set()).add(source)
    return neighbours


def select_neighbourhood(
    pool_ids: list[str],
    scores: dict[str, float],
    edges: pd.DataFrame,
    seed_count: int = SEED_COUNT,
    hops: int = HOPS,
    max_nodes: int = MAX_NODES,
) -> Selection:
    ranked = sorted(pool_ids, key=lambda node: (-scores[node], node))
    seeds = ranked[:seed_count]
    hop_of = dict.fromkeys(seeds, 0)
    neighbours = _undirected_neighbours(edges, set(pool_ids))
    frontier = seeds
    for distance in range(1, hops + 1):
        reached = sorted(
            {
                neighbour
                for node in frontier
                for neighbour in neighbours.get(node, ())
                if neighbour not in hop_of
            }
        )
        for node in reached:
            hop_of[node] = distance
        frontier = reached
    seed_rank = {node: rank for rank, node in enumerate(seeds, start=1)}
    frame = pd.DataFrame(
        {
            "node_id": list(hop_of),
            "hop": list(hop_of.values()),
        }
    )
    frame["seed_rank"] = frame["node_id"].map(seed_rank).astype("Int64")
    frame["sort_score"] = frame["node_id"].map(lambda node: -round(scores[node], FLOAT_DIGITS))
    frame = frame.sort_values(["hop", "sort_score", "node_id"], ignore_index=True)
    truncated = len(frame) > max_nodes
    frame = frame.head(max_nodes).drop(columns="sort_score").reset_index(drop=True)
    return Selection(nodes=frame, truncated=truncated)


def induced_edges(edges: pd.DataFrame, node_ids: set[str]) -> pd.DataFrame:
    inside = edges["source"].isin(node_ids) & edges["target"].isin(node_ids)
    subset = edges.loc[inside, ["source", "target"]]
    subset = subset[subset["source"] != subset["target"]].drop_duplicates()
    return subset.sort_values(["source", "target"], ignore_index=True)


def normalise_positions(coordinates: np.ndarray) -> np.ndarray:
    if coordinates.size == 0:
        return coordinates
    low = coordinates.min(axis=0)
    high = coordinates.max(axis=0)
    centre = (low + high) / 2.0
    half_range = float(((high - low) / 2.0).max())
    if half_range == 0.0:
        return np.zeros_like(coordinates)
    return np.clip((coordinates - centre) / half_range, -1.0, 1.0)


def _component_layout(graph: nx.Graph, members: list[str], seed: int) -> np.ndarray:
    if len(members) == 1:
        return np.zeros((1, 2))
    component = nx.Graph()
    component.add_nodes_from(members)
    component.add_edges_from(sorted({(min(u, v), max(u, v)) for u, v in graph.edges(members)}))
    positions = nx.spring_layout(component, seed=seed, iterations=LAYOUT_ITERATIONS, method="force")
    return normalise_positions(np.array([positions[node] for node in members]))


def layout_positions(
    node_ids: list[str], edges: pd.DataFrame, seed: int
) -> dict[str, tuple[float, float]]:
    graph = nx.Graph()
    graph.add_nodes_from(sorted(node_ids))
    graph.add_edges_from(sorted(zip(edges["source"], edges["target"], strict=True)))
    if graph.number_of_nodes() == 0:
        return {}
    components = sorted(
        (sorted(component) for component in nx.connected_components(graph)),
        key=lambda members: (-len(members), members[0]),
    )
    columns = math.ceil(math.sqrt(len(components)))
    largest = len(components[0])
    ordered: list[str] = []
    blocks: list[np.ndarray] = []
    for index, members in enumerate(components):
        row, column = divmod(index, columns)
        centre = np.array([2.0 * column, -2.0 * row])
        radius = COMPONENT_FILL * math.sqrt(len(members) / largest)
        blocks.append(centre + radius * _component_layout(graph, members, seed))
        ordered.extend(members)
    coordinates = normalise_positions(np.vstack(blocks))
    return {
        node: (float(coordinates[index, 0]), float(coordinates[index, 1]))
        for index, node in enumerate(ordered)
    }


def nodes_payload(
    dataset: Dataset,
    output: BaselineOutput,
    split: TemporalSplit,
    seed: int,
    seed_count: int = SEED_COUNT,
    hops: int = HOPS,
    max_nodes: int = MAX_NODES,
) -> tuple[dict[str, Any], set[str]]:
    nodes = dataset.nodes
    test_mask = split_masks(nodes, split)["test"]
    pool_ids = nodes.loc[test_mask, "node_id"].tolist()
    scores = output.scores.set_index("node_id")
    iforest = scores["score_iforest"].to_dict()
    selection = select_neighbourhood(pool_ids, iforest, dataset.edges, seed_count, hops, max_nodes)
    selected_ids = set(selection.nodes["node_id"])
    sub_edges = induced_edges(dataset.edges, selected_ids)
    positions = layout_positions(list(selected_ids), sub_edges, seed)
    info = nodes.set_index("node_id")
    measures = output.measures.set_index("node_id")
    entries = []
    for row in selection.nodes.itertuples(index=False):
        node = row.node_id
        x, y = positions[node]
        entries.append(
            {
                "id": node,
                "x": x,
                "y": y,
                "label": str(info.at[node, "label"]),
                "timeStep": int(info.at[node, "time_step"]),
                "scoreZscore": float(scores.at[node, "score_zscore"]),
                "scoreIforest": float(scores.at[node, "score_iforest"]),
                "scoreGnn": None,
                "seedRank": None if pd.isna(row.seed_rank) else int(row.seed_rank),
                "hop": int(row.hop),
                "inDegree": int(measures.at[node, "g_in_degree"]),
                "outDegree": int(measures.at[node, "g_out_degree"]),
            }
        )
    payload = {
        "schemaVersion": SCHEMA_VERSION,
        "selection": {
            "scoreField": SCORE_FIELD,
            "pool": POOL,
            "seeds": seed_count,
            "hops": hops,
            "maxNodes": max_nodes,
            "truncated": selection.truncated,
        },
        "nodes": entries,
    }
    return payload, selected_ids


def edges_payload(edges: pd.DataFrame, node_ids: set[str]) -> dict[str, Any]:
    subset = induced_edges(edges, node_ids)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "edges": [
            {"source": source, "target": target}
            for source, target in zip(subset["source"], subset["target"], strict=True)
        ],
    }


def build_exports(
    dataset: Dataset,
    output: BaselineOutput,
    generated_at: str,
    seed_count: int = SEED_COUNT,
    hops: int = HOPS,
    max_nodes: int = MAX_NODES,
) -> ExportBundle:
    split = split_for(dataset.name, dataset.time_steps)
    nodes, selected = nodes_payload(
        dataset, output, split, output.seed, seed_count, hops, max_nodes
    )
    return ExportBundle(
        metrics=round_floats(metrics_payload(output, generated_at)),
        nodes=round_floats(nodes),
        edges=edges_payload(dataset.edges, selected),
    )


def write_exports(bundle: ExportBundle, out_dir: Path) -> list[Path]:
    paths = []
    for name, payload in zip(
        EXPORT_FILES, (bundle.nodes, bundle.edges, bundle.metrics), strict=True
    ):
        path = out_dir / name
        write_json(path, payload)
        paths.append(path)
    return paths
