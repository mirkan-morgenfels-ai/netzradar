import numpy as np
import pandas as pd

from k3_train.graph import GRAPH_FEATURES

FEATURE_SETS = ("local", "local+graph")


def local_columns(nodes: pd.DataFrame) -> list[str]:
    return [column for column in nodes.columns if column.startswith("f_")]


def feature_columns(nodes: pd.DataFrame, feature_set: str) -> list[str]:
    if feature_set == "local":
        return local_columns(nodes)
    if feature_set == "local+graph":
        return [*local_columns(nodes), *GRAPH_FEATURES]
    raise ValueError(f"Unbekannter Merkmalssatz: {feature_set}")


def feature_matrix(
    nodes: pd.DataFrame, measures: pd.DataFrame | None, feature_set: str
) -> tuple[np.ndarray, list[str]]:
    columns = feature_columns(nodes, feature_set)
    frame = nodes
    if feature_set == "local+graph":
        if measures is None:
            raise ValueError("Fuer local+graph werden Graphmasse benoetigt")
        if measures["node_id"].astype(str).tolist() != nodes["node_id"].astype(str).tolist():
            raise ValueError("Graphmasse und Knoten haben unterschiedliche Reihenfolge")
        frame = pd.concat(
            [nodes.reset_index(drop=True), measures[list(GRAPH_FEATURES)].reset_index(drop=True)],
            axis=1,
        )
    values = frame[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Merkmalsmatrix enthaelt nicht endliche Werte")
    return values, columns
