import copy
import platform
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, Linear, SAGEConv

from k3_train.metrics import pr_auc
from k3_train.training import EarlyStopping

ARCHITECTURES = ("gcn", "graphsage", "mlp")
LAYER_TYPES = {"gcn": "GCNConv", "graphsage": "SAGEConv(aggr=mean)", "mlp": "Linear"}
DTYPES = {"float64": torch.float64, "float32": torch.float32}
THREADS = 1


@dataclass(frozen=True)
class GnnConfig:
    layers: int = 2
    hidden: int = 64
    dropout: float = 0.5
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    max_epochs: int = 300
    patience: int = 50
    dtype: str = "float64"
    spread_seeds: tuple[int, ...] = (42, 43, 44, 45, 46)


@dataclass(frozen=True)
class GraphTensors:
    x: torch.Tensor
    edge_index: torch.Tensor


@dataclass
class TrainResult:
    model: nn.Module
    best_epoch: int
    stopped_epoch: int
    validation_pr_auc: float | None


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, 2)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden, aggr="mean")
        self.conv2 = SAGEConv(hidden, 2, aggr="mean")
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.lin1 = Linear(in_dim, hidden, weight_initializer="glorot", bias_initializer="zeros")
        self.lin2 = Linear(hidden, 2, weight_initializer="glorot", bias_initializer="zeros")
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)


MODELS: dict[str, type[nn.Module]] = {"gcn": GCN, "graphsage": GraphSAGE, "mlp": MLP}


def set_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(THREADS)


def environment() -> dict[str, Any]:
    return {
        "torch": torch.__version__,
        "torchGeometric": torch_geometric.__version__,
        "python": platform.python_version(),
        "platform": platform.system(),
        "threads": torch.get_num_threads(),
        "deterministicAlgorithms": torch.are_deterministic_algorithms_enabled(),
    }


def undirected_edge_index(node_ids: Sequence[str], edges: pd.DataFrame) -> torch.Tensor:
    position = {node: index for index, node in enumerate(node_ids)}
    if len(position) != len(node_ids):
        raise ValueError("Knoten-IDs sind nicht eindeutig")
    try:
        source = np.array([position[node] for node in edges["source"]], dtype=np.int64)
        target = np.array([position[node] for node in edges["target"]], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"Kante verweist auf unbekannten Knoten {error}") from error
    pairs = np.concatenate([np.stack([source, target]), np.stack([target, source])], axis=1)
    pairs = pairs[:, pairs[0] != pairs[1]]
    if pairs.shape[1] == 0:
        return torch.empty((2, 0), dtype=torch.int64)
    return torch.from_numpy(np.ascontiguousarray(np.unique(pairs, axis=1)))


def graph_tensors(values: np.ndarray, edge_index: torch.Tensor, dtype: str) -> GraphTensors:
    x = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float64)).to(DTYPES[dtype])
    return GraphTensors(x=x, edge_index=edge_index)


def build_model(architecture: str, in_dim: int, config: GnnConfig) -> nn.Module:
    if architecture not in MODELS:
        raise ValueError(f"Unbekannte Architektur: {architecture}")
    if config.layers != 2:
        raise ValueError("Umgesetzt sind nur zwei Schichten")
    model = MODELS[architecture](in_dim, config.hidden, config.dropout)
    return model.to(DTYPES[config.dtype])


def weighted_loss(
    logits: torch.Tensor, index: torch.Tensor, targets: torch.Tensor, positive_weight: float
) -> torch.Tensor:
    weight = torch.tensor([1.0, positive_weight], dtype=logits.dtype)
    return F.cross_entropy(logits[index], targets, weight=weight)


def logit_scores(model: nn.Module, graph: GraphTensors) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(graph.x, graph.edge_index)
    return (logits[:, 1] - logits[:, 0]).to(torch.float64).numpy().copy()


def _as_index(values: Any) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(values, dtype=np.int64))


def train_model(
    architecture: str,
    graph: GraphTensors,
    loss_index: Any,
    loss_targets: Any,
    positive_weight: float,
    config: GnnConfig,
    seed: int,
    validation: tuple[Any, Any] | None = None,
    epochs: int | None = None,
) -> TrainResult:
    set_determinism(seed)
    model = build_model(architecture, graph.x.shape[1], config)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    index = _as_index(loss_index)
    targets = _as_index(loss_targets)
    limit = config.max_epochs if epochs is None else epochs
    if limit < 1:
        raise ValueError("Mindestens eine Epoche")
    tracker = EarlyStopping(config.patience)
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, limit + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(graph.x, graph.edge_index)
        loss = weighted_loss(logits, index, targets, positive_weight)
        loss.backward()
        optimizer.step()
        if validation is None:
            continue
        validation_index, validation_targets = validation
        value = pr_auc(validation_targets, logit_scores(model, graph)[validation_index])
        if tracker.update(epoch, value):
            best_state = copy.deepcopy(model.state_dict())
        if tracker.stop:
            break
    if validation is None or best_state is None:
        return TrainResult(
            model=model, best_epoch=limit, stopped_epoch=limit, validation_pr_auc=None
        )
    model.load_state_dict(best_state)
    return TrainResult(
        model=model,
        best_epoch=tracker.best_epoch,
        stopped_epoch=tracker.last_epoch,
        validation_pr_auc=tracker.best_value,
    )
