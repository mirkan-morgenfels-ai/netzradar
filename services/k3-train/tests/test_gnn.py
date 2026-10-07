import math
import warnings

import numpy as np
import pandas as pd
import pytest

SKIP_REASON = "Extra gnn fehlt (torch, torch_geometric): uv sync --extra gnn"
torch = pytest.importorskip("torch", reason=SKIP_REASON)
pytest.importorskip("torch_geometric", reason=SKIP_REASON)

import torch_geometric
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import add_self_loops, scatter

from k3_train.features import feature_matrix
from k3_train.gnn import (
    GCN,
    MLP,
    GnnConfig,
    GraphSAGE,
    GraphTensors,
    environment,
    graph_tensors,
    logit_scores,
    train_model,
    undirected_edge_index,
    weighted_loss,
)
from k3_train.load import Dataset
from k3_train.metrics import pr_auc
from k3_train.split import split_for
from k3_train.training import fit_scaler, loss_targets, training_parts, transform

CHAIN = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
DIRECTED_CHAIN = torch.tensor([[0, 1, 2], [1, 2, 3]])
X_CHAIN = torch.tensor([[1.0], [0.0], [0.0], [1.0]], dtype=torch.float64)
SMALL_CONFIG_GNN = GnnConfig(max_epochs=5, patience=5)


def values_of(tensor: torch.Tensor) -> list[float]:
    return tensor.flatten().tolist()


def small_graph(dataset: Dataset, phase: str = "train") -> GraphTensors:
    parts = training_parts(dataset.nodes, split_for(dataset.name, dataset.time_steps))
    values, _ = feature_matrix(dataset.nodes, None, "local")
    scaled = transform(values, fit_scaler(values, parts[phase]))
    edge_index = undirected_edge_index(dataset.nodes["node_id"].tolist(), dataset.edges)
    return graph_tensors(scaled, edge_index, "float64")


def test_versions_match_the_lock_and_numpy_interop_is_silent() -> None:
    assert torch.__version__.split("+")[0] == "2.8.0"
    assert torch_geometric.__version__ == "2.7.0"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tensor = torch.from_numpy(np.array([0.25, 1.5], dtype=np.float64))
        back = tensor.numpy()
    assert caught == []
    assert tensor.dtype == torch.float64
    assert back.tolist() == [0.25, 1.5]


def test_mean_over_neighbourhood_and_self_by_hand() -> None:
    loops, _ = add_self_loops(CHAIN, num_nodes=4)

    def step(x: torch.Tensor) -> torch.Tensor:
        return scatter(x[loops[0]], loops[1], dim=0, dim_size=4, reduce="mean")

    first = step(X_CHAIN)
    second = step(first)
    assert values_of(first) == pytest.approx([0.5, 1 / 3, 1 / 3, 0.5])
    assert values_of(second) == pytest.approx([5 / 12, 7 / 18, 7 / 18, 5 / 12])
    assert values_of(second) == pytest.approx([0.4167, 0.3889, 0.3889, 0.4167], abs=1e-4)


def test_gcn_layer_by_hand() -> None:
    conv = GCNConv(1, 1, bias=False).double()
    with torch.no_grad():
        conv.lin.weight.fill_(1.0)
        first = conv(X_CHAIN, CHAIN)
        second = conv(first, CHAIN)
    root6 = math.sqrt(6.0)
    assert values_of(first) == pytest.approx([0.5, 1 / root6, 1 / root6, 0.5])
    assert values_of(first) == pytest.approx([0.5, 0.408248, 0.408248, 0.5], abs=1e-6)
    assert values_of(second) == pytest.approx(
        [5 / 12, 1 / (2 * root6) + 2 / (3 * root6), 1 / (2 * root6) + 2 / (3 * root6), 5 / 12]
    )
    assert values_of(second) == pytest.approx([0.416667, 0.476290, 0.476290, 0.416667], abs=1e-6)


def test_graphsage_layer_and_directed_gcn_by_hand() -> None:
    sage = SAGEConv(1, 1, aggr="mean").double()
    gcn = GCNConv(1, 1, bias=False).double()
    with torch.no_grad():
        sage.lin_l.weight.fill_(1.0)
        sage.lin_l.bias.fill_(0.0)
        sage.lin_r.weight.fill_(1.0)
        gcn.lin.weight.fill_(1.0)
        assert values_of(sage(X_CHAIN, CHAIN)) == pytest.approx([1.0, 0.5, 0.5, 1.0])
        directed = gcn(X_CHAIN, DIRECTED_CHAIN)
    assert values_of(directed) == pytest.approx([1.0, 1 / math.sqrt(2.0), 0.0, 0.5])
    assert values_of(directed) == pytest.approx([1.0, 0.707107, 0.0, 0.5], abs=1e-6)


@pytest.mark.parametrize("model_class", [GCN, GraphSAGE])
def test_receptive_field_has_two_hops(model_class: type) -> None:
    torch.manual_seed(0)
    model = model_class(3, 8, 0.5).double()
    model.eval()
    x = torch.randn(4, 3, dtype=torch.float64)
    with torch.no_grad():
        base = model(x, CHAIN)
        far = x.clone()
        far[3] += 5.0
        near = x.clone()
        near[2] += 5.0
        assert torch.equal(model(far, CHAIN)[0], base[0])
        assert not torch.equal(model(near, CHAIN)[0], base[0])
        two_parts = torch.tensor([[0, 1, 1, 2, 2, 3, 4, 5, 5, 6], [1, 0, 2, 1, 3, 2, 5, 4, 6, 5]])
        y = torch.randn(7, 3, dtype=torch.float64)
        other = y.clone()
        other[4:] = other[4:] * 1000.0 + 50.0
        assert torch.equal(model(other, two_parts)[:4], model(y, two_parts)[:4])


def test_weighted_loss_by_hand() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    index = torch.tensor([0, 1])
    targets = torch.tensor([0, 1])
    first = math.log(1.0 + math.exp(-2.0))
    assert first == pytest.approx(0.126928, abs=1e-6)
    assert weighted_loss(logits, index, targets, 4.0).item() == pytest.approx(
        (first + 4.0 * math.log(2.0)) / 5.0
    )
    assert weighted_loss(logits, index, targets, 4.0).item() == pytest.approx(0.579903, abs=1e-6)
    assert weighted_loss(logits, index, targets, 1.0).item() == pytest.approx(0.410038, abs=1e-6)
    zeros = torch.zeros((2, 2), dtype=torch.float64)
    for weight in (1.0, 4.0, 20.0):
        assert weighted_loss(zeros, index, targets, weight).item() == pytest.approx(math.log(2.0))


def test_undirected_edge_index_by_hand(default_dataset: Dataset) -> None:
    edges = pd.DataFrame({"source": ["A", "C"], "target": ["B", "B"]})
    index = undirected_edge_index(["A", "B", "C"], edges)
    assert index.tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    assert index.dtype == torch.int64
    repeated = pd.DataFrame({"source": ["A", "B", "A"], "target": ["B", "A", "B"]})
    assert undirected_edge_index(["A", "B"], repeated).tolist() == [[0, 1], [1, 0]]
    with pytest.raises(ValueError, match="unbekannten Knoten"):
        undirected_edge_index(["A"], edges)
    full = undirected_edge_index(default_dataset.nodes["node_id"].tolist(), default_dataset.edges)
    assert full.shape == (2, 2 * 13_757)


@pytest.mark.parametrize("architecture", ["gcn", "graphsage", "mlp"])
def test_training_is_deterministic(small_dataset: Dataset, architecture: str) -> None:
    graph = small_graph(small_dataset)
    parts = training_parts(small_dataset.nodes, split_for("synthetic", small_dataset.time_steps))
    index, targets = loss_targets(small_dataset.nodes["label"], parts["train"])

    def run(seed: int) -> tuple[list[torch.Tensor], np.ndarray]:
        model = train_model(architecture, graph, index, targets, 3.0, SMALL_CONFIG_GNN, seed).model
        return [p.detach().clone() for p in model.parameters()], logit_scores(model, graph)

    first, first_scores = run(42)
    second, second_scores = run(42)
    other, other_scores = run(43)
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    np.testing.assert_array_equal(first_scores, second_scores)
    assert not all(torch.equal(a, b) for a, b in zip(first, other, strict=True))
    assert not np.array_equal(first_scores, other_scores)
    assert np.isfinite(first_scores).all()
    assert first_scores.shape == (960,)
    assert environment()["threads"] == 1
    assert environment()["deterministicAlgorithms"] is True


@pytest.mark.parametrize("architecture", ["gcn", "graphsage", "mlp"])
def test_test_nodes_do_not_influence_training(small_dataset: Dataset, architecture: str) -> None:
    split = split_for("synthetic", small_dataset.time_steps)
    parts = training_parts(small_dataset.nodes, split)
    values, _ = feature_matrix(small_dataset.nodes, None, "local")
    changed_values = values.copy()
    changed_values[parts["test"]] = changed_values[parts["test"]] * 1000.0 + 50.0
    labels = small_dataset.nodes["label"].to_numpy()
    changed_labels = labels.copy()
    changed_labels[parts["test"]] = "illicit"
    edge_index = undirected_edge_index(small_dataset.nodes["node_id"].tolist(), small_dataset.edges)

    def run(features: np.ndarray, node_labels: np.ndarray):
        scaled = transform(features, fit_scaler(features, parts["train"]))
        graph = graph_tensors(scaled, edge_index, "float64")
        index, targets = loss_targets(node_labels, parts["train"])
        model = train_model(architecture, graph, index, targets, 3.0, SMALL_CONFIG_GNN, 42).model
        return [p.detach().clone() for p in model.parameters()], logit_scores(model, graph)

    params, scores = run(values, labels)
    changed_params, changed_scores = run(changed_values, changed_labels)
    assert all(torch.equal(a, b) for a, b in zip(params, changed_params, strict=True))
    np.testing.assert_array_equal(scores[parts["train"]], changed_scores[parts["train"]])
    assert not np.array_equal(scores[parts["test"]], changed_scores[parts["test"]])


@pytest.mark.parametrize("architecture", ["gcn", "mlp"])
def test_early_stopping_restores_the_best_epoch(small_dataset: Dataset, architecture: str) -> None:
    parts = training_parts(small_dataset.nodes, split_for("synthetic", small_dataset.time_steps))
    graph = small_graph(small_dataset, "fit")
    labels = small_dataset.nodes["label"]
    fit_index, fit_targets = loss_targets(labels, parts["fit"])
    validation = loss_targets(labels, parts["validation"])
    config = GnnConfig(max_epochs=60, patience=4)
    result = train_model(
        architecture, graph, fit_index, fit_targets, 4.24, config, 42, validation=validation
    )
    assert 1 <= result.best_epoch <= result.stopped_epoch <= 60
    assert result.stopped_epoch == 60 or result.stopped_epoch - result.best_epoch == 4
    restored = pr_auc(validation[1], logit_scores(result.model, graph)[validation[0]])
    assert restored == result.validation_pr_auc
    replay = train_model(
        architecture, graph, fit_index, fit_targets, 4.24, config, 42, epochs=result.best_epoch
    )
    assert all(
        torch.equal(a, b)
        for a, b in zip(result.model.parameters(), replay.model.parameters(), strict=True)
    )


def test_mlp_ignores_edges() -> None:
    torch.manual_seed(0)
    model = MLP(3, 8, 0.5).double()
    model.eval()
    x = torch.randn(4, 3, dtype=torch.float64)
    with torch.no_grad():
        assert torch.equal(model(x, CHAIN), model(x, torch.empty((2, 0), dtype=torch.int64)))
