import re
from typing import Any

SCHEMA_VERSION = 3
METRICS_KEYS = ["schemaVersion", "generatedAt", "dataset", "split", "evaluation", "seed", "runs"]
DATASET_KEYS = [
    "name",
    "displayName",
    "license",
    "source",
    "generator",
    "nodes",
    "edges",
    "timeSteps",
    "features",
    "labelCounts",
    "homophily",
]
HOMOPHILY_KEYS = [
    "labelledEdges",
    "sameLabelShare",
    "illicitIllicitEdges",
    "licitLicitEdges",
    "illicitLicitEdges",
    "illicitWithIllicitNeighbour",
    "licitWithIllicitNeighbour",
]
LABEL_KEYS = ["illicit", "licit", "unknown"]
SPLIT_KEYS = ["kind", "train", "validation", "test", "crossSplitEdges"]
RANGE_KEYS = ["from", "to"]
EVALUATION_KEYS = [
    "positiveLabel",
    "excludedLabel",
    "testPositives",
    "testNegatives",
    "prevalence",
    "allLicitAccuracy",
    "randomPrAucExpected",
    "randomPrAucQ95",
    "randomPermutations",
]
METRIC_KEYS = [
    "prAuc",
    "precisionAtRecall50",
    "recallAtPrecision50",
    "accuracy",
    "accuracyThreshold",
    "accuracyFlagged",
    "accuracyTruePositives",
    "prCurve",
]
RUN_KEYS = [
    "method",
    "displayName",
    "featureSet",
    "seed",
    "date",
    "hyperparameters",
    *METRIC_KEYS,
]
RUN_LOG_KEYS = [
    "schemaVersion",
    "method",
    "displayName",
    "featureSet",
    "date",
    "seed",
    "dataset",
    "split",
    "evaluation",
    "hyperparameters",
    *METRIC_KEYS,
]
SELECTION_KEYS = ["scoreField", "pool", "seeds", "hops", "maxNodes", "truncated"]
NODE_KEYS = [
    "id",
    "x",
    "y",
    "label",
    "timeStep",
    "scoreZscore",
    "scoreIforest",
    "scoreGnn",
    "seedRank",
    "hop",
    "inDegree",
    "outDegree",
]
NODES_FILE_KEYS = ["schemaVersion", "selection", "scoreGnnMethod", "nodes"]
METHODS = ["zscore", "iforest", "gcn", "graphsage", "mlp"]
GNN_METHODS = ["gcn", "graphsage", "mlp"]
GRAPH_METHODS = ["gcn", "graphsage"]
FEATURE_SETS = {"local", "local+graph"}
WEIGHT_RULES = {"trainRatio", "fixed"}
GNN_HYPERPARAMETER_KEYS = [
    "architecture",
    "layers",
    "hidden",
    "activation",
    "dropout",
    "optimizer",
    "learningRate",
    "weightDecay",
    "loss",
    "score",
    "positiveWeight",
    "positiveWeightRule",
    "edges",
    "dtype",
    "scaling",
    "features",
    "maxEpochs",
    "patience",
    "selectedEpoch",
    "selectionMetric",
    "validationPrAuc",
    "selectionSteps",
    "validationSteps",
    "finalFitSteps",
    "search",
    "environment",
]
GNN_LOG_HYPERPARAMETER_KEYS = [*GNN_HYPERPARAMETER_KEYS[:-1], "seedSpread", "environment"]
SCALING_KEYS = [
    "center",
    "scale",
    "madScale",
    "fallback",
    "clip",
    "fitOn",
    "zeroMadFeatures",
    "unitScaleFeatures",
]
SEARCH_KEYS = [
    "featureSet",
    "positiveWeightRule",
    "positiveWeight",
    "validationPrAuc",
    "bestEpoch",
    "stoppedEpoch",
    "selected",
]
SPREAD_KEYS = ["seeds", "reportedSeed", "prAuc", "mean", "min", "max"]
ENVIRONMENT_KEYS = [
    "torch",
    "torchGeometric",
    "python",
    "platform",
    "threads",
    "deterministicAlgorithms",
]
ISO_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class Checker:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)

    def keys(self, value: Any, expected: list[str], path: str) -> bool:
        if not isinstance(value, dict):
            self.errors.append(f"{path}: Objekt erwartet")
            return False
        self.require(list(value) == expected, f"{path}: Schlüssel {list(value)} statt {expected}")
        return list(value) == expected

    def integer(self, value: Any, path: str, minimum: int | None = None) -> None:
        ok = isinstance(value, int) and not isinstance(value, bool)
        self.require(ok, f"{path}: ganze Zahl erwartet")
        if ok and minimum is not None:
            self.require(value >= minimum, f"{path}: mindestens {minimum} erwartet")

    def number(
        self, value: Any, path: str, low: float | None = None, high: float | None = None
    ) -> None:
        ok = isinstance(value, int | float) and not isinstance(value, bool)
        self.require(ok, f"{path}: Zahl erwartet")
        if not ok:
            return
        self.require(round(value, 4) == value, f"{path}: mehr als 4 Nachkommastellen")
        if low is not None:
            self.require(value >= low, f"{path}: kleiner als {low}")
        if high is not None:
            self.require(value <= high, f"{path}: größer als {high}")

    def text(self, value: Any, path: str) -> None:
        self.require(isinstance(value, str) and value != "", f"{path}: Text erwartet")


def _check_range(checker: Checker, value: Any, path: str) -> None:
    if checker.keys(value, RANGE_KEYS, path):
        checker.integer(value["from"], f"{path}.from", 1)
        checker.integer(value["to"], f"{path}.to", 1)


def check_dataset(checker: Checker, dataset: Any, path: str) -> None:
    if not checker.keys(dataset, DATASET_KEYS, path):
        return
    for key in ("name", "displayName", "license", "source"):
        checker.text(dataset[key], f"{path}.{key}")
    checker.require(
        dataset["generator"] is None or isinstance(dataset["generator"], dict),
        f"{path}.generator: Objekt oder null erwartet",
    )
    for key in ("nodes", "edges", "timeSteps", "features"):
        checker.integer(dataset[key], f"{path}.{key}", 0)
    if checker.keys(dataset["labelCounts"], LABEL_KEYS, f"{path}.labelCounts"):
        for label in LABEL_KEYS:
            checker.integer(dataset["labelCounts"][label], f"{path}.labelCounts.{label}", 0)
        checker.require(
            sum(dataset["labelCounts"].values()) == dataset["nodes"],
            f"{path}.labelCounts: Summe passt nicht zur Knotenzahl",
        )
    homophily = dataset["homophily"]
    if checker.keys(homophily, HOMOPHILY_KEYS, f"{path}.homophily"):
        for key in HOMOPHILY_KEYS:
            if key == "sameLabelShare":
                checker.number(homophily[key], f"{path}.homophily.{key}", 0.0, 1.0)
            else:
                checker.integer(homophily[key], f"{path}.homophily.{key}", 0)
        edge_sum = sum(
            homophily[key]
            for key in ("illicitIllicitEdges", "licitLicitEdges", "illicitLicitEdges")
        )
        checker.require(
            edge_sum == homophily["labelledEdges"],
            f"{path}.homophily: Kantenzahlen passen nicht zusammen",
        )


def check_split(checker: Checker, split: Any, path: str) -> None:
    if not checker.keys(split, SPLIT_KEYS, path):
        return
    checker.require(split["kind"] == "temporal", f"{path}.kind: temporal erwartet")
    for key in ("train", "validation", "test"):
        _check_range(checker, split[key], f"{path}.{key}")
    checker.require(split["crossSplitEdges"] == 0, f"{path}.crossSplitEdges: muss 0 sein")


def check_evaluation(checker: Checker, evaluation: Any, path: str) -> None:
    if not checker.keys(evaluation, EVALUATION_KEYS, path):
        return
    checker.require(evaluation["positiveLabel"] == "illicit", f"{path}.positiveLabel")
    checker.require(evaluation["excludedLabel"] == "unknown", f"{path}.excludedLabel")
    checker.integer(evaluation["testPositives"], f"{path}.testPositives", 1)
    checker.integer(evaluation["testNegatives"], f"{path}.testNegatives", 0)
    checker.number(evaluation["prevalence"], f"{path}.prevalence", 0.0, 1.0)
    checker.number(evaluation["allLicitAccuracy"], f"{path}.allLicitAccuracy", 0.0, 1.0)
    checker.number(evaluation["randomPrAucExpected"], f"{path}.randomPrAucExpected", 0.0, 1.0)
    checker.number(evaluation["randomPrAucQ95"], f"{path}.randomPrAucQ95", 0.0, 1.0)
    checker.integer(evaluation["randomPermutations"], f"{path}.randomPermutations", 1)


def check_metric_values(checker: Checker, run: dict[str, Any], path: str) -> None:
    for key in ("prAuc", "precisionAtRecall50", "recallAtPrecision50", "accuracy"):
        checker.number(run[key], f"{path}.{key}", 0.0, 1.0)
    checker.text(run["accuracyThreshold"], f"{path}.accuracyThreshold")
    checker.integer(run["accuracyFlagged"], f"{path}.accuracyFlagged", 1)
    checker.integer(run["accuracyTruePositives"], f"{path}.accuracyTruePositives", 0)
    checker.require(
        run["accuracyTruePositives"] <= run["accuracyFlagged"],
        f"{path}.accuracyTruePositives: mehr Treffer als markierte Knoten",
    )
    curve = run["prCurve"]
    checker.require(isinstance(curve, list) and 1 <= len(curve) <= 101, f"{path}.prCurve: Länge")
    if not isinstance(curve, list):
        return
    for index, point in enumerate(curve):
        if checker.keys(point, ["recall", "precision"], f"{path}.prCurve[{index}]"):
            checker.number(point["recall"], f"{path}.prCurve[{index}].recall", 0.0, 1.0)
            checker.number(point["precision"], f"{path}.prCurve[{index}].precision", 0.0, 1.0)
    recalls = [point.get("recall", 0.0) for point in curve if isinstance(point, dict)]
    checker.require(recalls == sorted(recalls), f"{path}.prCurve: nicht nach recall sortiert")


def _check_scaling(checker: Checker, scaling: Any, features: Any, path: str) -> None:
    if not checker.keys(scaling, SCALING_KEYS, path):
        return
    checker.require(scaling["center"] == "median", f"{path}.center")
    checker.require(scaling["scale"] == "mad", f"{path}.scale")
    checker.require(scaling["madScale"] == 1.4826, f"{path}.madScale")
    checker.require(scaling["fallback"] == ["std", "one"], f"{path}.fallback")
    checker.number(scaling["clip"], f"{path}.clip", 0.0)
    checker.require(scaling["fitOn"] == "train", f"{path}.fitOn")
    names = features if isinstance(features, list) else []
    for key in ("zeroMadFeatures", "unitScaleFeatures"):
        checker.require(
            isinstance(scaling[key], list) and all(name in names for name in scaling[key]),
            f"{path}.{key}",
        )


def _check_search(checker: Checker, search: Any, path: str) -> dict[str, Any] | None:
    if not isinstance(search, list) or not search:
        checker.errors.append(f"{path}: Liste erwartet")
        return None
    selected = []
    for index, entry in enumerate(search):
        entry_path = f"{path}[{index}]"
        if not checker.keys(entry, SEARCH_KEYS, entry_path):
            continue
        checker.require(entry["featureSet"] in FEATURE_SETS, f"{entry_path}.featureSet")
        checker.require(entry["positiveWeightRule"] in WEIGHT_RULES, f"{entry_path}.rule")
        checker.number(entry["positiveWeight"], f"{entry_path}.positiveWeight", 0.0)
        checker.number(entry["validationPrAuc"], f"{entry_path}.validationPrAuc", 0.0, 1.0)
        checker.integer(entry["bestEpoch"], f"{entry_path}.bestEpoch", 1)
        checker.integer(entry["stoppedEpoch"], f"{entry_path}.stoppedEpoch", 1)
        checker.require(
            isinstance(entry["bestEpoch"], int)
            and isinstance(entry["stoppedEpoch"], int)
            and entry["bestEpoch"] <= entry["stoppedEpoch"],
            f"{entry_path}: bestEpoch nach stoppedEpoch",
        )
        checker.require(isinstance(entry["selected"], bool), f"{entry_path}.selected")
        if entry["selected"] is True:
            selected.append(entry)
    checker.require(len(selected) == 1, f"{path}: genau ein gewählter Kandidat erwartet")
    entries = [entry for entry in search if isinstance(entry, dict)]
    pairs = [(entry.get("featureSet"), entry.get("positiveWeightRule")) for entry in entries]
    checker.require(len(pairs) == len(set(pairs)), f"{path}: Kandidat doppelt")
    values = [entry.get("validationPrAuc") for entry in entries]
    if len(selected) == 1 and all(isinstance(value, int | float) for value in values):
        first_best = entries[values.index(max(values))]
        checker.require(
            first_best is selected[0], f"{path}: gewählt ist nicht der erste beste Kandidat"
        )
    return selected[0] if len(selected) == 1 else None


def check_seed_spread(checker: Checker, spread: Any, path: str) -> None:
    if not checker.keys(spread, SPREAD_KEYS, path):
        return
    seeds = spread["seeds"]
    values = spread["prAuc"]
    checker.require(isinstance(seeds, list) and len(seeds) >= 1, f"{path}.seeds")
    checker.require(
        isinstance(values, list) and isinstance(seeds, list) and len(values) == len(seeds),
        f"{path}.prAuc: Länge passt nicht zu seeds",
    )
    checker.require(isinstance(seeds, list) and spread["reportedSeed"] in seeds, f"{path}.seed")
    for key in ("mean", "min", "max"):
        checker.number(spread[key], f"{path}.{key}", 0.0, 1.0)
    if isinstance(values, list) and values:
        for index, value in enumerate(values):
            checker.number(value, f"{path}.prAuc[{index}]", 0.0, 1.0)
        checker.require(spread["min"] == min(values), f"{path}.min")
        checker.require(spread["max"] == max(values), f"{path}.max")


def check_gnn_hyperparameters(
    checker: Checker, run: dict[str, Any], path: str, with_spread: bool
) -> None:
    hyperparameters = run["hyperparameters"]
    expected = GNN_LOG_HYPERPARAMETER_KEYS if with_spread else GNN_HYPERPARAMETER_KEYS
    if not checker.keys(hyperparameters, expected, path):
        return
    checker.require(hyperparameters["layers"] == 2, f"{path}.layers")
    checker.require(hyperparameters["hidden"] == 64, f"{path}.hidden")
    checker.require(hyperparameters["dropout"] == 0.5, f"{path}.dropout")
    checker.require(hyperparameters["optimizer"] == "adam", f"{path}.optimizer")
    checker.require(hyperparameters["learningRate"] == 0.01, f"{path}.learningRate")
    checker.require(hyperparameters["weightDecay"] == 0.0005, f"{path}.weightDecay")
    checker.require(hyperparameters["loss"] == "weightedCrossEntropy", f"{path}.loss")
    checker.number(hyperparameters["positiveWeight"], f"{path}.positiveWeight", 0.0)
    checker.require(hyperparameters["positiveWeightRule"] in WEIGHT_RULES, f"{path}.rule")
    edges = "none" if run["method"] == "mlp" else "undirected"
    checker.require(hyperparameters["edges"] == edges, f"{path}.edges")
    checker.require(hyperparameters["dtype"] in ("float64", "float32"), f"{path}.dtype")
    _check_scaling(
        checker, hyperparameters["scaling"], hyperparameters["features"], f"{path}.scaling"
    )
    checker.integer(hyperparameters["maxEpochs"], f"{path}.maxEpochs", 1)
    checker.integer(hyperparameters["patience"], f"{path}.patience", 1)
    checker.integer(hyperparameters["selectedEpoch"], f"{path}.selectedEpoch", 1)
    checker.require(
        isinstance(hyperparameters["selectedEpoch"], int)
        and isinstance(hyperparameters["maxEpochs"], int)
        and hyperparameters["selectedEpoch"] <= hyperparameters["maxEpochs"],
        f"{path}.selectedEpoch: größer als maxEpochs",
    )
    checker.require(
        hyperparameters["selectionMetric"] == "validationPrAuc", f"{path}.selectionMetric"
    )
    checker.number(hyperparameters["validationPrAuc"], f"{path}.validationPrAuc", 0.0, 1.0)
    for key in ("selectionSteps", "validationSteps", "finalFitSteps"):
        _check_range(checker, hyperparameters[key], f"{path}.{key}")
    selected = _check_search(checker, hyperparameters["search"], f"{path}.search")
    if selected is not None:
        checker.require(selected["featureSet"] == run["featureSet"], f"{path}: featureSet")
        checker.require(
            selected["positiveWeightRule"] == hyperparameters["positiveWeightRule"],
            f"{path}: positiveWeightRule",
        )
        checker.require(
            selected["bestEpoch"] == hyperparameters["selectedEpoch"], f"{path}: selectedEpoch"
        )
        checker.require(
            selected["validationPrAuc"] == hyperparameters["validationPrAuc"],
            f"{path}: validationPrAuc",
        )
    if with_spread:
        check_seed_spread(checker, hyperparameters["seedSpread"], f"{path}.seedSpread")
    checker.keys(hyperparameters["environment"], ENVIRONMENT_KEYS, f"{path}.environment")


def check_run(checker: Checker, run: Any, path: str) -> None:
    if not checker.keys(run, RUN_KEYS, path):
        return
    checker.require(run["method"] in METHODS, f"{path}.method")
    checker.text(run["displayName"], f"{path}.displayName")
    checker.require(run["featureSet"] in FEATURE_SETS, f"{path}.featureSet")
    checker.integer(run["seed"], f"{path}.seed", 0)
    checker.require(
        isinstance(run["date"], str) and bool(ISO_DATE.match(run["date"])), f"{path}.date"
    )
    checker.require(isinstance(run["hyperparameters"], dict), f"{path}.hyperparameters")
    if run["method"] in GNN_METHODS and isinstance(run["hyperparameters"], dict):
        check_gnn_hyperparameters(checker, run, f"{path}.hyperparameters", with_spread=False)
    check_metric_values(checker, run, path)


def validate_metrics(payload: Any) -> list[str]:
    checker = Checker()
    if not checker.keys(payload, METRICS_KEYS, "metrics"):
        return checker.errors
    checker.require(payload["schemaVersion"] == SCHEMA_VERSION, "metrics.schemaVersion")
    checker.require(
        isinstance(payload["generatedAt"], str)
        and bool(ISO_TIMESTAMP.match(payload["generatedAt"])),
        "metrics.generatedAt: ISO-8601 UTC erwartet",
    )
    check_dataset(checker, payload["dataset"], "metrics.dataset")
    check_split(checker, payload["split"], "metrics.split")
    check_evaluation(checker, payload["evaluation"], "metrics.evaluation")
    checker.integer(payload["seed"], "metrics.seed", 0)
    checker.require(isinstance(payload["runs"], list) and payload["runs"], "metrics.runs")
    for index, run in enumerate(payload["runs"]):
        check_run(checker, run, f"metrics.runs[{index}]")
    methods = [run.get("method") for run in payload["runs"] if isinstance(run, dict)]
    checker.require(len(methods) == len(set(methods)), "metrics.runs: Verfahren doppelt")
    known = [method for method in methods if method in METHODS]
    checker.require(
        known == sorted(known, key=METHODS.index), "metrics.runs: Reihenfolge wie METHODS"
    )
    return checker.errors


def validate_run_log(payload: Any) -> list[str]:
    checker = Checker()
    if not checker.keys(payload, RUN_LOG_KEYS, "run"):
        return checker.errors
    checker.require(payload["schemaVersion"] == SCHEMA_VERSION, "run.schemaVersion")
    checker.require(payload["method"] in METHODS, "run.method")
    check_dataset(checker, payload["dataset"], "run.dataset")
    check_split(checker, payload["split"], "run.split")
    check_evaluation(checker, payload["evaluation"], "run.evaluation")
    if payload["method"] in GNN_METHODS and isinstance(payload["hyperparameters"], dict):
        check_gnn_hyperparameters(checker, payload, "run.hyperparameters", with_spread=True)
    check_metric_values(checker, payload, "run")
    return checker.errors


def expected_score_gnn_method(metrics: dict[str, Any]) -> str | None:
    graph_runs = [run for run in metrics["runs"] if run["method"] in GRAPH_METHODS]
    if not graph_runs:
        return None
    graph_runs.sort(key=lambda run: GRAPH_METHODS.index(run["method"]))
    best = graph_runs[0]
    for run in graph_runs[1:]:
        if run["hyperparameters"]["validationPrAuc"] > best["hyperparameters"]["validationPrAuc"]:
            best = run
    return best["method"]


def validate_nodes(payload: Any) -> list[str]:
    checker = Checker()
    if not checker.keys(payload, NODES_FILE_KEYS, "nodes"):
        return checker.errors
    checker.require(payload["schemaVersion"] == SCHEMA_VERSION, "nodes.schemaVersion")
    selection = payload["selection"]
    if checker.keys(selection, SELECTION_KEYS, "nodes.selection"):
        checker.require(selection["scoreField"] == "scoreIforest", "selection.scoreField")
        checker.require(selection["pool"] == "test", "selection.pool")
        checker.require(isinstance(selection["truncated"], bool), "selection.truncated")
    gnn_method = payload["scoreGnnMethod"]
    checker.require(
        gnn_method is None or gnn_method in GRAPH_METHODS, "nodes.scoreGnnMethod: gcn, graphsage"
    )
    for index, node in enumerate(payload["nodes"]):
        path = f"nodes[{index}]"
        if not checker.keys(node, NODE_KEYS, path):
            continue
        checker.text(node["id"], f"{path}.id")
        checker.number(node["x"], f"{path}.x", -1.0, 1.0)
        checker.number(node["y"], f"{path}.y", -1.0, 1.0)
        checker.require(node["label"] in LABEL_KEYS, f"{path}.label")
        checker.integer(node["timeStep"], f"{path}.timeStep", 1)
        checker.number(node["scoreZscore"], f"{path}.scoreZscore", 0.0)
        checker.number(node["scoreIforest"], f"{path}.scoreIforest")
        if gnn_method is None:
            checker.require(node["scoreGnn"] is None, f"{path}.scoreGnn: null erwartet")
        else:
            checker.number(node["scoreGnn"], f"{path}.scoreGnn")
        rank = node["seedRank"]
        checker.require(
            rank is None or (isinstance(rank, int) and 1 <= rank <= 50), f"{path}.seedRank"
        )
        checker.require(node["hop"] in (0, 1, 2), f"{path}.hop")
        checker.require((rank is not None) == (node["hop"] == 0), f"{path}: seedRank und hop")
        checker.integer(node["inDegree"], f"{path}.inDegree", 0)
        checker.integer(node["outDegree"], f"{path}.outDegree", 0)
    return checker.errors


def validate_edges(payload: Any, node_ids: set[str]) -> list[str]:
    checker = Checker()
    if not checker.keys(payload, ["schemaVersion", "edges"], "edges"):
        return checker.errors
    checker.require(payload["schemaVersion"] == SCHEMA_VERSION, "edges.schemaVersion")
    pairs = []
    for index, edge in enumerate(payload["edges"]):
        if checker.keys(edge, ["source", "target"], f"edges[{index}]"):
            checker.require(edge["source"] in node_ids, f"edges[{index}].source fehlt")
            checker.require(edge["target"] in node_ids, f"edges[{index}].target fehlt")
            checker.require(edge["source"] != edge["target"], f"edges[{index}]: Selbstschleife")
            pairs.append((edge["source"], edge["target"]))
    checker.require(len(pairs) == len(set(pairs)), "edges: Duplikate")
    checker.require(pairs == sorted(pairs), "edges: nicht sortiert")
    return checker.errors
