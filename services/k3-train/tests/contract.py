import re
from typing import Any

SCHEMA_VERSION = 2
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
METHODS = {"zscore", "iforest", "gcn", "graphsage"}
FEATURE_SETS = {"local", "local+graph"}
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
    return checker.errors


def validate_run_log(payload: Any) -> list[str]:
    checker = Checker()
    if not checker.keys(payload, RUN_LOG_KEYS, "run"):
        return checker.errors
    checker.require(payload["schemaVersion"] == SCHEMA_VERSION, "run.schemaVersion")
    check_dataset(checker, payload["dataset"], "run.dataset")
    check_split(checker, payload["split"], "run.split")
    check_evaluation(checker, payload["evaluation"], "run.evaluation")
    check_metric_values(checker, payload, "run")
    return checker.errors


def validate_nodes(payload: Any) -> list[str]:
    checker = Checker()
    if not checker.keys(payload, ["schemaVersion", "selection", "nodes"], "nodes"):
        return checker.errors
    checker.require(payload["schemaVersion"] == SCHEMA_VERSION, "nodes.schemaVersion")
    selection = payload["selection"]
    if checker.keys(selection, SELECTION_KEYS, "nodes.selection"):
        checker.require(selection["scoreField"] == "scoreIforest", "selection.scoreField")
        checker.require(selection["pool"] == "test", "selection.pool")
        checker.require(isinstance(selection["truncated"], bool), "selection.truncated")
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
        checker.require(node["scoreGnn"] is None, f"{path}.scoreGnn: null erwartet")
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
