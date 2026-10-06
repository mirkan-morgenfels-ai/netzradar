from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from k3_train.jsonio import camel_case_keys

PATTERN_TYPES = ("fan_in", "fan_out", "cycle", "chain")

LOCAL_FEATURES = (
    "f_log_amount",
    "f_log_fee",
    "f_log_fee_rate",
    "f_num_inputs",
    "f_num_outputs",
    "f_log_size",
    "f_round_amount",
    "f_change_output",
    "f_max_output_share",
    "f_input_amount_cv",
    "f_log_hours_since_last",
    "f_log_address_age_days",
)

FEATURE_DECIMALS = 6
MAX_NODES = 999_999


@dataclass(frozen=True)
class SynthConfig:
    time_steps: int = 30
    nodes_per_step: int = 400
    seed: int = 42
    illicit_share: float = 0.04
    min_patterns_per_step: int = 1
    illicit_visibility: float = 0.55
    licit_visibility: float = 0.22
    attachment_power: float = 1.0
    extra_edge_ratio: float = 0.15
    pattern_link_probability: float = 0.3
    fan_size_min: int = 5
    fan_size_max: int = 12
    cycle_length_min: int = 3
    cycle_length_max: int = 6
    chain_length_min: int = 4
    chain_length_max: int = 8
    local_shift: float = 0.5
    log_amount_mean: float = 5.5
    log_amount_sd: float = 1.5
    smurf_log_amount_sd: float = 0.5
    hop_noise_sd: float = 0.02
    cycle_fee_rate: float = 0.01
    chain_retention: float = 0.97
    fan_out_concentration: float = 5.0
    log_hours_mean: float = 3.0
    log_hours_sd: float = 1.2
    log_age_mean: float = 5.0
    log_age_sd: float = 1.0
    round_amount_rate: float = 0.10
    change_output_rate: float = 0.70
    fee_slope: float = 0.4
    fee_intercept: float = -2.0
    fee_sd: float = 0.6
    extra_io_rate: float = 1.0
    max_output_share_alpha: float = 4.0
    max_output_share_beta: float = 1.5
    input_cv_shape: float = 2.0
    input_cv_scale: float = 0.35

    def as_dict(self) -> dict[str, int | float]:
        return camel_case_keys(asdict(self))

    def validate(self) -> None:
        if self.time_steps < 1:
            raise ValueError("time_steps muss mindestens 1 sein")
        if self.nodes_per_step < 4 * self.fan_size_max:
            raise ValueError("nodes_per_step ist zu klein fuer die eingebetteten Muster")
        if self.time_steps * self.nodes_per_step > MAX_NODES:
            raise ValueError("Zu viele Knoten fuer das ID-Format")
        for name in (
            "illicit_share",
            "illicit_visibility",
            "licit_visibility",
            "extra_edge_ratio",
            "pattern_link_probability",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} muss zwischen 0 und 1 liegen")
        ranges = (
            (self.fan_size_min, self.fan_size_max, 2),
            (self.cycle_length_min, self.cycle_length_max, 3),
            (self.chain_length_min, self.chain_length_max, 2),
        )
        for low, high, minimum in ranges:
            if not minimum <= low <= high:
                raise ValueError("Ungueltige Mustergroessen")


SMALL_CONFIG = SynthConfig(time_steps=12, nodes_per_step=80)


@dataclass(frozen=True)
class Pattern:
    kind: str
    members: tuple[int, ...]
    roles: tuple[str, ...]
    edges: tuple[tuple[int, int], ...]
    entry: int
    exit: int


@dataclass(frozen=True)
class SynthResult:
    config: SynthConfig
    nodes: pd.DataFrame
    edges: pd.DataFrame
    truth: pd.DataFrame


def expected_pattern_size(config: SynthConfig) -> float:
    fan = (config.fan_size_min + config.fan_size_max) / 2 + 1
    cycle = (config.cycle_length_min + config.cycle_length_max) / 2
    chain = (config.chain_length_min + config.chain_length_max) / 2
    return (2 * fan + cycle + chain) / len(PATTERN_TYPES)


def _pattern_parameter(kind: str, rng: np.random.Generator, config: SynthConfig) -> int:
    bounds = {
        "fan_in": (config.fan_size_min, config.fan_size_max),
        "fan_out": (config.fan_size_min, config.fan_size_max),
        "cycle": (config.cycle_length_min, config.cycle_length_max),
        "chain": (config.chain_length_min, config.chain_length_max),
    }[kind]
    return int(rng.integers(bounds[0], bounds[1] + 1))


def _member_count(kind: str, parameter: int) -> int:
    return parameter + 1 if kind in ("fan_in", "fan_out") else parameter


def build_pattern(kind: str, parameter: int, start: int) -> Pattern:
    if kind == "fan_in":
        sources = tuple(range(start, start + parameter))
        collector = start + parameter
        return Pattern(
            kind=kind,
            members=(*sources, collector),
            roles=("smurf",) * parameter + ("collector",),
            edges=tuple((source, collector) for source in sources),
            entry=sources[0],
            exit=collector,
        )
    if kind == "fan_out":
        distributor = start
        receivers = tuple(range(start + 1, start + 1 + parameter))
        return Pattern(
            kind=kind,
            members=(distributor, *receivers),
            roles=("distributor",) + ("receiver",) * parameter,
            edges=tuple((distributor, receiver) for receiver in receivers),
            entry=distributor,
            exit=receivers[0],
        )
    members = tuple(range(start, start + parameter))
    if kind == "cycle":
        return Pattern(
            kind=kind,
            members=members,
            roles=("round_trip",) * parameter,
            edges=tuple((members[i], members[(i + 1) % parameter]) for i in range(parameter)),
            entry=members[0],
            exit=members[parameter // 2],
        )
    if kind == "chain":
        return Pattern(
            kind=kind,
            members=members,
            roles=("layer",) * parameter,
            edges=tuple((members[i], members[i + 1]) for i in range(parameter - 1)),
            entry=members[0],
            exit=members[-1],
        )
    raise ValueError(f"Unbekannter Mustertyp: {kind}")


def _benign_edges(
    benign_count: int, rng: np.random.Generator, config: SynthConfig
) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    degree = np.zeros(benign_count)
    for node in range(1, benign_count):
        weights = (degree[:node] + 1.0) ** config.attachment_power
        parent = int(rng.choice(node, p=weights / weights.sum()))
        edges.append((parent, node) if rng.random() < 0.5 else (node, parent))
        degree[node] += 1
        degree[parent] += 1
    seen = {(min(u, v), max(u, v)) for u, v in edges}
    target = round(config.extra_edge_ratio * benign_count)
    added = 0
    attempts = 0
    while added < target and attempts < 100 * (target + 1):
        attempts += 1
        u, v = (int(value) for value in rng.integers(0, benign_count, size=2))
        key = (min(u, v), max(u, v))
        if u == v or key in seen:
            continue
        seen.add(key)
        edges.append((u, v))
        added += 1
    return edges


def _pattern_edges(
    patterns: list[Pattern], benign_count: int, rng: np.random.Generator, config: SynthConfig
) -> list[tuple[int, int]]:
    edges = [edge for pattern in patterns for edge in pattern.edges]
    linked = [False] + [bool(rng.random() < config.pattern_link_probability) for _ in patterns[1:]]
    for index, pattern in enumerate(patterns):
        if linked[index]:
            edges.append((patterns[index - 1].exit, pattern.entry))
        elif pattern.kind != "fan_in":
            edges.append((int(rng.integers(0, benign_count)), pattern.entry))
        if index + 1 < len(patterns) and linked[index + 1]:
            continue
        edges.append((pattern.exit, int(rng.integers(0, benign_count))))
    return edges


def _apply_pattern_amounts(
    pattern: Pattern,
    log_amount: np.ndarray,
    max_output_share: np.ndarray,
    input_amount_cv: np.ndarray,
    rng: np.random.Generator,
    config: SynthConfig,
) -> None:
    members = np.array(pattern.members)
    shift = config.local_shift * config.log_amount_sd
    if pattern.kind == "fan_in":
        smurfs = members[:-1]
        collector = members[-1]
        log_amount[smurfs] = rng.normal(
            config.log_amount_mean - shift, config.smurf_log_amount_sd, smurfs.size
        )
        amounts = np.exp(log_amount[smurfs])
        log_amount[collector] = np.log(amounts.sum())
        input_amount_cv[collector] = amounts.std() / amounts.mean()
        return
    if pattern.kind == "fan_out":
        distributor = members[0]
        receivers = members[1:]
        log_amount[distributor] = rng.normal(config.log_amount_mean + shift, config.log_amount_sd)
        shares = rng.dirichlet(np.full(receivers.size, config.fan_out_concentration))
        max_output_share[distributor] = shares.max()
        log_amount[receivers] = log_amount[distributor] + np.log(shares)
        return
    base = rng.normal(config.log_amount_mean + shift, config.log_amount_sd)
    hops = np.arange(members.size)
    noise = rng.normal(0.0, config.hop_noise_sd, members.size)
    if pattern.kind == "cycle":
        log_amount[members] = base + np.log1p(-config.cycle_fee_rate * hops) + noise
    else:
        log_amount[members] = base + hops * np.log(config.chain_retention) + noise


def _features(
    node_count: int,
    is_pattern: np.ndarray,
    patterns: list[Pattern],
    edges: list[tuple[int, int]],
    rng: np.random.Generator,
    config: SynthConfig,
) -> dict[str, np.ndarray]:
    in_degree = np.bincount([target for _, target in edges], minlength=node_count)
    out_degree = np.bincount([source for source, _ in edges], minlength=node_count)
    log_amount = rng.normal(config.log_amount_mean, config.log_amount_sd, node_count)
    log_hours = rng.normal(config.log_hours_mean, config.log_hours_sd, node_count)
    log_age = rng.normal(config.log_age_mean, config.log_age_sd, node_count)
    max_output_share = rng.beta(
        config.max_output_share_alpha, config.max_output_share_beta, node_count
    )
    input_amount_cv = rng.gamma(config.input_cv_shape, config.input_cv_scale, node_count)
    fee_noise = rng.normal(0.0, config.fee_sd, node_count)
    round_draw = rng.random(node_count)
    change_draw = rng.random(node_count)
    extra_inputs = rng.poisson(config.extra_io_rate, node_count)
    extra_outputs = rng.poisson(config.extra_io_rate, node_count)

    shift = config.local_shift
    log_hours[is_pattern] -= shift * config.log_hours_sd
    log_age[is_pattern] -= shift * config.log_age_sd
    fee_noise[is_pattern] += shift * config.fee_sd
    round_rate = np.where(
        is_pattern, config.round_amount_rate * (1.0 + 2.0 * shift), config.round_amount_rate
    )
    change_rate = np.where(
        is_pattern, config.change_output_rate * (1.0 - 0.3 * shift), config.change_output_rate
    )
    for pattern in patterns:
        _apply_pattern_amounts(pattern, log_amount, max_output_share, input_amount_cv, rng, config)

    log_fee = config.fee_slope * log_amount + config.fee_intercept + fee_noise
    num_inputs = np.maximum(1, in_degree + extra_inputs)
    num_outputs = np.maximum(1, out_degree + extra_outputs)
    log_size = np.log(10.0 + 148.0 * num_inputs + 34.0 * num_outputs)
    columns = {
        "f_log_amount": log_amount,
        "f_log_fee": log_fee,
        "f_log_fee_rate": log_fee - log_size,
        "f_num_inputs": num_inputs.astype(np.int64),
        "f_num_outputs": num_outputs.astype(np.int64),
        "f_log_size": log_size,
        "f_round_amount": (round_draw < round_rate).astype(np.int64),
        "f_change_output": (change_draw < change_rate).astype(np.int64),
        "f_max_output_share": max_output_share,
        "f_input_amount_cv": input_amount_cv,
        "f_log_hours_since_last": log_hours,
        "f_log_address_age_days": log_age,
    }
    return {
        name: np.round(values, FEATURE_DECIMALS) if values.dtype.kind == "f" else values
        for name, values in columns.items()
    }


def _generate_step(
    step: int, rng: np.random.Generator, config: SynthConfig
) -> tuple[pd.DataFrame, list[tuple[str, str]], list[dict[str, object]]]:
    node_count = config.nodes_per_step
    rate = config.illicit_share * node_count / expected_pattern_size(config)
    pattern_total = max(config.min_patterns_per_step, int(rng.poisson(rate)))
    kinds = [PATTERN_TYPES[int(i)] for i in rng.integers(0, len(PATTERN_TYPES), pattern_total)]
    parameters = [_pattern_parameter(kind, rng, config) for kind in kinds]
    chosen: list[tuple[str, int]] = []
    used = 0
    for kind, parameter in zip(kinds, parameters, strict=True):
        size = _member_count(kind, parameter)
        if used + size > node_count // 2:
            break
        chosen.append((kind, parameter))
        used += size
    benign_count = node_count - used

    edges = _benign_edges(benign_count, rng, config)
    patterns: list[Pattern] = []
    start = benign_count
    for kind, parameter in chosen:
        pattern = build_pattern(kind, parameter, start)
        patterns.append(pattern)
        start += len(pattern.members)
    edges.extend(_pattern_edges(patterns, benign_count, rng, config))

    is_pattern = np.zeros(node_count, dtype=bool)
    is_pattern[benign_count:] = True
    features = _features(node_count, is_pattern, patterns, edges, rng, config)

    visibility = rng.random(node_count)
    labels = np.where(
        is_pattern,
        np.where(visibility < config.illicit_visibility, "illicit", "unknown"),
        np.where(visibility < config.licit_visibility, "licit", "unknown"),
    )
    permutation = rng.permutation(node_count)
    offset = (step - 1) * node_count
    ids = [f"tx{offset + int(position) + 1:06d}" for position in permutation]

    frame = pd.DataFrame({"node_id": ids, "time_step": step, "label": labels, **features})
    edge_ids = [(ids[source], ids[target]) for source, target in edges]
    truth = [
        {
            "node_id": ids[member],
            "time_step": step,
            "pattern_id": f"p{step:02d}_{index:02d}",
            "pattern_type": pattern.kind,
            "role": role,
        }
        for index, pattern in enumerate(patterns, start=1)
        for member, role in zip(pattern.members, pattern.roles, strict=True)
    ]
    return frame, edge_ids, truth


def generate(config: SynthConfig | None = None) -> SynthResult:
    config = config or SynthConfig()
    config.validate()
    rng = np.random.default_rng(config.seed)
    frames: list[pd.DataFrame] = []
    edge_rows: list[tuple[str, str]] = []
    truth_rows: list[dict[str, object]] = []
    for step in range(1, config.time_steps + 1):
        frame, step_edges, step_truth = _generate_step(step, rng, config)
        frames.append(frame)
        edge_rows.extend(step_edges)
        truth_rows.extend(step_truth)
    nodes = pd.concat(frames, ignore_index=True).sort_values("node_id", ignore_index=True)
    edges = pd.DataFrame(edge_rows, columns=["source", "target"]).sort_values(
        ["source", "target"], ignore_index=True
    )
    truth = pd.DataFrame(
        truth_rows, columns=["node_id", "time_step", "pattern_id", "pattern_type", "role"]
    ).sort_values("node_id", ignore_index=True)
    return SynthResult(config=config, nodes=nodes, edges=edges, truth=truth)


def pattern_summary(truth: pd.DataFrame, nodes: pd.DataFrame) -> dict[str, object]:
    labels = nodes.set_index("node_id")["label"]
    member_labels = labels.reindex(truth["node_id"]).to_numpy()
    per_type = truth.drop_duplicates("pattern_id")["pattern_type"].value_counts()
    return {
        "patterns": {kind: int(per_type.get(kind, 0)) for kind in PATTERN_TYPES},
        "patternNodes": len(truth),
        "patternNodesLabeledIllicit": int((member_labels == "illicit").sum()),
        "patternNodesUnknown": int((member_labels == "unknown").sum()),
    }
