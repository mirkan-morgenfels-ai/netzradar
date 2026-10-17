import csv
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from k3_train.datasets import IBM_AML
from k3_train.load import Dataset, build_dataset

IBM_AML_HEADER = (
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
)
RAW_COLUMNS = (
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
)
TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"
MINUTES_PER_DAY = 1440
LAUNDERING_LABELS = {"0": "licit", "1": "illicit"}
MANGLED_ACCOUNT = re.compile(r"Account\.\d+")
NODE_ID_WIDTH = 7
FEATURE_DIGITS = 6
BASE_FEATURES = (
    "f_log_amount_paid",
    "f_log_amount_received",
    "f_cross_currency",
    "f_same_bank",
    "f_self_transfer",
    "f_whole_amount_paid",
    "f_time_sin",
    "f_time_cos",
)
CATEGORY_FEATURES = (
    ("payment_format", "f_format_", "paymentFormats"),
    ("payment_currency", "f_paid_", "paymentCurrencies"),
    ("receiving_currency", "f_received_", "receivingCurrencies"),
)


@dataclass(frozen=True)
class IbmAmlConfig:
    file_name: str = "HI-Small_Trans.csv"
    documented_days: int = 10
    window_minutes: int = 360
    max_successors: int = 10
    max_predecessors: int = 10

    def validate(self) -> None:
        if not 0 <= self.window_minutes < MINUTES_PER_DAY:
            raise ValueError(
                f"Das Zeitfenster muss zwischen 0 und {MINUTES_PER_DAY - 1} Minuten liegen"
            )
        if self.max_successors < 1 or self.max_predecessors < 1:
            raise ValueError("Die Kantengrenzen je Knoten müssen mindestens 1 sein")
        if self.documented_days < 1:
            raise ValueError("Die Zahl der dokumentierten Tage muss mindestens 1 sein")

    def as_dict(self) -> dict[str, Any]:
        return {
            "file": self.file_name,
            "documentedDays": self.documented_days,
            "timeStep": "calendarDay",
            "windowMinutes": self.window_minutes,
            "maxSuccessors": self.max_successors,
            "maxPredecessors": self.max_predecessors,
            "sameTimeStepOnly": True,
        }


@dataclass(frozen=True)
class TransactionEdges:
    source: np.ndarray
    target: np.ndarray
    window_pairs: int
    cross_step_pairs: int

    @property
    def removed_by_limit(self) -> int:
        return self.window_pairs - len(self.source)


def missing_ibm_aml_files(raw_dir: Path, config: IbmAmlConfig | None = None) -> list[str]:
    name = (config or IbmAmlConfig()).file_name
    return [] if (raw_dir / name).is_file() else [name]


def header_problems(header: Sequence[str]) -> list[str]:
    names = [
        "Account" if MANGLED_ACCOUNT.fullmatch(name.strip()) else name.strip() for name in header
    ]
    if names == list(IBM_AML_HEADER):
        return []
    available = Counter(names)
    missing = []
    for name in IBM_AML_HEADER:
        if available[name] > 0:
            available[name] -= 1
        else:
            missing.append(name)
    expected = Counter(IBM_AML_HEADER)
    seen: Counter[str] = Counter()
    extra = []
    for name in names:
        seen[name] += 1
        if seen[name] > expected[name]:
            extra.append(name)
    problems = []
    if missing:
        problems.append("fehlende Spalten: " + ", ".join(missing))
    if extra:
        problems.append("unerwartete Spalten: " + ", ".join(extra))
    if not problems:
        problems.append("Reihenfolge der Spalten weicht ab")
    return problems


def check_header(header: Sequence[str], file_name: str) -> None:
    problems = header_problems(header)
    if problems:
        raise ValueError(
            f"Die Kopfzeile von {file_name} passt nicht zum IBM-AML-Format "
            f"({'; '.join(problems)}). Erwartet: {', '.join(IBM_AML_HEADER)}"
        )


def read_transactions(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), None)
    if header is None:
        raise ValueError(f"{path.name} ist leer")
    check_header(header, path.name)
    raw = pd.read_csv(
        path,
        header=0,
        names=list(RAW_COLUMNS),
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    return raw.apply(lambda column: column.str.strip())


def _first_row(mask: np.ndarray) -> int:
    return int(np.flatnonzero(mask)[0]) + 1


def _check_filled(raw: pd.DataFrame) -> None:
    empty = [
        f"{IBM_AML_HEADER[index]} (Spalte {index + 1})"
        for index, column in enumerate(RAW_COLUMNS)
        if (raw[column] == "").any()
    ]
    if empty:
        raise ValueError(f"Leere Werte in den Spalten: {', '.join(empty)}")


def _timestamps(raw: pd.DataFrame) -> pd.Series:
    stamps = pd.to_datetime(raw["timestamp"], format=TIMESTAMP_FORMAT, errors="coerce")
    bad = stamps.isna().to_numpy()
    if bad.any():
        row = _first_row(bad)
        raise ValueError(
            f"{int(bad.sum())} Zeitstempel passen nicht zum Format {TIMESTAMP_FORMAT} "
            f"(erste Datenzeile {row}: {raw['timestamp'].iloc[row - 1]!r})"
        )
    return stamps


def _amounts(raw: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(raw[column], errors="coerce").to_numpy(dtype=np.float64)
    bad = ~np.isfinite(values) | (values < 0)
    if bad.any():
        title = IBM_AML_HEADER[RAW_COLUMNS.index(column)]
        row = _first_row(bad)
        raise ValueError(
            f"{title}: {int(bad.sum())} Werte sind keine nicht negativen Zahlen "
            f"(erste Datenzeile {row}: {raw[column].iloc[row - 1]!r})"
        )
    return values


def _labels(raw: pd.DataFrame) -> np.ndarray:
    flags = raw["is_laundering"].to_numpy(dtype=str)
    unknown = sorted(set(flags) - set(LAUNDERING_LABELS))
    if unknown:
        raise ValueError(f"Is Laundering enthält andere Werte als 0 und 1: {unknown[:5]}")
    return flags


def category_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not slug:
        raise ValueError(f"Kategorie ohne verwertbaren Namen: {name!r}")
    return slug


def local_features(
    frame: pd.DataFrame,
    paid: np.ndarray,
    received: np.ndarray,
    minute_of_day: np.ndarray,
    senders: np.ndarray,
    receivers: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    angle = 2 * np.pi * minute_of_day / MINUTES_PER_DAY
    base = (
        np.round(np.log1p(paid), FEATURE_DIGITS),
        np.round(np.log1p(received), FEATURE_DIGITS),
        frame["payment_currency"].to_numpy() != frame["receiving_currency"].to_numpy(),
        frame["from_bank"].to_numpy() == frame["to_bank"].to_numpy(),
        senders == receivers,
        paid == np.floor(paid),
        np.round(np.sin(angle), FEATURE_DIGITS) + 0.0,
        np.round(np.cos(angle), FEATURE_DIGITS) + 0.0,
    )
    columns: dict[str, np.ndarray] = {
        name: values.astype(np.int64) if values.dtype == bool else values
        for name, values in zip(BASE_FEATURES, base, strict=True)
    }
    vocabularies: dict[str, list[str]] = {}
    for column, prefix, key in CATEGORY_FEATURES:
        values = frame[column].to_numpy(dtype=str)
        names = sorted(set(values))
        slugs = [category_slug(name) for name in names]
        clashes = sorted({slug for slug in slugs if slugs.count(slug) > 1})
        if clashes:
            raise ValueError(f"Kategorien mit gleichem Spaltennamen in {column}: {clashes}")
        for name, slug in zip(names, slugs, strict=True):
            columns[prefix + slug] = (values == name).astype(np.int64)
        vocabularies[key] = names
    return pd.DataFrame(columns), vocabularies


def _rank_in_runs(keys: np.ndarray) -> np.ndarray:
    if len(keys) == 0:
        return np.zeros(0, dtype=np.int64)
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    lengths = np.diff(np.r_[starts, len(keys)])
    return np.arange(len(keys)) - np.repeat(starts, lengths)


def _window_bounds(
    sender_groups: np.ndarray,
    receiver_groups: np.ndarray,
    minutes: np.ndarray,
    window: int,
    span: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.lexsort((np.arange(len(minutes)), minutes, sender_groups))
    keys = sender_groups[order] * span + minutes[order]
    start = receiver_groups * span + minutes
    lower = np.searchsorted(keys, start, side="left")
    upper = np.searchsorted(keys, start + window, side="right")
    return order, lower, upper


def transaction_edges(
    steps: Any,
    minutes: Any,
    senders: Any,
    receivers: Any,
    config: IbmAmlConfig,
) -> TransactionEdges:
    steps = np.asarray(steps, dtype=np.int64)
    minutes = np.asarray(minutes, dtype=np.int64)
    senders = np.asarray(senders, dtype=np.int64)
    receivers = np.asarray(receivers, dtype=np.int64)
    count = len(steps)
    if count == 0:
        empty = np.zeros(0, dtype=np.int64)
        return TransactionEdges(empty, empty, 0, 0)
    window = config.window_minutes
    span = int(minutes.max()) + window + 1
    accounts = int(max(senders.max(), receivers.max())) + 1
    loops = int((senders == receivers).sum())
    _, lower_all, upper_all = _window_bounds(senders, receivers, minutes, window, span)
    all_pairs = int((upper_all - lower_all).sum()) - loops
    order, lower, upper = _window_bounds(
        steps * accounts + senders, steps * accounts + receivers, minutes, window, span
    )
    window_pairs = int((upper - lower).sum()) - loops
    take = np.minimum(upper - lower, config.max_successors + 1)
    query = np.repeat(np.arange(count), take)
    offsets = np.arange(len(query)) - np.repeat(np.cumsum(take) - take, take)
    target = order[np.repeat(lower, take) + offsets]
    distinct = target != query
    query, target = query[distinct], target[distinct]
    keep = _rank_in_runs(query) < config.max_successors
    source, target = query[keep], target[keep]
    by_target = np.lexsort((source, -minutes[source], target))
    source, target = source[by_target], target[by_target]
    keep = _rank_in_runs(target) < config.max_predecessors
    source, target = source[keep], target[keep]
    final = np.lexsort((target, source))
    return TransactionEdges(
        source=source[final],
        target=target[final],
        window_pairs=window_pairs,
        cross_step_pairs=all_pairs - window_pairs,
    )


def node_ids(rows: np.ndarray, total_rows: int) -> np.ndarray:
    width = max(NODE_ID_WIDTH, len(str(total_rows)))
    return ("t" + pd.Series(rows).astype(str).str.zfill(width)).to_numpy()


def load_ibm_aml(raw_dir: Path, config: IbmAmlConfig | None = None) -> Dataset:
    config = config or IbmAmlConfig()
    config.validate()
    missing = missing_ibm_aml_files(raw_dir, config)
    if missing:
        raise FileNotFoundError(f"Fehlende IBM-AML-Datei in {raw_dir}: {', '.join(missing)}")
    raw = read_transactions(raw_dir / config.file_name)
    if raw.empty:
        raise ValueError(f"{config.file_name} enthält keine Transaktionen")
    _check_filled(raw)
    stamps = _timestamps(raw)
    paid_all = _amounts(raw, "amount_paid")
    received_all = _amounts(raw, "amount_received")
    flags_all = _labels(raw)

    first_day = stamps.min().normalize()
    minutes_all = ((stamps - first_day) // pd.Timedelta(minutes=1)).to_numpy(dtype=np.int64)
    steps_all = minutes_all // MINUTES_PER_DAY + 1
    inside = steps_all <= config.documented_days
    present = np.unique(steps_all[inside])
    gaps = sorted(set(range(1, int(present.max()) + 1)) - {int(step) for step in present})
    if gaps:
        raise ValueError(f"Tage ohne Transaktionen im dokumentierten Zeitraum: {gaps}")

    rows = np.flatnonzero(inside) + 1
    frame = raw[inside].reset_index(drop=True)
    steps = steps_all[inside]
    minutes = minutes_all[inside]
    flags = flags_all[inside]
    sender_keys = (frame["from_bank"] + ":" + frame["from_account"]).to_numpy()
    receiver_keys = (frame["to_bank"] + ":" + frame["to_account"]).to_numpy()
    codes, accounts = pd.factorize(np.concatenate([sender_keys, receiver_keys]))
    senders = codes[: len(frame)].astype(np.int64)
    receivers = codes[len(frame) :].astype(np.int64)

    features, vocabularies = local_features(
        frame,
        paid=paid_all[inside],
        received=received_all[inside],
        minute_of_day=minutes % MINUTES_PER_DAY,
        senders=senders,
        receivers=receivers,
    )
    ids = node_ids(rows, len(raw))
    nodes = pd.concat(
        [
            pd.DataFrame(
                {
                    "node_id": ids,
                    "time_step": steps,
                    "label": [LAUNDERING_LABELS[flag] for flag in flags],
                }
            ),
            features,
        ],
        axis=1,
    )
    links = transaction_edges(steps, minutes, senders, receivers, config)
    edges = pd.DataFrame({"source": ids[links.source], "target": ids[links.target]})
    outside = ~inside
    notes = {
        **config.as_dict(),
        "inputRows": len(raw),
        "firstDay": first_day.date().isoformat(),
        "lastDay": (first_day + pd.Timedelta(days=int(steps.max()) - 1)).date().isoformat(),
        "removedAfterRange": int(outside.sum()),
        "removedAfterRangeLaundering": int((outside & (flags_all == "1")).sum()),
        "accounts": len(accounts),
        "windowPairs": links.window_pairs,
        "windowPairsAcrossTimeSteps": links.cross_step_pairs,
        "removedByDegreeLimit": links.removed_by_limit,
        **vocabularies,
    }
    return build_dataset(IBM_AML.name, nodes, edges, notes=notes)
