import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

FLOAT_DIGITS = 4


def camel_case(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in rest)


def camel_case_keys(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {camel_case(key): value for key, value in mapping.items()}


def to_builtin(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_builtin(item) for item in value.tolist()]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def round_floats(value: Any, digits: int = FLOAT_DIGITS) -> Any:
    value = to_builtin(value)
    if isinstance(value, dict):
        return {key: round_floats(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [round_floats(item, digits) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Nicht endlicher Wert im Export: {value}")
        return round(value, digits) + 0.0
    return value


def dumps(payload: Any) -> str:
    return json.dumps(to_builtin(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(payload), encoding="utf-8", newline="\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
