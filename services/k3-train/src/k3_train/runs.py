from pathlib import Path

from k3_train.jsonio import round_floats, write_json
from k3_train.pipeline import BaselineOutput, run_log


def run_log_name(date: str, dataset: str, method: str) -> str:
    return f"{date}_{dataset}_{method}.json"


def latest_run_log(runs_dir: Path, dataset: str, method: str) -> Path | None:
    candidates = sorted(runs_dir.glob(run_log_name("*", dataset, method)))
    return candidates[-1] if candidates else None


def write_run_logs(output: BaselineOutput, out_dir: Path) -> list[Path]:
    paths = []
    for run in output.runs:
        path = out_dir / run_log_name(run["date"], output.dataset["name"], run["method"])
        write_json(path, round_floats(run_log(output, run)))
        paths.append(path)
    return paths
