from pathlib import Path

from k3_train.gnn_results import GnnOutput
from k3_train.jsonio import round_floats, write_json
from k3_train.pipeline import BaselineOutput


def run_log_name(date: str, dataset: str, method: str) -> str:
    return f"{date}_{dataset}_{method}.json"


def latest_run_log(runs_dir: Path, dataset: str, method: str) -> Path | None:
    candidates = sorted(runs_dir.glob(run_log_name("*", dataset, method)))
    return candidates[-1] if candidates else None


def write_run_logs(output: BaselineOutput | GnnOutput, out_dir: Path) -> list[Path]:
    paths = []
    for log in output.run_logs():
        path = out_dir / run_log_name(log["date"], output.dataset["name"], log["method"])
        write_json(path, round_floats(log))
        paths.append(path)
    return paths
