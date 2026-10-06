import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SERVICE_DIR = PACKAGE_DIR.parents[1]
REPO_ROOT = SERVICE_DIR.parents[1]
DEFAULT_DATA_DIR = SERVICE_DIR / "data"
DATA_DIR_ENV = "K3_DATA_DIR"
EXPORT_DIR = REPO_ROOT / "data" / "k3"
RUNS_DIR = REPO_ROOT / "docs" / "runs"
REPO_MARKERS = (REPO_ROOT / "pnpm-workspace.yaml", SERVICE_DIR / "pyproject.toml")


class RepositoryLayoutError(RuntimeError):
    pass


def check_repo_layout() -> None:
    missing = [marker for marker in REPO_MARKERS if not marker.is_file()]
    if missing:
        raise RepositoryLayoutError(
            f"Projektordner nicht gefunden: {missing[0]} fehlt. k3-train muss editierbar aus "
            "services/k3-train installiert sein (uv sync), sonst landen data/k3, docs/runs und "
            "die aufbereiteten Daten außerhalb des Repositorys."
        )


def data_root(override: str | Path | None = None) -> Path:
    chosen = str(override) if override else os.environ.get(DATA_DIR_ENV, "").strip()
    if chosen:
        return Path(chosen).expanduser().resolve()
    check_repo_layout()
    return DEFAULT_DATA_DIR


def raw_dir(root: Path, dataset: str) -> Path:
    return root / "raw" / dataset


def processed_dir(root: Path, dataset: str) -> Path:
    return root / "processed" / dataset


def private_runs_dir(root: Path, dataset: str) -> Path:
    return processed_dir(root, dataset) / "runs"
