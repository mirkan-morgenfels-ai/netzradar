from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    display_name: str
    license: str
    source: str
    publishable: bool


SYNTHETIC = DatasetInfo(
    name="synthetic",
    display_name="Synthetisches Transaktionsnetz",
    license="MIT",
    source="services/k3-train/src/k3_train/synth.py",
    publishable=True,
)

ELLIPTIC = DatasetInfo(
    name="elliptic",
    display_name="Elliptic Bitcoin Dataset",
    license="CC BY-NC-ND 4.0",
    source="Kaggle ellipticco/elliptic-data-set",
    publishable=False,
)

DATASETS = {info.name: info for info in (SYNTHETIC, ELLIPTIC)}
DATASET_NAMES = tuple(DATASETS)


def dataset_info(name: str) -> DatasetInfo:
    if name not in DATASETS:
        raise ValueError(f"Unbekannter Datensatz: {name}")
    return DATASETS[name]
