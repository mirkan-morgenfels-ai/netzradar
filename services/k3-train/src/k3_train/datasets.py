from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    display_name: str
    license: str
    source: str
    publishable: bool
    withheld_reason: str = ""


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
    withheld_reason="Die Lizenz erlaubt keine Weitergabe bearbeiteter Fassungen.",
)

IBM_AML = DatasetInfo(
    name="ibm-aml",
    display_name="IBM Transactions for Anti-Money Laundering (HI-Small)",
    license="CDLA-Sharing-1.0",
    source="Kaggle ealtman2019/ibm-transactions-for-anti-money-laundering-aml",
    publishable=False,
    withheld_reason=(
        "Die öffentliche Demo bleibt das synthetische Netz; ein Ausschnitt wäre eine bearbeitete "
        "Fassung, die unter CDLA-Sharing-1.0 mit Änderungshinweis stehen müsste, data/k3 steht "
        "unter MIT."
    ),
)

DATASETS = {info.name: info for info in (SYNTHETIC, ELLIPTIC, IBM_AML)}
DATASET_NAMES = tuple(DATASETS)


def dataset_info(name: str) -> DatasetInfo:
    if name not in DATASETS:
        raise ValueError(f"Unbekannter Datensatz: {name}")
    return DATASETS[name]
