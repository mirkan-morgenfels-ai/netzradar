import argparse
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from k3_train.datasets import DATASET_NAMES, dataset_info
from k3_train.export import build_exports, write_exports
from k3_train.gnn_results import GnnOutput, has_gnn, load_gnn, save_gnn
from k3_train.load import (
    Dataset,
    load_elliptic,
    load_synthetic,
    missing_elliptic_files,
    read_processed,
    write_processed,
)
from k3_train.paths import (
    DATA_DIR_ENV,
    EXPORT_DIR,
    RUNS_DIR,
    RepositoryLayoutError,
    check_repo_layout,
    data_root,
    private_runs_dir,
    processed_dir,
    raw_dir,
)
from k3_train.pipeline import (
    BaselineOutput,
    dataset_block,
    iso_date,
    iso_timestamp,
    load_baseline,
    run_baseline,
    save_baseline,
    utc_now,
)
from k3_train.runs import write_run_logs
from k3_train.synth import SynthConfig
from k3_train.verify import GNN_MODULES, verify

MISSING_GNN_MESSAGE = (
    "Für k3-train gnn fehlen torch und torch_geometric. Bitte das Extra installieren: "
    "uv sync --extra gnn, dann uv run --extra gnn k3-train gnn (mit make: make gnn)."
)
NO_DATASET_MESSAGE = (
    "Kein Datensatz gewählt.\n"
    "  Synthetisches Netz: make data SYNTH=1 (oder k3-train data --synth)\n"
    "  Elliptic: Rohdaten nach <Datenordner>/raw/elliptic legen oder per --raw-dir angeben, "
    "dann make data DATASET=elliptic\n"
    f"  Datenordner: --data-dir oder {DATA_DIR_ENV}, sonst services/k3-train/data\n"
    "  IBM-AML: Lader folgt nach der Datensatzentscheidung."
)
DATA_DIR_HELP = (
    "Datenordner für raw/ und processed/ (Vorrang vor der Umgebungsvariablen "
    f"{DATA_DIR_ENV}, Standard services/k3-train/data)"
)


def _print_dataset(dataset: Dataset) -> None:
    counts = dataset.label_counts()
    total = len(dataset.nodes)
    cleaning = dataset.cleaning
    print(f"Datensatz: {dataset.name}")
    print(f"  Knoten: {total}, Kanten: {len(dataset.edges)}, Zeitschritte: {dataset.time_steps}")
    local = len(dataset.feature_columns)
    aggregated = len(dataset.aggregated_columns)
    if aggregated:
        print(
            f"  Merkmale (gezählt): {local + aggregated}, davon lokal {local}, "
            f"über Nachbarn aggregiert {aggregated} (nicht in den Baselines)"
        )
    else:
        print(f"  Merkmale (gezählt): {local}")
    for label, count in counts.items():
        print(f"  {label}: {count} ({count / total:.2%})")
    print(
        f"  Entfernte Kanten: {cleaning.removed_total} "
        f"(fehlender Endpunkt {cleaning.removed_missing_endpoint}, "
        f"Selbstschleifen {cleaning.removed_self_loops}, Duplikate {cleaning.removed_duplicates})"
    )
    if "patterns" in dataset.notes:
        notes = dataset.notes
        print(
            f"  Eingebettete Muster: {notes['patterns']}, Musterknoten: {notes['patternNodes']} "
            f"(davon als illicit sichtbar {notes['patternNodesLabeledIllicit']}, "
            f"unknown {notes['patternNodesUnknown']})"
        )


def _print_baseline(output: BaselineOutput) -> None:
    evaluation = output.evaluation
    print(
        f"Test: {evaluation['testPositives']} illicit, {evaluation['testNegatives']} licit, "
        f"Prävalenz {evaluation['prevalence']:.4f}, "
        f"Accuracy 'alles licit' {evaluation['allLicitAccuracy']:.4f}"
    )
    print(
        f"Zufällige Rangfolge: PR-AUC im Erwartungswert {evaluation['randomPrAucExpected']:.4f}, "
        f"95-%-Quantil {evaluation['randomPrAucQ95']:.4f} "
        f"({evaluation['randomPermutations']} Permutationen)"
    )
    for run in output.runs:
        print(
            f"  {run['method']:8s} PR-AUC {run['prAuc']:.4f}  "
            f"P@R0.5 {run['precisionAtRecall50']:.4f}  "
            f"R@P0.5 {run['recallAtPrecision50']:.4f}  "
            f"Accuracy {run['accuracy']:.4f} "
            f"({run['accuracyFlagged']} markiert, {run['accuracyTruePositives']} Treffer)"
        )


def _print_gnn(output: GnnOutput) -> None:
    for run in output.runs:
        hyperparameters = run["hyperparameters"]
        spread = output.seed_spread[run["method"]]
        print(
            f"  {run['method']:9s} Merkmale {run['featureSet']}, "
            f"Gewicht {hyperparameters['positiveWeight']:.4f} "
            f"({hyperparameters['positiveWeightRule']}), "
            f"Epoche {hyperparameters['selectedEpoch']}, "
            f"Validierung PR-AUC {hyperparameters['validationPrAuc']:.4f}"
        )
        print(
            f"            Test PR-AUC {run['prAuc']:.4f}  "
            f"P@R0.5 {run['precisionAtRecall50']:.4f}  "
            f"R@P0.5 {run['recallAtPrecision50']:.4f}  "
            f"Accuracy {run['accuracy']:.4f} "
            f"({run['accuracyFlagged']} markiert, {run['accuracyTruePositives']} Treffer); "
            f"Seeds {spread['seeds'][0]}-{spread['seeds'][-1]}: "
            f"PR-AUC {spread['min']:.4f} bis {spread['max']:.4f}, Mittel {spread['mean']:.4f}"
        )


def command_data(args: argparse.Namespace) -> int:
    name = "synthetic" if args.synth else args.dataset
    if name is None:
        print(NO_DATASET_MESSAGE, file=sys.stderr)
        return 2
    root = data_root(args.data_dir)
    if name == "synthetic":
        dataset = load_synthetic(SynthConfig())
    else:
        source = Path(args.raw_dir) if args.raw_dir else raw_dir(root, name)
        missing = missing_elliptic_files(source)
        if missing:
            print(
                f"Elliptic-Rohdaten fehlen in {source}: {', '.join(missing)}\n"
                "Bezug: Kaggle ellipticco/elliptic-data-set (CC BY-NC-ND 4.0, nur lokal nutzen).",
                file=sys.stderr,
            )
            return 2
        dataset = load_elliptic(source)
    target = processed_dir(root, dataset.name)
    write_processed(dataset, target)
    _print_dataset(dataset)
    print(f"Geschrieben: {target}")
    return 0


def command_baseline(args: argparse.Namespace) -> int:
    info = dataset_info(args.dataset)
    if info.publishable:
        check_repo_layout()
    root = data_root(args.data_dir)
    dataset = read_processed(processed_dir(root, args.dataset))
    output = run_baseline(dataset, run_date=iso_date(utc_now()))
    save_baseline(output, processed_dir(root, args.dataset))
    runs_dir = RUNS_DIR if info.publishable else private_runs_dir(root, args.dataset)
    paths = write_run_logs(output, runs_dir)
    _print_baseline(output)
    for path in paths:
        print(f"Protokoll: {path}")
    return 0


def command_gnn(args: argparse.Namespace) -> int:
    try:
        from k3_train.gnn import GnnConfig
        from k3_train.gnn_pipeline import run_gnn
    except ModuleNotFoundError as error:
        if error.name not in GNN_MODULES:
            raise
        print(MISSING_GNN_MESSAGE, file=sys.stderr)
        return 2
    info = dataset_info(args.dataset)
    if info.publishable:
        check_repo_layout()
    root = data_root(args.data_dir)
    target = processed_dir(root, args.dataset)
    dataset = read_processed(target)
    baseline = load_baseline(target)
    if baseline.dataset != dataset_block(dataset):
        print(
            "Die Baseline-Ergebnisse passen nicht zu den aufbereiteten Daten. "
            "Bitte zuerst k3-train baseline ausführen.",
            file=sys.stderr,
        )
        return 1
    config = GnnConfig() if args.max_epochs is None else GnnConfig(max_epochs=args.max_epochs)
    output = run_gnn(dataset, baseline.measures, run_date=iso_date(utc_now()), config=config)
    save_gnn(output, target)
    runs_dir = RUNS_DIR if info.publishable else private_runs_dir(root, args.dataset)
    paths = write_run_logs(output, runs_dir)
    _print_gnn(output)
    for path in paths:
        print(f"Protokoll: {path}")
    return 0


def command_export(args: argparse.Namespace) -> int:
    info = dataset_info(args.dataset)
    if not info.publishable:
        print(
            f"Export abgelehnt: {info.display_name} ({info.license}) wird nicht veröffentlicht. "
            "Nach data/k3 gelangen nur frei lizenzierte Datensätze.",
            file=sys.stderr,
        )
        return 2
    check_repo_layout()
    root = data_root(args.data_dir)
    dataset = read_processed(processed_dir(root, args.dataset))
    output = load_baseline(processed_dir(root, args.dataset))
    if output.dataset != dataset_block(dataset):
        print(
            "Die Baseline-Ergebnisse passen nicht zu den aufbereiteten Daten. "
            "Bitte zuerst k3-train baseline ausführen.",
            file=sys.stderr,
        )
        return 1
    gnn = None
    if has_gnn(processed_dir(root, args.dataset)):
        gnn = load_gnn(processed_dir(root, args.dataset))
    else:
        print("Hinweis: keine GNN-Ergebnisse gefunden, scoreGnn bleibt leer (k3-train gnn).")
    bundle = build_exports(dataset, output, generated_at=iso_timestamp(utc_now()), gnn=gnn)
    paths = write_exports(bundle, EXPORT_DIR)
    selection = bundle.nodes["selection"]
    print(
        f"Ausschnitt: {len(bundle.nodes['nodes'])} Knoten, {len(bundle.edges['edges'])} Kanten, "
        f"gekürzt: {'ja' if selection['truncated'] else 'nein'}, "
        f"scoreGnn aus: {bundle.nodes['scoreGnnMethod'] or 'keinem Verfahren'}"
    )
    for path in paths:
        print(f"Geschrieben: {path}")
    return 0


def command_verify(_: argparse.Namespace) -> int:
    check_repo_layout()
    return verify(EXPORT_DIR, RUNS_DIR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="k3-train",
        description="NetzRadar: Daten aufbereiten, Baseline und GNN rechnen, exportieren",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data", help="Datensatz laden oder erzeugen und aufbereiten")
    data.add_argument("--synth", action="store_true", help="synthetisches Netz erzeugen")
    data.add_argument("--dataset", choices=DATASET_NAMES, help="Datensatz")
    data.add_argument("--raw-dir", help="Ordner mit den Rohdaten")
    data.add_argument("--data-dir", help=DATA_DIR_HELP)
    data.set_defaults(handler=command_data)

    for name, handler, help_text in (
        ("baseline", command_baseline, "robuste Z-Scores und Isolation Forest rechnen"),
        ("export", command_export, "nodes.json, edges.json und metrics.json schreiben"),
    ):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--dataset", choices=DATASET_NAMES, default="synthetic")
        sub.add_argument("--data-dir", help=DATA_DIR_HELP)
        sub.set_defaults(handler=handler)

    gnn = commands.add_parser(
        "gnn", help="GCN, GraphSAGE und MLP-Kontrolle trainieren (braucht das Extra gnn)"
    )
    gnn.add_argument("--dataset", choices=DATASET_NAMES, default="synthetic")
    gnn.add_argument("--data-dir", help=DATA_DIR_HELP)
    gnn.add_argument(
        "--max-epochs",
        type=int,
        help="höchstens so viele Epochen je Training (Standard 300); nur zum Testen",
    )
    gnn.set_defaults(handler=command_gnn)

    check = commands.add_parser("verify", help="synthetische Pipeline nachrechnen und vergleichen")
    check.set_defaults(handler=command_verify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler
    started = time.perf_counter()
    try:
        status = handler(args)
    except RepositoryLayoutError as error:
        print(f"Fehler: {error}", file=sys.stderr)
        status = 2
    except (FileNotFoundError, ValueError) as error:
        print(f"Fehler: {error}", file=sys.stderr)
        status = 1
    print(f"Laufzeit {args.command}: {time.perf_counter() - started:.1f} s")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
