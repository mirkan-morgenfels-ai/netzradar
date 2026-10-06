# NetzRadar

Anomalie-Erkennung in Transaktionsnetzwerken: klassische Baseline gegen Graph Neural Networks. Zeitlicher Split, PR-AUC statt Accuracy, feste Seeds, vorberechnete Ergebnisse, statische Visualisierung.

[![CI](https://github.com/mirkan-morgenfels-ai/AI-Project-3/actions/workflows/ci.yml/badge.svg)](https://github.com/mirkan-morgenfels-ai/AI-Project-3/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

Live-Demo: [Platzhalter: noch nicht deployt. Geplant ist ein Vercel-Projekt mit Root `apps/web`, Seite unter `/projects/netzradar`.]

**English summary.** NetzRadar is a reproducible case study in graph-based fraud and anti-money-laundering detection. It compares a classical baseline (robust z-scores, Isolation Forest with graph centrality features) with Graph Neural Networks (GCN and GraphSAGE in PyTorch Geometric) on a transaction network, using a strict temporal train/test split and precision-recall metrics suited to a rare positive class. Models are trained offline; the web page renders precomputed results (an interactive subgraph around the top anomalies plus a metrics table) from static JSON, with no model and no API call at runtime. Status on 6 October 2026: data preparation, temporal split, baseline and export run on a synthetic transaction network with planted laundering patterns. On its test period the robust z-scores (PR-AUC 0.1633) lie above the 95th percentile of 10,000 random rankings (0.1420), while the Isolation Forest (0.1281) cannot be told apart from a random ranking (expected PR-AUC 0.1181; a constant score reaches the prevalence of 0.1115). The GNN models have not been trained yet. Raw datasets are not redistributed; see the dataset section for licences.

## Fragestellung

Betrug und Geldwäsche sind Netzwerkphänomene. Ein Konto mit unauffälligen eigenen Merkmalen wird verdächtig, wenn seine Nachbarn verdächtig sind, wenn Geld im Kreis läuft oder wenn viele kleine Beträge auf einen Knoten zulaufen. Die Frage dieses Projekts: Wie viel gewinnt ein Modell, das die Nachbarschaft einbezieht (GNN), gegenüber einem Modell, das nur die Merkmale des einzelnen Knotens sieht (Baseline), bei gleichem Split und gleichen Metriken?

## Ergebnisse

Datensatz: synthetisches Transaktionsnetz aus dem eigenen Generator (MIT, Seed 42) mit 12.000 Knoten, 13.757 gerichteten Kanten, 30 Zeitschritten und 12 lokalen Merkmalen. Split: zeitlich, Training auf den Zeitschritten 1 bis 21 (Validierungsteil 18 bis 21 innerhalb des Trainings), Test auf 22 bis 30, 0 Kanten zwischen Trainings- und Testknoten. Seed: 42. Stand: 06.10.2026.

Bewertet werden nur gelabelte Testknoten: 95 illicit und 757 licit; `unknown` ist ausgeschlossen. Die Prävalenz beträgt 95 / 852 = 0,1115; so hoch ist die PR-AUC eines konstanten Scores. Eine zufällige Rangfolge erreicht im Erwartungswert etwas mehr, 0,1181, weil die ersten Ränge stark in die Average Precision eingehen. 95 % von 10.000 zufälligen Rangfolgen (Seed 42) bleiben unter 0,1420. Gegen diese Werte sind die Zeilen zu lesen (`evaluation.randomPrAucExpected` und `evaluation.randomPrAucQ95` in `metrics.json`).

| Verfahren | Merkmale | PR-AUC | Precision bei Recall ≥ 0,5 | Recall bei Precision ≥ 0,5 |
|---|---|---|---|---|
| Robuste Z-Scores (Einzelmerkmale) | 12 lokale, davon 10 wirksam | 0,1633 | 0,1324 | 0,0211 |
| Isolation Forest (Einzelmerkmale + Graphmaße) | 12 lokale + 4 Graphmaße | 0,1281 | 0,1340 | 0,0000 |
| GCN (2 Schichten) | lokale + Nachbarschaft | Schritt 4, noch nicht gemessen | – | – |
| GraphSAGE (2 Schichten) | lokale + Nachbarschaft | Schritt 4, noch nicht gemessen | – | – |
| Zufällige Rangfolge (Erwartungswert; 95 % unter 0,1420) | – | 0,1181 | – | – |
| Konstanter Score (Prävalenz) | – | 0,1115 | – | – |

Bei den Z-Scores haben `f_round_amount` und `f_change_output` auf den Trainingsknoten MAD = 0 und tragen nichts bei; wirksam sind 10 der 12 lokalen Merkmale (`hyperparameters.zeroMadFeatures`).

Precision bei Recall ≥ 0,5 ist die höchste Precision über alle Schwellen, bei denen mindestens die Hälfte der illicit-Testknoten gefunden wird. Recall bei Precision ≥ 0,5 ist der höchste Recall über alle Schwellen, bei denen mindestens jeder zweite markierte Knoten illicit ist; 0 heißt, dass keine Schwelle das erreicht.

Accuracy als Nebenwert, wenn die obersten 2 % der Test-Scores als auffällig gelten; Gleichstände an der Schwelle zählen mit: beide Verfahren 0,8744. Ein Modell, das alles licit nennt, erreicht 0,8885. Der Isolation Forest markiert 18 Knoten mit 3 Treffern: (852 − 15 − 92) / 852. Die robusten Z-Scores markieren wegen Gleichständen an der Schwelle 22 Knoten mit 5 Treffern: (852 − 17 − 90) / 852. Beides ergibt 745 / 852 = 0,8744 (`accuracyFlagged` und `accuracyTruePositives` in `metrics.json`). Genau deshalb ist Accuracy hier keine Hauptmetrik.

**Einordnung.** Die Zahlen sind auf einem synthetischen Netz mit eingebauten Mustern gemessen und sagen nichts über reale Transaktionsdaten aus. Die robusten Z-Scores liegen mit 0,1633 über dem 95-%-Quantil zufälliger Rangfolgen (0,1420) und ordnen auffällige Knoten damit etwas besser als Zufall. Der Isolation Forest liegt mit 0,1281 darunter und ist von einer zufälligen Rangfolge (Erwartungswert 0,1181) nicht zu unterscheiden. Ob das am Verfahren oder an den zusätzlichen Graphmaßen liegt, lässt sich aus diesen zwei Läufen nicht ablesen, weil sich beides gleichzeitig unterscheidet. Von seinen 50 höchsten Testknoten (die Startknoten in `data/k3/nodes.json`) sind 2 illicit, 10 licit und 38 unknown. Ihr Median-Grad (ein- plus ausgehend) liegt bei 13, der mittlere Grad im ganzen Netz bei 2 · 13.757 / 12.000 ≈ 2,29. Der Isolation Forest hält also vor allem stark vernetzte gutartige Knoten für auffällig: Der gutartige Hintergrund des Generators wächst durch Preferential Attachment und bildet Hubs, und Hubs sind strukturelle Ausreißer. Die meisten lokalen Merkmale der Musterknoten sind bewusst nur um 0,5 Standardabweichungen verschoben (`localShift`). Einzelne weichen stärker ab: der größte Ausgangsanteil der Fan-out-Verteiler, die geringere Streuung der Fan-in-Beträge und die doppelte Rate runder Beträge (Einzelheiten in [`docs/daten.md`](docs/daten.md)). Das Signal steckt vor allem in der Netzstruktur (Fan-in, Fan-out, Kreise, Ketten, illicit-Nachbarschaften). Ob GCN und GraphSAGE es nutzen können, misst Schritt 4; ein Vorsprung wäre auf diesem Netz allerdings zum Teil eingebaut (siehe Grenzen). Die Generator-Parameter wurden nach dem ersten Ergebnis nicht nachjustiert.

Die vollständigen Läufe mit Konfiguration, Seeds und Precision-Recall-Kurven liegen in `data/k3/metrics.json` und `docs/runs/`. Ein Ergebnis, bei dem das GNN die Baseline nicht schlägt, wird hier genauso berichtet.

## Datensatz

Rohdaten sind nicht Teil dieses Repositories. Sie werden lokal nach `services/k3-train/data/raw/` (gitignored) oder in einen Datenordner außerhalb des Repositorys geladen (`K3_DATA_DIR` oder `--data-dir`). Veröffentlicht werden nur Code, Metriken, Plots und Stichproben synthetischer Daten. Bezug, Lizenzen und Ablage stehen in [`docs/daten.md`](docs/daten.md).

| Datensatz | Verwendung | Lizenz | Bezug | Stand |
|---|---|---|---|---|
| Eigenes synthetisches Netz (eigener Generator, NumPy) | Tests, CI, derzeit öffentliche Demo | MIT | `services/k3-train/src/k3_train/synth.py` | umgesetzt |
| IBM Transactions for Anti-Money Laundering (synthetisch) | öffentliche Demo, falls gewählt | CDLA-Sharing-1.0 (Daten), Apache-2.0 (Code) | Kaggle `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`, GitHub IBM/AML-Data | Lader offen, Datensatzentscheidung steht aus |
| Elliptic Bitcoin Dataset (203.769 Knoten, 234.355 Kanten, 49 Zeitschritte, 2 % illicit) | nur lokaler Methodenvergleich | CC BY-NC-ND 4.0 | Kaggle `ellipticco/elliptic-data-set` | Lader nur an erfundener Fixture getestet |

Das synthetische Netz: je Zeitschritt 400 Knoten. Der gutartige Hintergrund ist ein Preferential-Attachment-Baum mit zufälliger Kantenrichtung und 15 % Zusatzkanten. Eingebettet sind Muster vom Typ Fan-in (viele kleine Beträge auf einen Sammler), Fan-out (ein Verteiler auf viele Empfänger), Kreis und Kette, die mit Wahrscheinlichkeit 0,3 aneinandergehängt werden. Kanten verlaufen nur innerhalb eines Zeitschritts. Musterknoten sind mit Wahrscheinlichkeit 0,55 als illicit gelabelt, sonst `unknown`; gutartige Knoten sind mit Wahrscheinlichkeit 0,22 als licit gelabelt, sonst `unknown`. Ergebnis laut `metrics.json`: 292 illicit (2,43 %), 2.587 licit (21,56 %), 9.121 unknown (76,01 %). Alle 38 Generator-Parameter stehen in `metrics.json` unter `dataset.generator` und in `docs/daten.md`.

Zum Elliptic-Datensatz: Der Originaldatensatz steht unter CC BY-NC-ND 4.0. Deshalb werden daraus keine Rohdaten, keine bearbeiteten Fassungen und keine Ausschnitte veröffentlicht. Der Code verweigert den Export (`k3-train export --dataset elliptic` endet mit Exit 2) und legt Elliptic-Laufprotokolle nur lokal ab. Einzelne Spiegelungen im Netz nennen andere Lizenzen; hier gilt die restriktivere Angabe. Die Merkmalszahl wird in der Literatur als 165, 166 oder 167 angegeben. Der Lader erwartet nach Transaktions-ID und Zeitschritt 165 Merkmalsspalten und trennt die letzten 72, über Nachbarn aggregierten (`a_*`), von den lokalen (`f_*`); die Baselines nutzen nur die lokalen. Bei einer anderen Spaltenzahl bricht er ab. Geprüft ist das bisher nur an einer erfundenen Fixture (siehe `docs/daten.md`).

## Methodik

- **Zeitlicher Split.** Alle Trainingszeitschritte liegen vor allen Testzeitschritten. Synthetisch: Training 1 bis 21, Test 22 bis 30; Elliptic: Training 1 bis 34, Test 35 bis 49. Der Validierungsteil (synthetisch 18 bis 21, Elliptic 30 bis 34) liegt innerhalb des Trainingszeitraums und ist für die Hyperparameterwahl der GNN in Schritt 4 vorgesehen; die Baselines haben keine abgestimmten Hyperparameter. Ein zufälliger Split würde Knoten derselben Zeitkomponente auf beide Seiten verteilen und die Ergebnisse überschätzen. Vor jedem Lauf prüft `check_split`, dass kein Zeitschritt auf beiden Seiten liegt und keine Kante Trainings- und Testknoten verbindet (`crossSplitEdges` = 0).
- **Lokale Merkmale** (12, synthetisch): Betrag, Gebühr, Gebührenrate, Zahl der Ein- und Ausgänge, Größe, runder Betrag, Wechselgeld-Ausgang, größter Ausgangsanteil, Variationskoeffizient der Eingangsbeträge, Stunden seit der letzten Transaktion, Alter der Adresse. Wie bei Elliptic hängen Ein- und Ausgänge teilweise vom Grad ab (Grad + Poisson(1)).
- **Graphmaße** je Zeitschritt: Ein- und Ausgangsgrad, Betweenness-Zentralität auf dem gerichteten Graphen (exakt bis 5.000 Knoten je Zeitschritt, darüber mit 1.000 gezogenen Quellknoten, Seed 42) und Eigenvektor-Zentralität auf der ungerichteten Version (höchstens 1.000 Iterationen, Toleranz 1e-6, bei Nichtkonvergenz 0 mit Warnung). Diese Einstellungen und die Zeitschritte mit Stichprobe oder Rückfall auf 0 stehen je Lauf unter `hyperparameters.graphMeasures`.
- **Robuste Z-Scores.** zᵢⱼ = (xᵢⱼ − medianⱼ) / (1,4826 · MADⱼ) mit Median und MAD nur aus den Trainingsknoten; Merkmale mit MAD = 0 tragen 0 bei. Score = maxⱼ |zᵢⱼ| über die lokalen Merkmale. Der Faktor 1,4826 macht die MAD bei Normalverteilung zu einem Schätzer der Standardabweichung. Welche Merkmale wegen MAD = 0 nichts beitragen, steht unter `hyperparameters.zeroMadFeatures`.
- **Isolation Forest** mit 200 Bäumen, `contamination` 0,02, `max_samples` auto, `random_state` 42, angepasst nur auf den Trainingsknoten ohne Labels, auf lokalen Merkmalen plus Graphmaßen. Score = −`score_samples`: Punkte, die mit wenigen zufälligen Splits isoliert werden, gelten als anomal.
- **GNN** (Schritt 4, geplant): GCN und GraphSAGE mit zwei Schichten, 64 versteckten Einheiten, Dropout 0,5, Adam mit Lernrate 0,01 und Weight Decay 5e-4, gewichtete Kreuzentropie für die seltene Positivklasse; Hyperparameter nur am Validierungsteil gewählt.
- **Metriken.** Hauptmetrik ist PR-AUC als Average Precision: AP = Σₙ (Rₙ − Rₙ₋₁) · Pₙ über die Schwellen n. Dazu Precision bei Recall ≥ 0,5 und Recall bei Precision ≥ 0,5, beide auf der vollen Kurve. Die exportierte PR-Kurve beginnt bei der höchsten Schwelle, ohne den künstlichen Startpunkt (Recall 0, Precision 1) von scikit-learn, und ist auf höchstens 101 Punkte ausgedünnt. Als Zufallsreferenz dienen die Prävalenz (PR-AUC eines konstanten Scores), der exakte Erwartungswert der Average Precision einer zufälligen Rangfolge, E[AP] = (1/N) Σₖ (1 + (k − 1)(P − 1)/(N − 1)) / k mit N gelabelten Testknoten und P positiven, und das 95-%-Quantil aus 10.000 zufälligen Rangfolgen mit Seed 42. Accuracy nur als Nebenwert. Warum: Im synthetischen Test erreicht „alles licit“ 0,8885 Accuracy und ist wertlos. Bei Elliptic sind 2 % aller Knoten illicit, unter den gelabelten aber 4.545 / 46.564 ≈ 9,8 %; ein Alles-licit-Modell hätte dort auf den gelabelten Knoten etwa 90 % Accuracy.
- **Reproduzierbarkeit.** Ein Seed (42) für Generator, Isolation Forest, Betweenness-Stichprobe, Zufallsreferenz und Layout. Merkmale sind auf 6, exportierte Floats auf 4 Nachkommastellen gerundet. Jeder Lauf schreibt Datensatz, Split, Seed, Hyperparameter und Datum nach `docs/runs/`. `k3-train verify` rechnet die synthetische Pipeline neu und vergleicht jeden Wert in `metrics.json`, `nodes.json`, `edges.json` und in den jüngsten Protokollen unter `docs/runs/` mit Toleranz 1e-4. Zwei Läufe in getrennten Prozessen sind bis auf Zeitstempel byte-gleich.

## So funktioniert es

```
synth.py (Seed 42)  oder  Rohdaten lokal (<Datenordner>/raw/, nicht im Repo)
  -> k3-train data       Laden, Kanten bereinigen (fehlende Endpunkte, Selbstschleifen, Duplikate)
                         -> <Datenordner>/processed/<datensatz>/ (nicht im Repo;
                            Datenordner: --data-dir, K3_DATA_DIR oder services/k3-train/data)
  -> k3-train baseline   zeitlicher Split + Split-Pruefung
                         -> Graphmasse je Zeitschritt (Grad, Betweenness, Eigenvektor)
                         -> robuste Z-Scores (lokal) | Isolation Forest (lokal + Graph)
                         -> Kennzahlen auf gelabelten Testknoten
                         -> docs/runs/<datum>_<datensatz>_<verfahren>.json
  -> (Schritt 4)         GCN / GraphSAGE in PyTorch Geometric -> Kennzahlen
  -> k3-train export     50 Testknoten mit hoechstem Isolation-Forest-Score + 2-Hop-Nachbarschaft,
                         Layout vorberechnet -> data/k3/nodes.json, edges.json, metrics.json
  -> k3-train verify     Neuberechnung, Vergleich mit metrics.json, nodes.json, edges.json
                         und den juengsten Laufprotokollen (Toleranz 1e-4)
  -> Next.js-Seite /projects/netzradar bindet nur diese JSON-Dateien beim Build ein:
     Graph-Ausschnitt (Sigma.js), Metriktafel und PR-Kurven (Recharts)
```

Vorberechnung statt Live-Inferenz: GNN-Inferenz wäre für Serverless-Funktionen zu schwer und würde Kosten und Latenz erzeugen. Statische JSON-Dateien sind kostenlos und schnell. Der Browser rechnet kein Modell und kein Layout. Der Vollgraph wird nicht gerendert.

## Screenshots

Die Seite zeigt die 2-Hop-Nachbarschaft der 50 Testknoten mit dem höchsten Isolation-Forest-Score als interaktiven Graphen (derzeit 1.635 Knoten und 1.784 Kanten), kodiert nach Label: auffällig in Bordeaux mit Goldring, unauffällig in Grün, unbekannt hohl in Grau. Je Knoten stehen beide Scores. Daneben die Metriktafel mit PR-Kurven.

![Graph-Ansicht: 2-Hop-Nachbarschaft der 50 auffälligsten Testknoten](docs/screenshots/netzradar-graph.png)

![Metriktafel mit PR-AUC, Precision, Recall und Precision-Recall-Kurven](docs/screenshots/netzradar-metriken.png)

Beide Ansichten zeigen das synthetische Netz.

## Datenschutz

Die Seite nimmt keine Eingaben und keine Dateien entgegen. Es gibt keine Anmeldung, keine Cookies, kein Tracking und keine Werbung. Beim Aufruf lädt der Browser nur eigene statische Dateien vom selben Server: HTML, JavaScript und CSS. Die vorberechneten Ergebnisse aus `data/k3/` werden beim Build in die Seite eingebunden. Zur Laufzeit wird kein Modell ausgeführt und keine Programmierschnittstelle aufgerufen; es werden keine Inhalte Dritter nachgeladen. Die Content-Security-Policy erlaubt Verbindungen nur zum eigenen Origin (`connect-src 'self'`). Ein Playwright-Test prüft, dass während der Nutzung keine Anfrage an einen fremden Origin und keine Nicht-GET-Anfrage entsteht. Die gezeigten Daten sind synthetisch und enthalten keine personenbezogenen Daten.

## Reproduktion

Voraussetzungen: Python 3.12 und [uv](https://docs.astral.sh/uv/). Für die geplanten GNN-Läufe auf Elliptic optional eine GPU; Google Colab reicht.

Linux und macOS mit make:

```bash
git clone https://github.com/mirkan-morgenfels-ai/AI-Project-3.git
cd AI-Project-3/services/k3-train
uv sync
make verify
make data SYNTH=1
make baseline
make export
make verify
```

`make all` führt zusätzlich ruff und pytest aus. `make gnn` folgt in Schritt 4.

Windows (PowerShell) ohne make:

```powershell
git clone https://github.com/mirkan-morgenfels-ai/AI-Project-3.git
cd AI-Project-3\services\k3-train
uv sync
uv run k3-train verify
uv run k3-train data --synth
uv run k3-train baseline
uv run k3-train export
uv run k3-train verify
```

Liegt das Repository in einem synchronisierten Ordner wie OneDrive, die virtuelle Umgebung vor `uv sync` außerhalb anlegen lassen, zum Beispiel mit `$env:UV_PROJECT_ENVIRONMENT = "$env:LOCALAPPDATA\netzradar\venv"`.

Das erste `verify` prüft die eingecheckten Dateien in `data/k3/` und die jüngsten Protokolle in `docs/runs/` gegen eine Neuberechnung im Speicher; es braucht keine aufbereiteten Daten. `data` erzeugt das synthetische Netz und schreibt es aufbereitet nach `services/k3-train/data/processed/`. `baseline` rechnet beide Verfahren und schreibt je ein Protokoll nach `docs/runs/`. `export` schreibt `nodes.json`, `edges.json` und `metrics.json` nach `data/k3/`, von wo die Seite sie statisch lädt. Das zweite `verify` prüft die neu geschriebenen Dateien und meldet „Reproduzierbar“ oder jede Abweichung über 1e-4. Weil `baseline` und `export` das heutige Datum schreiben, ändern sich bei einem Neulauf nur `generatedAt`, `date` und die Namen der Protokolldateien; `git diff` zeigt das.

Elliptic lokal (Rohdaten von Kaggle, Anleitung in `docs/daten.md`): einen Datenordner außerhalb des Repositorys wählen, die Rohdaten nach `<Datenordner>/raw/elliptic/` legen und `uv run k3-train data --dataset elliptic --data-dir <Datenordner>`, danach `uv run k3-train baseline --dataset elliptic --data-dir <Datenordner>` ausführen. Statt `--data-dir` geht auch die Umgebungsvariable `K3_DATA_DIR`, mit make `make data DATASET=elliptic DATA_DIR=<Datenordner>`; Rohdaten an anderer Stelle per `--raw-dir` bzw. `RAW_DIR`. Aufbereitete Dateien und Protokolle landen dann ebenfalls im Datenordner und bleiben lokal. Ohne Datenordner gilt `services/k3-train/data/`; liegt das Repository in OneDrive, würde das mitsynchronisiert.

Laufzeiten: Auf dem synthetischen Netz braucht jeder Befehl auf einer Laptop-CPU wenige Sekunden Rechenzeit; die CLI gibt sie am Ende aus. Elliptic-Volllauf: [Platzhalter: noch nicht gemessen].

## Frontend lokal

Voraussetzungen: Node.js 20.19 oder 22.12 und neuer (`engines` in `package.json`), pnpm 10.

```bash
pnpm install
pnpm dev
```

Danach unter http://localhost:3000/projects/netzradar. Umgebungsvariablen sind nicht nötig; `NEXT_PUBLIC_SITE_URL` setzt optional die Basis-URL für Metadaten.

## Tests

```bash
cd services/k3-train
uv run ruff check
uv run ruff format --check
uv run pytest
uv run k3-train verify
```

```bash
pnpm typecheck
pnpm lint
pnpm test
pnpm build
pnpm test:e2e
```

pytest (117 Tests, Stand 06.10.2026) deckt ab:

- Generator: Determinismus bei gleichem Seed, Knoten je Zeitschritt, Kanten nur innerhalb eines Zeitschritts, je Zeitschritt eine schwache Zusammenhangskomponente, keine Selbstschleifen und Duplikate, Form der Muster, homophile illicit-Nachbarschaften.
- Laden: Elliptic-Format an einer erfundenen Mini-Fixture einschließlich der Trennung lokaler und aggregierter Merkmale, Kantenbereinigung an einem Handbeispiel, verlustfreies Schreiben und Lesen der aufbereiteten Dateien.
- Split: Standard-Splits; überlappende Splits, Zeitschritte auf beiden Seiten und Kanten zwischen Training und Test in beiden Richtungen werden erkannt.
- Graphmaße: Grad, Betweenness und Eigenvektor-Zentralität an einem Pfad von Hand gerechnet, Berechnung je Zeitschritt, Protokoll der Einstellungen mit Stichproben- und Rückfall-Zeitschritten, Label-Homophilie an einem Handbeispiel.
- Baseline: robuster Z-Score mit Ausreißer von Hand gerechnet, MAD = 0 und die Liste der betroffenen Merkmale, Normalisierung nur mit Trainingsdaten, Isolation Forest deterministisch und nur auf Trainingsknoten angepasst.
- Pipeline gegen Lecks: Geänderte Testmerkmale ändern keinen Trainingsscore, vertauschte Labels ändern keinen Score; Golden-Werte des kleinen Testnetzes.
- Metriken: PR-AUC, Precision bei Recall, Recall bei Precision, Accuracy der obersten 2 % mit markierten Knoten und Treffern, Prävalenz, Alles-licit-Accuracy und Erwartungswert zufälliger Rangfolgen jeweils an Handbeispielen, auch mit Gleichständen; `unknown` wird ausgeschlossen; PR-Kurve ohne künstlichen Startpunkt und mit höchstens 101 Punkten.
- Export: Datenvertrag aller drei Dateien, Startknoten, Hops als ungerichtete Distanzen, Koordinaten in [-1, 1], Kürzung, byte-gleiche Ausgabe, Layout unabhängig vom Python-Hash-Seed, deterministische Gesamtpipeline.
- Protokolle, Verifikation von `metrics.json`, `nodes.json`, `edges.json` und Protokollen sowie CLI, darunter die Ablehnung des Elliptic-Exports, der Datenordner außerhalb des Repositorys und der Abbruch außerhalb des Projektordners.

Vitest prüft die Seitenlogik in `apps/web/lib/`, Playwright die Seiten im Browser einschließlich des Wächters gegen fremde Anfragen, Konsolenfehler und CSP-Verstöße sowie der Tastaturbedienung im Detailfeld.

Die CI führt bei jedem Push auf `main` und jedem Pull Request zwei Jobs aus: `web` mit install, typecheck, lint, test, build und E2E sowie `train` mit `uv sync --locked` (uv 0.12.23, `UV_LOCKED=1` für alle `uv run`), `make lint` (ruff check und ruff format --check), `make test`, `make verify` gegen die eingecheckten Dateien und danach als Rauchtest `make data SYNTH=1`, `make baseline`, `make export` und ein zweites `make verify`. Ein GNN-Smoke-Test kommt mit Schritt 4 dazu; das volle Training läuft lokal, sein Ergebnis wird als `metrics.json` committet.

## Projektstruktur

```
services/k3-train/                    Python 3.12, uv-Projekt, Paket k3_train
  pyproject.toml, uv.lock             Abhaengigkeiten, fixiert
  Makefile                            all, data, baseline, export, verify, test, lint (gnn folgt)
  src/k3_train/cli.py                 Kommandozeile k3-train: data, baseline, export, verify
  src/k3_train/synth.py               synthetischer Netzgenerator
  src/k3_train/load.py                Laden, Aufbereiten, Kantenbereinigung
  src/k3_train/datasets.py            Datensatzregister mit Lizenz und Freigabe
  src/k3_train/split.py               zeitlicher Split und Split-Pruefung
  src/k3_train/graph.py               Graphmasse je Zeitschritt
  src/k3_train/features.py            Merkmalssaetze
  src/k3_train/baseline.py            robuste Z-Scores, Isolation Forest
  src/k3_train/metrics.py             PR-AUC, Precision, Recall, PR-Kurve
  src/k3_train/pipeline.py            Baseline-Lauf und Protokolle
  src/k3_train/export.py              JSON-Export mit Ausschnitt und Layout
  src/k3_train/verify.py              Nachrechnen gegen data/k3 und die juengsten Protokolle
  src/k3_train/paths.py               Pfade, Datenordner (--data-dir, K3_DATA_DIR)
  src/k3_train/runs.py, jsonio.py     Protokolle, JSON-Format
  tests/                              pytest, Datenvertrag, erfundene Elliptic-Fixture
  data/raw/, data/processed/          lokal, nicht im Repository
data/k3/                              nodes.json, edges.json, metrics.json
apps/web/app/projects/netzradar/      Projektseite
apps/web/components/netzradar/        NetzRadarExplorer, GraphView, GraphLegend, NodeDetail, NodeSymbol,
                                      TopNodesTable, MetricsTable
apps/web/lib/netzradar/               types.ts, data.ts (Datenvertrag und Pruefung), graph.ts, format.ts,
                                      summary.ts, curves.ts, Unit-Tests
apps/web/e2e/                         Playwright-Tests
packages/ui/                          Basiskomponenten
packages/legal/                       Betreiberangaben, Disclaimer, Datenschutztexte
packages/charts/                      Palette (Export ./theme ohne Recharts), PR-Kurve (Recharts)
docs/runs/                            Protokolle einzelner Laeufe
docs/daten.md                         Bezug, Lizenzen und Ablage der Datensaetze
docs/screenshots/                     Screenshots fuer dieses README
.github/workflows/ci.yml              Jobs web und train
```

## Tech-Stack

Python 3.12, uv, pandas, NumPy, scikit-learn, NetworkX, pytest, ruff, Makefile; für Schritt 4 PyTorch 2.x und PyTorch Geometric 2.7. Frontend: Next.js 15 (App Router), TypeScript (strict), Tailwind CSS 4, Sigma.js, Recharts, Vitest, Playwright, pnpm Workspaces. CI: GitHub Actions. Hosting: Vercel (statisch).

## Grenzen

- Bisher gibt es nur Ergebnisse auf einem synthetischen Netz mit eingebauten Mustern. Sie sagen nichts über reale Transaktionsdaten.
- Die Annahmen des Generators (Musterformen, schwache Verschiebung der meisten lokalen Merkmale, Hubs im gutartigen Hintergrund) bestimmen, was Baseline und GNN finden können. Ein anderer Generator kann zu einer anderen Rangfolge führen.
- Die Nachbarschaft ist im Generator stark homophil: Musterknoten hängen fast nur an anderen Musterknoten, und Kreise und Ketten tragen entlang der Kanten fast denselben Betrag (`hopNoiseSd` 0,02). Von den 847 Kanten zwischen zwei gelabelten Knoten verbinden 161 zwei illicit-, 666 zwei licit-Knoten und nur 20 ein gemischtes Paar; 216 von 292 illicit-Knoten haben einen illicit-Nachbarn, aber nur 19 von 2.587 licit-Knoten (`dataset.homophily` in `metrics.json`). Ein Modell mit Nachbarschaft bekommt dieses Signal teilweise geschenkt. Ein Vorsprung von GCN oder GraphSAGE in Schritt 4 wäre auf diesem Netz deshalb zum Teil eingebaut und nicht auf andere Daten übertragbar.
- Die Labels sind knapp: im synthetischen Netz 76 % `unknown`, darunter bewusst verdeckte Musterknoten, bei Elliptic 77 %. Die Merkmale des Elliptic-Datensatzes sind anonymisiert und nicht dokumentiert.
- Das Testset ist klein (95 illicit-Testknoten). PR-AUC-Unterschiede in der zweiten Nachkommastelle sind entsprechend unsicher: Schon zufällige Rangfolgen streuen bis 0,1420 (95-%-Quantil). Konfidenzintervalle für die Verfahren werden noch nicht berichtet.
- Die Reproduzierbarkeit ist bisher nur unter Windows geprüft. Die CI vergleicht die eingecheckten Dateien unter Linux mit einer Neuberechnung; ihr erster Lauf steht aus.
- Kein Live-Scoring, kein Echtzeitbetrieb; das Projekt ist eine Methodenstudie.
- Nur zwei GNN-Architekturen sind geplant; GAT und temporale Modelle sind Roadmap.

## Roadmap

- v1: Baseline, GCN und GraphSAGE, zeitlicher Split, Export, statische Seite. Stand 06.10.2026: Schritte 1 bis 3 auf dem synthetischen Netz umgesetzt (Generator, zeitlicher Split mit Prüfung, robuste Z-Scores, Isolation Forest mit Graphmaßen, Kennzahlen, Export, Verifikation, Seite `/projects/netzradar` mit Graph-Ausschnitt, Metriktafel und PR-Kurven). Offen: GNN (Schritt 4), Case-Study (Schritt 5), lokaler Elliptic-Lauf, Datensatzentscheidung IBM-AML, Deployment.
- v2: temporale GNN-Variante, GAT im Vergleich, Sampling gegen Klassengewichtung, dokumentierte Hyperparametersuche.
- v3: vorberechnete Erklärungen der markierten Muster in verständlicher Sprache (Sprachmodell im Batch-Modus, keine Laufzeitkosten).

## Disclaimer

NetzRadar ist eine Methodenstudie zur Anomalie-Erkennung in Transaktionsnetzwerken auf synthetischen beziehungsweise öffentlich zugänglichen Forschungsdatensätzen. Die gezeigten Anomalie-Scores sind Ausgaben statistischer Modelle. Sie sind keine Feststellung, dass eine Transaktion oder ein Konto rechtswidrig ist, und es werden keine realen Konten oder Personen geprüft. Die Inhalte stellen keine Anlage-, Rechts- oder Compliance-Beratung dar und ersetzen keine geldwäscherechtliche Prüfung. Alle Angaben ohne Gewähr.

## Zitation und Quellen

- Weber, M., Domeniconi, G., Chen, J., Weidele, D. K. I., Bellei, C., Robinson, T., Leiserson, C. E. (2019). Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics. arXiv:1908.02591.
- Elmougy, Y., Liu, L. (2023). Demystifying Fraudulent Transactions and Illicit Nodes in the Bitcoin Network for Financial Forensics. arXiv:2306.06108.
- Elliptic Bitcoin Dataset, Kaggle `ellipticco/elliptic-data-set`, CC BY-NC-ND 4.0.
- IBM/AML-Data (GitHub) und Kaggle `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`, Daten unter CDLA-Sharing-1.0, Code unter Apache-2.0.
- Hamilton, W. L., Ying, R., Leskovec, J. (2017). Inductive Representation Learning on Large Graphs (GraphSAGE). arXiv:1706.02216.
- Kipf, T. N., Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks (GCN). arXiv:1609.02907.
- Liu, F. T., Ting, K. M., Zhou, Z.-H. (2008). Isolation Forest. ICDM.
- safe-graph/graph-fraud-detection-papers (GitHub), Literaturliste.
- PyTorch Geometric, Dokumentation Version 2.7.

## Lizenz

Code: MIT, siehe [LICENSE](./LICENSE). Das synthetische Netz und die daraus exportierten Dateien in `data/k3/` stehen ebenfalls unter MIT. Die übrigen Datensätze unterliegen ihren eigenen Lizenzen (siehe Abschnitt Datensatz) und sind nicht Teil dieses Repositories.
