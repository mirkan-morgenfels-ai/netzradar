# Projekt: NetzRadar (K3)

Zweck: Reproduzierbare Fallstudie zur Anomalie-Erkennung in Transaktionsnetzwerken. Verglichen werden eine klassische Baseline (robuste Z-Scores und Isolation Forest mit Graphmaßen) und Graph Neural Networks (GCN, GraphSAGE), mit strikt zeitlichem Split und PR-AUC als Hauptmetrik. Training und Auswertung laufen offline in Python. Die Seite `/projects/netzradar` zeigt nur vorberechnete JSON-Ergebnisse. Kein Endnutzer-Werkzeug, kein Live-Scoring, keine echten personenbezogenen Daten. Drittes Portfolio-Projekt nach K1 DepotDoktor und K2 KontoKlar.

## Quellen

- Maßgeblich für Umfang, Methodik, Datensätze, Lizenzen, Zeitplan und Regeln ist das Umsetzungsdokument, Teil 3 „K3: NetzRadar“.
- Das Umsetzungsdokument, `Projektanweisungen_K3_NetzRadar.md` und der Plan für Schritt 4 (`Schritt-4-Plan_K3_NetzRadar.md`) liegen außerhalb des Repos in `AI-Projekte\AI-Projekt-3\` und werden nicht eingecheckt. Der README-Entwurf `README_K3_NetzRadar.md` ist in `README.md` aufgegangen.
- Widerspricht eine Anfrage einem Dokument oder ein Dokument dem Code: hinweisen und vorschlagen, welches Dokument angepasst wird, statt still abzuweichen.
- Bekannte Widersprüche, Stand 07.10.2026:
  - Umsetzungsdokument 3.6 nennt als Abnahme von Schritt 4 „GNN schlägt Baseline (PR-AUC)“, 3.12 „GNN schlägt Baseline messbar“. Das widerspricht Regel 5. Es gilt der Abschnitt „Abnahme Schritt 4“ unten.
  - Umsetzungsdokument 3.8: Das GNN-Training sei zu schwer für die CI, dort laufe nur ein Smoke-Test. Dennis hat am 07.10.2026 entschieden (E7), dass die CI die GNN auf dem synthetischen Netz voll trainiert (`make gnn`) und mit `verify` nachrechnet, weil das Netz mit 12.000 Knoten klein ist. Für Elliptic bleibt es beim lokalen Training. Vorschlag: 3.8 im Umsetzungsdokument entsprechend anpassen.
  - Umsetzungsdokument 0.5 und Projektanweisungen sprechen von einem gemeinsamen Monorepo mit K1 und K2. Tatsächlich ist K3 ein eigenes Repo (`AI-Project-3`), wie K1 und K2 auch.
  - Umsetzungsdokument 3.3.4: „alles licit ergibt ~98 % Accuracy“. Das gilt bezogen auf alle Knoten. Bewertet wird aber nur auf gelabelten Knoten. Bei Elliptic sind davon 42.019 / 46.564 ≈ 90,2 % licit, ein Alles-licit-Modell hätte dort also etwa 90 % Accuracy. Im README ist das korrigiert.
  - Umsetzungsdokument 3.7 exportiert beispielhaft `data/k3/result.json` mit `baseline_pr_auc`. Verbindlich ist der Datenvertrag unten: drei Dateien, Schlüssel in camelCase.
  - Projektanweisungen nennen die Projektstruktur `src/load.py`, `src/synth.py` usw. Tatsächlich liegt der Code im Paket `services/k3-train/src/k3_train/`.
  - Plan Schritt 4, Abschnitt 1 („Nicht Teil von Schritt 4: Seitentexte und Case-Study (Schritt 5)“): Auf Anfrage vom 07.10.2026 enthält die Seite schon in Schritt 4 die Texte zu den GNN (Methodik mit Message Passing, Klassengewichtung, Auswahl am Validierungsteil, aus den Zahlen berechnete Einordnung, Homophilie-Vorbehalt). Für Schritt 5 bleiben Case-Study, Sensitivität gegen Homophilie und Startknoten nach dem GNN-Score (S2). Vorschlag: Abschnitt 1 des Plans und die Schrittbeschreibung im Umsetzungsdokument entsprechend anpassen.
  - Plan Schritt 4, Abschnitt 6.3 erwartete, dass `e2e/netzradar.spec.ts` ohne Änderung grün bleibt. Mit der Kontrolle `mlp` und `scoreGnn` auf der Seite wurde die Spezifikation erweitert (Hinweise „noch nicht gemessen“ nur bei fehlenden Runs, `scoreGnn` mit Verfahren in Tabelle und Detailfeld). Keine Dokumentänderung nötig, nur zur Kenntnis.

## Stack

- Root: pnpm 10.34.5 (`packageManager`), Node `^20.19.0 || >=22.12.0` (`engines`, Untergrenze von vite 8 für Vitest); lokal Node 22.23.2, CI Node 22.
- Web, aufgelöst laut `pnpm-lock.yaml`: Next.js 15.5.27 (App Router), React 19.3.0, TypeScript 5.9.3 (strict, `noUncheckedIndexedAccess`), Tailwind CSS 4.3.3, Recharts 3.10.1 (in `packages/charts`), Vitest 4.1.11, Playwright 1.63.0, ESLint 9.39.5. Für die Graph-Ansicht: Sigma.js 3.0.3, graphology 0.26.0 und @sigma/node-border 3.0.0 für die Knotenringe. axe-core 4.14.0 als devDependency von `apps/web` nur für die E2E-Prüfung der Zugänglichkeit; das Paket stand vorher schon transitiv (über eslint-config-next) im Lock, die Aufnahme hat nichts Neues heruntergeladen.
- Python in `services/k3-train`, aufgelöst laut `uv.lock` (48 Pakete): Python 3.12 (`requires-python >=3.12,<3.13`, lokal 3.12.15), uv 0.12.23 mit Build-Backend `uv_build`, networkx 3.7, numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.18.1, pytest 9.1.1, ruff 0.16.10. Die Untergrenze `networkx>=3.5` ist nötig, weil `spring_layout(method="force")` erst ab 3.5 existiert. System-Python 3.14 ist für PyG 2.7 zu neu, deshalb uv mit Python 3.12.
- GNN als optionales Extra `gnn` (`[project.optional-dependencies]`, Entscheidung E2): `torch>=2.8,<2.9` und `torch-geometric>=2.7,<2.8` (E1), aufgelöst torch 2.8.0 und PyG 2.7.0. torch kommt über `[tool.uv.sources]` aus dem Index `pytorch-cpu` (https://download.pytorch.org/whl/cpu, `explicit = true`): `2.8.0+cpu` für Linux und Windows, `2.8.0` für macOS. PyG führt torch nicht als Abhängigkeit, der Resolver prüft die Verträglichkeit also nicht; deshalb ist torch auf 2.8 gebunden (laut Release-Notes von PyG 2.7.0 die Hauptversion). Das Extra bringt 27 Pakete mit; das Windows-Wheel von torch hat 590,7 MiB. PyG 2.7.0 gibt beim Import eine DeprecationWarning zu `torch_geometric.distributed` aus; pytest filtert genau diese Meldung (`filterwarnings` in `pyproject.toml`). Code ohne torch-Bedarf importiert torch nicht: `data`, `baseline`, `export` und `verify` ohne GNN-Runs laufen ohne das Extra; `cli.py` lädt `gnn_pipeline` erst im Befehl `gnn`, `verify.py` erst beim Nachrechnen von GNN-Runs.
- CI: GitHub Actions `.github/workflows/ci.yml` mit den Jobs `web`, `train` und `train-core` (`uv sync --locked` ohne Extra, `uv run pytest`; zeigt, dass die Kernmodule ohne torch laufen). Actions auf Node-24-Runtime: `actions/checkout@v5`, `pnpm/action-setup@v6`, `actions/setup-node@v5`, `actions/upload-artifact@v7`. `astral-sh/setup-uv@v10.2.0` ist fest eingetragen, weil setup-uv seit v7 keinen wandernden Major-Tag mehr veröffentlicht; die uv-Version ist dort auf 0.12.23 gesetzt. Der Job `train` setzt `UV_LOCKED=1`, damit kein `uv run` die Lockdatei still neu auflöst, und `K3_REQUIRE_GNN=1`, damit fehlendes torch die Tests rot statt übersprungen macht.
- Hosting: Vercel Hobby, statisch. Das Projekt ist noch nicht angelegt.

## Struktur

```
.github/workflows/ci.yml              web: install, typecheck, lint, test, build, E2E; train: uv sync --locked --extra gnn, make lint,
                                      make test, make verify (eingecheckte Dateien), data, baseline, gnn, export;
                                      train-core: uv sync --locked ohne Extra, uv run pytest
apps/web/                             Next.js-App (Paket web)
  app/                                Startseite, layout, error, not-found, robots, impressum, datenschutz, nutzungsbedingungen
  app/projects/netzradar/page.tsx     Projektseite (Server-Komponente, Texte aus metrics.json)
  components/LegalPage.tsx            Rahmen der Rechtsseiten
  components/netzradar/               NetzRadarExplorer (Client, Auswahlzustand), GraphView (Sigma.js, dynamisch geladen),
                                      GraphLegend, NodeDetail (Detailfeld mit Fokusführung und scoreGnn), NodeSymbol,
                                      TopNodesTable (Startknoten, Spalte scoreGnn mit Verfahren), MetricsTable (alle Runs,
                                      Zufallsreferenz), SearchTable (Kandidaten der GNN-Suche am Validierungsteil),
                                      ScrollRegion (waagrecht scrollbarer Bereich mit role="region", tabIndex 0 und Namen)
  lib/site.ts                         Projekte K1 bis K3, Navigation, siteUrl()
  lib/netzradar/types.ts              Datenvertrag als TypeScript-Typen, SCHEMA_VERSION, METHODS, LEARNED_METHODS,
                                      GRAPH_METHODS, Training (typisierte Hyperparameter der gelernten Runs)
  lib/netzradar/data.ts               statischer JSON-Import aus data/k3 mit strenger Prüfung (bricht den Build ab)
  lib/netzradar/graph.ts              NODE_STYLES (Label-Kodierung), Knotengrößen, Kantenstile, Nachbarschaft, Komponenten
  lib/netzradar/format.ts             deutsche Zahlen- und Datumsformate, LABEL_TEXT, METHOD_TEXT, METHOD_SHORT_TEXT,
                                      METHOD_KIND_TEXT, FEATURE_SET_TEXT, joinList
  lib/netzradar/assessment.ts         Sätze der Einordnung aus metrics.json (Zufallsvergleich, Baselines, GNN gegen Baseline,
                                      GCN gegen GraphSAGE, Zerlegung mit MLP, Accuracy, Vorbehalt nur bei gemessenem Vorsprung)
  lib/netzradar/summary.ts            Tabellenzeilen, Zufallsreferenz, Mindestabstand für Vergleiche, erwartetes
                                      scoreGnnMethod, Accuracy-Obergrenze, Klassenabstand nach dem Mitteln,
                                      Hyperparameterliste, Kennzahlen für die Texte
  lib/netzradar/curves.ts             METHOD_CURVE_STYLES (Farbe und Strichmuster je Verfahren), Prävalenzlinie
  lib/netzradar/messagePassing.ts     Rechenbeispiel Kette A – B – C – D (Mittelwert, GCN, GraphSAGE, MLP)
  lib/netzradar/__tests__/            Vitest; fixtures.ts baut Runs und Trainingsangaben für die Tests
  e2e/                                Playwright-Tests (netzradar.spec.ts liest data/k3 direkt)
  vitest.config.mts                   als .mts, damit Vite die Konfiguration als ESM lädt
packages/ui/                          Card, Button, StatTile, cx
packages/legal/                       OPERATOR (lastUpdated = Stand der K3-Rechtstexte), Disclaimer, Datenschutz-Kurztext
packages/charts/                      theme.ts (Palette, Fallback-Farben und -Strichmuster), PrCurveChart (Recharts, eigene Legende);
                                      Export `@portfolio/charts/theme` liefert die Palette ohne Recharts
services/k3-train/                    Python-Paket k3_train, uv-Projekt, nicht im pnpm-Workspace
  Makefile                            all, data, baseline, gnn, export, verify, test, lint; GNN_RUN = uv run --extra gnn
                                      für gnn, verify, test
  src/k3_train/cli.py                 Einstieg k3-train mit data, baseline, gnn, export, verify
  src/k3_train/synth.py               synthetischer Netzgenerator (SynthConfig, Seed 42)
  src/k3_train/load.py                Laden (synthetisch, Elliptic), Kantenbereinigung, data/processed lesen und schreiben
  src/k3_train/datasets.py            Datensatzregister mit Lizenz und Freigabe (publishable)
  src/k3_train/split.py               zeitliche Splits und Split-Prüfung
  src/k3_train/graph.py               Graphmaße je Zeitschritt mit Protokoll der Einstellungen, Label-Homophilie
  src/k3_train/features.py            Merkmalssätze local und local+graph
  src/k3_train/baseline.py            robuste Z-Scores, Isolation Forest
  src/k3_train/metrics.py             PR-AUC, Precision und Recall an Zielwerten, Accuracy-Nebenwert mit markierten Knoten
                                      und Treffern, Zufallsreferenz (exakter Erwartungswert, Permutationsquantil), PR-Kurve
  src/k3_train/pipeline.py            Baseline-Lauf, SCHEMA_VERSION, METHODS (Reihenfolge der Runs), metrics-Payload, Laufprotokolle
  src/k3_train/training.py            ohne torch: training_parts (fit, validation, train, test), loss_targets, positive_weight,
                                      RobustScaler (fit_scaler, transform), Candidate, SEARCH_SPACE, select_candidate
                                      (vergleicht auf 4 Nachkommastellen wie im Export, bei Gleichstand der frühere),
                                      EarlyStopping, check_fit_validation_edges
  src/k3_train/gnn.py                 mit torch und PyG: GnnConfig, set_determinism, undirected_edge_index, GCN, GraphSAGE, MLP,
                                      weighted_loss, train_model, logit_scores, environment
  src/k3_train/gnn_pipeline.py        mit torch: run_gnn (Suche, Auswahl, Endmodell, Seed-Streuung), gnn_hyperparameters
  src/k3_train/gnn_results.py         ohne torch: GnnOutput, save_gnn, load_gnn, score_gnn_method (auf 4 Nachkommastellen,
                                      bei Gleichstand gcn), with_seed_spread
  src/k3_train/export.py              Ausschnitt, Layout, nodes.json (mit scoreGnn), edges.json, metrics.json
  src/k3_train/runs.py                Laufprotokolle für Baseline- und GNN-Runs schreiben
  src/k3_train/verify.py              Neuberechnung (mit GNN, wenn metrics.json GNN-Runs enthält) und Vergleich mit metrics.json,
                                      nodes.json, edges.json und den jüngsten Protokollen; K3_REQUIRE_GNN=1 verlangt GNN-Runs
  src/k3_train/jsonio.py              JSON-Format (4 Nachkommastellen, LF, 2 Leerzeichen)
  src/k3_train/paths.py               Pfade, Datenordner (--data-dir, K3_DATA_DIR), Prüfung des Projektordners
  tests/                              pytest; contract.py prüft den Datenvertrag; test_pipeline.py prüft Lecks und Golden-Werte;
                                      test_training.py (ohne torch), test_gnn.py und test_gnn_pipeline.py (mit torch, sonst
                                      übersprungen; mit K3_REQUIRE_GNN=1 Abbruch); fixtures/elliptic_mini ist erfunden
  data/raw/, data/processed/          Standard-Datenordner, nur lokal, gitignored
data/k3/                              nodes.json, edges.json, metrics.json (Export, eingecheckt)
docs/runs/                            Laufprotokolle <YYYY-MM-DD>_<datensatz>_<verfahren>.json
docs/daten.md                         Bezug, Lizenzen und Ablage der Datensätze
docs/screenshots/                     netzradar-graph.png, netzradar-metriken.png
```

## Feste Regeln

Aus den Projektanweisungen:

1. Rohdaten werden nie ins Repository committet. `services/k3-train/data/raw/` und `services/k3-train/data/processed/` sind gitignored (Root-`.gitignore` und `services/k3-train/.gitignore`). Veröffentlicht werden nur Code, Metriken, Plots und Stichproben synthetischer Daten. Aus Elliptic werden keine Rohdaten, keine bearbeiteten Fassungen und keine Ausschnitte veröffentlicht. Bei widersprüchlichen Lizenzangaben gilt die restriktivere. Technisch: `k3-train export --dataset elliptic` bricht mit Exit 2 ab, Elliptic-Laufprotokolle landen in `<Datenordner>/processed/elliptic/runs/` (Datenordner per `--data-dir` oder `K3_DATA_DIR`, Standard `services/k3-train/data`).
2. Der Split ist zeitlich. Kein zufälliger Split und kein Split, bei dem Knoten derselben Zeitkomponente auf beiden Seiten liegen. Jeder Split bekommt einen Test (`tests/test_split.py`; `check_split` prüft: keine geteilten Zeitschritte, keine Kante zwischen Trainings- und Testknoten, Validierung liegt im Trainingszeitraum).
3. Hauptmetrik ist PR-AUC (average precision), dazu Precision und Recall. Accuracy wird nie als Hauptergebnis berichtet, nur als Nebenwert.
4. Jeder Lauf schreibt Datensatz, Split, Seed, Hyperparameter und Datum nach `docs/runs/` und über den Export nach `data/k3/metrics.json`. Ergebnisse ohne diese Angaben gelten als nicht reproduzierbar und werden nicht ins README übernommen.
5. Ergebnisse werden so berichtet, wie sie sind. Schlägt das GNN die Baseline nicht, wird das dokumentiert und erklärt, nicht kaschiert.
6. Kein Modell im Live-Pfad, keine API-Aufrufe zur Laufzeit. Die Seite liest nur die statisch exportierten JSON-Dateien aus `data/k3/`. Erklärungen per Sprachmodell (v3) werden im Batch vorberechnet.
7. Zahlen zu Datensätzen (Größe, Labelanteile, Merkmalszahl) nur aus dem Umsetzungsdokument, aus `data/k3/metrics.json` oder aus einer im Repository dokumentierten Quelle. Unklarheiten offen benennen, etwa 165, 166 oder 167 Merkmale bei Elliptic. Nichts erfinden, nichts schönen.
8. Englisch in Code-Bezeichnern, Dateinamen und JSON-Schlüsseln; Deutsch in allen Seitentexten, Ansprache „Sie“. Keine Kommentarzeilen im Code: keine `//` oder `/* */` in TS, TSX und JS, keine `#`-Kommentare und keine Docstrings in Python, keine Kommentare in YAML, CSS, Makefile und TOML. Erklärungen gehören ins README oder nach `docs/`.

Zusätzlich verbindlich:

- Jede Kennzahl bekommt einen Unit-Test mit von Hand gerechnetem Erwartungswert (`tests/test_metrics.py`, `tests/test_baseline.py`, `tests/test_graph.py`, auf der Webseite in `lib/netzradar/__tests__/`).
- Kein Blau, siehe Design.

## Arbeitsweise

- Jede Aufgabe einem Schritt zuordnen und den Schritt am Anfang nennen. Stand 07.10.2026:
  1. Daten laden, aufbereiten, zeitlicher Split: auf dem synthetischen Netz umgesetzt. Der Elliptic-Lader ist nur an einer erfundenen Fixture getestet. Der IBM-AML-Lader fehlt, er hängt an der Datensatzentscheidung.
  2. Baseline (robuste Z-Scores, Isolation Forest, Graphmaße): umgesetzt, Ergebnisse in `data/k3/metrics.json` und `docs/runs/`.
  3. Visualisierung und Export: umgesetzt. Export (`nodes.json`, `edges.json`, `metrics.json`) und Seite `/projects/netzradar` mit Graph-Ausschnitt, Metriktafel und PR-Kurven; Deployment offen.
  4. GNN (GCN, GraphSAGE in PyTorch Geometric): auf `feat/k3-gnn` umgesetzt. Python: Extra `gnn`, `training.py`, `gnn.py`, `gnn_pipeline.py`, `gnn_results.py`, Befehl `gnn`, Kontrollvariante MLP, Datenvertrag Version 3, `verify` mit GNN, CI mit `make gnn`; Ergebnisse in `data/k3/metrics.json` und `docs/runs/`. Seite: liest Vertrag 3 mit strengem Parser, zeigt alle fünf Runs mit Zufallsreferenz und PR-Kurven, `scoreGnn` mit Verfahren in Detailfeld und Startknoten-Tabelle, die Suche am Validierungsteil, die Methodik der GNN und eine aus `metrics.json` berechnete Einordnung mit Homophilie-Vorbehalt; Screenshots erneuert. Offen: erster CI-Lauf mit GNN (Abgleich Windows gegen Linux, Laufzeit), Commit und PR. Plan in `AI-Projekte\AI-Projekt-3\Schritt-4-Plan_K3_NetzRadar.md`, außerhalb des Repos; Entscheidungen E1 bis E11 am 07.10.2026 nach den Empfehlungen des Plans getroffen.
  5. Case-Study und Seite: offen.
- Bei neuen Aufgaben zuerst ein kurzer Plan mit Dateien, Funktionen, Tests, Abnahmekriterium, geschätzten Stunden und Laufzeit. Kleine Änderungen direkt umsetzen.
- Mathematik erklären, wenn sie eine Entscheidung trägt: Formeln und Mini-Beispiele sind erwünscht, Floskeln nicht.
- Code-Reviews in dieser Reihenfolge: Datenlecks (Split, Merkmale und Normalisierung nur aus Trainingsdaten), Korrektheit der Metriken, Reproduzierbarkeit (Seeds, Konfiguration), Lizenz (was landet im Repository), Struktur und Stil.
- Am Ende jeder Antwort drei Zeilen: was geliefert wurde; was Dennis prüfen muss (Split, Metriken, Reproduzierbarkeit mit gleichem Seed); was Dennis selbst erledigen muss (Kaggle-Download, Lizenzentscheidung, Colab-Lauf, Anthropic-Key nur für v3).
- Offene Entscheidungen mit Optionen und Empfehlung vorlegen und auf Dennis warten.
- Branches `feat/<thema>` und `fix/<thema>`. `main` ist immer deploybar. Ein PR pro abgeschlossenem Schritt, CI muss grün sein, Dennis merged selbst. Commit und Push nur auf ausdrücklichen Wunsch.
- Commit-Stil wie in K2: Conventional Commits auf Deutsch, z. B. `feat(k3): …`, `fix(k3): …`, `test(k3): …`, `docs(readme): …`, `ci: …`, `chore: …`.

## Befehle

PowerShell unter Windows: Vorspann in jedem Aufruf, weil Node, pnpm und uv nicht im PATH liegen:

```powershell
$env:PATH = "$env:LOCALAPPDATA\Programs\nodejs;$env:LOCALAPPDATA\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe;$env:PATH"
$env:UV_PROJECT_ENVIRONMENT = "$env:LOCALAPPDATA\netzradar\venv"
```

- Node 22.23.2 und pnpm 10.34.5 liegen portabel unter `%LOCALAPPDATA%\Programs\nodejs`. Der pnpm-Shim funktioniert nur in PowerShell, nicht in Git Bash.
- uv 0.12.23 kommt aus WinGet. Python 3.12 ist über uv installiert (`uv python find 3.12`).
- Die venv liegt bewusst außerhalb von OneDrive (`UV_PROJECT_ENVIRONMENT`), weil das Repo in OneDrive liegt. Ohne die Variable legt uv `.venv` im synchronisierten Ordner an.

Web, im Repo-Root:

- `pnpm install` (CI: `pnpm install --frozen-lockfile`)
- `pnpm dev` startet die Entwicklung unter http://localhost:3000/projects/netzradar
- `pnpm typecheck`, `pnpm lint`, `pnpm test`, `pnpm build`; `pnpm ci` führt alle vier nacheinander aus
- `pnpm test:e2e` startet Playwright; ohne `CI` gegen `next dev`, mit `CI=true` gegen `next start` (vorher `pnpm build`). Lokal ist Chromium-Revision 1234 installiert, Playwright 1.63.0 erwartet 1243. Deshalb lokal `$env:PLAYWRIGHT_CHROMIUM_PATH = "$env:LOCALAPPDATA\ms-playwright\chromium_headless_shell-1234\chrome-headless-shell-win64\chrome-headless-shell.exe"` setzen; die volle `chrome.exe` 1234 startet hier mit „spawn UNKNOWN“. Die CI installiert die passende Revision selbst.

Python unter Windows ohne make, in `services/k3-train`:

- `uv sync --extra gnn` (CI: `uv sync --locked --extra gnn` mit `UV_LOCKED=1`; lokal prüft `uv lock --check`, ob `uv.lock` zu `pyproject.toml` passt). `uv sync` ohne `--extra gnn` synchronisiert exakt und entfernt torch wieder; `uv run` ohne `--extra gnn` lässt ein vorhandenes torch stehen, installiert ein fehlendes aber nicht.
- `uv run ruff check` und `uv run ruff format --check`
- `uv run --extra gnn pytest` (ohne Extra laufen die GNN-Tests nicht, sie werden mit Grund übersprungen)
- `uv run k3-train data --synth` erzeugt das synthetische Netz und schreibt `data/processed/synthetic/`
- `uv run k3-train baseline` rechnet beide Baselines und schreibt `docs/runs/<heute>_synthetic_{zscore,iforest}.json`
- `uv run --extra gnn k3-train gnn` trainiert GCN, GraphSAGE und die MLP-Kontrolle auf denselben aufbereiteten Daten (braucht `baseline` vorher, wegen der Graphmaße), schreibt `<Datenordner>/processed/<datensatz>/gnn.json` und `gnn_scores.csv` und `docs/runs/<heute>_synthetic_{gcn,graphsage,mlp}.json`; `--max-epochs` nur zum Testen
- `uv run k3-train export` schreibt `data/k3/nodes.json`, `edges.json` und `metrics.json`, mit GNN-Runs und `scoreGnn`, wenn `gnn.json` vorliegt (sonst Hinweis und `scoreGnn` null); prüft, dass Baseline und GNN zu denselben Daten gehören
- `uv run --extra gnn k3-train verify` rechnet das synthetische Netz im Speicher neu, die GNN nur, wenn `metrics.json` GNN-Runs enthält, und vergleicht mit `data/k3/metrics.json`, `nodes.json`, `edges.json` und dem jeweils jüngsten Protokoll `docs/runs/*_synthetic_<verfahren>.json` (Toleranz 1e-4 plus 1e-9 Gleitkomma-Spielraum, damit ein Rundungsschritt der 4. Nachkommastelle wie 0,1234 gegen 0,1235 durchgeht; `generatedAt`, `date` und `environment` ausgenommen). Braucht keine aufbereiteten Daten und schreibt nichts. Ohne torch und mit GNN-Runs: Exit 1 mit Hinweis auf `uv sync --extra gnn`. Mit `K3_REQUIRE_GNN=1` (CI-Job `train`) Exit 1, wenn `metrics.json` keine GNN-Runs enthält.
- Datenordner für `raw/` und `processed/`: `--data-dir <Ordner>` bei `data`, `baseline`, `gnn` und `export`, sonst `K3_DATA_DIR`, sonst `services/k3-train/data`. Ohne Option und Variable prüft die CLI, dass sie im Projektordner läuft (`pnpm-workspace.yaml`, `services/k3-train/pyproject.toml`), und endet sonst mit Exit 2; `export`, `verify` sowie `baseline` und `gnn` für freigegebene Datensätze prüfen das immer.
- Elliptic nur lokal, mit Datenordner außerhalb von OneDrive: `$env:K3_DATA_DIR = "$env:LOCALAPPDATA\netzradar\data"`, Rohdaten nach `$env:K3_DATA_DIR\raw\elliptic`, dann `uv run k3-train data --dataset elliptic` und `uv run k3-train baseline --dataset elliptic`. Der Lader erwartet 165 Merkmalsspalten (93 lokal `f_*`, 72 aggregiert `a_*`). Einzelheiten in `docs/daten.md`.

Linux und CI mit make, in `services/k3-train`:

- `make verify`, `make data SYNTH=1`, `make baseline`, `make gnn`, `make export`, `make verify`
- `make test`, `make lint` (ruff check und ruff format --check)
- `make all` führt lint, test, verify (eingecheckte Dateien), data --synth, baseline, gnn, export und verify aus
- `make gnn`, `make verify` und `make test` nutzen `uv run --extra gnn` (Variable `GNN_RUN`); `data`, `baseline`, `export` und `lint` bleiben bei `uv run`.
- `make data` ohne `SYNTH=1` und ohne `DATASET` endet mit einer Erklärung und Exit 2. `DATA_DIR=<Ordner>` reicht `--data-dir` an `data`, `baseline`, `gnn` und `export` durch, `RAW_DIR=<Ordner>` reicht `--raw-dir` an `data` durch.

Hinweise:

- `baseline`, `gnn` und `export` schreiben Dateien mit dem heutigen Datum in UTC: `docs/runs/<YYYY-MM-DD>_…` und `generatedAt` in `metrics.json`. Ein Neulauf an einem anderen Tag erzeugt neue Protokolldateien neben den alten. Vor einem Commit `git diff` prüfen: Außer `generatedAt`, `date` und den Dateinamen darf sich nichts ändern, sonst ist die Reproduzierbarkeit gebrochen.
- `export` setzt voraus, dass `baseline` auf denselben aufbereiteten Daten lief, sonst Exit 1; liegen GNN-Ergebnisse vor, müssen `dataset`, `split`, `evaluation` und `seed` mit der Baseline übereinstimmen.
- Laufzeiten auf diesem Rechner, gemessen am 06.10.2026 (UTC): Start und Importe je Aufruf rund 8 bis 13 s (der allererste Import von torch nach der Installation 35 s); `uv run --extra gnn pytest` 59 s (166 Tests), davon rund 22 s für den Hash-Seed-Regressionstest, am 07.10.2026 44 bis 49 s laut pytest (171 Tests); ohne Extra 20 s (139 Tests, `test_gnn.py` und `test_gnn_pipeline.py` mit zusammen 27 Tests übersprungen), am 07.10.2026 in einer eigenen venv ohne torch 27 s laut pytest (143 Tests, die beiden Module mit 28 Tests übersprungen); `uv sync --extra gnn` aus dem uv-Cache 11 s, beim ersten Mal 527 s (Download 8,5 min); `data` 0,7 s, `baseline` 4,1 s, `export` 1,9 s, `gnn` 334,5 s, `verify` mit GNN 302 bis 357 s (am 07.10.2026 226 und 197 s). Eine GCN-Epoche auf dem Standardnetz kostet in float64 mit einem Thread rund 55 ms (gemessen über 20 Epochen ohne Validierung), GraphSAGE rund 60 ms, das MLP rund 20 ms.

- Web, gemessen am 07.10.2026: `pnpm test` rund 1,5 s (167 Vitest-Tests in 8 Dateien), `pnpm test:e2e` mit `CI=true` gegen `next start`: Playwright meldet in zwei Läufen 8,9 s und 14,3 s für 13 Tests, davon 3 axe-Läufe. Die Screenshots in `docs/screenshots/` entstehen mit einem Playwright-Skript außerhalb des Repos gegen `next start` (1280 × 900, Graph ohne Auswahl bis zur sechsten Zeile der Startknoten-Tabelle, Metriken bis zur PR-Kurve).

## Git und Deployment

- Remote: https://github.com/mirkan-morgenfels-ai/AI-Project-3 (derzeit privat), Standardzweig `main`.
- Repo-lokale Git-Identität ist Pflicht: `user.email` = `324466065+mirkan-morgenfels-ai@users.noreply.github.com`, `user.name` derzeit `mirkan-morgenfels-ai`. Ohne die noreply-Adresse blockiert Vercel das Deployment („commit email could not be matched“). Global ist eine andere Adresse eingetragen.
- Das Home-Verzeichnis `C:\Users\MirkanDeniz` ist selbst ein Git-Repo (Branch `master`). Git-Befehle immer im Repo-Root `netzradar` ausführen.
- Push und PR über die GitHub CLI als `mirkan-morgenfels-ai`. Repo-lokal ist `credential.https://github.com.helper` gesetzt: erst leer, dann `!<gh.exe> auth git-credential`. Die globale Git-Konfiguration bleibt unverändert, andere Projekte laufen weiter über den Windows Credential Manager. gh liegt unter `%LOCALAPPDATA%\Microsoft\WinGet\Packages\GitHub.cli_Microsoft.Winget.Source_8wekyb3d8bbwe\bin\gh.exe`, nicht im PATH. PRs mit `gh pr create --base main`.
- Vercel: Projekt noch anzulegen, Root Directory `apps/web`, Framework Next.js. Danach deployt Vercel jeden Push auf `main`. Umgebungsvariablen sind nicht nötig; `NEXT_PUBLIC_SITE_URL` ist optional für `metadataBase` (leer bedeutet http://localhost:3000). Python läuft nicht auf Vercel.
- `.gitattributes` erzwingt LF (`* text=auto eol=lf`, `*.pdf` und `*.png` binär), weil `core.autocrlf=true` systemweit gesetzt ist.
- `next-env.d.ts` ist gitignored: Die Datei enthält Kommentare und verweist auf `.next/types`, ohne `.next` bricht `tsc` ab.

## Datenvertrag data/k3

Verbindlich für den Python-Export und die Seite. TypeScript-Typen in `apps/web/lib/netzradar/types.ts`, Python-Prüfung in `services/k3-train/tests/contract.py`. JSON in UTF-8, 2 Leerzeichen Einrückung, LF, Schlüssel in camelCase und fester Reihenfolge, Floats auf 4 Nachkommastellen gerundet, `schemaVersion` 3 in jeder Datei (seit 07.10.2026; Version 2 hatte nur `zscore` und `iforest`, `scoreGnn` immer null und kein `scoreGnnMethod`; Version 1 hatte keine Homophilie, keine Zufallsreferenz, keine Trefferzahlen und eine PR-Kurve mit künstlichem Startpunkt). Stand 07.10.2026: Python-Export und Seite auf Version 3. `data.ts` prüft die festen Werte wie `tests/contract.py`: `layers` 2, `hidden` 64, `activation` `relu`, `dropout` 0,5, `optimizer` `adam`, `learningRate` 0,01, `weightDecay` 0,0005, `loss` `weightedCrossEntropy`, `score` `logitIllicit - logitLicit`, `dtype` `float64` oder `float32`, Skalierung `median`, `mad`, 1,4826 und `fallback` genau [`std`, `one`], `scoreZscore` ≥ 0; Kandidaten der Suche eindeutig nach Merkmalssatz und Gewichtsregel, bei Gleichstand der frühere gewählt (beides prüft auch `contract.py`). Nur `contract.py` prüft die Rundung auf 4 Nachkommastellen. Darüber hinaus prüft `data.ts`: Runs in der Reihenfolge von `METHODS`; bei `gcn`, `graphsage` und `mlp` alle Hyperparameter-Schlüssel ohne `seedSpread`, `architecture` passend zum Verfahren, `edges` `none` genau beim MLP, genau ein gewählter Kandidat, kein Kandidat mit höherer Validierungs-PR-AUC als der gewählte, `stoppedEpoch` = min(`bestEpoch` + `patience`, `maxEpochs`), `selectedEpoch`, `featureSet`, Gewichtsregel und Validierungs-PR-AUC passend zum gewählten Kandidaten, festes Gewicht gleich dem des Kandidaten, `scaling.zeroMadFeatures` und `unitScaleFeatures` nur aus `features`, `finalFitSteps` = `split.train`, `validationSteps` = `split.validation`, `selectionSteps` direkt davor, Merkmalszahl passend zu `featureSet` (`local` gleich `dataset.features`, `local+graph` größer); `scoreGnnMethod` gleich dem Graph-Verfahren mit der höheren Validierungs-PR-AUC (bei Gleichstand `gcn`, ohne Graph-Runs `null`).

- `metrics.json`: `generatedAt` (ISO-8601 UTC); `dataset` mit `name`, `displayName`, `license`, `source`, `generator` (alle Generator-Parameter inklusive `seed`, oder `null`), `nodes`, `edges`, `timeSteps`, `features` (Anzahl lokaler Merkmale), `labelCounts` (`illicit`, `licit`, `unknown`) und `homophily` (`labelledEdges`, `sameLabelShare`, `illicitIllicitEdges`, `licitLicitEdges`, `illicitLicitEdges` über Kanten zwischen zwei gelabelten Knoten; `illicitWithIllicitNeighbour`, `licitWithIllicitNeighbour` als Knotenzahlen, Richtung ignoriert); `split` mit `kind: "temporal"`, `train`, `validation`, `test` (je `from`, `to`) und `crossSplitEdges` (muss 0 sein); `evaluation` mit `positiveLabel: "illicit"`, `excludedLabel: "unknown"`, `testPositives`, `testNegatives`, `prevalence` (= PR-AUC eines konstanten Scores), `allLicitAccuracy`, `randomPrAucExpected` (exakter Erwartungswert der Average Precision einer zufälligen Rangfolge), `randomPrAucQ95` (95-%-Quantil über `randomPermutations` zufällige Rangfolgen mit dem Seed des Laufs) und `randomPermutations`; `seed`; `runs[]` mit `method` (`zscore`, `iforest`, `gcn`, `graphsage`, `mlp`; jedes Verfahren höchstens einmal, Reihenfolge wie in dieser Liste), `displayName`, `featureSet` (`local`, `local+graph`), `seed`, `date`, `hyperparameters`, `prAuc`, `precisionAtRecall50` (maximale Precision über alle Schwellen mit Recall ≥ 0,5), `recallAtPrecision50` (maximaler Recall über alle Schwellen mit Precision ≥ 0,5, sonst 0), `accuracy` (Nebenwert), `accuracyThreshold` (deutscher Text), `accuracyFlagged` (markierte Knoten an der Schwelle, Gleichstände eingeschlossen), `accuracyTruePositives` (davon illicit) und `prCurve` (1 bis 101 Punkte `{recall, precision}`, nach `recall` aufsteigend, beginnend bei der höchsten Schwelle ohne den künstlichen Punkt Recall 0 / Precision 1). `hyperparameters` von `zscore` enthält `zeroMadFeatures`, von `iforest` den Block `graphMeasures` (Grenzen der exakten Betweenness, Stichprobe, Seed, Eigenvektor-Einstellungen, `sampledSteps`, `eigenvectorFallbackSteps`). `hyperparameters` von `gcn`, `graphsage` und `mlp` hat die Schlüssel in dieser Reihenfolge: `architecture` (`GCNConv`, `SAGEConv(aggr=mean)`, `Linear`), `layers` (2), `hidden` (64), `activation` (`relu`), `dropout` (0,5), `optimizer` (`adam`), `learningRate` (0,01), `weightDecay` (0,0005), `loss` (`weightedCrossEntropy`), `score` (`logitIllicit - logitLicit`), `positiveWeight` (Gewicht des Endmodells), `positiveWeightRule` (`trainRatio` oder `fixed`), `edges` (`undirected`, beim MLP `none`), `dtype` (`float64`), `scaling` (`center` `median`, `scale` `mad`, `madScale` 1,4826, `fallback` [`std`, `one`], `clip` 10, `fitOn` `train`, `zeroMadFeatures`, `unitScaleFeatures`), `features` (Namen in Spaltenreihenfolge), `maxEpochs` (300), `patience` (50), `selectedEpoch`, `selectionMetric` (`validationPrAuc`), `validationPrAuc`, `selectionSteps` (`from`, `to`; synthetisch 1 bis 17), `validationSteps` (18 bis 21), `finalFitSteps` (1 bis 21), `search` (je Kandidat `featureSet`, `positiveWeightRule`, `positiveWeight`, `validationPrAuc`, `bestEpoch`, `stoppedEpoch`, `selected`; genau einer `selected`) und `environment` (`torch`, `torchGeometric`, `python`, `platform`, `threads`, `deterministicAlgorithms`). `featureSet` des Runs ist der gewählte Merkmalssatz.
- `nodes.json`: `selection` (`scoreField: "scoreIforest"`, `pool: "test"`, `seeds: 50`, `hops: 2`, `maxNodes: 2000`, `truncated`), `scoreGnnMethod` (`"gcn"`, `"graphsage"` oder `null`; das GNN-Verfahren mit der höheren Validierungs-PR-AUC, bei Gleichstand `gcn`, nie das MLP, nie nach dem Test) und `nodes[]` mit `id`, `x` und `y` in [-1, 1], `label`, `timeStep`, `scoreZscore`, `scoreIforest`, `scoreGnn` (Zahl, wenn `scoreGnnMethod` gesetzt ist, sonst `null`), `seedRank` (1 bis 50 für Startknoten, sonst `null`), `hop` (0 bis 2), `inDegree`, `outDegree`. Höherer Score bedeutet auffälliger. Startknoten sind die 50 Testknoten aller Labels mit dem höchsten `scoreIforest` (Option S1, Entscheidung E5); Nachbarschaft ungerichtet bis 2 Hops; Layout beim Export vorberechnet (Seed 42). Über `maxNodes` wird nach `hop`, dann nach Score gekürzt und `truncated` gesetzt.
- `edges.json`: `edges[]` mit `source` und `target`; induzierter gerichteter Teilgraph auf den exportierten Knoten, ohne Selbstschleifen und Duplikate, sortiert.
- `docs/runs/<YYYY-MM-DD>_<dataset>_<method>.json`: vollständiges, eigenständiges Protokoll je Lauf mit denselben Schlüsseln. Nur in den Protokollen der GNN-Runs steht in `hyperparameters` zusätzlich `seedSpread` vor `environment` (`seeds` 42 bis 46, `reportedSeed`, `prAuc` je Seed, `mean`, `min`, `max`; Entscheidung E8). `verify` vergleicht alles außer `generatedAt`, `date` und `environment`.
- Score-Semantik: `scoreZscore` = maxⱼ |zᵢⱼ| über die lokalen Merkmale mit zᵢⱼ = (xᵢⱼ − medianⱼ) / (1,4826 · MADⱼ), Median und MAD nur aus Trainingsknoten, Merkmale mit MAD = 0 tragen 0 bei. `scoreIforest` = −`score_samples`. `scoreGnn` = Logit illicit − Logit licit des Endmodells (keine Wahrscheinlichkeit, nach oben und unten unbeschränkt).
- Eine Änderung am Vertrag betrifft immer gleichzeitig `jsonio.py`, `pipeline.py`, `metrics.py`, `export.py`, `tests/contract.py`, `apps/web/lib/netzradar/types.ts` (`SCHEMA_VERSION`), `apps/web/lib/netzradar/data.ts` (strenger Parser mit festen Schlüsseln), `apps/web/lib/netzradar/__tests__/data.test.ts`, gegebenenfalls `apps/web/e2e/netzradar.spec.ts` (liest `metrics.json` und `nodes.json` direkt) und diesen Abschnitt; dabei `schemaVersion` erhöhen und `data`, `baseline`, `gnn`, `export` neu laufen lassen. Kommt ein Verfahren dazu, außerdem `METHODS` in `pipeline.py`, `contract.py` und `types.ts`, `METHOD_TEXT` in `apps/web/lib/netzradar/format.ts` und `METHOD_CURVE_STYLES` in `apps/web/lib/netzradar/curves.ts`.

## Betrieb lokal

- `next dev` und `next build` nie gleichzeitig in `apps/web` laufen lassen. Der Build überschreibt `.next`, der Dev-Server antwortet danach mit 500.
- Die Seite braucht keine Umgebungsvariablen und keinen Python-Prozess; sie liest nur `data/k3/`.
- Nach Änderungen an der Python-Pipeline in dieser Reihenfolge: `verify` (zeigt, ob sich Kennzahlen ändern), `data`, `baseline`, `export`, `verify`, danach die Seite prüfen.
- `services/k3-train/data/processed/` ist ein lokaler Zwischenstand und lässt sich jederzeit mit `data --synth` neu erzeugen.
- Bricht `pnpm build` mit „TypeError: Cannot read properties of undefined (reading 'length') at WasmHash._updateWithBuffer“ ab, ist `apps/web/.next` unvollständig oder veraltet. Abhilfe: `apps/web/.next` löschen und neu bauen.
- Lokale E2E-Läufe brauchen `PLAYWRIGHT_CHROMIUM_PATH` auf die Headless-Shell 1234 (siehe Befehle), weil Playwright 1.63.0 die Revision 1243 erwartet; alternativ `pnpm --filter web exec playwright install chromium`. Nach dem Lauf Port 3000 freigeben.

## Sicherheit und Zugänglichkeit

- Sicherheits-Header in `apps/web/next.config.ts`: CSP mit `default-src 'self'` und `connect-src 'self'`, `'unsafe-eval'` nur im Dev-Modus; HSTS zwei Jahre, `X-Frame-Options: DENY`, `nosniff`, Referrer-Policy, Permissions-Policy und COOP `same-origin`.
- Keine externen Schriften, Skripte, Fetches, Cookies oder Tracker. Eine neue fremde Quelle verstieße gegen Regel 6 und gegen die Datenschutzerklärung (`apps/web/app/datenschutz/page.tsx`). Änderungen an dem, was die Seite lädt oder überträgt, ziehen Änderungen an der Datenschutzerklärung und am README-Abschnitt „Datenschutz“ nach sich.
- E2E-Wächter: Während der Nutzung entsteht keine Anfrage an einen fremden Origin und keine Nicht-GET-Anfrage; außerdem keine Konsolenfehler, keine `pageerror` und keine CSP-Verstöße (`securitypolicyviolation`).
- Skip-Link auf `#main`, globaler Fokusring in `gold-deep`, `prefers-reduced-motion` wird respektiert, `lang="de"`.
- Gold für Text nur als `text-gold-deep` (#7d5f17, AA-Kontrast); `gold` (#b8912f) nur für Flächen, Rahmen und Ringe.
- Labels nie nur über Farbe kodieren, siehe Design. Linien in Diagrammen zusätzlich über Strichmuster unterscheiden: Farbe und Strichmuster je Verfahren in `METHOD_CURVE_STYLES` (`apps/web/lib/netzradar/curves.ts`: Z-Scores gold durchgezogen, Isolation Forest wine „8 3“, GCN moss „2 3“, GraphSAGE ink „8 3 2 3“, MLP stone „12 4“, Prävalenz stone „4 4“ dünner); `SERIES_COLORS` und `SERIES_DASHES` in `packages/charts/src/theme.ts` sind nur Rückfallwerte. Ein Test prüft, dass alle Kurvenfarben aus der Palette stammen und keine bläulich ist.
- PR-Diagramm: eigene Legende in `PrCurveChart` mit Text in ink und `aria-hidden`-Symbolen, Tooltip-Text in ink, `accessibilityLayer={false}` (kein namenloser Tab-Stopp `role="application"`; die Werte stehen in der Metriktafel). Recharts färbt Legenden- und Tooltip-Text sonst in der Serienfarbe, bei Gold also mit zu wenig Kontrast.
- Detailfeld: Nach Auswahl eines Nachbarn oder „Auswahl aufheben“ geht der Fokus auf die Überschrift `#node-detail-title` (`tabIndex={-1}`), damit er nicht auf `<body>` fällt.
- Waagrecht scrollbare Bereiche (Tabellen der Labels, Metriken, Suche und der Kette A – D sowie alle Formeln) liegen in `ScrollRegion` mit `role="region"`, `tabIndex={0}` und eindeutigem Namen (Tabellenüberschrift oder „Formel: …“), damit sie per Tastatur scrollbar sind; die Startknoten-Tabelle hat dasselbe Muster. Die E2E lässt axe-core (WCAG 2.x A und AA, Best Practices) bei 390, 768 und 1280 px laufen und erwartet 0 Verstöße. Ohne `tabIndex` melden die Läufe bei 390 und 768 px `scrollable-region-focusable` (geprüft am 07.10.2026).
- Rechtsseiten über `@portfolio/legal` und `components/LegalPage.tsx`; Disclaimer auf der Projektseite.

## Design

- Kein Blau, nirgends: keine Tailwind-Farbklassen aus dem Blau-, Himmelblau-, Violettblau- oder Türkisbereich und keine bläulichen Hex-Werte, auch nicht in Diagrammen, SVGs oder der Graph-Ansicht.
- Palette (Tokens in `apps/web/app/globals.css`, Diagrammfarben in `packages/charts/src/theme.ts`): ink #111111, paper #fbfaf6, surface #ffffff, gold #b8912f, gold-deep #7d5f17 (Gold für Text), gold-soft #f3e9c9, moss #2f6b3a, moss-soft #dfeadf, wine #7a1f2b, wine-soft #f1dcdf, stone #6b6b66, line #e3e0d6. Dieselben Werte wie K2; K1 weicht in einzelnen Hex-Werten leicht ab.
- Label-Kodierung in Graph, Legende und Tabellen, ohne Blau und auch bei Rot-Grün-Schwäche unterscheidbar (Werte in `NODE_STYLES`, `apps/web/lib/netzradar/graph.ts`):
  - auffällig (`illicit`): gefüllt in wine mit Goldring
  - unauffällig (`licit`): gefüllt in moss
  - unbekannt (`unknown`): hohl (Füllung paper), nur Kontur in stone
- Zahlen mit deutschem Dezimalkomma und `tabular-nums`. Lieber mehr Weißraum als gedrängt.

## Definition of Done pro Schritt

Aus den Projektanweisungen: pytest grün, Split-Test grün, Metriken reproduzierbar, Export in `data/k3/` vorhanden, Seite rendert Graph-Ausschnitt und Metriktafel, Rohdaten nicht im Diff.

Konkret prüfbar:

- `uv run ruff check`, `uv run ruff format --check` und `uv run --extra gnn pytest` grün, ohne übersprungene GNN-Tests
- `uv run --extra gnn k3-train verify` mit Exit 0
- `pnpm typecheck`, `pnpm lint`, `pnpm test`, `pnpm build` und `pnpm test:e2e` grün, E2E-Wächter ohne fremde Anfragen
- jede neue Kennzahl mit Handtest
- `git status` zeigt nichts unter `services/k3-train/data/`
- Zahlen im README stammen aus `data/k3/metrics.json`
- CI grün

## Abnahme Schritt 4 (geändert)

Das Umsetzungsdokument (3.6) nennt als Abnahme „GNN schlägt Baseline (PR-AUC)“. Das widerspricht Regel 5. Es gilt: **Vergleich GNN gegen Baseline gemessen und berichtet.**

- Gleicher Datensatz, gleicher zeitlicher Split, gleiche gelabelte Testknoten, gleiche Metriken, Seed 42.
- Hyperparameter nur am Validierungsteil gewählt (synthetisch Zeitschritte 18 bis 21, Elliptic 30 bis 34), nie am Testteil. Normalisierung und Klassengewichte nur aus Trainingsdaten.
- Ergebnis als Runs `gcn` und `graphsage` in `metrics.json` und `docs/runs/`, `scoreGnn` in `nodes.json` gefüllt.
- Im README ehrlich eingeordnet, auch wenn das GNN die Baseline nicht schlägt.
- Gleiches gilt für die Case-Study (3.12): statt „GNN schlägt Baseline messbar“ das gemessene Ergebnis.
- Stand 07.10.2026: lokal erfüllt. Python: Runs `gcn`, `graphsage` und Kontrolle `mlp`, `scoreGnn` aus GraphSAGE, README eingeordnet, `verify` lokal Exit 0. Seite: Vertrag 3, alle Runs, Einordnung mit Bedingungen; `pnpm typecheck`, `lint`, `test`, `build` und `test:e2e` grün. Für die volle Definition of Done fehlt ein grüner CI-Lauf.
- Regel der Seite für vergleichende Sätze: Ein Unterschied zweier PR-AUC-Werte gilt nur dann als Rangfolge, wenn er größer ist als `randomPrAucQ95` − `randomPrAucExpected` (derzeit 0,0239); sonst sagt die Seite, dass er nicht belastbar ist. Der Satz „größerer Teil aus der Nachbarschaft“ erscheint nur, wenn jedes Graph-Verfahren um mehr als diesen Abstand über dem MLP liegt und dieser Gewinn den Gewinn des MLP über die bessere Baseline um mehr als den Abstand übersteigt. Auch der Satz Isolation Forest gegen Z-Scores folgt dem Abstand. Überschrift und Schluss des Homophilie-Vorbehalts sowie der Punkt „Eingebaute Nachbarschaft“ unter „Grenzen“ sprechen nur dann von einem gemessenen Vorsprung, wenn jedes Graph-Verfahren um mehr als den Abstand über der besseren Baseline und über dem MLP liegt; sonst im Konjunktiv. Die Satzbausteine stehen in `lib/netzradar/assessment.ts` mit Handtests, die E2E prüft sie gegen `metrics.json`. Das ist eine grobe Schwelle, kein Test (offener Punkt unten).

## Offene Entscheidungen (Dennis)

1. **Datensatz für die öffentliche Demo: IBM-AML oder synthetisches Netz.** Derzeit ist nur das synthetische Netz veröffentlicht. IBM-AML (CDLA-Sharing-1.0) trägt das Label je Transaktion, also je Kante zwischen Konten; die Methodik ist aber Knotenklassifikation. Optionen für die Abbildung:
   - a) Transaktionen als Knoten, wie bei Elliptic: Kante t₁ → t₂, wenn das Empfängerkonto von t₁ das Senderkonto von t₂ ist und t₂ in einem begrenzten Zeitfenster nach t₁ liegt. Das Label bleibt das Original-Tag, Pipeline und zeitlicher Split bleiben unverändert. Aufwand: Graphaufbau, Kantenzahl über das Fenster begrenzen, Zeitschritte festlegen.
   - b) Konten als Knoten, Label illicit bei mindestens einer Laundering-Transaktion. Einfach, aber das Label ist abgeleitet, und Konten existieren über den ganzen Zeitraum. Ein zeitlicher Split hätte dann Knoten auf beiden Seiten, im Konflikt mit Regel 2.
   - c) Kantenklassifikation. Passt zu den Labels, ändert aber Baseline, Metriken, Datenvertrag und Export.
   - Empfehlung: a), weil Label, Methodik und Split unverändert bleiben. Bis zur Entscheidung bleibt das synthetische Netz die öffentliche Demo.
2. **Laufzeit des CI-Jobs `train` mit GNN.** Gemessen ist nur lokal unter Windows: `gnn` 334,5 s, `verify` mit GNN 302 bis 357 s, pytest 59 s. Am 07.10.2026 ist das zweite `verify` nach `make export` gestrichen (es prüfte dieselbe Rechnung wie das erste). Der Job rechnet die GNN damit zweimal (`verify` gegen die eingecheckten Dateien, `make gnn`); bei ähnlicher Geschwindigkeit des Runners rund 12 bis 14 min einschließlich Installation. Weitere Option, falls das zu lang ist: die Seed-Streuung (4 zusätzliche Endmodelle je Verfahren, geschätzt rund 110 der 334,5 s) nur bei `gnn` rechnen und in `verify` ausnehmen.
3. **Live-Demo auf Hugging Face Spaces** (Free CPU, pausiert bei Inaktivität): ja oder nein. Die Seite selbst bleibt statisch (Regel 6); ein Space wäre höchstens ein externer Link.
4. **Repo öffentlich oder privat.** Davon hängen das CI-Badge im README (bei privatem Repo nicht öffentlich sichtbar) und die Sichtbarkeit von `docs/runs/` ab. Planungsdokumente bleiben in jedem Fall außerhalb.

Weitere offene Punkte:

- Generator-Annahmen fachlich bestätigen (`localShift` 0,5, gutartiger Hintergrund mit Hubs aus Preferential Attachment). Die GNN-Ergebnisse sind inzwischen dagegen gemessen; die Parameter wurden weder nach dem ersten Ergebnis noch nach den GNN-Läufen nachjustiert.
- Kontrollvariante K2 aus dem Plan (Isolation Forest auf lokalen Merkmalen plus Mittelwert der Nachbarmerkmale, ohne Labels) ist nicht umgesetzt; als eigener Run `iforestNeighbours` wäre sie eine weitere Vertragsänderung (neues Verfahren). Sensitivität gegen Homophilie (K3 Tarnkanten, K4 gradtreue Umverdrahtung) für Schritt 5 oder v2.
- GraphSAGE wählte auf dem Standardnetz Epoche 300, also die Obergrenze `maxEpochs`; das Early Stopping griff nicht. Ob die Obergrenze erhöht wird, ist eine Entscheidung vor einem neuen Lauf und darf sich nicht am Testergebnis orientieren.
- Golden-Werte des GNN auf `SMALL_CONFIG` (wie `test_small_network_matches_golden_values`) erst, wenn der erste CI-Lauf zeigt, dass Windows und Linux innerhalb von 1e-4 übereinstimmen.
- Mindestabstand für vergleichende Sätze auf der Seite (Abschnitt „Abnahme Schritt 4“). Optionen: so lassen (grobe Schwelle aus der Streuung zufälliger Rangfolgen); gepaarter Bootstrap über die Testknoten mit Konfidenzintervall für jede Differenz (neue Kennzahl, Vertragsänderung); Seed-Streuung aus `docs/runs/` in `metrics.json` übernehmen (Vertragsänderung). Empfehlung: Bootstrap in Schritt 5 zusammen mit der Case-Study, bis dahin die Schwelle.
- Die Seed-Streuung (Seeds 42 bis 46) steht nur in `docs/runs/` und im README; die Seite zeigt sie nicht, weil sie nur `data/k3/` liest (Regel 6).
- Ob aggregierte Elliptic-Kennzahlen (PR-AUC je Verfahren) ins README dürfen. Das Umsetzungsdokument (3.4, 3.10) erlaubt aggregierte Ergebnismetriken; derzeit wird aus Elliptic gar nichts veröffentlicht.
- Impressum für K1 bis K3 einheitlich gestalten. `OPERATOR.lastUpdated` steht in K3 auf 06.10.2026 (Stand der K3-Rechtstexte); K1 nutzt in seinem eigenen Repo weiter 03.09.2026.
- Die Ausrichtung der Elliptic-Merkmale (165 Spalten, die letzten 72 aggregiert) beim ersten echten Lauf bestätigen.
- Umgang mit alten `docs/runs/`-Dateien nach Neuläufen an anderen Tagen.
- Die Monorepo-Aussage in Projektanweisungen und Umsetzungsdokument an die getrennten Repos anpassen.
