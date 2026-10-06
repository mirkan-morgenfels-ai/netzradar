# Projekt: NetzRadar (K3)

Zweck: Reproduzierbare Fallstudie zur Anomalie-Erkennung in Transaktionsnetzwerken. Verglichen werden eine klassische Baseline (robuste Z-Scores und Isolation Forest mit Graphmaßen) und Graph Neural Networks (GCN, GraphSAGE), mit strikt zeitlichem Split und PR-AUC als Hauptmetrik. Training und Auswertung laufen offline in Python. Die Seite `/projects/netzradar` zeigt nur vorberechnete JSON-Ergebnisse. Kein Endnutzer-Werkzeug, kein Live-Scoring, keine echten personenbezogenen Daten. Drittes Portfolio-Projekt nach K1 DepotDoktor und K2 KontoKlar.

## Quellen

- Maßgeblich für Umfang, Methodik, Datensätze, Lizenzen, Zeitplan und Regeln ist das Umsetzungsdokument, Teil 3 „K3: NetzRadar“.
- Das Umsetzungsdokument und `Projektanweisungen_K3_NetzRadar.md` liegen außerhalb des Repos in `AI-Projekte\AI-Projekt-3\` und werden nicht eingecheckt. Der README-Entwurf `README_K3_NetzRadar.md` ist in `README.md` aufgegangen.
- Widerspricht eine Anfrage einem Dokument oder ein Dokument dem Code: hinweisen und vorschlagen, welches Dokument angepasst wird, statt still abzuweichen.
- Bekannte Widersprüche, Stand 06.10.2026:
  - Umsetzungsdokument 3.6 nennt als Abnahme von Schritt 4 „GNN schlägt Baseline (PR-AUC)“, 3.12 „GNN schlägt Baseline messbar“. Das widerspricht Regel 5. Es gilt der Abschnitt „Abnahme Schritt 4“ unten.
  - Umsetzungsdokument 0.5 und Projektanweisungen sprechen von einem gemeinsamen Monorepo mit K1 und K2. Tatsächlich ist K3 ein eigenes Repo (`AI-Project-3`), wie K1 und K2 auch.
  - Umsetzungsdokument 3.3.4: „alles licit ergibt ~98 % Accuracy“. Das gilt bezogen auf alle Knoten. Bewertet wird aber nur auf gelabelten Knoten. Bei Elliptic sind davon 42.019 / 46.564 ≈ 90,2 % licit, ein Alles-licit-Modell hätte dort also etwa 90 % Accuracy. Im README ist das korrigiert.
  - Umsetzungsdokument 3.7 exportiert beispielhaft `data/k3/result.json` mit `baseline_pr_auc`. Verbindlich ist der Datenvertrag unten: drei Dateien, Schlüssel in camelCase.
  - Projektanweisungen nennen die Projektstruktur `src/load.py`, `src/synth.py` usw. Tatsächlich liegt der Code im Paket `services/k3-train/src/k3_train/`.

## Stack

- Root: pnpm 10.34.5 (`packageManager`), Node `^20.19.0 || >=22.12.0` (`engines`, Untergrenze von vite 8 für Vitest); lokal Node 22.23.2, CI Node 22.
- Web, aufgelöst laut `pnpm-lock.yaml`: Next.js 15.5.27 (App Router), React 19.3.0, TypeScript 5.9.3 (strict, `noUncheckedIndexedAccess`), Tailwind CSS 4.3.3, Recharts 3.10.1 (in `packages/charts`), Vitest 4.1.11, Playwright 1.63.0, ESLint 9.39.5. Für die Graph-Ansicht: Sigma.js 3.0.3, graphology 0.26.0 und @sigma/node-border 3.0.0 für die Knotenringe.
- Python in `services/k3-train`, aufgelöst laut `uv.lock`: Python 3.12 (`requires-python >=3.12,<3.13`, lokal 3.12.15), uv 0.12.23 mit Build-Backend `uv_build`, networkx 3.7, numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.18.1, pytest 9.1.1, ruff 0.16.10. Die Untergrenze `networkx>=3.5` ist nötig, weil `spring_layout(method="force")` erst ab 3.5 existiert. PyTorch 2.x und PyTorch Geometric 2.7 kommen mit Schritt 4; für die CI aus dem CPU-Index. System-Python 3.14 ist für PyG 2.7 zu neu, deshalb uv mit Python 3.12.
- CI: GitHub Actions `.github/workflows/ci.yml` mit den Jobs `web` und `train`. Actions auf Node-24-Runtime: `actions/checkout@v5`, `pnpm/action-setup@v6`, `actions/setup-node@v5`, `actions/upload-artifact@v7`. `astral-sh/setup-uv@v10.2.0` ist fest eingetragen, weil setup-uv seit v7 keinen wandernden Major-Tag mehr veröffentlicht; die uv-Version ist dort auf 0.12.23 gesetzt. Der Job `train` setzt `UV_LOCKED=1`, damit kein `uv run` die Lockdatei still neu auflöst.
- Hosting: Vercel Hobby, statisch. Das Projekt ist noch nicht angelegt.

## Struktur

```
.github/workflows/ci.yml              web: install, typecheck, lint, test, build, E2E; train: uv sync --locked, make lint, make test,
                                      make verify (eingecheckte Dateien), data, baseline, export, make verify
apps/web/                             Next.js-App (Paket web)
  app/                                Startseite, layout, error, not-found, robots, impressum, datenschutz, nutzungsbedingungen
  app/projects/netzradar/page.tsx     Projektseite (Server-Komponente, Texte aus metrics.json)
  components/LegalPage.tsx            Rahmen der Rechtsseiten
  components/netzradar/               NetzRadarExplorer (Client, Auswahlzustand), GraphView (Sigma.js, dynamisch geladen),
                                      GraphLegend, NodeDetail (Detailfeld mit Fokusführung), NodeSymbol,
                                      TopNodesTable (Startknoten), MetricsTable
  lib/site.ts                         Projekte K1 bis K3, Navigation, siteUrl()
  lib/netzradar/types.ts              Datenvertrag als TypeScript-Typen, SCHEMA_VERSION
  lib/netzradar/data.ts               statischer JSON-Import aus data/k3 mit strenger Prüfung (bricht den Build ab)
  lib/netzradar/graph.ts              NODE_STYLES (Label-Kodierung), Knotengrößen, Kantenstile, Nachbarschaft, Komponenten
  lib/netzradar/format.ts             deutsche Zahlen- und Datumsformate, LABEL_TEXT, METHOD_TEXT
  lib/netzradar/summary.ts            Tabellenzeilen, Zufallsreferenz, Hyperparameterliste, Kennzahlen für die Texte
  lib/netzradar/curves.ts             METHOD_CURVE_STYLES (Farbe und Strichmuster je Verfahren), Prävalenzlinie
  lib/netzradar/__tests__/            Vitest
  e2e/                                Playwright-Tests (netzradar.spec.ts liest data/k3 direkt)
  vitest.config.mts                   als .mts, damit Vite die Konfiguration als ESM lädt
packages/ui/                          Card, Button, StatTile, cx
packages/legal/                       OPERATOR (lastUpdated = Stand der K3-Rechtstexte), Disclaimer, Datenschutz-Kurztext
packages/charts/                      theme.ts (Palette, Fallback-Farben und -Strichmuster), PrCurveChart (Recharts, eigene Legende);
                                      Export `@portfolio/charts/theme` liefert die Palette ohne Recharts
services/k3-train/                    Python-Paket k3_train, uv-Projekt, nicht im pnpm-Workspace
  Makefile                            all, data, baseline, export, verify, test, lint
  src/k3_train/cli.py                 Einstieg k3-train mit data, baseline, export, verify
  src/k3_train/synth.py               synthetischer Netzgenerator (SynthConfig, Seed 42)
  src/k3_train/load.py                Laden (synthetisch, Elliptic), Kantenbereinigung, data/processed lesen und schreiben
  src/k3_train/datasets.py            Datensatzregister mit Lizenz und Freigabe (publishable)
  src/k3_train/split.py               zeitliche Splits und Split-Prüfung
  src/k3_train/graph.py               Graphmaße je Zeitschritt mit Protokoll der Einstellungen, Label-Homophilie
  src/k3_train/features.py            Merkmalssätze local und local+graph
  src/k3_train/baseline.py            robuste Z-Scores, Isolation Forest
  src/k3_train/metrics.py             PR-AUC, Precision und Recall an Zielwerten, Accuracy-Nebenwert mit markierten Knoten
                                      und Treffern, Zufallsreferenz (exakter Erwartungswert, Permutationsquantil), PR-Kurve
  src/k3_train/pipeline.py            Baseline-Lauf, metrics-Payload, Laufprotokolle
  src/k3_train/export.py              Ausschnitt, Layout, nodes.json, edges.json, metrics.json
  src/k3_train/runs.py                Laufprotokolle schreiben
  src/k3_train/verify.py              Neuberechnung und Vergleich mit metrics.json, nodes.json, edges.json und den jüngsten Protokollen
  src/k3_train/jsonio.py              JSON-Format (4 Nachkommastellen, LF, 2 Leerzeichen)
  src/k3_train/paths.py               Pfade, Datenordner (--data-dir, K3_DATA_DIR), Prüfung des Projektordners
  tests/                              pytest; contract.py prüft den Datenvertrag; test_pipeline.py prüft Lecks und Golden-Werte;
                                      fixtures/elliptic_mini ist erfunden
  data/raw/, data/processed/          Standard-Datenordner, nur lokal, gitignored
data/k3/                              nodes.json, edges.json, metrics.json (Export, eingecheckt)
docs/runs/                            Laufprotokolle <YYYY-MM-DD>_<datensatz>_<verfahren>.json
docs/daten.md                         Bezug, Lizenzen und Ablage der Datensätze
docs/screenshots/                     netzradar-graph.png, netzradar-metriken.png
```

`src/k3_train/gnn.py` und das Make-Ziel `gnn` entstehen in Schritt 4.

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

- Jede Aufgabe einem Schritt zuordnen und den Schritt am Anfang nennen. Stand 06.10.2026:
  1. Daten laden, aufbereiten, zeitlicher Split: auf dem synthetischen Netz umgesetzt. Der Elliptic-Lader ist nur an einer erfundenen Fixture getestet. Der IBM-AML-Lader fehlt, er hängt an der Datensatzentscheidung.
  2. Baseline (robuste Z-Scores, Isolation Forest, Graphmaße): umgesetzt, Ergebnisse in `data/k3/metrics.json` und `docs/runs/`.
  3. Visualisierung und Export: umgesetzt. Export (`nodes.json`, `edges.json`, `metrics.json`) und Seite `/projects/netzradar` mit Graph-Ausschnitt, Metriktafel und PR-Kurven; Deployment offen.
  4. GNN (GCN, GraphSAGE in PyTorch Geometric): offen.
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

- `uv sync` (CI: `uv sync --locked` mit `UV_LOCKED=1`; lokal prüft `uv lock --check`, ob `uv.lock` zu `pyproject.toml` passt)
- `uv run ruff check` und `uv run ruff format --check`
- `uv run pytest`
- `uv run k3-train data --synth` erzeugt das synthetische Netz und schreibt `data/processed/synthetic/`
- `uv run k3-train baseline` rechnet beide Baselines und schreibt `docs/runs/<heute>_synthetic_{zscore,iforest}.json`
- `uv run k3-train export` schreibt `data/k3/nodes.json`, `edges.json` und `metrics.json`
- `uv run k3-train verify` rechnet das synthetische Netz im Speicher neu und vergleicht mit `data/k3/metrics.json`, `nodes.json`, `edges.json` und dem jeweils jüngsten Protokoll `docs/runs/*_synthetic_<verfahren>.json` (Toleranz 1e-4, `generatedAt` und `date` ausgenommen). Braucht keine aufbereiteten Daten und schreibt nichts.
- Datenordner für `raw/` und `processed/`: `--data-dir <Ordner>` bei `data`, `baseline` und `export`, sonst `K3_DATA_DIR`, sonst `services/k3-train/data`. Ohne Option und Variable prüft die CLI, dass sie im Projektordner läuft (`pnpm-workspace.yaml`, `services/k3-train/pyproject.toml`), und endet sonst mit Exit 2; `export`, `verify` und `baseline` für freigegebene Datensätze prüfen das immer.
- Elliptic nur lokal, mit Datenordner außerhalb von OneDrive: `$env:K3_DATA_DIR = "$env:LOCALAPPDATA\netzradar\data"`, Rohdaten nach `$env:K3_DATA_DIR\raw\elliptic`, dann `uv run k3-train data --dataset elliptic` und `uv run k3-train baseline --dataset elliptic`. Der Lader erwartet 165 Merkmalsspalten (93 lokal `f_*`, 72 aggregiert `a_*`). Einzelheiten in `docs/daten.md`.

Linux und CI mit make, in `services/k3-train`:

- `make verify`, `make data SYNTH=1`, `make baseline`, `make export`, `make verify`
- `make test`, `make lint` (ruff check und ruff format --check)
- `make all` führt lint, test, verify (eingecheckte Dateien), data --synth, baseline, export und verify aus
- `make data` ohne `SYNTH=1` und ohne `DATASET` endet mit einer Erklärung und Exit 2. `DATA_DIR=<Ordner>` reicht `--data-dir` an `data`, `baseline` und `export` durch, `RAW_DIR=<Ordner>` reicht `--raw-dir` an `data` durch.
- `make gnn` folgt in Schritt 4.

Hinweise:

- `baseline` und `export` schreiben Dateien mit dem heutigen Datum: `docs/runs/<YYYY-MM-DD>_…` und `generatedAt` in `metrics.json`. Ein Neulauf an einem anderen Tag erzeugt neue Protokolldateien neben den alten. Vor einem Commit `git diff` prüfen: Außer `generatedAt`, `date` und den Dateinamen darf sich nichts ändern, sonst ist die Reproduzierbarkeit gebrochen.
- `export` setzt voraus, dass `baseline` auf denselben aufbereiteten Daten lief, sonst Exit 1.
- Ein Python-Start dauert auf diesem Rechner etwa 8 s Importzeit; `uv run pytest` braucht rund 50 s, davon rund 17 s für den Hash-Seed-Regressionstest mit zwei Subprozessen. `k3-train verify` braucht rund 10 s.

## Git und Deployment

- Remote: https://github.com/mirkan-morgenfels-ai/AI-Project-3 (derzeit privat), Standardzweig `main`.
- Repo-lokale Git-Identität ist Pflicht: `user.email` = `324466065+mirkan-morgenfels-ai@users.noreply.github.com`, `user.name` derzeit `mirkan-morgenfels-ai`. Ohne die noreply-Adresse blockiert Vercel das Deployment („commit email could not be matched“). Global ist eine andere Adresse eingetragen.
- Das Home-Verzeichnis `C:\Users\MirkanDeniz` ist selbst ein Git-Repo (Branch `master`). Git-Befehle immer im Repo-Root `netzradar` ausführen.
- Push und PR über die GitHub CLI als `mirkan-morgenfels-ai`. Repo-lokal ist `credential.https://github.com.helper` gesetzt: erst leer, dann `!<gh.exe> auth git-credential`. Die globale Git-Konfiguration bleibt unverändert, andere Projekte laufen weiter über den Windows Credential Manager. gh liegt unter `%LOCALAPPDATA%\Microsoft\WinGet\Packages\GitHub.cli_Microsoft.Winget.Source_8wekyb3d8bbwe\bin\gh.exe`, nicht im PATH. PRs mit `gh pr create --base main`.
- Vercel: Projekt noch anzulegen, Root Directory `apps/web`, Framework Next.js. Danach deployt Vercel jeden Push auf `main`. Umgebungsvariablen sind nicht nötig; `NEXT_PUBLIC_SITE_URL` ist optional für `metadataBase` (leer bedeutet http://localhost:3000). Python läuft nicht auf Vercel.
- `.gitattributes` erzwingt LF (`* text=auto eol=lf`, `*.pdf` und `*.png` binär), weil `core.autocrlf=true` systemweit gesetzt ist.
- `next-env.d.ts` ist gitignored: Die Datei enthält Kommentare und verweist auf `.next/types`, ohne `.next` bricht `tsc` ab.

## Datenvertrag data/k3

Verbindlich für den Python-Export und die Seite. TypeScript-Typen in `apps/web/lib/netzradar/types.ts`, Python-Prüfung in `services/k3-train/tests/contract.py`. JSON in UTF-8, 2 Leerzeichen Einrückung, LF, Schlüssel in camelCase und fester Reihenfolge, Floats auf 4 Nachkommastellen gerundet, `schemaVersion` 2 in jeder Datei (seit 06.10.2026; Version 1 hatte keine Homophilie, keine Zufallsreferenz, keine Trefferzahlen und eine PR-Kurve mit künstlichem Startpunkt).

- `metrics.json`: `generatedAt` (ISO-8601 UTC); `dataset` mit `name`, `displayName`, `license`, `source`, `generator` (alle Generator-Parameter inklusive `seed`, oder `null`), `nodes`, `edges`, `timeSteps`, `features` (Anzahl lokaler Merkmale), `labelCounts` (`illicit`, `licit`, `unknown`) und `homophily` (`labelledEdges`, `sameLabelShare`, `illicitIllicitEdges`, `licitLicitEdges`, `illicitLicitEdges` über Kanten zwischen zwei gelabelten Knoten; `illicitWithIllicitNeighbour`, `licitWithIllicitNeighbour` als Knotenzahlen, Richtung ignoriert); `split` mit `kind: "temporal"`, `train`, `validation`, `test` (je `from`, `to`) und `crossSplitEdges` (muss 0 sein); `evaluation` mit `positiveLabel: "illicit"`, `excludedLabel: "unknown"`, `testPositives`, `testNegatives`, `prevalence` (= PR-AUC eines konstanten Scores), `allLicitAccuracy`, `randomPrAucExpected` (exakter Erwartungswert der Average Precision einer zufälligen Rangfolge), `randomPrAucQ95` (95-%-Quantil über `randomPermutations` zufällige Rangfolgen mit dem Seed des Laufs) und `randomPermutations`; `seed`; `runs[]` mit `method` (`zscore`, `iforest`, `gcn`, `graphsage`), `displayName`, `featureSet` (`local`, `local+graph`), `seed`, `date`, `hyperparameters`, `prAuc`, `precisionAtRecall50` (maximale Precision über alle Schwellen mit Recall ≥ 0,5), `recallAtPrecision50` (maximaler Recall über alle Schwellen mit Precision ≥ 0,5, sonst 0), `accuracy` (Nebenwert), `accuracyThreshold` (deutscher Text), `accuracyFlagged` (markierte Knoten an der Schwelle, Gleichstände eingeschlossen), `accuracyTruePositives` (davon illicit) und `prCurve` (1 bis 101 Punkte `{recall, precision}`, nach `recall` aufsteigend, beginnend bei der höchsten Schwelle ohne den künstlichen Punkt Recall 0 / Precision 1). `hyperparameters` von `zscore` enthält `zeroMadFeatures`, von `iforest` den Block `graphMeasures` (Grenzen der exakten Betweenness, Stichprobe, Seed, Eigenvektor-Einstellungen, `sampledSteps`, `eigenvectorFallbackSteps`). Bis Schritt 3 genau zwei Runs: `zscore` und `iforest`.
- `nodes.json`: `selection` (`scoreField: "scoreIforest"`, `pool: "test"`, `seeds: 50`, `hops: 2`, `maxNodes: 2000`, `truncated`) und `nodes[]` mit `id`, `x` und `y` in [-1, 1], `label`, `timeStep`, `scoreZscore`, `scoreIforest`, `scoreGnn` (`null` bis Schritt 4), `seedRank` (1 bis 50 für Startknoten, sonst `null`), `hop` (0 bis 2), `inDegree`, `outDegree`. Höherer Score bedeutet auffälliger. Startknoten sind die 50 Testknoten aller Labels mit dem höchsten `scoreIforest`; Nachbarschaft ungerichtet bis 2 Hops; Layout beim Export vorberechnet (Seed 42). Über `maxNodes` wird nach `hop`, dann nach Score gekürzt und `truncated` gesetzt.
- `edges.json`: `edges[]` mit `source` und `target`; induzierter gerichteter Teilgraph auf den exportierten Knoten, ohne Selbstschleifen und Duplikate, sortiert.
- `docs/runs/<YYYY-MM-DD>_<dataset>_<method>.json`: vollständiges, eigenständiges Protokoll je Lauf mit denselben Schlüsseln.
- Score-Semantik: `scoreZscore` = maxⱼ |zᵢⱼ| über die lokalen Merkmale mit zᵢⱼ = (xᵢⱼ − medianⱼ) / (1,4826 · MADⱼ), Median und MAD nur aus Trainingsknoten, Merkmale mit MAD = 0 tragen 0 bei. `scoreIforest` = −`score_samples`.
- Eine Änderung am Vertrag betrifft immer gleichzeitig `jsonio.py`, `pipeline.py`, `metrics.py`, `export.py`, `tests/contract.py`, `apps/web/lib/netzradar/types.ts` (`SCHEMA_VERSION`), `apps/web/lib/netzradar/data.ts` (strenger Parser mit festen Schlüsseln), `apps/web/lib/netzradar/__tests__/data.test.ts`, gegebenenfalls `apps/web/e2e/netzradar.spec.ts` (liest `metrics.json` und `nodes.json` direkt) und diesen Abschnitt; dabei `schemaVersion` erhöhen und `data`, `baseline`, `export` neu laufen lassen.

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
- Labels nie nur über Farbe kodieren, siehe Design. Linien in Diagrammen zusätzlich über Strichmuster unterscheiden: Farbe und Strichmuster je Verfahren in `METHOD_CURVE_STYLES` (`apps/web/lib/netzradar/curves.ts`); `SERIES_COLORS` und `SERIES_DASHES` in `packages/charts/src/theme.ts` sind nur Rückfallwerte.
- PR-Diagramm: eigene Legende in `PrCurveChart` mit Text in ink und `aria-hidden`-Symbolen, Tooltip-Text in ink, `accessibilityLayer={false}` (kein namenloser Tab-Stopp `role="application"`; die Werte stehen in der Metriktafel). Recharts färbt Legenden- und Tooltip-Text sonst in der Serienfarbe, bei Gold also mit zu wenig Kontrast.
- Detailfeld: Nach Auswahl eines Nachbarn oder „Auswahl aufheben“ geht der Fokus auf die Überschrift `#node-detail-title` (`tabIndex={-1}`), damit er nicht auf `<body>` fällt.
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

- `uv run ruff check`, `uv run ruff format --check` und `uv run pytest` grün
- `uv run k3-train verify` mit Exit 0
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

## Offene Entscheidungen (Dennis)

1. **Datensatz für die öffentliche Demo: IBM-AML oder synthetisches Netz.** Derzeit ist nur das synthetische Netz veröffentlicht. IBM-AML (CDLA-Sharing-1.0) trägt das Label je Transaktion, also je Kante zwischen Konten; die Methodik ist aber Knotenklassifikation. Optionen für die Abbildung:
   - a) Transaktionen als Knoten, wie bei Elliptic: Kante t₁ → t₂, wenn das Empfängerkonto von t₁ das Senderkonto von t₂ ist und t₂ in einem begrenzten Zeitfenster nach t₁ liegt. Das Label bleibt das Original-Tag, Pipeline und zeitlicher Split bleiben unverändert. Aufwand: Graphaufbau, Kantenzahl über das Fenster begrenzen, Zeitschritte festlegen.
   - b) Konten als Knoten, Label illicit bei mindestens einer Laundering-Transaktion. Einfach, aber das Label ist abgeleitet, und Konten existieren über den ganzen Zeitraum. Ein zeitlicher Split hätte dann Knoten auf beiden Seiten, im Konflikt mit Regel 2.
   - c) Kantenklassifikation. Passt zu den Labels, ändert aber Baseline, Metriken, Datenvertrag und Export.
   - Empfehlung: a), weil Label, Methodik und Split unverändert bleiben. Bis zur Entscheidung bleibt das synthetische Netz die öffentliche Demo.
2. **MVP-Umfang Schritt 4:** GCN und GraphSAGE wie geplant, oder Minimalpfad mit nur einem GCN ohne Architekturvergleich.
3. **Live-Demo auf Hugging Face Spaces** (Free CPU, pausiert bei Inaktivität): ja oder nein. Die Seite selbst bleibt statisch (Regel 6); ein Space wäre höchstens ein externer Link.
4. **Repo öffentlich oder privat.** Davon hängen das CI-Badge im README (bei privatem Repo nicht öffentlich sichtbar) und die Sichtbarkeit von `docs/runs/` ab. Planungsdokumente bleiben in jedem Fall außerhalb.

Weitere offene Punkte:

- Generator-Annahmen fachlich bestätigen (`localShift` 0,5, gutartiger Hintergrund mit Hubs aus Preferential Attachment), bevor in Schritt 4 GNN-Ergebnisse dagegen verglichen werden. Sie wurden nach dem ersten Ergebnis nicht nachjustiert.
- Ob aggregierte Elliptic-Kennzahlen (PR-AUC je Verfahren) ins README dürfen. Das Umsetzungsdokument (3.4, 3.10) erlaubt aggregierte Ergebnismetriken; derzeit wird aus Elliptic gar nichts veröffentlicht.
- Impressum für K1 bis K3 einheitlich gestalten. `OPERATOR.lastUpdated` steht in K3 auf 06.10.2026 (Stand der K3-Rechtstexte); K1 nutzt in seinem eigenen Repo weiter 03.09.2026.
- Die Ausrichtung der Elliptic-Merkmale (165 Spalten, die letzten 72 aggregiert) beim ersten echten Lauf bestätigen.
- Umgang mit alten `docs/runs/`-Dateien nach Neuläufen an anderen Tagen.
- Die Monorepo-Aussage in Projektanweisungen und Umsetzungsdokument an die getrennten Repos anpassen.
