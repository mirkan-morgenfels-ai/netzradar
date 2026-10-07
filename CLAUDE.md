# Projekt: NetzRadar (K3)

Arbeitsanweisungen für Claude Code (KI-gestützte Entwicklung).
Autor: Mirkan Deniz Günkaya.

## Zweck und Stand

Reproduzierbare Fallstudie zur Anomalie-Erkennung in Transaktionsnetzwerken. Verglichen werden eine klassische Baseline (robuste Z-Scores und Isolation Forest mit Graphmaßen) und Graph Neural Networks (GCN, GraphSAGE) mit einer MLP-Kontrolle ohne Kanten, mit strikt zeitlichem Split und PR-AUC als Hauptmetrik. Training und Auswertung laufen offline in Python. Die Seite `/projects/netzradar` zeigt nur vorberechnete JSON-Ergebnisse. Kein Endnutzer-Werkzeug, kein Live-Scoring, keine echten personenbezogenen Daten. Drittes Portfolio-Projekt nach K1 DepotDoktor und K2 KontoKlar; die Startseite verlinkt alle drei Projekte und ihre öffentlichen Repos.

Stand 07.10.2026: Schritte 1 bis 4 auf dem synthetischen Netz umgesetzt, Schritt 4 gemergt (35574f2), CI grün (Lauf 37554990175 vom 07.10.2026, Windows und Linux innerhalb 1e-4), deployt unter https://netzradar.vercel.app/projects/netzradar. Offen ist Schritt 5 (siehe Arbeitsweise).

## Quellen

- Die ursprüngliche Planung ist nicht Teil des Repos. Maßgeblich sind dieses Dokument, `README.md`, `docs/daten.md` und `docs/tests.md`. Rechnerspezifisches und Hinweise zu den Planungsunterlagen stehen in der nicht versionierten `CLAUDE.local.md`.
- Widerspricht eine Anfrage einem Dokument oder ein Dokument dem Code: hinweisen und vorschlagen, welches Dokument angepasst wird, statt still abzuweichen.
- Zahlen zu Datensätzen stammen aus `data/k3/metrics.json`, aus einer im Repository dokumentierten Quelle oder aus einer zitierten Veröffentlichung (für Elliptic Weber et al. 2019 und Elmougy und Liu 2023).

## Stack

- Root: pnpm 10.34.5 (`packageManager`), Node `^20.19.0 || >=22.12.0` (`engines`, Untergrenze von vite 8 für Vitest); lokal Node 22.23.2, CI Node 22.
- Web, aufgelöst laut `pnpm-lock.yaml`: Next.js 15.5.27 (App Router), React 19.3.0, TypeScript 5.9.3 (strict, `noUncheckedIndexedAccess`), Tailwind CSS 4.3.3, Recharts 3.10.1 (in `packages/charts`), Vitest 4.1.11, Playwright 1.63.0, ESLint 9.39.5. Für die Graph-Ansicht: Sigma.js 3.0.3, graphology 0.26.0 und @sigma/node-border 3.0.0 für die Knotenringe. axe-core 4.14.0 als devDependency von `apps/web` nur für die E2E-Prüfung der Zugänglichkeit. Vorschaubild und Apple-Icon entstehen beim Build mit `next/og` (in Next.js enthalten, Schrift Noto Sans aus dem Paket, kein Abruf zur Laufzeit).
- Python in `services/k3-train`, aufgelöst laut `uv.lock` (48 Pakete): Python 3.12 (`requires-python >=3.12,<3.13`, lokal 3.12.15), uv 0.12.23 mit Build-Backend `uv_build`, networkx 3.7, numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.18.1, pytest 9.1.1, ruff 0.16.10. Die Untergrenze `networkx>=3.5` ist nötig, weil `spring_layout(method="force")` erst ab 3.5 existiert. Python 3.14 ist für PyG 2.7 zu neu, deshalb uv mit Python 3.12.
- GNN als optionales Extra `gnn` (`[project.optional-dependencies]`): `torch>=2.8,<2.9` und `torch-geometric>=2.7,<2.8`, aufgelöst torch 2.8.0 und PyG 2.7.0. torch kommt über `[tool.uv.sources]` aus dem Index `pytorch-cpu` (https://download.pytorch.org/whl/cpu, `explicit = true`): `2.8.0+cpu` für Linux und Windows, `2.8.0` für macOS. PyG führt torch nicht als Abhängigkeit, der Resolver prüft die Verträglichkeit also nicht; deshalb ist torch auf 2.8 gebunden (laut Release-Notes von PyG 2.7.0 die Hauptversion). Das Extra bringt 27 Pakete mit; das torch-Wheel hat unter Windows 590,7 MiB, unter Linux laut CI-Log 175,4 MiB. PyG 2.7.0 gibt beim Import eine DeprecationWarning zu `torch_geometric.distributed` aus; pytest filtert genau diese Meldung (`filterwarnings` in `pyproject.toml`). Code ohne torch-Bedarf importiert torch nicht: `data`, `baseline`, `export` und `verify` ohne GNN-Runs laufen ohne das Extra; `cli.py` lädt `gnn_pipeline` erst im Befehl `gnn`, `verify.py` erst beim Nachrechnen von GNN-Runs.
- CI: GitHub Actions `.github/workflows/ci.yml` mit den Jobs `web`, `train` und `train-core` (`uv sync --locked` ohne Extra, `uv run pytest`; zeigt, dass die Kernmodule ohne torch laufen), alle auf `ubuntu-24.04` (fest, weil `ubuntu-latest` ab 19.10.2026 auf Ubuntu 26 wechselt und `verify` unter einem neuen Image erst bewusst geprüft werden soll), Workflow-Rechte `contents: read`, `concurrency` je Ref mit Abbruch älterer Läufe, `timeout-minutes` 15 (web), 20 (train) und 15 (train-core). Actions auf Node-24-Runtime: `actions/checkout@v5`, `pnpm/action-setup@v6`, `actions/setup-node@v5`, `actions/upload-artifact@v7`. `astral-sh/setup-uv@v10.2.0` ist fest eingetragen, weil setup-uv seit v7 keinen wandernden Major-Tag mehr veröffentlicht; die uv-Version ist dort auf 0.12.23 gesetzt. Der Job `train` setzt `UV_LOCKED=1`, damit kein `uv run` die Lockdatei still neu auflöst, und `K3_REQUIRE_GNN=1`, damit fehlendes torch die Tests rot statt übersprungen macht. `.github/dependabot.yml`: monatliche Updates für npm, GitHub Actions und uv (`services/k3-train`). npm in zwei Gruppen, `npm-minor-patch` und `npm-major`, damit ein Major-Sprung wie Next.js 16 oder TypeScript 7 die Patch-Updates nicht blockiert. uv in den Gruppen `python-tools` (pytest, ruff), `python-numerics` (networkx, numpy, pandas, scikit-learn, scipy; kann `make verify` rot machen, weil sich Kennzahlen verschieben können) und `python-gnn`; torch und torch-geometric nur als Patch-Version, weil eine neue Minor-Version die Kennzahlen verschieben kann.
- Hosting: Vercel Hobby, statisch, https://netzradar.vercel.app/projects/netzradar.

## Struktur

```
.github/workflows/ci.yml              web: install, typecheck, lint, test, build, E2E; train: uv sync --locked --extra gnn, make lint,
                                      make test, make verify (eingecheckte Dateien), data, baseline, gnn, export;
                                      train-core: uv sync --locked ohne Extra, uv run pytest
.github/dependabot.yml                npm, github-actions, uv (monatlich, gruppiert)
apps/web/                             Next.js-App (Paket web)
  app/                                Startseite, layout, error, not-found, robots, sitemap, opengraph-image, apple-icon,
                                      icon.svg, impressum, datenschutz, nutzungsbedingungen
  app/projects/netzradar/page.tsx     Projektseite (Server-Komponente, Texte aus metrics.json, Sprungnavigation)
  app/fonts.ts                        Cormorant Garamond und Inter über next/font/google (beim Build selbst gehostet)
  components/LegalPage.tsx            Rahmen der Rechtsseiten
  components/site/                    gemeinsame Bausteine mit K1 und K2: SiteHeader, Brand, NavLinks (Client, usePathname,
                                      aria-current), SiteFooter, HomeHero, ProjectHero, ProjectCards, SectionHeader, StatusPage,
                                      LegalNav, PillTabs, Shell, buttons.ts, motif.tsx (BrandMark, HeroOrnament), netMotif.ts
  components/ExternalLink.tsx         externer Link mit rel="noopener noreferrer" und sr-only „(externe Seite)“
  components/BrandMark.tsx            Bildmarke wie icon.svg für Vorschaubild und Apple-Icon
  components/netzradar/               NetzRadarExplorer (Client, Auswahlzustand), GraphView (Sigma.js, dynamisch geladen,
                                      noscript-Hinweis, Ladehinweis nur im Browser), GraphLegend, NodeDetail (Detailfeld mit
                                      Fokusführung und scoreGnn), NodeSymbol, TopNodesTable (Startknoten, Spalte scoreGnn mit
                                      Verfahren), MetricsTable (alle Runs, Zufallsreferenz), SearchTable (Kandidaten der
                                      GNN-Suche am Validierungsteil), ScrollRegion (waagrecht scrollbarer Bereich mit
                                      role="region", tabIndex 0 und Namen), MethodIndex (Inhaltsverzeichnis der Methodik,
                                      öffnet den Ziel-details)
  lib/og-glyphs.ts                    Glyphenpfade für Titel und Untertitel des Vorschaubilds
  lib/site.ts                         Projekte K1 bis K3 (kicker, repo), Navigation, Rechtslinks, Sitemap-Pfade,
                                      Repository-Links, DEFAULT_SITE_URL, siteUrl()
  lib/metadata.ts                     pageMetadata (Titel, Canonical, Open Graph mit Bild), Texte der Link-Vorschau,
                                      OG_IMAGE_VERSION (derzeit 3; Abfrage ?v= am Vorschaubild, weil es ein Jahr immutable
                                      ausgeliefert wird; bei jeder Änderung am Bild zusammen mit dem Fingerabdruck in
                                      lib/__tests__/metadata.test.ts anheben)
  lib/netzradar/types.ts              Datenvertrag als TypeScript-Typen, SCHEMA_VERSION, METHODS, LEARNED_METHODS,
                                      GRAPH_METHODS, Training (typisierte Hyperparameter der gelernten Runs)
  lib/netzradar/data.ts               statischer JSON-Import aus data/k3 mit strenger Prüfung (bricht den Build ab)
  lib/netzradar/graph.ts              NODE_STYLES (Label-Kodierung), Knotengrößen, Kantenstile, Nachbarschaft, Komponenten
  lib/netzradar/format.ts             deutsche Zahlen- und Datumsformate (Minus U+2212), LABEL_TEXT, METHOD_TEXT (lange
                                      Namen, nur Metriktafel), METHOD_SHORT_TEXT (kurze Namen im Text), METHOD_KIND_TEXT,
                                      FEATURE_SET_TEXT, joinList
  lib/netzradar/assessment.ts         Sätze der Einordnung aus metrics.json (Zufallsvergleich, Baselines, GNN gegen Baseline,
                                      GCN gegen GraphSAGE, Zerlegung mit MLP, Recall, Accuracy, Vorbehalt nur bei
                                      gemessenem Vorsprung)
  lib/netzradar/summary.ts            Tabellenzeilen, Zufallsreferenz, Mindestabstand für Vergleiche, erwartetes
                                      scoreGnnMethod, Accuracy-Obergrenze, Klassenabstand nach dem Mitteln,
                                      Hyperparameterliste, Zeitraum der Läufe, Kennzahlen für die Texte
  lib/netzradar/curves.ts             METHOD_CURVE_STYLES (Farbe, Strichmuster und Strichstärke je Verfahren), Prävalenzlinie
  lib/netzradar/messagePassing.ts     Rechenbeispiel Kette A – B – C – D (Mittelwert, GCN, GraphSAGE, MLP)
  lib/__tests__/, lib/netzradar/__tests__/
                                      Vitest; fixtures.ts baut Runs und Trainingsangaben für die Tests
  e2e/                                Playwright-Tests (netzradar.spec.ts liest data/k3 direkt), Port über PORT (Standard 3000)
  vitest.config.mts                   als .mts, damit Vite die Konfiguration als ESM lädt
packages/ui/                          Card, Button, StatTile, cx
packages/legal/                       OPERATOR (lastUpdated = Stand der Rechtstexte), Disclaimer, Datenschutz-Kurztext
packages/charts/                      theme.ts (Palette, Fallback-Farben und -Strichmuster), PrCurveChart (Recharts, eigene Legende
                                      und eigener Tooltip in Verfahrensreihenfolge);
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
docs/tests.md                         was pytest, Vitest und Playwright prüfen
docs/screenshots/                     netzradar-graph.png, netzradar-metriken.png
```

## Datenvertrag data/k3

Verbindlich für den Python-Export und die Seite. TypeScript-Typen in `apps/web/lib/netzradar/types.ts`, Python-Prüfung in `services/k3-train/tests/contract.py`. JSON in UTF-8, 2 Leerzeichen Einrückung, LF, Schlüssel in camelCase und fester Reihenfolge, Floats auf 4 Nachkommastellen gerundet, `schemaVersion` 3 in jeder Datei (seit 07.10.2026; Version 2 hatte nur `zscore` und `iforest`, `scoreGnn` immer null und kein `scoreGnnMethod`; Version 1 hatte keine Homophilie, keine Zufallsreferenz, keine Trefferzahlen und eine PR-Kurve mit künstlichem Startpunkt). Stand 07.10.2026: Python-Export und Seite auf Version 3. `data.ts` prüft die festen Werte wie `tests/contract.py`: `layers` 2, `hidden` 64, `activation` `relu`, `dropout` 0,5, `optimizer` `adam`, `learningRate` 0,01, `weightDecay` 0,0005, `loss` `weightedCrossEntropy`, `score` `logitIllicit - logitLicit`, `dtype` `float64` oder `float32`, Skalierung `median`, `mad`, 1,4826 und `fallback` genau [`std`, `one`], `scoreZscore` ≥ 0; Kandidaten der Suche eindeutig nach Merkmalssatz und Gewichtsregel, bei Gleichstand der frühere gewählt (beides prüft auch `contract.py`). Nur `contract.py` prüft die Rundung auf 4 Nachkommastellen. Darüber hinaus prüft `data.ts`: Runs in der Reihenfolge von `METHODS`; bei `gcn`, `graphsage` und `mlp` alle Hyperparameter-Schlüssel ohne `seedSpread`, `architecture` passend zum Verfahren, `edges` `none` genau beim MLP, genau ein gewählter Kandidat, kein Kandidat mit höherer Validierungs-PR-AUC als der gewählte, `stoppedEpoch` = min(`bestEpoch` + `patience`, `maxEpochs`), `selectedEpoch`, `featureSet`, Gewichtsregel und Validierungs-PR-AUC passend zum gewählten Kandidaten, festes Gewicht gleich dem des Kandidaten, `scaling.zeroMadFeatures` und `unitScaleFeatures` nur aus `features`, `finalFitSteps` = `split.train`, `validationSteps` = `split.validation`, `selectionSteps` direkt davor, Merkmalszahl passend zu `featureSet` (`local` gleich `dataset.features`, `local+graph` größer); `scoreGnnMethod` gleich dem Graph-Verfahren mit der höheren Validierungs-PR-AUC (bei Gleichstand `gcn`, ohne Graph-Runs `null`).

- `metrics.json`: `generatedAt` (ISO-8601 UTC); `dataset` mit `name`, `displayName`, `license`, `source`, `generator` (alle Generator-Parameter inklusive `seed`, oder `null`), `nodes`, `edges`, `timeSteps`, `features` (Anzahl lokaler Merkmale), `labelCounts` (`illicit`, `licit`, `unknown`) und `homophily` (`labelledEdges`, `sameLabelShare`, `illicitIllicitEdges`, `licitLicitEdges`, `illicitLicitEdges` über Kanten zwischen zwei gelabelten Knoten; `illicitWithIllicitNeighbour`, `licitWithIllicitNeighbour` als Knotenzahlen, Richtung ignoriert); `split` mit `kind: "temporal"`, `train`, `validation`, `test` (je `from`, `to`) und `crossSplitEdges` (muss 0 sein); `evaluation` mit `positiveLabel: "illicit"`, `excludedLabel: "unknown"`, `testPositives`, `testNegatives`, `prevalence` (= PR-AUC eines konstanten Scores), `allLicitAccuracy`, `randomPrAucExpected` (exakter Erwartungswert der Average Precision einer zufälligen Rangfolge), `randomPrAucQ95` (95-%-Quantil über `randomPermutations` zufällige Rangfolgen mit dem Seed des Laufs) und `randomPermutations`; `seed`; `runs[]` mit `method` (`zscore`, `iforest`, `gcn`, `graphsage`, `mlp`; jedes Verfahren höchstens einmal, Reihenfolge wie in dieser Liste), `displayName`, `featureSet` (`local`, `local+graph`), `seed`, `date`, `hyperparameters`, `prAuc`, `precisionAtRecall50` (maximale Precision über alle Schwellen mit Recall ≥ 0,5), `recallAtPrecision50` (maximaler Recall über alle Schwellen mit Precision ≥ 0,5, sonst 0), `accuracy` (Nebenwert), `accuracyThreshold` (deutscher Text), `accuracyFlagged` (markierte Knoten an der Schwelle, Gleichstände eingeschlossen), `accuracyTruePositives` (davon illicit) und `prCurve` (1 bis 101 Punkte `{recall, precision}`, nach `recall` aufsteigend, beginnend bei der höchsten Schwelle ohne den künstlichen Punkt Recall 0 / Precision 1). `hyperparameters` von `zscore` enthält `zeroMadFeatures`, von `iforest` den Block `graphMeasures` (Grenzen der exakten Betweenness, Stichprobe, Seed, Eigenvektor-Einstellungen, `sampledSteps`, `eigenvectorFallbackSteps`). `hyperparameters` von `gcn`, `graphsage` und `mlp` hat die Schlüssel in dieser Reihenfolge: `architecture` (`GCNConv`, `SAGEConv(aggr=mean)`, `Linear`), `layers` (2), `hidden` (64), `activation` (`relu`), `dropout` (0,5), `optimizer` (`adam`), `learningRate` (0,01), `weightDecay` (0,0005), `loss` (`weightedCrossEntropy`), `score` (`logitIllicit - logitLicit`), `positiveWeight` (Gewicht des Endmodells), `positiveWeightRule` (`trainRatio` oder `fixed`), `edges` (`undirected`, beim MLP `none`), `dtype` (`float64`), `scaling` (`center` `median`, `scale` `mad`, `madScale` 1,4826, `fallback` [`std`, `one`], `clip` 10, `fitOn` `train`, `zeroMadFeatures`, `unitScaleFeatures`), `features` (Namen in Spaltenreihenfolge), `maxEpochs` (300), `patience` (50), `selectedEpoch`, `selectionMetric` (`validationPrAuc`), `validationPrAuc`, `selectionSteps` (`from`, `to`; synthetisch 1 bis 17), `validationSteps` (18 bis 21), `finalFitSteps` (1 bis 21), `search` (je Kandidat `featureSet`, `positiveWeightRule`, `positiveWeight`, `validationPrAuc`, `bestEpoch`, `stoppedEpoch`, `selected`; genau einer `selected`) und `environment` (`torch`, `torchGeometric`, `python`, `platform`, `threads`, `deterministicAlgorithms`). `featureSet` des Runs ist der gewählte Merkmalssatz.
- `nodes.json`: `selection` (`scoreField: "scoreIforest"`, `pool: "test"`, `seeds: 50`, `hops: 2`, `maxNodes: 2000`, `truncated`), `scoreGnnMethod` (`"gcn"`, `"graphsage"` oder `null`; das GNN-Verfahren mit der höheren Validierungs-PR-AUC, bei Gleichstand `gcn`, nie das MLP, nie nach dem Test) und `nodes[]` mit `id`, `x` und `y` in [-1, 1], `label`, `timeStep`, `scoreZscore`, `scoreIforest`, `scoreGnn` (Zahl, wenn `scoreGnnMethod` gesetzt ist, sonst `null`), `seedRank` (1 bis 50 für Startknoten, sonst `null`), `hop` (0 bis 2), `inDegree`, `outDegree`. Höherer Score bedeutet auffälliger. Startknoten sind die 50 Testknoten aller Labels mit dem höchsten `scoreIforest`; Nachbarschaft ungerichtet bis 2 Hops; Layout beim Export vorberechnet (Seed 42). Über `maxNodes` wird nach `hop`, dann nach Score gekürzt und `truncated` gesetzt.
- `edges.json`: `edges[]` mit `source` und `target`; induzierter gerichteter Teilgraph auf den exportierten Knoten, ohne Selbstschleifen und Duplikate, sortiert.
- `docs/runs/<YYYY-MM-DD>_<dataset>_<method>.json`: vollständiges, eigenständiges Protokoll je Lauf mit denselben Schlüsseln. Nur in den Protokollen der GNN-Runs steht in `hyperparameters` zusätzlich `seedSpread` vor `environment` (`seeds` 42 bis 46, `reportedSeed`, `prAuc` je Seed, `mean`, `min`, `max`). `verify` vergleicht alles außer `generatedAt`, `date` und `environment`.
- Score-Semantik: `scoreZscore` = maxⱼ |zᵢⱼ| über die lokalen Merkmale mit zᵢⱼ = (xᵢⱼ − medianⱼ) / (1,4826 · MADⱼ), Median und MAD nur aus Trainingsknoten, Merkmale mit MAD = 0 tragen 0 bei. `scoreIforest` = −`score_samples`. `scoreGnn` = Logit illicit − Logit licit des Endmodells (keine Wahrscheinlichkeit, nach oben und unten unbeschränkt).
- Eine Änderung am Vertrag betrifft immer gleichzeitig `jsonio.py`, `pipeline.py`, `metrics.py`, `export.py`, `tests/contract.py`, `apps/web/lib/netzradar/types.ts` (`SCHEMA_VERSION`), `apps/web/lib/netzradar/data.ts` (strenger Parser mit festen Schlüsseln), `apps/web/lib/netzradar/__tests__/data.test.ts`, gegebenenfalls `apps/web/e2e/netzradar.spec.ts` (liest `metrics.json` und `nodes.json` direkt) und diesen Abschnitt; dabei `schemaVersion` erhöhen und `data`, `baseline`, `gnn`, `export` neu laufen lassen. Kommt ein Verfahren dazu, außerdem `METHODS` in `pipeline.py`, `contract.py` und `types.ts`, `METHOD_TEXT` und `METHOD_SHORT_TEXT` in `apps/web/lib/netzradar/format.ts` und `METHOD_CURVE_STYLES` in `apps/web/lib/netzradar/curves.ts`.

## Feste Regeln

1. Rohdaten werden nie ins Repository committet. `services/k3-train/data/raw/` und `services/k3-train/data/processed/` sind gitignored (Root-`.gitignore` und `services/k3-train/.gitignore`). Veröffentlicht werden nur Code, Metriken, Plots und Stichproben synthetischer Daten. Aus Elliptic werden keine Rohdaten, keine bearbeiteten Fassungen und keine Ausschnitte veröffentlicht. Bei widersprüchlichen Lizenzangaben gilt die restriktivere. Technisch: `k3-train export --dataset elliptic` bricht mit Exit 2 ab, Elliptic-Laufprotokolle landen in `<Datenordner>/processed/elliptic/runs/` (Datenordner per `--data-dir` oder `K3_DATA_DIR`, Standard `services/k3-train/data`).
2. Der Split ist zeitlich. Kein zufälliger Split und kein Split, bei dem Knoten derselben Zeitkomponente auf beiden Seiten liegen. Jeder Split bekommt einen Test (`tests/test_split.py`; `check_split` prüft: keine geteilten Zeitschritte, keine Kante zwischen Trainings- und Testknoten, Validierung liegt im Trainingszeitraum).
3. Hauptmetrik ist PR-AUC (average precision), dazu Precision und Recall. Accuracy wird nie als Hauptergebnis berichtet, nur als Nebenwert.
4. Jeder Lauf schreibt Datensatz, Split, Seed, Hyperparameter und Datum nach `docs/runs/` und über den Export nach `data/k3/metrics.json`. Ergebnisse ohne diese Angaben gelten als nicht reproduzierbar und werden nicht ins README übernommen.
5. Ergebnisse werden so berichtet, wie sie sind. Schlägt das GNN die Baseline nicht, wird das dokumentiert und erklärt, nicht kaschiert.
6. Kein Modell im Live-Pfad, keine API-Aufrufe zur Laufzeit. Die Seite bindet beim Build nur die statisch exportierten JSON-Dateien aus `data/k3/` ein. Erklärungen per Sprachmodell (v3) werden im Batch vorberechnet.
7. Zahlen zu Datensätzen (Größe, Labelanteile, Merkmalszahl) nur aus `data/k3/metrics.json`, aus einer im Repository dokumentierten Quelle oder aus einer zitierten Veröffentlichung. Unklarheiten offen benennen, etwa 165, 166 oder 167 Merkmale bei Elliptic. Nichts erfinden, nichts schönen.
8. Englisch in Code-Bezeichnern, Dateinamen und JSON-Schlüsseln; Deutsch in allen Seitentexten, Ansprache „Sie“. Keine Kommentarzeilen im Code: keine `//` oder `/* */` in TS, TSX und JS, keine `#`-Kommentare und keine Docstrings in Python, keine Kommentare in YAML, CSS, Makefile und TOML. Erklärungen gehören ins README oder nach `docs/`.

Zusätzlich verbindlich:

- Jede Kennzahl bekommt einen Unit-Test mit von Hand gerechnetem Erwartungswert (`tests/test_metrics.py`, `tests/test_baseline.py`, `tests/test_graph.py`, auf der Webseite in `lib/netzradar/__tests__/`).
- Farben nur aus den Tokens, siehe Design; moss und wine nur für Labels und Signale.
- Seitentexte ohne interne Schrittnummern und ohne Verweise auf nicht öffentliche Unterlagen; Labels heißen auf der Seite auffällig, unauffällig und unbekannt, nicht illicit oder licit.

## Arbeitsweise

- Jede Aufgabe einem Schritt zuordnen und den Schritt am Anfang nennen. Stand 07.10.2026:
  1. Daten laden, aufbereiten, zeitlicher Split: auf dem synthetischen Netz umgesetzt. Der Elliptic-Lader ist nur an einer erfundenen Fixture getestet. Der IBM-AML-Lader fehlt, er hängt an der Datensatzentscheidung.
  2. Baseline (robuste Z-Scores, Isolation Forest, Graphmaße): umgesetzt, Ergebnisse in `data/k3/metrics.json` und `docs/runs/`.
  3. Visualisierung und Export: umgesetzt und deployt unter https://netzradar.vercel.app.
  4. GNN (GCN, GraphSAGE und MLP-Kontrolle in PyTorch Geometric): umgesetzt und gemergt (35574f2). Datenvertrag Version 3, `verify` mit GNN, CI mit `make gnn` grün; der erste CI-Lauf (37554990175, 07.10.2026) hat Windows und Linux innerhalb 1e-4 bestätigt.
  5. Case-Study und Seite: offen. Dazu gehören Läufe mit gestörter Nachbarschaft (Sensitivität gegen Homophilie), eine Baseline mit gemittelten Nachbarmerkmalen und Startknoten nach dem GNN-Score.
- Nachkontrolle 07.10.2026 (`fix/nachkontrolle`): Repo-Links, gemeinsamer Projektblock und Navigation mit K1 und K2, Metadaten mit Vorschaubild, Sitemap, Reflow bei 320 px, Texte ohne interne Schrittnummern; keine Kennzahl geändert.
- Bei neuen Aufgaben zuerst ein kurzer Plan mit Dateien, Funktionen, Tests, Abnahmekriterium, geschätzten Stunden und Laufzeit. Kleine Änderungen direkt umsetzen.
- Mathematik erklären, wenn sie eine Entscheidung trägt: Formeln und Mini-Beispiele sind erwünscht, Floskeln nicht.
- Code-Reviews in dieser Reihenfolge: Datenlecks (Split, Merkmale und Normalisierung nur aus Trainingsdaten), Korrektheit der Metriken, Reproduzierbarkeit (Seeds, Konfiguration), Lizenz (was landet im Repository), Struktur und Stil.
- Offene Entscheidungen mit Optionen und Empfehlung vorlegen; der Autor entscheidet.

## Befehle

Unter Windows liegen Node, pnpm und uv je nach Installation nicht im PATH; der lokale Vorspann steht in `CLAUDE.local.md`. Liegt das Repo in einem synchronisierten Ordner wie OneDrive, die venv per `UV_PROJECT_ENVIRONMENT` außerhalb anlegen lassen.

Web, im Repo-Root:

- `pnpm install` (CI: `pnpm install --frozen-lockfile`)
- `pnpm dev` startet die Entwicklung unter `http://localhost:3000/projects/netzradar`
- `pnpm typecheck`, `pnpm lint`, `pnpm test`, `pnpm build`; `pnpm ci` führt alle vier nacheinander aus
- `pnpm test:e2e` startet Playwright; ohne `CI` gegen `next dev`, mit `CI=true` gegen `next start` (vorher `pnpm build`). Der Port kommt aus `PORT` (Standard 3000), etwa `PORT=3121` neben anderen Servern. `PLAYWRIGHT_CHROMIUM_PATH` setzt optional ein lokal installiertes Chromium. Die CI installiert die passende Revision selbst.

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
- Elliptic nur lokal, mit einem Datenordner außerhalb synchronisierter Ordner (`K3_DATA_DIR` oder `--data-dir`), Rohdaten nach `<Datenordner>/raw/elliptic`, dann `uv run k3-train data --dataset elliptic` und `uv run k3-train baseline --dataset elliptic`. Der Lader erwartet 165 Merkmalsspalten (93 lokal `f_*`, 72 aggregiert `a_*`). Einzelheiten in `docs/daten.md`.

Linux und CI mit make, in `services/k3-train`:

- `make verify`, `make data SYNTH=1`, `make baseline`, `make gnn`, `make export`, `make verify`
- `make test`, `make lint` (ruff check und ruff format --check)
- `make all` führt lint, test, verify (eingecheckte Dateien), data --synth, baseline, gnn, export und verify aus
- `make gnn`, `make verify` und `make test` nutzen `uv run --extra gnn` (Variable `GNN_RUN`); `data`, `baseline`, `export` und `lint` bleiben bei `uv run`.
- `make data` ohne `SYNTH=1` und ohne `DATASET` endet mit einer Erklärung und Exit 2. `DATA_DIR=<Ordner>` reicht `--data-dir` an `data`, `baseline`, `gnn` und `export` durch, `RAW_DIR=<Ordner>` reicht `--raw-dir` an `data` durch.

Hinweise:

- `baseline`, `gnn` und `export` schreiben Dateien mit dem heutigen Datum in UTC: `docs/runs/<YYYY-MM-DD>_…` und `generatedAt` in `metrics.json`. Ein Neulauf an einem anderen Tag erzeugt neue Protokolldateien neben den alten. Vor einem Commit `git diff` prüfen: Außer `generatedAt`, `date` und den Dateinamen darf sich nichts ändern, sonst ist die Reproduzierbarkeit gebrochen.
- `export` setzt voraus, dass `baseline` auf denselben aufbereiteten Daten lief, sonst Exit 1; liegen GNN-Ergebnisse vor, müssen `dataset`, `split`, `evaluation` und `seed` mit der Baseline übereinstimmen.
- Laufzeiten unter Windows auf dem Entwicklungsrechner (06.10.2026): `data` 0,7 s, `baseline` 4,1 s, `export` 1,9 s, `gnn` 334,5 s, `verify` mit GNN 302 bis 357 s, am 07.10.2026 226 und 197 s. In der CI unter Linux (07.10.2026): Job `train` rund 7 min, `verify` 186 s, `gnn` 178 s. Weitere lokale Messungen in `CLAUDE.local.md`.

## Git und Deployment

- Remote: https://github.com/mirkan-morgenfels-ai/netzradar, öffentlich seit 07.10.2026, Standardzweig `main`.
- Commit-Autor „Mirkan Deniz Günkaya“ mit der GitHub-noreply-Adresse `324466065+mirkan-morgenfels-ai@users.noreply.github.com`. Ohne die noreply-Adresse blockiert Vercel das Deployment („commit email could not be matched“). Die repo-lokale Einrichtung steht in `CLAUDE.local.md`.
- Commit-Stil: Conventional Commits auf Deutsch mit echten Umlauten, z. B. `feat(k3): …`, `fix(k3): …`, `test(k3): …`, `docs(readme): …`, `ci: …`, `chore: …`. Keine Verweise auf private Repos oder lokale Pfade in Commit-Nachrichten.
- Branches `feat/<thema>` und `fix/<thema>`. `main` ist immer deploybar. Ein PR pro abgeschlossenem Schritt, CI muss grün sein, der Autor merged selbst. Commit und Push nur auf ausdrücklichen Wunsch.
- Vercel: Root Directory `apps/web`, Framework Next.js, Produktions-URL https://netzradar.vercel.app. Vercel deployt jeden Push auf `main`. Umgebungsvariablen sind nicht nötig; `NEXT_PUBLIC_SITE_URL` ist optional für `metadataBase`, Sitemap und `robots.txt` (leer bedeutet `https://netzradar.vercel.app`). Python läuft nicht auf Vercel.
- `.gitattributes` erzwingt LF (`* text=auto eol=lf`, `*.pdf` und `*.png` binär), damit Checkouts mit `core.autocrlf=true` keine CRLF erzeugen.
- `next-env.d.ts` ist gitignored: Die Datei enthält Kommentare und verweist auf `.next/types`, ohne `.next` bricht `tsc` ab. `CLAUDE.local.md` ist ebenfalls gitignored.

## Betrieb lokal

- `next dev` und `next build` nie gleichzeitig in `apps/web` laufen lassen. Der Build überschreibt `.next`, der Dev-Server antwortet danach mit 500.
- Die Seite braucht keine Umgebungsvariablen und keinen Python-Prozess; sie bindet nur `data/k3/` ein.
- Nach Änderungen an der Python-Pipeline in dieser Reihenfolge: `verify` (zeigt, ob sich Kennzahlen ändern), `data`, `baseline`, `export`, `verify`, danach die Seite prüfen.
- `services/k3-train/data/processed/` ist ein lokaler Zwischenstand und lässt sich jederzeit mit `data --synth` neu erzeugen.
- Bricht `pnpm build` mit „TypeError: Cannot read properties of undefined (reading 'length') at WasmHash._updateWithBuffer“ ab, ist `apps/web/.next` unvollständig oder veraltet. Abhilfe: `apps/web/.next` löschen und neu bauen.
- Nach einem lokalen E2E-Lauf den Port wieder freigeben.

## Sicherheit und Zugänglichkeit

- Sicherheits-Header in `apps/web/next.config.ts`: CSP mit `default-src 'self'` und `connect-src 'self'`, `'unsafe-eval'` nur im Dev-Modus; HSTS zwei Jahre, `X-Frame-Options: DENY`, `nosniff`, Referrer-Policy, Permissions-Policy (`camera`, `microphone`, `geolocation`, `payment`, `usb`, `browsing-topics` aus, wie in K1 und K2) und COOP `same-origin`.
- Keine externen Schriften, Skripte, Fetches, Cookies oder Tracker. Eine neue fremde Quelle verstieße gegen Regel 6 und gegen die Datenschutzerklärung (`apps/web/app/datenschutz/page.tsx`). Änderungen an dem, was die Seite lädt oder überträgt, ziehen Änderungen an der Datenschutzerklärung und am README-Abschnitt „Datenschutz“ nach sich. Nach dem Laden holt der Browser nur Programmteile und Seitendaten verlinkter Seiten (Prefetch von `next/link`) vom selben Server.
- Externe Links (Schwesterprojekte, GitHub) mit `rel="noopener noreferrer"`, ohne `target`, mit sichtbarem oder sr-only „(externe Seite)“. Sie laden vorab nichts.
- E2E-Wächter: Während der Nutzung entsteht keine Anfrage an einen fremden Origin und keine Nicht-GET-Anfrage; außerdem keine Konsolenfehler, keine `pageerror` und keine CSP-Verstöße (`securitypolicyviolation`).
- Skip-Link auf `#main`, globaler Fokusring 2 px in `gold-deep` auf Hell und `gold-light` auf Navy, `prefers-reduced-motion` wird respektiert, `lang="de"`. Aktiver Navigationslink mit `aria-current="page"`.
- Goldtext auf Hell nur als `text-gold-deep` (#7d5f17), auf Navy als `text-gold-light` (#d8bd72); `gold` (#c9a548) nur für Linien, Flächen, Rahmen und Ringe.
- Labels nie nur über Farbe kodieren, siehe Design. Linien in Diagrammen zusätzlich über Strichmuster unterscheiden: Farbe, Strichmuster und Strichstärke je Verfahren in `METHOD_CURVE_STYLES` (`apps/web/lib/netzradar/curves.ts`, Werte im Abschnitt Design); `SERIES_COLORS` und `SERIES_DASHES` in `packages/charts/src/theme.ts` sind nur Rückfallwerte. Tests prüfen, dass alle Kurvenfarben aus `CHART_COLORS` stammen, jedes Verfahren eine eigene Farbe und ein eigenes Strichmuster hat und keine Kurve moss oder wine trägt.
- PR-Diagramm: eigene Legende in `PrCurveChart` mit Text in ink und `aria-hidden`-Symbolen, eigener Tooltip-Inhalt in der Reihenfolge der Legende mit Linienmuster je Eintrag und Text in ink mit Trenner „: “, `accessibilityLayer={false}` (kein namenloser Tab-Stopp `role="application"`; die Werte stehen in der Metriktafel). Recharts färbt Legenden- und Tooltip-Text sonst in der Serienfarbe, bei Gold also mit zu wenig Kontrast.
- Detailfeld: Nach Auswahl eines Nachbarn oder „Auswahl aufheben“ geht der Fokus auf die Überschrift `#node-detail-title` (`tabIndex={-1}`), damit er nicht auf `<body>` fällt.
- Waagrecht scrollbare Bereiche (Tabellen der Labels, Metriken, Suche und der Kette A – D sowie alle Formeln) liegen in `ScrollRegion` mit `role="region"`, `tabIndex={0}` und eindeutigem Namen (Tabellenüberschrift oder „Formel: …“), damit sie per Tastatur scrollbar sind; die Startknoten-Tabelle hat dasselbe Muster. Die E2E lässt axe-core (WCAG 2.x A und AA, Best Practices) bei 390, 768 und 1280 px laufen und erwartet 0 Verstöße. Ohne `tabIndex` melden die Läufe bei 390 und 768 px `scrollable-region-focusable` (geprüft am 07.10.2026). Kleine Textlinks in Navigation und Fußzeile brauchen mindestens 8 px Zeilenabstand (`gap-y-2` bzw. `gap-y-3`), sonst meldet axe `target-size`.
- Reflow: bei 320 px kein waagrechter Überlauf (E2E auf fünf Seiten); Überschriften der Rechtsseiten mit `hyphens-auto break-words` und `text-[clamp(2rem,9vw,4rem)]`, Lesespalte der Rechtsseiten `max-w-[36rem]`.
- Graph: Hinweis für Maus und Touch über dem Graphen, Zoom-Schaltflächen auf schmalen Viewports unter dem Graphen, `<noscript>`-Hinweis an seiner Stelle.
- Rechtsseiten über `@portfolio/legal` und `components/LegalPage.tsx`; Disclaimer auf der Projektseite.

## Design

Designsystem „Navy & Gold“ (seit 07.10.2026 live), gemeinsam mit K1 und K2: Kopf, Fuß, Startseite, Rechtsseiten, 404 und die Bausteine in `apps/web/components/site/` sind in allen drei Repos gleich, nur die Projektdaten unterscheiden sich. Leitidee: dunkler Rahmen in Navy mit Goldakzenten (Kopf, Hero, Fuß), helle Arbeitsflächen in Elfenbein (Tabellen, Diagramme, Graph). Blau ist erlaubt.

- Tokens (`@theme` in `apps/web/app/globals.css`):
  - Navy: navy-950 #0b1626 (Kopf, Hero, Fuß), navy-900 #101f35 (Karten auf Navy), navy-800 #16273f (Hover, aktive Pille), navy-700 #26354d (Haarlinien auf Navy), navy-300 #8f9bb0 (gedämpfter Text auf Navy).
  - Hell: ivory #f7f3ea (Seitenhintergrund), surface #fffdf8 (Karten), line #e4ddcc (Haarlinien), line-strong #858d9b (Ränder von Bedienelementen), ink #0f1b2d (Text), slate #5b6474 (gedämpfter Text).
  - Gold: gold #c9a548 (Linien, Ränder, Marke, Buttons auf Navy), gold-light #d8bd72 (Goldtext auf Navy), gold-deep #7d5f17 (Goldtext und Fokus auf Hell), gold-soft #f3e9c9 (zarte Flächen).
  - Signale: moss #2f6b3a, moss-light #93c9a0, moss-soft #dfeadf (unauffällig) sowie wine #7a1f2b, wine-light #e39aa4, wine-soft #f1dcdf (auffällig). moss und wine nur für Labels und Signale, nie für Verfahren.
  - sky #3e6a9e: einzige mittlere Blau-Datenfarbe, sparsam.
  - Aliase für ältere Klassen: paper = ivory, stone = slate.
- Schriften: Cormorant Garamond 500 (Display, Kursiv für Akzentwörter in gold-light) und Inter 400/500 über `next/font/google` in `app/fonts.ts`, beim Build selbst gehostet, zur Laufzeit keine fremde Anfrage. Klassen `display` (Display-Schrift mit lining-nums), `num` (tabellarische Ziffern) und `eyebrow` (Inter 11 px, Sperrung 0,16 em, Versalien, gold-deep, auf Navy gold-light). Fließtext 15–16 px mit Zeilenhöhe 1,75 und höchstens etwa 70 Zeichen je Zeile (`max-w-[40rem]`), Leads höchstens drei Zeilen.
- Diagrammfarben (`CHART_COLORS` in `packages/charts/src/theme.ts`): navy #1d3a5f, gold #b8912f, goldLine #a8832a, goldDeep #7d5f17, moss #2f6b3a, wine #7a1f2b, sky #3e6a9e, slate #5b6474, sand #c9b98f, ink #0f1b2d, line #e4ddcc, lineStrong #858d9b, grid #ece6d8, surface #ffffff. Kurven nach Verfahrensfamilie in `METHOD_CURVE_STYLES` (`apps/web/lib/netzradar/curves.ts`): Baselines warm (Z-Scores goldLine durchgezogen, Isolation Forest goldDeep „8 3“), Graph Neural Networks blau (GCN sky „2 3“ mit Strichstärke 2,5, GraphSAGE navy „8 3 2 3“), Kontrolle grau (MLP slate „12 4“), Prävalenz lineStrong „4 4“ dünner. Dieselben Farben tragen die Messbalken der Kacheln und der Metriktafel. `SERIES_COLORS` und `SERIES_DASHES` sind nur Rückfallwerte.
- Bausteine in `apps/web/components/site/`: SiteHeader (navy-950, ab 640 px klebend, darunter statisch, ab 768 px einzeilig), Brand mit Netzmarke (`BrandMark` in `motif.tsx`), NavLinks (aktiver Link mit Goldlinie und `aria-current`, externe Links immer mit ↗), HomeHero, ProjectHero (Kennfakten als `dl`, Begriff als `dt`, Zahl als `dd`; unter 640 px dreispaltige Leiste ohne Ornament), ProjectCards, SectionHeader (Eyebrow, Display-h2, kurzer Lead), StatusPage (404 und Fehler), SiteFooter (drei Spalten mit gemeinsamem 20-px-Zeilenrhythmus), LegalNav, PillTabs (unter 360 px als 2×2-Raster), Buttons in `buttons.ts` (Pillen: Gold und Kontur gold/70 auf Navy, Navy und Kontur line-strong auf Hell). In `packages/ui`: Card, Button, StatTile (Einheit klein hinter der Zahl, Leerwert „—“ klein in line-strong).
- Karten auf Hell: surface, Radius 16 px, Rand line, `shadow-card`. Tabellen `.data-table`: Kopf in Versalien slate 11 px, Zahlenspalten rechtsbündig mit tabellarischen Ziffern, Textspalten ohne (sonst wirkt der Bindestrich gesperrt). Auswahlfelder `select.field` mit eigenem Chevron.
- Projektseite: Abschnittsleiste mit allen sieben Abschnitten, ab 1024 px unter dem Kopf klebend (`lg:top-[4.25rem]`, Anker mit `lg:scroll-mt-36`), mobil einzeilig waagrecht scrollbar. Detailfeld ab 1024 px klebend mit eigener Scrollfläche (`max-h-[calc(100dvh-10rem)]`, unten ausgeblendet). Die Unterabschnitte der Methodik außer Split, Suche und PR-AUC sind `details` mit h3 im `summary`; das Inhaltsverzeichnis (`components/netzradar/MethodIndex.tsx`) öffnet den Zielabschnitt. Die Tabelle der Startknoten ist ein einziger Tab-Stopp mit Pfeiltasten. Graph-Zeichenfläche hell (surface mit Punktraster), Lade-, Fehler- und noscript-Zustand mit Goldraute auf dem Raster.
- Label-Kodierung in Graph, Legende und Tabellen (Werte in `NODE_STYLES`, `apps/web/lib/netzradar/graph.ts`), formunterscheidbar auch bei Rot-Grün-Schwäche: auffällig gefüllt in wine mit Goldring, unauffällig gefüllt in moss, unbekannt hohl (surface) mit Kontur in slate. Kanten slate mit Alpha, aktive Kanten navy.
- Vorschaubild, Apple-Icon und Favicon: Navy mit Goldraute und Netzmarke (drei Ringknoten sternförmig um einen gefüllten Mittelknoten), Projektname in Display aus Glyphenpfaden (`lib/og-glyphs.ts`). Das Netz-Motiv im Vorschaubild bleibt rechts vom Titel.
- Fokus: 2 px solid mit 2 px Abstand, gold-deep auf Hell, gold-light auf Navy (`.surface-navy`); Links und `summary` mit 0,375 rem Radius, Navigationslinks und Pillen rund. Skip-Link in Gold auf `#main`.
- Kontraste (nachgerechnet, WCAG 4,5:1 für Text, 3:1 für Grafik und Ränder): navy-300 auf navy-950 6,47:1, gold-light auf navy-950 9,89:1, slate auf ivory 5,39:1 und auf surface 5,87:1, gold-deep auf surface 5,87:1 und auf gold-soft 4,92:1, wine auf surface 10,0:1; Grafik und Ränder: goldLine 3,47:1, sky 5,49:1 und line-strong 3,29:1 auf surface, line-strong 3,02:1 auf ivory, gold/70 auf navy-950 4,36:1.
- Zahlen mit deutschem Dezimalkomma, typografischem Minus (U+2212) und tabellarischen Ziffern. Verfahrensnamen im Text kurz (Z-Scores, Isolation Forest, GCN, GraphSAGE, MLP (ohne Kanten)), die langen `displayName` aus dem Export nur in der Metriktafel. Lieber mehr Weißraum als gedrängt. `prefers-reduced-motion` schaltet Übergänge und die Linienanimation (`draw-line`) ab.

## Test-Stand

Stand 07.10.2026, Einzelheiten in `docs/tests.md`:

- pytest: 171 Tests mit dem Extra `gnn`; ohne Extra werden die 28 Tests in `test_gnn.py` und `test_gnn_pipeline.py` mit Grund übersprungen.
- Vitest: 194 Tests in 9 Dateien, rund 3 s.
- Playwright: 24 Tests, davon 3 axe-Läufe; mit `CI=true` gegen `next start` rund 30 s.
- CI: Lauf 37554990175 vom 07.10.2026 grün, Job `train` rund 7 min.

## Definition of Done

Je Schritt: pytest grün, Split-Test grün, Metriken reproduzierbar, Export in `data/k3/` vorhanden, Seite rendert Graph-Ausschnitt und Metriktafel, Rohdaten nicht im Diff.

Konkret prüfbar:

- `uv run ruff check`, `uv run ruff format --check` und `uv run --extra gnn pytest` grün, ohne übersprungene GNN-Tests
- `uv run --extra gnn k3-train verify` mit Exit 0
- `pnpm typecheck`, `pnpm lint`, `pnpm test`, `pnpm build` und `pnpm test:e2e` grün, E2E-Wächter ohne fremde Anfragen
- jede neue Kennzahl mit Handtest
- `git status` zeigt nichts unter `services/k3-train/data/`
- Zahlen im README stammen aus `data/k3/metrics.json` oder `docs/runs/`
- CI grün

Abnahme der GNN (Schritt 4, erfüllt): Vergleich GNN gegen Baseline gemessen und berichtet, nicht „GNN schlägt Baseline“ (Regel 5).

- Gleicher Datensatz, gleicher zeitlicher Split, gleiche gelabelte Testknoten, gleiche Metriken, Seed 42.
- Hyperparameter nur am Validierungsteil gewählt (synthetisch Zeitschritte 18 bis 21, Elliptic 30 bis 34), nie am Testteil. Normalisierung und Klassengewichte nur aus Trainingsdaten.
- Ergebnis als Runs `gcn` und `graphsage` in `metrics.json` und `docs/runs/`, `scoreGnn` in `nodes.json` gefüllt.
- Im README ehrlich eingeordnet, auch wenn das GNN die Baseline nicht schlägt. Gleiches gilt für die Case-Study: das gemessene Ergebnis, kein Zielwert.
- Erfüllt: Runs `gcn`, `graphsage` und Kontrolle `mlp`, `scoreGnn` aus GraphSAGE, README eingeordnet, `verify` unter Windows und in der CI unter Linux mit Exit 0, Seite mit Vertrag 3 und Einordnung mit Bedingungen, alle Web-Prüfungen grün, CI grün.
- Regel der Seite für vergleichende Sätze: Ein Unterschied zweier PR-AUC-Werte gilt nur dann als Rangfolge, wenn er größer ist als `randomPrAucQ95` − `randomPrAucExpected` (derzeit 0,0239); sonst sagt die Seite, dass er nicht belastbar ist. Der Satz „Gegenüber dem MLP gleicher Größe ohne Kanten kommt … der größere Teil des Vorsprungs … aus der Nachbarschaft“ erscheint nur, wenn jedes Graph-Verfahren um mehr als diesen Abstand über dem MLP liegt und dieser Gewinn den Gewinn des MLP über die bessere Baseline um mehr als den Abstand übersteigt. Er gilt nur gegenüber dem MLP, weil ein starkes überwachtes Verfahren ohne Kanten fehlt (Grenzen). Auch der Satz Isolation Forest gegen Z-Scores folgt dem Abstand. Überschrift und Schluss des Homophilie-Vorbehalts sowie der Punkt „Eingebaute Nachbarschaft“ unter „Grenzen“ sprechen nur dann von einem gemessenen Vorsprung, wenn jedes Graph-Verfahren um mehr als den Abstand über der besseren Baseline und über dem MLP liegt; sonst im Konjunktiv. Die Satzbausteine stehen in `lib/netzradar/assessment.ts` mit Handtests, die E2E prüft sie gegen `metrics.json`. Das ist eine grobe Schwelle, kein Test (offener Punkt unten).

## Offene Entscheidungen

1. **Datensatz für die öffentliche Demo: IBM-AML oder synthetisches Netz.** Derzeit ist nur das synthetische Netz veröffentlicht. IBM-AML (CDLA-Sharing-1.0) trägt das Label je Transaktion, also je Kante zwischen Konten; die Methodik ist aber Knotenklassifikation. Optionen für die Abbildung:
   - a) Transaktionen als Knoten, wie bei Elliptic: Kante t₁ → t₂, wenn das Empfängerkonto von t₁ das Senderkonto von t₂ ist und t₂ in einem begrenzten Zeitfenster nach t₁ liegt. Das Label bleibt das Original-Tag, Pipeline und zeitlicher Split bleiben unverändert. Aufwand: Graphaufbau, Kantenzahl über das Fenster begrenzen, Zeitschritte festlegen.
   - b) Konten als Knoten, Label illicit bei mindestens einer Laundering-Transaktion. Einfach, aber das Label ist abgeleitet, und Konten existieren über den ganzen Zeitraum. Ein zeitlicher Split hätte dann Knoten auf beiden Seiten, im Konflikt mit Regel 2.
   - c) Kantenklassifikation. Passt zu den Labels, ändert aber Baseline, Metriken, Datenvertrag und Export.
   - Empfehlung: a), weil Label, Methodik und Split unverändert bleiben. Bis zur Entscheidung bleibt das synthetische Netz die öffentliche Demo.
2. **Live-Demo auf Hugging Face Spaces** (Free CPU, pausiert bei Inaktivität): ja oder nein. Die Seite selbst bleibt statisch (Regel 6); ein Space wäre höchstens ein externer Link.
3. **Starkes überwachtes Verfahren ohne Kanten** (Random Forest oder HistGradientBoosting auf lokalen Merkmalen plus Graphmaßen, gleicher Split, gleiche Auswahl am Validierungsteil, Seed 42). Neues Verfahren, also Vertragsänderung. Bis dahin nennt „Grenzen“ die Lücke, und der Satz zur Nachbarschaft gilt nur gegenüber dem MLP.
4. **Seed-Streuung auf der Seite** (`seedSpread` in `metrics.json`, Vertrag v4, Spalte „Spanne über 5 Seeds“) und **Startknoten nach GNN-Score** als Umschalter. Beides ändert den Export.
5. ~~**Kicker „Projekt K1/K2/K3“**~~ Entschieden am 08.10.2026: „Projekt 01/02/03“ wie auf den Projektkarten, in allen drei Repos umgesetzt (`PROJECTS.kicker`, Eyebrow der Projektseite, Kleintext im Vorschaubild).
6. **CLAUDE.md öffentlich lassen** (bereinigt, Standard) oder aus dem Repo nehmen und nur eine kurze Architekturbeschreibung veröffentlichen.
7. **Impressum**: ladungsfähige Anschrift für K1 bis K3.
8. **Begriffe**: „auffällig“ steht auf der Seite für das Label illicit und zugleich für hohe Scores; eindeutige Begriffe für Label und Modellausgabe wählen.

Entschieden: Repo öffentlich seit 07.10.2026; CI trainiert die GNN voll (07.10.2026, Job `train` rund 7 min, damit ist die Laufzeitfrage erledigt).

Weitere offene Punkte:

- Generator-Annahmen fachlich bestätigen (`localShift` 0,5, gutartiger Hintergrund mit Hubs aus Preferential Attachment). Die GNN-Ergebnisse sind dagegen gemessen; die Parameter wurden weder nach dem ersten Ergebnis noch nach den GNN-Läufen nachjustiert.
- Baseline mit gemittelten Nachbarmerkmalen (Isolation Forest auf lokalen Merkmalen plus Mittelwert der Nachbarmerkmale, ohne Labels) ist nicht umgesetzt; als eigener Run `iforestNeighbours` wäre sie eine Vertragsänderung (neues Verfahren). Sensitivität gegen Homophilie (Tarnkanten, gradtreue Umverdrahtung) für Schritt 5 oder v2.
- GraphSAGE wählte auf dem Standardnetz Epoche 300, also die Obergrenze `maxEpochs`; das Early Stopping griff nicht. Ob die Obergrenze erhöht wird, ist eine Entscheidung vor einem neuen Lauf und darf sich nicht am Testergebnis orientieren.
- Golden-Werte des GNN auf `SMALL_CONFIG` (wie `test_small_network_matches_golden_values`) sind jetzt möglich, weil der CI-Lauf vom 07.10.2026 gezeigt hat, dass Windows und Linux innerhalb von 1e-4 übereinstimmen.
- Mindestabstand für vergleichende Sätze auf der Seite (Abschnitt „Definition of Done“). Optionen: so lassen (grobe Schwelle aus der Streuung zufälliger Rangfolgen); gepaarter Bootstrap über die Testknoten mit Konfidenzintervall für jede Differenz (neue Kennzahl, Vertragsänderung); Seed-Streuung aus `docs/runs/` in `metrics.json` übernehmen (Vertragsänderung). Empfehlung: Bootstrap in Schritt 5 zusammen mit der Case-Study, bis dahin die Schwelle.
- Die Seed-Streuung (Seeds 42 bis 46) steht nur in `docs/runs/` und im README; die Seite zeigt sie nicht, weil sie nur `data/k3/` einbindet (Regel 6), verlinkt aber auf `docs/runs`.
- Ob aggregierte Elliptic-Kennzahlen (PR-AUC je Verfahren) ins README dürfen; derzeit wird aus Elliptic gar nichts veröffentlicht.
- `OPERATOR.lastUpdated` steht in K1, K2 und K3 auf 07.10.2026 (gemeinsames Datum der Rechtstexte).
- Die Ausrichtung der Elliptic-Merkmale (165 Spalten, die letzten 72 aggregiert) beim ersten echten Lauf bestätigen.
- Umgang mit alten `docs/runs/`-Dateien nach Neuläufen an anderen Tagen.
- Abhängigkeiten beobachten: `pnpm audit` meldet braces ≤ 3.0.3 (GHSA-vfj7-8cjw-p6xm, high) nur über `eslint-config-next` › `@next/eslint-plugin-next` › `fast-glob` › `micromatch`, also nur beim Linting, ohne Patch. Mit dem nächsten Update von `eslint-config-next` erneut prüfen.
- GitHub-Einstellungen (vom Autor selbst): Secret Scanning mit Push Protection, Dependabot-Alerts und Security Updates einschalten; optional ein Ruleset für `main` mit den Pflicht-Checks `web`, `train` und `train-core`.
