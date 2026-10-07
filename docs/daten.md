# Datensätze: Bezug, Lizenzen, Ablage

Grundsatz: Rohdaten kommen nie ins Repository. `services/k3-train/data/raw/` und `services/k3-train/data/processed/` stehen in der Root-`.gitignore` und in `services/k3-train/.gitignore`. Der Datenordner, unter dem `raw/` und `processed/` liegen, lässt sich mit `--data-dir` (Befehle `data`, `baseline`, `export`) oder der Umgebungsvariablen `K3_DATA_DIR` aus dem Repository herausverlegen; ohne beides gilt `services/k3-train/data/`. Veröffentlicht werden nur Code, Metriken, Plots und Stichproben synthetischer Daten. Bei widersprüchlichen Lizenzangaben gilt die restriktivere.

| Datensatz | Lizenz | Bezug | Ablage lokal | Veröffentlicht | Stand 06.10.2026 |
|---|---|---|---|---|---|
| Synthetisches Transaktionsnetz | MIT | `services/k3-train/src/k3_train/synth.py` | wird erzeugt, aufbereitet unter `services/k3-train/data/processed/synthetic/` | Ausschnitt und Kennzahlen in `data/k3/`, Protokolle in `docs/runs/` | umgesetzt, öffentliche Demo |
| IBM Transactions for Anti-Money Laundering | CDLA-Sharing-1.0 (Daten), Apache-2.0 (Code) | Kaggle `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`, GitHub IBM/AML-Data | `<Datenordner>/raw/ibm-aml/` | noch nichts | Lader fehlt, Datensatzentscheidung offen |
| Elliptic Bitcoin Dataset | CC BY-NC-ND 4.0 | Kaggle `ellipticco/elliptic-data-set` | `<Datenordner>/raw/elliptic/` mit Datenordner außerhalb von OneDrive (`--data-dir` oder `K3_DATA_DIR`), Rohdaten auch per `--raw-dir` | nichts | Lader nur an erfundener Fixture getestet |

Welche Datensätze veröffentlicht werden dürfen, steht im Code: `services/k3-train/src/k3_train/datasets.py` setzt `publishable` für `synthetic` auf `True` und für `elliptic` auf `False`. Nur freigegebene Datensätze gelangen nach `data/k3/` und `docs/runs/`.

## Was ins Repository darf

Darf:

- Code unter `services/k3-train/src/` und `services/k3-train/tests/`, einschließlich des Generators `synth.py`
- `data/k3/nodes.json`, `edges.json` und `metrics.json`, aber nur aus freigegebenen Datensätzen; derzeit nur aus dem synthetischen Netz
- `docs/runs/*.json`, ebenfalls nur aus freigegebenen Datensätzen
- Plots und Screenshots aus synthetischen Daten (`docs/screenshots/`)
- erfundene Test-Fixtures mit Herkunftsvermerk, etwa `services/k3-train/tests/fixtures/elliptic_mini/` mit `PROVENANCE.md`. Diese Dateien bilden nur das Elliptic-Dateiformat nach und enthalten keine echten Transaktionen.

Darf nicht:

- alles unter `services/k3-train/data/raw/`, also jede Rohdatei aus jedem Datensatz
- alles unter `services/k3-train/data/processed/`, auch die aufbereiteten synthetischen Dateien; sie lassen sich jederzeit mit `k3-train data --synth` neu erzeugen
- aus Elliptic irgendetwas: keine Rohdaten, keine bearbeiteten Fassungen, keine Ausschnitte, keine Knoten-IDs, keine Merkmalswerte, keine Scores je Knoten, keine Layouts, keine Laufprotokolle. Der Code erzwingt das: `k3-train export --dataset elliptic` endet mit Exit 2, und `k3-train baseline --dataset elliptic` schreibt seine Protokolle nach `<Datenordner>/processed/elliptic/runs/` statt nach `docs/runs/`.
- aus IBM-AML vor der Datensatzentscheidung nichts. Danach ist ein Ausschnitt in `data/k3/` möglich, mit Quellenangabe und unter CDLA-Sharing-1.0 für diese Datendateien.

Offen: Das Umsetzungsdokument (3.4, 3.10) erlaubt für Elliptic aggregierte Ergebnismetriken mit Verweis auf die Originalquelle. Derzeit wird aus Elliptic gar nichts veröffentlicht, auch keine Kennzahl. Ob einzelne aggregierte Werte, etwa die PR-AUC je Verfahren, ins README dürfen, entscheidet Dennis.

## Synthetisches Transaktionsnetz (MIT)

- Quelle: `services/k3-train/src/k3_train/synth.py`, Lizenz MIT wie das Repository.
- Erzeugung: `uv run k3-train data --synth` oder `make data SYNTH=1`. Der Generator ist über den Seed vollständig bestimmt; gleicher Seed ergibt dasselbe Netz.
- Kennzahlen laut `data/k3/metrics.json`: 12.000 Knoten, 13.757 gerichtete Kanten, 30 Zeitschritte, 12 lokale Merkmale. Labels: 292 illicit (2,43 %), 2.587 licit (21,56 %), 9.121 unknown (76,01 %).
- Homophilie laut `dataset.homophily`: Von 847 Kanten zwischen zwei gelabelten Knoten verbinden 161 zwei illicit-, 666 zwei licit-Knoten und 20 ein gemischtes Paar (Anteil gleicher Labels 0,9764). 216 von 292 illicit-Knoten haben mindestens einen illicit-Nachbarn (Kantenrichtung ignoriert), aber nur 19 von 2.587 licit-Knoten. Das folgt aus dem Aufbau unten: Muster sind untereinander verkettet und nur über einzelne Kanten an den Hintergrund angeschlossen. Ein Modell mit Nachbarschaft bekommt dieses Signal teilweise geschenkt.

Aufbau je Zeitschritt:

- Gutartiger Hintergrund: ein Baum durch Preferential Attachment. Jeder neue Knoten hängt sich an einen bisherigen Knoten mit Gewicht (Grad + 1)^`attachmentPower`, die Kantenrichtung ist zufällig. Dazu kommen Zusatzkanten im Umfang von `extraEdgeRatio` der gutartigen Knoten zwischen zufälligen Paaren.
- Eingebettete Muster: Die Zahl der Muster ist Poisson-verteilt mit Erwartungswert `illicitShare` · `nodesPerStep` / mittlere Mustergröße, mindestens `minPatternsPerStep`. Musterknoten belegen höchstens die Hälfte eines Zeitschritts. Die vier Typen sind gleich wahrscheinlich:
  - Fan-in: `fanSizeMin` bis `fanSizeMax` Zubringer mit kleinen Beträgen auf einen Sammler
  - Fan-out: ein Verteiler auf `fanSizeMin` bis `fanSizeMax` Empfänger
  - Kreis: `cycleLengthMin` bis `cycleLengthMax` Knoten, geschlossen
  - Kette: `chainLengthMin` bis `chainLengthMax` Knoten, offen
- Ein Muster schließt mit Wahrscheinlichkeit `patternLinkProbability` an das vorherige an (Ausgang auf Eingang). Sonst ist es über einzelne Kanten mit dem Hintergrund verbunden.
- Kanten verlaufen nur innerhalb eines Zeitschritts. Jeder Zeitschritt ist eine schwach zusammenhängende Komponente, wie bei Elliptic. Die Tests prüfen das.
- Labels: Musterknoten sind mit Wahrscheinlichkeit `illicitVisibility` als illicit gelabelt, sonst `unknown`. Gutartige Knoten sind mit Wahrscheinlichkeit `licitVisibility` als licit gelabelt, sonst `unknown`. Unter `unknown` liegen also bewusst verdeckte Musterknoten.
- IDs: `tx000001` bis `tx012000`, innerhalb eines Zeitschritts zufällig permutiert.

Lokale Merkmale; Floats auf 6 Nachkommastellen gerundet, damit plattformabhängige Rundungsunterschiede verschwinden:

| Merkmal | Gutartige Knoten | Musterknoten |
|---|---|---|
| `f_log_amount` | N(`logAmountMean`, `logAmountSd`) | Fan-in-Zubringer N(5,5 − 0,75; `smurfLogAmountSd`), Sammler = log der Summe; Fan-out-Verteiler N(5,5 + 0,75; 1,5), Empfänger = Verteiler + log(Dirichlet-Anteil); Kreis und Kette starten bei N(5,5 + 0,75; 1,5) und verlieren je Station `cycleFeeRate` bzw. behalten `chainRetention`, plus Rauschen N(0, `hopNoiseSd`) |
| `f_log_fee` | `feeSlope` · log Betrag + `feeIntercept` + N(0, `feeSd`) | Rauschen um 0,5 · `feeSd` erhöht |
| `f_log_fee_rate` | log Gebühr − log Größe | gleich |
| `f_num_inputs`, `f_num_outputs` | max(1, Ein- bzw. Ausgangsgrad + Poisson(`extraIoRate`)) | gleich; damit teilweise graphabhängig wie bei Elliptic |
| `f_log_size` | log(10 + 148 · Eingänge + 34 · Ausgänge) | gleich |
| `f_round_amount` | Bernoulli(`roundAmountRate`) = 0,1 | Bernoulli(0,1 · (1 + 2 · 0,5)) = 0,2 |
| `f_change_output` | Bernoulli(`changeOutputRate`) = 0,7 | Bernoulli(0,7 · (1 − 0,3 · 0,5)) = 0,595 |
| `f_max_output_share` | Beta(`maxOutputShareAlpha`, `maxOutputShareBeta`) | Fan-out-Verteiler: größter Dirichlet-Anteil |
| `f_input_amount_cv` | Gamma(`inputCvShape`, `inputCvScale`) | Fan-in-Sammler: Variationskoeffizient der Zubringerbeträge |
| `f_log_hours_since_last` | N(`logHoursMean`, `logHoursSd`) | um 0,5 · `logHoursSd` verringert |
| `f_log_address_age_days` | N(`logAgeMean`, `logAgeSd`) | um 0,5 · `logAgeSd` verringert |

Die Verschiebungen der Musterknoten skalieren mit `localShift` = 0,5; die Werte 0,75 = 0,5 · 1,5 und die Raten oben sind mit diesem Wert eingesetzt.

Generator-Parameter, wörtlich aus `metrics.json` unter `dataset.generator`:

| Parameter | Wert | Bedeutung |
|---|---|---|
| `timeSteps` | 30 | Zahl der Zeitschritte |
| `nodesPerStep` | 400 | Knoten je Zeitschritt |
| `seed` | 42 | Seed des Zufallsgenerators |
| `illicitShare` | 0.04 | angestrebter Anteil der Musterknoten |
| `minPatternsPerStep` | 1 | Mindestzahl der Muster je Zeitschritt |
| `illicitVisibility` | 0.55 | Wahrscheinlichkeit, dass ein Musterknoten als illicit gelabelt ist |
| `licitVisibility` | 0.22 | Wahrscheinlichkeit, dass ein gutartiger Knoten als licit gelabelt ist |
| `attachmentPower` | 1.0 | Exponent im Preferential Attachment |
| `extraEdgeRatio` | 0.15 | Zusatzkanten im Hintergrund je gutartigem Knoten |
| `patternLinkProbability` | 0.3 | Wahrscheinlichkeit, dass ein Muster an das vorherige anschließt |
| `fanSizeMin`, `fanSizeMax` | 5, 12 | Zubringer bei Fan-in, Empfänger bei Fan-out |
| `cycleLengthMin`, `cycleLengthMax` | 3, 6 | Kreislänge |
| `chainLengthMin`, `chainLengthMax` | 4, 8 | Kettenlänge |
| `localShift` | 0.5 | Verschiebung der Musterknoten in den lokalen Merkmalen, in Standardabweichungen |
| `logAmountMean`, `logAmountSd` | 5.5, 1.5 | log Betrag |
| `smurfLogAmountSd` | 0.5 | Streuung der kleinen Fan-in-Beträge |
| `hopNoiseSd` | 0.02 | Rauschen je Station in Kreis und Kette |
| `cycleFeeRate` | 0.01 | Abschlag je Station im Kreis |
| `chainRetention` | 0.97 | weitergegebener Anteil je Station in der Kette |
| `fanOutConcentration` | 5.0 | Dirichlet-Konzentration der Fan-out-Anteile |
| `logHoursMean`, `logHoursSd` | 3.0, 1.2 | log Stunden seit der letzten Transaktion |
| `logAgeMean`, `logAgeSd` | 5.0, 1.0 | log Alter der Adresse in Tagen |
| `roundAmountRate` | 0.1 | Anteil runder Beträge |
| `changeOutputRate` | 0.7 | Anteil mit Wechselgeld-Ausgang |
| `feeSlope`, `feeIntercept`, `feeSd` | 0.4, -2.0, 0.6 | lineares Gebührenmodell auf log-Skala |
| `extraIoRate` | 1.0 | Poisson-Rate zusätzlicher Ein- und Ausgänge |
| `maxOutputShareAlpha`, `maxOutputShareBeta` | 4.0, 1.5 | Beta-Verteilung des größten Ausgangsanteils |
| `inputCvShape`, `inputCvScale` | 2.0, 0.35 | Gamma-Verteilung des Variationskoeffizienten der Eingänge |

Die Parameter wurden vor dem ersten Ergebnis festgelegt und danach nicht nachjustiert. Für Tests gibt es eine kleine Konfiguration `SMALL_CONFIG` mit 12 Zeitschritten zu je 80 Knoten und dem Split 1 bis 8 (Validierung 7 bis 8) gegen 9 bis 12.

Split des synthetischen Netzes: Training 1 bis 21, Validierung 18 bis 21 (innerhalb des Trainings), Test 22 bis 30.

## Elliptic Bitcoin Dataset (CC BY-NC-ND 4.0, nur lokal)

Kennzahlen laut Umsetzungsdokument 3.4:

- 203.769 Transaktionsknoten, 234.355 gerichtete Kanten, 49 Zeitschritte. Die Zeitschritte liegen im Abstand von etwa zwei Wochen; jeder enthält eine zusammenhängende Komponente von Transaktionen, die innerhalb von unter drei Stunden auf der Blockchain erschienen.
- Labels nach Elmougy und Liu (arXiv:2306.06108): 4.545 illicit (2 %), 42.019 licit (21 %), 157.205 unknown (77 %); gelabelt sind 46.564 von 203.769 Knoten. Unter den gelabelten Knoten sind 4.545 / 46.564 ≈ 9,8 % illicit.
- Merkmalszahl offen: Das Umsetzungsdokument nennt 166 Merkmale (94 lokal, 72 aggregiert), in der Literatur wird teils 165 oder 167 gezählt.

Wie der Lader die Merkmale zählt und benennt (`load_elliptic` in `services/k3-train/src/k3_train/load.py`): Spalte 1 der Merkmalsdatei ist die Transaktions-ID, Spalte 2 der Zeitschritt, alle weiteren Spalten sind Merkmale. Der Lader erwartet davon 165 und geht davon aus, dass die letzten 72 die über die Nachbarn aggregierten Merkmale sind (Umsetzungsdokument 3.4: 94 lokal, 72 aggregiert; der Zeitschritt zählt dort zu den 94 lokalen). Die ersten 93 heißen `f_001` bis `f_093` und bilden den Merkmalssatz `local`, die letzten 72 heißen `a_001` bis `a_072` und gehen in keine Baseline ein; sonst sähen die „robusten Z-Scores (Einzelmerkmale)“ Nachbarschaftsinformation. Hat die Datei eine andere Spaltenzahl, bricht der Lader mit einer Meldung ab, weil die Trennung dann nicht geprüft ist. Die gezählten Zahlen erscheinen nach `k3-train data --dataset elliptic` in der Ausgabe („Merkmale (gezählt)“ mit lokal und aggregiert) und stehen in `<Datenordner>/processed/elliptic/meta.json` (`featureColumns`, `aggregatedColumns`, `notes`). Die Annahme zur Spaltenfolge ist nur an einer erfundenen Fixture getestet. Gezählte Zahl nach dem ersten Lauf: [Platzhalter: noch nicht gezählt].

Bezug:

1. Kaggle-Konto anlegen (Dennis), Datensatz `ellipticco/elliptic-data-set` im Browser oder per Kaggle-CLI herunterladen.
2. Die drei CSV-Dateien direkt in den Zielordner legen, gegebenenfalls aus einem Unterordner des Archivs herausnehmen. Der Lader erwartet genau diese Namen:
   - `elliptic_txs_features.csv`: ohne Kopfzeile, Transaktions-ID, Zeitschritt, Merkmale
   - `elliptic_txs_classes.csv`: Transaktions-ID und Klasse (`1` = illicit, `2` = licit, `unknown`)
   - `elliptic_txs_edgelist.csv`: Kantenliste aus zwei Transaktions-IDs
3. Zielordner: `<Datenordner>/raw/elliptic/` mit einem Datenordner außerhalb von OneDrive, siehe unten. Ohne Datenordner gilt `services/k3-train/data/raw/elliptic/`.
4. `uv run k3-train data --dataset elliptic --data-dir <Datenordner>`; fehlt eine Datei, nennt der Befehl sie und endet mit Exit 2.
5. `uv run k3-train baseline --dataset elliptic --data-dir <Datenordner>`. Split laut `split.py`: Training 1 bis 34 (Validierung 30 bis 34), Test 35 bis 49. Für Zeitschritte mit mehr als 5.000 Knoten wird die Betweenness mit 1.000 gezogenen Quellknoten geschätzt (Seed 42).

Lizenz: CC BY-NC-ND 4.0, also nur nicht-kommerziell und keine Weitergabe bearbeiteter Fassungen. Das Portfolio ist nicht-kommerziell; die ND-Klausel wird eingehalten, indem aus dem Datensatz nichts veröffentlicht oder weitergegeben wird. Lokal sollen Roh- und aufbereitete Daten außerhalb von OneDrive liegen, siehe „Ablage außerhalb von OneDrive“. Die Hugging-Face-Spiegelung `yhoma/elliptic-bitcoin-dataset` nennt MIT. Das widerspricht der offiziellen Angabe; es gilt die restriktivere Lizenz, und die Spiegelung wird nicht als Quelle genutzt. Bei Nutzung wird die Originalquelle genannt (Weber et al. 2019, arXiv:1908.02591).

Laufzeit eines Elliptic-Volllaufs: [Platzhalter: noch nicht gemessen].

## IBM Transactions for Anti-Money Laundering (CDLA-Sharing-1.0)

- Synthetische Transaktionsdaten, CSV mit einem Laundering-Tag je Transaktion. Bezug über Kaggle `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`; das GitHub-Repo IBM/AML-Data steht unter Apache-2.0, die Daten unter CDLA-Sharing-1.0. Teilen ist unter gleichen Bedingungen erlaubt.
- Zielordner für den künftigen Lader: `<Datenordner>/raw/ibm-aml/`. Den Lader gibt es noch nicht, und `datasets.py` kennt `ibm-aml` noch nicht; `--dataset ibm-aml` wird abgelehnt.
- Welche Dateivariante genutzt wird, legt die Datensatzentscheidung fest; der frühere README-Entwurf nannte die Variante „Small“. Dateinamen und Größen werden erst beim Bau des Laders aus den heruntergeladenen Dateien übernommen, nicht vorher geschätzt.
- Methodische Lücke: Die Labels hängen an Transaktionen, also an Kanten zwischen Konten. NetzRadar klassifiziert Knoten. Die Optionen zur Abbildung stehen in `CLAUDE.md` unter „Offene Entscheidungen“.
- Weitere Quellen laut Umsetzungsdokument 3.4: der AMLSim-Generator (GitHub IBM/AMLSim) und SAML-D (9.504.852 Transaktionen, davon etwa 0,1039 % verdächtig).

## Ablage außerhalb von OneDrive

Das Repository liegt in OneDrive. Die `.gitignore` hält Rohdaten nur aus Git heraus, nicht aus der OneDrive-Synchronisierung. Dateien unter `services/k3-train/data/` würden also in die Cloud gespiegelt. Für Elliptic ist das nicht gewünscht, für IBM-AML ebenfalls nicht nötig.

- Einen Datenordner außerhalb wählen, zum Beispiel neben der venv, und die Rohdaten nach `<Datenordner>/raw/elliptic/` legen:

  ```powershell
  $env:K3_DATA_DIR = "$env:LOCALAPPDATA\netzradar\data"
  New-Item -ItemType Directory -Force "$env:K3_DATA_DIR\raw\elliptic"
  uv run k3-train data --dataset elliptic
  uv run k3-train baseline --dataset elliptic
  ```

- Statt der Umgebungsvariablen geht bei `data`, `baseline` und `export` die Option `--data-dir <Datenordner>`; sie hat Vorrang vor `K3_DATA_DIR`. Mit make: `make data DATASET=elliptic DATA_DIR=<Datenordner>`, ebenso bei `make baseline`. Liegen die Rohdaten woanders, nimmt `data` sie per `--raw-dir <Ordner>` (make: `RAW_DIR=<Ordner>`).
- Mit gesetztem Datenordner landen Rohdaten, aufbereitete Dateien (`processed/elliptic/`: `nodes.csv` mit allen Merkmalen, `edges.csv`, `meta.json`, Baseline-Scores in `scores.csv`, Graphmaße in `measures.csv`, GNN-Ergebnisse in `gnn.json` und `gnn_scores.csv`) und die Elliptic-Laufprotokolle aller Verfahren einschließlich `gcn`, `graphsage` und `mlp` (`processed/elliptic/runs/`) dort und nicht im Repo-Ordner. Ein Test prüft, dass `data --data-dir` nichts unter `services/k3-train/data/` anlegt.
- Für die GNN auf Elliptic: `uv run --extra gnn k3-train gnn --dataset elliptic --data-dir <Datenordner>` nach `baseline`. Eingabe sind nur die 93 lokalen Merkmale `f_*`, wahlweise mit den 4 Graphmaßen; die 72 aggregierten `a_*` bleiben außen vor, damit der Vergleich mit und ohne Nachbarschaft sauber bleibt. Bisher nicht gelaufen.
- `export` und `verify` schreiben bzw. lesen weiterhin nur `data/k3/` und `docs/runs/` im Repository; der Export von Elliptic bleibt gesperrt.
## Prüfung vor jedem Commit

- `git status --short` zeigt nichts unter `services/k3-train/data/`.
- `git check-ignore -v services/k3-train/data/raw/elliptic/elliptic_txs_features.csv` nennt die passende `.gitignore`-Regel.
- `data/k3/metrics.json` hat `dataset.name` = `"synthetic"`, solange keine andere Freigabe beschlossen ist.
- `docs/runs/` enthält nur Dateien `<datum>_synthetic_<verfahren>.json` (Verfahren `zscore`, `iforest`, `gcn`, `graphsage`, `mlp`).
- Knoten-IDs in `data/k3/nodes.json` haben die synthetische Form `tx000001`.
