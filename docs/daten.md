# Datensätze: Bezug, Lizenzen, Ablage

Grundsatz: Rohdaten kommen nie ins Repository. `services/k3-train/data/raw/` und `services/k3-train/data/processed/` stehen in der Root-`.gitignore` und in `services/k3-train/.gitignore`. Der Datenordner, unter dem `raw/` und `processed/` liegen, lässt sich mit `--data-dir` (Befehle `data`, `baseline`, `gnn`, `export`, `runs`) oder der Umgebungsvariablen `K3_DATA_DIR` aus dem Repository herausverlegen; ohne beides gilt `services/k3-train/data/`. Veröffentlicht werden nur Code, Metriken, Plots und Stichproben synthetischer Daten, und zwar nur aus dem eigenen Netz; IBM-AML ist zwar ebenfalls synthetisch, bleibt aber lokal (siehe unten). Bei widersprüchlichen Lizenzangaben gilt die restriktivere.

| Datensatz | Lizenz | Bezug | Ablage lokal | Veröffentlicht | Stand 16.10.2026 |
|---|---|---|---|---|---|
| Synthetisches Transaktionsnetz | MIT | `services/k3-train/src/k3_train/synth.py` | wird erzeugt, aufbereitet unter `services/k3-train/data/processed/synthetic/` | Ausschnitt und Kennzahlen in `data/k3/`, Protokolle in `docs/runs/` | umgesetzt, öffentliche Demo |
| IBM Transactions for Anti-Money Laundering (HI-Small) | CDLA-Sharing-1.0 (Daten), Apache-2.0 (Code in IBM/AML-Data) | Kaggle `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`, Datei `HI-Small_Trans.csv` | `<Datenordner>/raw/ibm-aml/` mit Datenordner außerhalb synchronisierter Ordner, Rohdatei auch per `--raw-dir` | nichts | entschieden: nur lokal; Lader (Transaktionen als Knoten) nur an erfundener Fixture getestet |
| Elliptic Bitcoin Dataset | CC BY-NC-ND 4.0 | Kaggle `ellipticco/elliptic-data-set` | `<Datenordner>/raw/elliptic/` mit Datenordner außerhalb synchronisierter Ordner (`--data-dir` oder `K3_DATA_DIR`), Rohdaten auch per `--raw-dir` | nichts | Lader nur an erfundener Fixture getestet |

Welche Datensätze veröffentlicht werden dürfen, steht im Code: `services/k3-train/src/k3_train/datasets.py` setzt `publishable` für `synthetic` auf `True`, für `elliptic` und `ibm-aml` auf `False`; `withheld_reason` enthält den Grund, den `k3-train export` beim Abbruch ausgibt. Nur freigegebene Datensätze gelangen nach `data/k3/` und `docs/runs/`.

## Was ins Repository darf

Darf:

- Code unter `services/k3-train/src/` und `services/k3-train/tests/`, einschließlich des Generators `synth.py`
- `data/k3/nodes.json`, `edges.json` und `metrics.json`, aber nur aus freigegebenen Datensätzen; derzeit nur aus dem synthetischen Netz
- `docs/runs/*.json`, ebenfalls nur aus freigegebenen Datensätzen
- Plots und Screenshots aus synthetischen Daten (`docs/screenshots/`)
- erfundene Test-Fixtures mit Herkunftsvermerk: `services/k3-train/tests/fixtures/elliptic_mini/` und `services/k3-train/tests/fixtures/ibm_aml_mini/`, je mit `PROVENANCE.md`. Diese Dateien bilden nur das Dateiformat von Elliptic bzw. von `HI-Small_Trans.csv` nach und enthalten keine echten Transaktionen und keine Zeile aus den Originaldaten.

Darf nicht:

- alles unter `services/k3-train/data/raw/`, also jede Rohdatei aus jedem Datensatz
- alles unter `services/k3-train/data/processed/`, auch die aufbereiteten synthetischen Dateien; sie lassen sich jederzeit mit `k3-train data --synth` neu erzeugen
- aus Elliptic irgendetwas: keine Rohdaten, keine bearbeiteten Fassungen, keine Ausschnitte, keine Knoten-IDs, keine Merkmalswerte, keine Scores je Knoten, keine Layouts, keine Laufprotokolle. Der Code erzwingt das: `k3-train export --dataset elliptic` endet mit Exit 2, und `k3-train baseline --dataset elliptic` schreibt seine Protokolle nach `<Datenordner>/processed/elliptic/runs/` statt nach `docs/runs/`.
- aus IBM-AML nichts: keine Rohdaten, keine aufbereiteten Knoten, Kanten oder Merkmale, keine Ausschnitte, keine Scores je Transaktion, keine Layouts, keine Laufprotokolle. Die Lizenz würde das Teilen unter gleichen Bedingungen erlauben; entschieden ist trotzdem, dass die öffentliche Demo das synthetische Netz bleibt (Abschnitt IBM-AML unten). Der Code erzwingt das wie bei Elliptic: `k3-train export --dataset ibm-aml` endet mit Exit 2, und `baseline` und `gnn` schreiben ihre Protokolle nach `<Datenordner>/processed/ibm-aml/runs/`.

Offen: Für Elliptic sind höchstens aggregierte Ergebnismetriken mit Verweis auf die Originalquelle vorgesehen (wie in Abschnitt 6 der Nutzungsbedingungen). Derzeit wird aus Elliptic gar nichts veröffentlicht, auch keine Kennzahl. Ob einzelne aggregierte Werte, etwa die PR-AUC je Verfahren, ins README dürfen, entscheidet der Autor.

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
- Labels: Musterknoten sind mit Wahrscheinlichkeit `illicitVisibility` als illicit gelabelt, sonst `unknown`. Gutartige Knoten sind mit Wahrscheinlichkeit `licitVisibility` als licit gelabelt, sonst `unknown`. Unter `unknown` liegen also bewusst verdeckte Musterknoten. Die Seite nennt die Labels „illegal“ (`illicit`), „legal“ (`licit`) und „ohne Label“ (`unknown`); Modellausgaben heißen dort nur Score, hoher Score oder markiert (Begriffsregel in `CLAUDE.md`).
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

Kennzahlen laut Weber et al. (2019, arXiv:1908.02591):

- 203.769 Transaktionsknoten, 234.355 gerichtete Kanten, 49 Zeitschritte. Die Zeitschritte liegen im Abstand von etwa zwei Wochen; jeder enthält eine zusammenhängende Komponente von Transaktionen, die innerhalb von unter drei Stunden auf der Blockchain erschienen.
- Labels nach Elmougy und Liu (arXiv:2306.06108): 4.545 illicit (2 %), 42.019 licit (21 %), 157.205 unknown (77 %); gelabelt sind 46.564 von 203.769 Knoten. Unter den gelabelten Knoten sind 4.545 / 46.564 ≈ 9,8 % illicit.
- Merkmalszahl offen: Weber et al. (2019) nennen 166 Merkmale (94 lokal einschließlich des Zeitschritts, 72 aggregiert), in der Literatur wird teils 165 oder 167 gezählt.

Wie der Lader die Merkmale zählt und benennt (`load_elliptic` in `services/k3-train/src/k3_train/load.py`): Spalte 1 der Merkmalsdatei ist die Transaktions-ID, Spalte 2 der Zeitschritt, alle weiteren Spalten sind Merkmale. Der Lader erwartet davon 165 und geht davon aus, dass die letzten 72 die über die Nachbarn aggregierten Merkmale sind (Weber et al. 2019: 94 lokal, 72 aggregiert; der Zeitschritt zählt dort zu den 94 lokalen). Die ersten 93 heißen `f_001` bis `f_093` und bilden den Merkmalssatz `local`, die letzten 72 heißen `a_001` bis `a_072` und gehen in keine Baseline ein; sonst sähen die „robusten Z-Scores (Einzelmerkmale)“ Nachbarschaftsinformation. Hat die Datei eine andere Spaltenzahl, bricht der Lader mit einer Meldung ab, weil die Trennung dann nicht geprüft ist. Die gezählten Zahlen erscheinen nach `k3-train data --dataset elliptic` in der Ausgabe („Merkmale (gezählt)“ mit lokal und aggregiert) und stehen in `<Datenordner>/processed/elliptic/meta.json` (`featureColumns`, `aggregatedColumns`, `notes`). Die Annahme zur Spaltenfolge ist nur an einer erfundenen Fixture getestet. Gezählte Zahl nach dem ersten Lauf: [Platzhalter: noch nicht gezählt].

Bezug:

1. Kaggle-Konto nötig (für die Läufe dieses Projekts das Konto des Autors), Datensatz `ellipticco/elliptic-data-set` im Browser oder per Kaggle-CLI herunterladen.
2. Die drei CSV-Dateien direkt in den Zielordner legen, gegebenenfalls aus einem Unterordner des Archivs herausnehmen. Der Lader erwartet genau diese Namen:
   - `elliptic_txs_features.csv`: ohne Kopfzeile, Transaktions-ID, Zeitschritt, Merkmale
   - `elliptic_txs_classes.csv`: Transaktions-ID und Klasse (`1` = illicit, `2` = licit, `unknown`)
   - `elliptic_txs_edgelist.csv`: Kantenliste aus zwei Transaktions-IDs
3. Zielordner: `<Datenordner>/raw/elliptic/` mit einem Datenordner außerhalb synchronisierter Ordner, siehe unten. Ohne Datenordner gilt `services/k3-train/data/raw/elliptic/`.
4. `uv run k3-train data --dataset elliptic --data-dir <Datenordner>`; fehlt eine Datei, nennt der Befehl sie und endet mit Exit 2.
5. `uv run k3-train baseline --dataset elliptic --data-dir <Datenordner>`. Split laut `split.py`: Training 1 bis 34 (Validierung 30 bis 34), Test 35 bis 49. Für Zeitschritte mit mehr als 5.000 Knoten wird die Betweenness mit 1.000 gezogenen Quellknoten geschätzt (Seed 42).

Lizenz: CC BY-NC-ND 4.0, also nur nicht-kommerziell und keine Weitergabe bearbeiteter Fassungen. Das Portfolio ist nicht-kommerziell; die ND-Klausel wird eingehalten, indem aus dem Datensatz nichts veröffentlicht oder weitergegeben wird. Lokal sollen Roh- und aufbereitete Daten außerhalb synchronisierter Ordner liegen, siehe „Ablage außerhalb synchronisierter Ordner“. Die Hugging-Face-Spiegelung `yhoma/elliptic-bitcoin-dataset` nennt MIT. Das widerspricht der offiziellen Angabe; es gilt die restriktivere Lizenz, und die Spiegelung wird nicht als Quelle genutzt. Bei Nutzung wird die Originalquelle genannt (Weber et al. 2019, arXiv:1908.02591).

Laufzeit eines Elliptic-Volllaufs: [Platzhalter: noch nicht gemessen].

## IBM Transactions for Anti-Money Laundering (CDLA-Sharing-1.0, nur lokal)

Entscheidung vom 16.10.2026 (vom Autor an Claude delegiert): Die öffentliche Demo bleibt das synthetische Netz. Es ist in der CI ohne Download, ohne Konto und ohne Lizenzfragen reproduzierbar, und `verify` kann jede veröffentlichte Zahl nachrechnen. IBM-AML wird nur lokal ausgewertet, mit Transaktionen als Knoten (Option a in `CLAUDE.md`, „Offene Entscheidungen“). Der Lader steht in `services/k3-train/src/k3_train/ibm_aml.py` und ist bisher nur an einer erfundenen Fixture getestet; ein Lauf auf den echten Daten steht aus.

### Quelle und Kennzahlen

- Altman, E., Blanuša, J., von Niederhäusern, L., Egressy, B., Anghel, A., Atasu, K. (2023). Realistic Synthetic Financial Transactions for Anti-Money Laundering Models. arXiv:2306.16424. Die Daten stammen aus einem agentenbasierten Generator (AMLworld), der an echten Transaktionen kalibriert ist; sie enthalten keine echten Konten.
- Kaggle-Beschreibung des Datensatzes (abgerufen am 16.10.2026 über die öffentliche Metadaten-Schnittstelle `www.kaggle.com/api/v1/datasets/view/…`, Version 8 vom 08.07.2025): sechs unabhängige Datensätze in zwei Gruppen, HI mit höherem und LI mit niedrigerem Anteil an Laundering, je Small, Medium und Large. Small umfasst den 1. bis 10. September (10 Tage), HI-Small rund 5 Mio. Transaktionen und rund 5,1 Tsd. Laundering-Transaktionen. Empfohlen wird eine zeitliche Aufteilung 60 % / 20 % / 20 %. Einige Transaktionen liegen nach dem angegebenen Zeitraum; laut der Beschreibung sind sie alle Laundering.
- Dateien je Variante laut Dateiliste derselben Schnittstelle (16.10.2026): `*_Trans.csv` (Transaktionen), `*_Patterns.txt` (Laundering-Transaktionen, die einem von 8 Mustern folgen; laut Beschreibung folgt nicht jede Laundering-Transaktion einem Muster) und seit Version 8 `*_accounts.csv`. `HI-Small_Trans.csv` hat 475.664.283 Bytes, alle 18 Dateien zusammen 41.608.549.656 Bytes. Der Lader braucht nur `HI-Small_Trans.csv`.
- Unklar ist die Zahl der Laundering-Transaktionen in HI-Small: Altman et al. (2023, arXiv-Fassung v3, Tabelle 4) nennen 3,6 Tsd. bei 515 Tsd. Konten, 5 Mio. Transaktionen und einer Rate von einer Laundering-Transaktion je 981. 5 Mio. / 981 ≈ 5,1 Tsd. passt zur Kaggle-Angabe, nicht zu 3,6 Tsd. Gezählt wird beim ersten Lauf; `k3-train data` gibt die Labelzahlen und die Zahl der entfernten Transaktionen nach dem Zeitraum mit Is Laundering 1 aus.

### Spalten von `HI-Small_Trans.csv`

Recherchiert am 16.10.2026, nicht an der Originaldatei geprüft. Der Lader erwartet diese Kopfzeile in dieser Reihenfolge:

| Nr. | Spalte | Bedeutung im Lader |
|---|---|---|
| 1 | `Timestamp` | Zeitpunkt, Format `JJJJ/MM/TT HH:MM` (`%Y/%m/%d %H:%M`), minutengenau |
| 2 | `From Bank` | Bank des Senders |
| 3 | `Account` | Konto des Senders |
| 4 | `To Bank` | Bank des Empfängers |
| 5 | `Account` | Konto des Empfängers; gleicher Name wie Spalte 3, pandas liest ihn als `Account.1` |
| 6 | `Amount Received` | empfangener Betrag |
| 7 | `Receiving Currency` | Währung des empfangenen Betrags |
| 8 | `Amount Paid` | gezahlter Betrag |
| 9 | `Payment Currency` | Währung des gezahlten Betrags |
| 10 | `Payment Format` | Zahlungsart, etwa ACH, Wire, Cheque, Credit Card, Cash, Bitcoin, Reinvestment |
| 11 | `Is Laundering` | 1 = Laundering, 0 = kein Laundering |

Quellen:

- IBM/Multi-GNN auf GitHub, Datei `format_kaggle_files.py` (abgerufen am 16.10.2026): Code aus dem Umfeld der Autoren, der genau diese Kaggle-Dateien aufbereitet (die README des Repos nennt `HI-Small_Trans.csv` als Beispiel). Er liest `Timestamp`, `From Bank`, `To Bank`, `Amount Received`, `Receiving Currency`, `Amount Paid`, `Payment Currency`, `Payment Format` und `Is Laundering` über den Namen, die beiden Konten über ihre Position (Spalte 3 und 5) und den Zeitstempel mit `%Y/%m/%d %H:%M`. Ein Konto ist dort Bank und Kontonummer zusammen.
- Dass beide Kontospalten `Account` heißen, steht dort nicht. Es stammt aus einer Sekundärquelle: Eine Spaltenliste mit `Account` und `Account.1` in einem Medium-Beitrag („Money laundering transactions Detection“), am 16.10.2026 nur über Suchergebnisse eingesehen, weil die Seite beim Abruf mit HTTP 403 antwortete. `Account.1` ist die Form, die pandas einer doppelten Spalte gibt. Der Lader nimmt deshalb für Spalte 3 und 5 sowohl `Account` als auch `Account.<Zahl>` an.
- Die Kaggle-Metadaten enthalten keine Spaltenliste (`columns` ist leer), das README von IBM/AML-Data ebenfalls nicht.
- Ob die Kopfzeile zeichengenau stimmt, zeigt erst der erste echte Lauf. Der Lader prüft sie vor dem Einlesen und nennt fehlende, unerwartete oder vertauschte Spalten samt der erwarteten Kopfzeile.

### Abbildung auf Knoten, Kanten und Zeitschritte

- Knoten: eine Transaktion je Zeile. Die ID ist `t` plus die Nummer der Datenzeile mit 7 Stellen, etwa `t0000001`; sie unterscheidet sich damit von den synthetischen IDs `tx000001`. Ein Konto ist Bank und Kontonummer zusammen, dieselbe Kontonummer bei zwei Banken sind zwei Konten.
- Zeitschritt: Kalendertag, Tag 1 ist der Tag der frühesten Transaktion. Begründung: Ein Split darf keinen Zeitschritt teilen (Regel 2), und die empfohlene Aufteilung 60/20/20 lässt sich bei 10 Tagen mit ganzen Tagen annähern (6/2/2 Tage; wie gut das die Anteile der Transaktionen trifft, siehe Split unten). Stunden als Zeitschritte würden bei einem Fenster von mehreren Stunden fast jede Kette an Schrittgrenzen abschneiden. Zeitschritte aus mehreren Tagen wären möglich: Gleich lang gehen sie bei 10 Tagen nur mit 2 Tagen auf, und `ibm_aml_split(5)` ergibt Training 1 bis 4 mit Validierung 4 und Test 5, also dieselbe Aufteilung der Tage (Auswahl 1 bis 6, Validierung 7 und 8, Test 9 und 10) mit eigenem Validierungsteil. Abzuwägen sind weniger abgeschnittene Paare (das Fenster dürfte dann über jede zweite Mitternacht reichen) gegen gröbere Zeitschritte (Graphmaße über zwei Tage, Validierung und Test aus je einem Zeitschritt). Gewählt ist der Kalendertag; den Preis, die Zahl der über Tagesgrenzen nicht verbundenen Paare, gibt `data` aus.
- Bereinigung: Transaktionen nach dem dokumentierten Zeitraum von 10 Tagen werden entfernt. Laut Kaggle-Beschreibung sind sie alle Laundering; behalten, ergäben sie am Ende Zeitschritte nur aus Laundering, also einen Testteil mit geschenkter Trennung. Gezählt und ausgegeben werden die entfernten Zeilen und wie viele davon Is Laundering 1 haben. Fehlt im Zeitraum ein Tag ganz, bricht der Lader ab.
- Kante t₁ → t₂ genau dann, wenn das Empfängerkonto von t₁ das Senderkonto von t₂ ist, beide am selben Tag liegen und 0 ≤ Zeit(t₂) − Zeit(t₁) ≤ 360 Minuten gilt. Die Kante folgt also dem Geld: was auf einem Konto ankommt, kann kurz danach von dort weitergehen. Weil die Zeitstempel minutengenau sind und die Reihenfolge innerhalb einer Minute nicht feststeht, dürfen zwei Transaktionen derselben Minute einander in beide Richtungen folgen. Eine Überweisung auf dasselbe Konto ist nicht ihr eigener Nachfolger; Selbstschleifen und doppelte Kanten entstehen nicht.
- Fenster 6 Stunden: Es soll mehrere Weiterleitungen am selben Tag erfassen und deutlich kürzer als ein Zeitschritt sein, damit die Tagesgrenze wenig abschneidet. Bei gleichmäßiger Aktivität über den Tag reicht das Fenster nur bei Transaktionen aus den letzten 6 Stunden (einem Viertel des Tages) über Mitternacht. Gemittelt über alle Transaktionen liegt dann ein Achtel der Fensterlänge hinter der Tagesgrenze: (1/1440) · ∫₁₀₈₀¹⁴⁴⁰ (t − 1080)/360 dt = 180/1440 = 0,125; allgemein ist der Anteil w / 2880 bei einem Fenster von w Minuten, bei 12 Stunden also ein Viertel. Ein kürzeres Fenster verlöre Weiterleitungen mit längerer Pause. Die 6 Stunden sind eine Setzung zwischen beidem, kein gemessenes Optimum. Solche Paare werden nicht verbunden (Regel 2). `k3-train data` gibt ihre Zahl als „über Tagesgrenzen nicht verbunden“ aus, damit der Verlust gemessen statt angenommen ist. Das Fenster ist vor jedem Lauf auf echten Daten festgelegt und darf nicht nach Testergebnissen geändert werden; eine Änderung (`IbmAmlConfig` in `ibm_aml.py`) gehört vor einen Lauf und in dieses Dokument.
- Kantenzahl je Knoten: höchstens 10 Nachfolger, nämlich die frühesten im Fenster, und höchstens 10 Vorgänger, die spätesten; bei Gleichstand gewinnt die frühere Zeile. Ohne Grenze wächst die Zahl der Paare bei Konten mit vielen Transaktionen quadratisch: Ein Konto mit k eingehenden und k ausgehenden Transaktionen im selben Fenster ergäbe bis zu k² Kanten. Mit Grenze hat jeder Knoten höchstens 10 ausgehende und 10 eingehende Kanten, der Graph also höchstens 10 · n Kanten bei n Transaktionen. Breite Fan-in- oder Fan-out-Muster mit mehr als 10 Gliedern werden dadurch nur teilweise verbunden. `k3-train data` gibt die Zahl der Paare im Fenster und die durch die Grenze entfernten aus.
- Label: `Is Laundering` 1 wird `illicit`, 0 wird `licit`; `unknown` gibt es in IBM-AML nicht. Auf der Seite hießen sie „Label illegal“ und „Label legal“ (Begriffsregel), die Seite zeigt IBM-AML aber nicht.
- Split laut `split.py` (`ibm_aml_split`): 60/20/20 nach Tagen, kaufmännisch auf ganze Tage gerundet. Bei 10 Tagen: Training 1 bis 8, darin Validierung 7 und 8 (die Auswahl lernt auf 1 bis 6), Test 9 und 10. Die Kaggle-Beschreibung nennt nur eine chronologische Aufteilung 60 % / 20 % / 20 % ohne Einheit. Altman et al. (2023, arXiv-Fassung v3, Abschnitt 4, „Data Split“, abgerufen am 16.10.2026) teilen die nach Zeitstempel geordneten Transaktionen an zwei Zeitpunkten t₁ und t₂, also nach Transaktionsanteilen. Ein solcher Zeitpunkt läge mitten in einem Tag und teilte einen Zeitschritt; hier wird deshalb nach ganzen Tagen geteilt (Regel 2), was die Anteile nur annähert. Wie viele Transaktionen tatsächlich in Training ohne Validierung, Validierung und Test liegen, gibt `k3-train data` als „Anteile nach Transaktionen“ aus (Fixture: 18, 4 und 4 von 26); für HI-Small wird das beim ersten Lauf gezählt. Da keine Kante eine Tagesgrenze überquert, verbindet auch keine Kante Training und Test; `check_split` prüft das wie bei den anderen Datensätzen.
- Die Einstellungen der Graphkonstruktion und alle Zählungen stehen in `<Datenordner>/processed/ibm-aml/meta.json` unter `notes`. Im Datensatzblock der Laufprotokolle stehen sie nicht (dort ist `generator` wie bei Elliptic `null`); maßgeblich sind die festen Werte in `IbmAmlConfig`.

Lokale Merkmale (nur aus der eigenen Zeile; Floats auf 6 Nachkommastellen gerundet):

| Merkmal | Bedeutung |
|---|---|
| `f_log_amount_paid`, `f_log_amount_received` | log(1 + Betrag) |
| `f_cross_currency` | 1, wenn Zahlungs- und Empfangswährung verschieden sind |
| `f_same_bank` | 1, wenn Sender- und Empfängerbank gleich sind |
| `f_self_transfer` | 1, wenn Sender- und Empfängerkonto gleich sind |
| `f_whole_amount_paid` | 1, wenn der gezahlte Betrag ganzzahlig ist |
| `f_time_sin`, `f_time_cos` | Uhrzeit auf dem Einheitskreis: sin und cos von 2π · Minute des Tages / 1440, damit 23:59 und 00:00 nah beieinander liegen |
| `f_format_*`, `f_paid_*`, `f_received_*` | One-Hot-Spalten für Zahlungsart, Zahlungs- und Empfangswährung, eine Spalte je vorkommendem Wert |

Kein Leck: Kein Merkmal nutzt Labels, andere Transaktionen, die Kontohistorie oder spätere Zeitpunkte, und die Kontonummern gehen nicht als Zahl ein. Die Liste der Kategorien stammt aus allen Zeilen; das verrät nichts über andere Zeilen, weil jede Zeile nur ihren eigenen Wert kodiert, und eine Kategorie, die erst im Testteil vorkommt, ist im Training überall 0. Tests prüfen, dass vertauschte Labels weder Merkmale noch Kanten ändern und dass die Tage 1 bis 8 dieselben Merkmale und Kanten haben, wenn die Tage 9 und 10 fehlen. Die Zahl der Merkmale hängt an den vorkommenden Kategorien (Fixture: 8 + 15 = 23) und wird beim ersten Lauf gezählt. Graphmaße und Nachbarschaft kommen wie bei den anderen Datensätzen erst in der Pipeline dazu.

### Bezug, Ablage und Befehle

1. Kaggle-Konto nötig (für die Läufe dieses Projekts das Konto des Autors). Vom Datensatz `ealtman2019/ibm-transactions-for-anti-money-laundering-aml` nur `HI-Small_Trans.csv` herunterladen, im Browser oder per Kaggle-CLI; das ganze Archiv ist rund 41,6 GB groß.
2. Die Datei nach `<Datenordner>/raw/ibm-aml/HI-Small_Trans.csv` legen, mit einem Datenordner außerhalb synchronisierter Ordner (siehe unten), oder ihren Ordner per `--raw-dir` angeben.
3. Befehle unter Windows (PowerShell, in `services/k3-train`):

   ```powershell
   $env:K3_DATA_DIR = "$env:LOCALAPPDATA\netzradar\data"
   New-Item -ItemType Directory -Force "$env:K3_DATA_DIR\raw\ibm-aml"
   uv run k3-train data --dataset ibm-aml
   uv run k3-train baseline --dataset ibm-aml
   uv run --extra gnn k3-train gnn --dataset ibm-aml
   ```

   Mit make: `make data DATASET=ibm-aml DATA_DIR=<Datenordner>` (Rohdatei an anderer Stelle: `RAW_DIR=<Ordner>`), dann `make baseline DATASET=ibm-aml DATA_DIR=<Datenordner>` und `make gnn DATASET=ibm-aml DATA_DIR=<Datenordner>`. Ohne Datei endet `data` mit Exit 2 und nennt Datei, Bezug und Lizenz; bei unpassender Kopfzeile, ungültigen Zeitstempeln, Beträgen oder Labels, leeren Werten oder einem fehlenden Tag mit Exit 1 und einer Meldung.
4. `uv run k3-train export --dataset ibm-aml` endet mit Exit 2: Der Datensatz ist in `datasets.py` nicht freigegeben.

Lizenz: Die Daten stehen unter CDLA-Sharing-1.0 (Kaggle-Metadaten: „Community Data License Agreement - Sharing - Version 1.0“; README von IBM/AML-Data), der Code im Repo IBM/AML-Data unter Apache-2.0. Wer Daten oder bearbeitete Daten veröffentlicht, muss sie laut Lizenztext (cdla.dev/sharing-1-0, abgerufen am 16.10.2026) unter derselben, unveränderten Lizenz veröffentlichen (§3.1 a, §3.3), bearbeitete Dateien deutlich als geändert kennzeichnen (§3.1 b) und die Nennung der Datenanbieter erhalten (§3.1 c). Für Ergebnisse (Results), etwa eine PR-AUC je Verfahren, sieht die Lizenz keine Pflichten vor (§3.5). Der Export ist trotzdem gesperrt: `data/k3/` steht unter MIT, ein Ausschnitt wäre eine bearbeitete Fassung unter einer anderen Lizenz, und die öffentliche Demo bleibt das synthetische Netz. Ob einzelne Kennzahlen eines lokalen Laufs ins README dürfen, ist eine eigene Entscheidung nach einem echten Lauf. Bei Nutzung werden Altman et al. (2023) und IBM/AML-Data genannt, wie die Kaggle-Beschreibung es erbittet.

Offen und nicht gemessen:

- Laufzeit und Speicherbedarf eines Volllaufs. HI-Small hätte nach der Bereinigung rund 5 Mio. Knoten, also rund 500 Tsd. je Zeitschritt und rund 1 Mio. in Validierung und Test (je 2 Tage). Die Graphmaße rechnet networkx je Zeitschritt (Betweenness ab 5.000 Knoten mit 1.000 gezogenen Quellknoten), die GNN trainieren den ganzen Graphen im Speicher in float64. Ob das auf einem Laptop in vertretbarer Zeit läuft, zeigt erst ein Lauf.
- Zwei Stellen, deren Speicher mit der Zahl der Test- bzw. Validierungsknoten wächst, sind seit 16.10.2026 begrenzt; die Werte für das synthetische Netz ändern sich dadurch nicht (`verify` mit Exit 0):
  - Zufallsreferenz (`random_ranking_pr_aucs` in `metrics.py`, 10.000 zufällige Rangfolgen): Gezogen wird in Blöcken von höchstens 1.000 Rangfolgen und höchstens 1 Mio. Zellen (Rangfolgen × Testknoten, `RANDOM_MAX_CELLS`). Vorher waren es immer 1.000 Rangfolgen je Block; mit `tracemalloc` gemessen 90,1 MB bei 5.000 und 180,2 MB bei 10.000 Testknoten, also rund 18 KB je Testknoten und bei 1.016.000 Testknoten hochgerechnet 18,3 GB. Mit der Grenze an einem erfundenen Vektor mit 1.016.000 Einträgen gemessen (16.10.2026, Entwicklungsrechner): Spitze 27,6 MB, rund 0,09 s je Rangfolge, für 10.000 Rangfolgen also rund 15 min. Die gezogenen Rangfolgen hängen nicht von der Blockgröße ab (`test_metrics.py` prüft das mit kleineren Grenzen auf gleiche Werte); beim synthetischen Netz (852 Testknoten) bleibt es bei Blöcken von 1.000.
  - Validierungskurve des Gradient Boosting (`validation_curve` in `boosting.py`): Die PR-AUC wird nach jedem Baum sofort berechnet, statt alle bis zu 300 Stufen der Scores gleichzeitig zu halten (vorher 300 · n · 8 Byte, bei rund 1 Mio. Validierungsknoten rund 2,4 GB je Kandidat).
- Kopfzeile, Zeitstempelformat, Zeitraum (erwartet 2022-09-01 bis 2022-09-10) und die Zahl der Laundering-Transaktionen an der Originaldatei bestätigen.
- Ob Fenster und Kantengrenze die Laundering-Muster gut abbilden, ist nicht geprüft. Möglich wäre ein Abgleich mit `HI-Small_Patterns.txt` (Anteil der Musterpaare, die als Kante erscheinen), ohne dabei Testergebnisse anzusehen.
- Weitere mögliche Quellen, bisher nicht geprüft: der AMLSim-Generator (GitHub IBM/AMLSim) und der synthetische Datensatz SAML-D. Größe und Anteil verdächtiger Transaktionen werden erst bei einer Nutzung aus der jeweiligen Originalquelle übernommen.

## Ablage außerhalb synchronisierter Ordner

Liegt das Repository in einem synchronisierten Ordner wie OneDrive, hält die `.gitignore` Rohdaten nur aus Git heraus, nicht aus der Synchronisierung. Dateien unter `services/k3-train/data/` würden dann in die Cloud gespiegelt. Für Elliptic ist das wegen der Lizenz nicht gewünscht, für IBM-AML wegen der Größe nicht (`HI-Small_Trans.csv` rund 476 MB, aufbereitet mehr).

- Einen Datenordner außerhalb wählen, zum Beispiel neben der venv, und die Rohdaten nach `<Datenordner>/raw/elliptic/` legen:

  ```powershell
  $env:K3_DATA_DIR = "$env:LOCALAPPDATA\netzradar\data"
  New-Item -ItemType Directory -Force "$env:K3_DATA_DIR\raw\elliptic"
  uv run k3-train data --dataset elliptic
  uv run k3-train baseline --dataset elliptic
  ```

- Statt der Umgebungsvariablen geht bei `data`, `baseline`, `gnn`, `export` und `runs` die Option `--data-dir <Datenordner>`; sie hat Vorrang vor `K3_DATA_DIR`. Mit make: `make data DATASET=elliptic DATA_DIR=<Datenordner>`, ebenso bei `make baseline`. Liegen die Rohdaten woanders, nimmt `data` sie per `--raw-dir <Ordner>` (make: `RAW_DIR=<Ordner>`).
- Mit gesetztem Datenordner landen Rohdaten, aufbereitete Dateien (`processed/elliptic/`: `nodes.csv` mit allen Merkmalen, `edges.csv`, `meta.json`, Baseline-Scores in `scores.csv`, Graphmaße in `measures.csv`, GNN-Ergebnisse in `gnn.json` und `gnn_scores.csv`) und die Elliptic-Laufprotokolle aller Verfahren einschließlich `gcn`, `graphsage` und `mlp` (`processed/elliptic/runs/`) dort und nicht im Repo-Ordner. Ein Test prüft, dass `data --data-dir` nichts unter `services/k3-train/data/` anlegt.
- Für die GNN auf Elliptic: `uv run --extra gnn k3-train gnn --dataset elliptic --data-dir <Datenordner>` nach `baseline`. Eingabe sind nur die 93 lokalen Merkmale `f_*`, wahlweise mit den 4 Graphmaßen; die 72 aggregierten `a_*` bleiben außen vor, damit der Vergleich mit und ohne Nachbarschaft sauber bleibt. Bisher nicht gelaufen.
- Für IBM-AML gilt dasselbe mit `ibm-aml` statt `elliptic`: Rohdatei nach `<Datenordner>/raw/ibm-aml/`, aufbereitete Dateien und Protokolle unter `<Datenordner>/processed/ibm-aml/`. Ein Test prüft an der Fixture, dass `data` und `baseline` mit `--data-dir` nur dort schreiben und `export` abbricht.
- `export` und `verify` schreiben bzw. lesen weiterhin nur `data/k3/` und `docs/runs/` im Repository; der Export von Elliptic und IBM-AML bleibt gesperrt.

## Prüfung vor jedem Commit

- `git status --short` zeigt nichts unter `services/k3-train/data/`.
- `git check-ignore -v services/k3-train/data/raw/elliptic/elliptic_txs_features.csv` und `git check-ignore -v services/k3-train/data/raw/ibm-aml/HI-Small_Trans.csv` nennen die passende `.gitignore`-Regel.
- `data/k3/metrics.json` hat `dataset.name` = `"synthetic"`, solange keine andere Freigabe beschlossen ist.
- `docs/runs/` enthält nur Dateien `<datum>_synthetic_<verfahren>.json` (Verfahren `zscore`, `iforest`, `hgb`, `gcn`, `graphsage`, `mlp`), keine mit `elliptic` oder `ibm-aml`.
- Knoten-IDs in `data/k3/nodes.json` haben die synthetische Form `tx000001`.
