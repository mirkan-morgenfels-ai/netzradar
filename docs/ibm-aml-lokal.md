# IBM-AML lokal auswerten: Anleitung für den Autor

Stand 21.10.2026. Diese Anleitung beschreibt den ersten Lauf von NetzRadar auf echten IBM-AML-Daten (Variante HI-Small) auf Ihrem Rechner. Hintergrund, Spalten, Graphkonstruktion, Split und Lizenz stehen in [`daten.md`](daten.md), Abschnitt „IBM Transactions for Anti-Money Laundering“; hier steht nur, was Sie in welcher Reihenfolge tun und worauf Sie achten müssen. Der Lader ist bisher nur an der erfundenen Fixture `services/k3-train/tests/fixtures/ibm_aml_mini` getestet.

Kurz: Datei herunterladen, außerhalb von OneDrive ablegen, `data` und `baseline` laufen lassen, Zählungen und Laufzeiten notieren, erst danach entscheiden, ob `gnn` auf diesem Rechner sinnvoll ist. Veröffentlicht wird nichts.

## 1. Kaggle-Konto und Lizenz

1. Mit Ihrem Kaggle-Konto anmelden (für die Läufe dieses Projekts das Konto des Autors; ohne Anmeldung bietet Kaggle keinen Download an).
2. Datensatzseite öffnen: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
3. Lizenz lesen: Community Data License Agreement – Sharing – Version 1.0 (CDLA-Sharing-1.0, Wortlaut unter https://cdla.dev/sharing-1-0/). Die öffentliche Metadaten-Schnittstelle von Kaggle (`www.kaggle.com/api/v1/datasets/view/ealtman2019/ibm-transactions-for-anti-money-laundering-aml`, abgerufen am 21.10.2026) nennt genau diese Lizenz und keine gesonderten Nutzungsregeln. Verlangt Kaggle beim Download trotzdem eine Zustimmung, ist es die Zustimmung zu dieser Lizenz. Was die Lizenz für dieses Projekt bedeutet, steht in `daten.md` (Absatz „Lizenz“ im IBM-AML-Abschnitt): Daten und bearbeitete Daten dürften nur unter derselben Lizenz weitergegeben werden, für Ergebnisse legt sie keine Pflichten auf (§3.5).

## 2. Nur `HI-Small_Trans.csv` herunterladen

Stand der Dateiliste (Kaggle-Schnittstelle `www.kaggle.com/api/v1/datasets/list/ealtman2019/ibm-transactions-for-anti-money-laundering-aml`, abgerufen am 21.10.2026): Version 8 vom 08.07.2025, 18 Dateien mit zusammen 41.608.549.656 Bytes. Gebraucht wird nur `HI-Small_Trans.csv` mit 475.664.283 Bytes. Laden Sie nicht das ganze Archiv herunter.

Weg im Browser: auf der Datensatzseite den Reiter „Data“ öffnen, links `HI-Small_Trans.csv` wählen und die Datei einzeln herunterladen. Kommt sie als ZIP, entpacken Sie sie.

Weg über die Kaggle-Kommandozeile (optional, nicht Teil des Projekts): API-Token in den Kontoeinstellungen von Kaggle erzeugen und als `kaggle.json` nach `%USERPROFILE%\.kaggle\` legen, nie ins Repository. Dann zum Beispiel mit uv ohne Installation ins Projekt:

```powershell
uvx kaggle datasets download ealtman2019/ibm-transactions-for-anti-money-laundering-aml -f HI-Small_Trans.csv -p "$env:LOCALAPPDATA\netzradar\data\raw\ibm-aml"
```

Entpacken Sie eine gelieferte ZIP-Datei in denselben Ordner und prüfen Sie die Größe: Die entpackte Datei sollte 475.664.283 Bytes haben (`(Get-Item ...\HI-Small_Trans.csv).Length`). Eine Prüfsumme veröffentlicht Kaggle nicht.

## 3. Ablage außerhalb von OneDrive

Das Repository liegt in OneDrive. Die `.gitignore` hält Rohdaten aus Git heraus, aber nicht aus der Synchronisierung. Legen Sie die Daten deshalb in einen Datenordner außerhalb, wie für Elliptic:

```powershell
$env:K3_DATA_DIR = "$env:LOCALAPPDATA\netzradar\data"
New-Item -ItemType Directory -Force "$env:K3_DATA_DIR\raw\ibm-aml"
```

Die Datei gehört nach `%LOCALAPPDATA%\netzradar\data\raw\ibm-aml\HI-Small_Trans.csv`. Alles, was die Läufe erzeugen (aufbereitete Dateien, Scores, Graphmaße, Laufprotokolle), landet dann unter `%LOCALAPPDATA%\netzradar\data\processed\ibm-aml\`. Statt der Variablen geht bei jedem Befehl `--data-dir <Ordner>`; eine Rohdatei an anderer Stelle nimmt `data` mit `--raw-dir <Ordner>`.

## 4. Befehle

In PowerShell, mit dem Vorspann aus `CLAUDE.local.md` (Node, uv und `UV_PROJECT_ENVIRONMENT`) und in `services/k3-train`. Lassen Sie die Schritte einzeln laufen und messen Sie die Zeit; die Ausgaben brauchen Sie für Abschnitt 6.

```powershell
$env:K3_DATA_DIR = "$env:LOCALAPPDATA\netzradar\data"
Measure-Command { uv run k3-train data --dataset ibm-aml | Out-Host }
Measure-Command { uv run k3-train baseline --dataset ibm-aml | Out-Host }
```

- `data` prüft die Kopfzeile und die Werte, entfernt die Transaktionen nach dem dokumentierten Zeitraum und gibt aus: Knoten, Kanten, Zeitschritte, Labelzahlen, entfernte Zeilen (davon mit `Is Laundering` 1), Paare im Fenster, durch die Grenze von je 10 Nachfolgern und Vorgängern entfernte Paare, über Tagesgrenzen nicht verbundene Paare und die Anteile nach Transaktionen in Training ohne Validierung, Validierung und Test. Bricht `data` mit einer Meldung zur Kopfzeile ab, ist die Kopfzeile der Originaldatei anders als recherchiert (`daten.md`, „Spalten von `HI-Small_Trans.csv`“); dann nichts von Hand umbenennen, sondern die Meldung notieren, der Lader wird angepasst.
- `baseline` rechnet robuste Z-Scores, Isolation Forest mit Graphmaßen und Gradient Boosting und schreibt die Protokolle `<Datum>_ibm-aml_{zscore,iforest,hgb}.json` nach `processed\ibm-aml\runs\`, nicht nach `docs/runs/`.
- Erst wenn `data` und `baseline` durchgelaufen sind und Abschnitt 5 nicht dagegen spricht:

  ```powershell
  Measure-Command { uv run --extra gnn k3-train gnn --dataset ibm-aml | Out-Host }
  ```

  Das schreibt `gnn.json`, `gnn_scores.csv` und die Protokolle `<Datum>_ibm-aml_{gcn,graphsage,mlp}.json` nach `processed\ibm-aml\`.
- `uv run k3-train export --dataset ibm-aml` endet absichtlich mit Exit 2: Der Datensatz ist in `datasets.py` nicht freigegeben.

Mit make (Linux) lauten die Schritte `make data DATASET=ibm-aml DATA_DIR=<Datenordner>`, `make baseline DATASET=ibm-aml DATA_DIR=<Datenordner>` und `make gnn DATASET=ibm-aml DATA_DIR=<Datenordner>`.

## 5. Laufzeit und Speicher: was zu erwarten ist

Gemessen ist noch nichts; die folgenden Zahlen sind Überschläge aus `daten.md` („Offen und nicht gemessen“) und dem Code, keine Messungen.

- Größe: rund 5 Mio. Transaktionen in 10 Tagen, also rund 500 Tsd. Knoten je Zeitschritt und rund 1 Mio. in Validierung und Test (je 2 Tage). Die Kantengrenze lässt höchstens 10 · n, also rund 50 Mio. gerichtete Kanten zu; wie viele es wirklich sind, gibt `data` aus.
- `data`: liest rund 476 MB CSV mit pandas und baut die Kanten je Tag; Arbeitsspeicher eher einige GB als einige hundert MB.
- `baseline`, Graphmaße: networkx baut je Tag einen Graphen und schätzt die Betweenness ab 5.000 Knoten mit 1.000 gezogenen Quellknoten (`EXACT_BETWEENNESS_LIMIT`, `BETWEENNESS_SAMPLES` in `graph.py`). Das sind je Tag 1.000 Breitensuchen über bis zu 500 Tsd. Knoten und die Kanten dieses Tages in reinem Python. Rechnen Sie mit Stunden für alle 10 Tage, nicht mit Minuten; das ist der größte Unsicherheitsfaktor von `baseline`.
- `baseline`, Zufallsreferenz und Gradient Boosting: seit 16.10.2026 im Speicher begrenzt (Blöcke mit höchstens 1 Mio. Zellen; PR-AUC je Stufe statt aller Stufen gleichzeitig). An einem erfundenen Vektor mit 1.016.000 Einträgen gemessen: 27,6 MB Spitze und rund 15 min für die 10.000 zufälligen Rangfolgen. Das Gradient Boosting rechnet 8 Kandidaten mit je bis zu 300 Bäumen, das Endmodell und 4 weitere Seeds auf einem Thread.
- `gnn`: GCN, GraphSAGE und MLP trainieren den ganzen Graphen auf einmal im Speicher, in float64 und ohne Mini-Batches. Überschlag für eine einzige Nachrichtenmatrix mit 64 Kanälen: 2 · E Einträge (ungerichtet) · 64 · 8 Byte. Bei 50 Mio. gerichteten Kanten sind das rund 51 GB, bei 10 Mio. rund 10 GB, und Autograd hält mehrere solcher Matrizen. Rechnen Sie das nach `data` mit der ausgegebenen Kantenzahl nach. Liegt es über dem freien Arbeitsspeicher, starten Sie `gnn` nicht; ein Lauf, der in die Auslagerungsdatei gerät, kann den Rechner für Stunden blockieren. Wie es dann weitergeht (weniger Tage, float32, Mini-Batches mit Nachbarschafts-Stichproben), ist eine offene Entscheidung und braucht Code.
- `uv run --extra gnn k3-train gnn --dataset ibm-aml --max-epochs 1` zeigt Speicherspitze und Zeit je Epoche, ohne voll zu trainieren. Die so geschriebenen Protokolle sind kein Ergebnis (nur eine Epoche) und werden vom vollen Lauf am selben Tag überschrieben; an einem anderen Tag löschen Sie sie.
- Beobachten Sie den Speicher im Task-Manager (Spalte „Arbeitsspeicher“ des Python-Prozesses) und lassen Sie während der Läufe keine anderen großen Prozesse laufen.

## 6. Was Sie festhalten

Gleich nach jedem Schritt, bevor Sie etwas anderes ändern:

1. Stand des Codes: `git -C <Repo> rev-parse HEAD` und `git status --short` (sollte nichts unter `services/k3-train/data/` zeigen).
2. Die vollständige Konsolenausgabe von `data`, `baseline` und `gnn` als Textdatei im Datenordner, etwa `processed\ibm-aml\notes-<Datum>.txt`.
3. Laufzeit je Befehl (`Measure-Command`) und die beobachtete Speicherspitze, dazu Rechner, Python- und torch-Version (stehen auch unter `environment` in den Protokollen).
4. Die Zählungen aus `meta.json` (`processed\ibm-aml\meta.json`, Schlüssel `notes`) gegen die Erwartungen in `daten.md` halten: Kopfzeile, Zeitraum 01.09. bis 10.09.2022, Zahl der Laundering-Transaktionen (Kaggle rund 5,1 Tsd., Altman et al. 2023 3,6 Tsd.), entfernte Zeilen nach dem Zeitraum.
5. Die Protokolle unter `processed\ibm-aml\runs\` bleiben dort. Sie haben dasselbe Format wie `docs/runs/` (Datensatz, Split, Seed, Hyperparameter, Datum, Kennzahlen) und sind damit reproduzierbar dokumentiert, aber nicht veröffentlicht.

Ins Repository dürfen nach dem Lauf nur Korrekturen der Beschreibung des Datensatzes in `daten.md` (zum Beispiel eine abweichende Kopfzeile oder die gezählte Zahl der Zeilen) und Laufzeit- und Speichermessungen; rechnerspezifische Messungen gehören in `CLAUDE.local.md`. Keine Scores, keine Knoten, keine Kennzahlen der Verfahren, keine Protokolle.

## 7. Was veröffentlicht werden darf

Derzeit nichts aus den Läufen: keine Rohdaten, keine aufbereiteten Dateien, keine Ausschnitte, keine Scores je Transaktion, keine Laufprotokolle und auch keine aggregierten Kennzahlen wie die PR-AUC je Verfahren. Der Code erzwingt den ersten Teil (`export --dataset ibm-aml` endet mit Exit 2, Protokolle landen nicht in `docs/runs/`); den letzten Teil müssen Sie selbst einhalten.

Offene Entscheidung (siehe `CLAUDE.md`, „Weitere offene Punkte“): Ob aggregierte IBM-AML-Kennzahlen ins README dürfen, entscheiden Sie nach einem echten Lauf. Die Lizenz stünde dem nicht entgegen (§3.5, Ergebnisse ohne Pflichten). Dafür sprechen ein zweiter, realistischerer Datensatz neben dem synthetischen Netz; dagegen, dass `verify` und die CI diese Zahlen nicht nachrechnen können (kein Download in der CI) und Regel 4 vollständige Protokolle verlangt, die dann ebenfalls öffentlich sein müssten. Empfehlung: erst veröffentlichen, wenn der Lauf vollständig ist (alle sechs Verfahren), die Zählungen zu `daten.md` passen und die Protokolle mit Commit-Stand vorliegen; dann nur eine Tabelle der PR-AUC je Verfahren mit Zufallsreferenz und Verweis auf Altman et al. (2023) und IBM/AML-Data, keine Daten.
