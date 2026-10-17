# IBM-AML-Mini-Fixture

Diese Datei ist frei erfunden. Sie bildet nur das Dateiformat von `HI-Small_Trans.csv` aus dem
Datensatz IBM Transactions for Anti-Money Laundering nach (Kopfzeile mit elf Spalten, davon zwei
mit dem Namen `Account`, Zeitstempel im Format `JJJJ/MM/TT HH:MM`). Banken, Konten, Beträge und
Labels sind ausgedacht; die Datei enthält keine Zeile und keinen Ausschnitt aus dem
Originaldatensatz. Eingebaut sind Kanten am Rand des Zeitfensters (genau 360 und 361 Minuten),
zwei Transaktionen in derselben Minute, eine Überweisung auf dasselbe Konto, gleiche Kontonummern
bei verschiedenen Banken, ein Paar über eine Tagesgrenze und zwei Zeilen nach dem dokumentierten
Zeitraum von zehn Tagen, damit Graphaufbau, Zeitschritte und Bereinigung getestet werden können.
Die Kopfzeile folgt den recherchierten Spaltennamen; Quellen und Abrufdatum stehen in
`docs/daten.md`. Ob sie zeichengenau zur Originaldatei passt, wird beim ersten echten Lauf geprüft.
