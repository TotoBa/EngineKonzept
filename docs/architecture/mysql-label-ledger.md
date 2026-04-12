# ExecPlan: MySQL-Label-Ledger ohne SQLite

## Status

Stand `2026-04-12`:

- ExecPlan dokumentiert
- MySQL-Ledger implementiert
- Builder und Worker auf MySQL-Ledger umgestellt
- Regressionstests und Doku aktualisiert

## Ziel

Das PGN/Stockfish-Labeling für `label_pgn_corpus` und die darauf aufbauenden
Idle-/Feedback-Pfade soll keinen SQLite-Zustand mehr verwenden.

Stattdessen soll derselbe MySQL-Server wie für die Orchestrator-Control-Plane
auch den resumierbaren Unique-Corpus-Ledger tragen.

Der neue Pfad muss:

- ohne `corpus.sqlite3` auskommen
- corpusweite FEN-Unique-Dedup pro Label-Workdir erhalten
- `train` und `verify` weiter disjunkt halten
- nach Worker-Abbrüchen sauber resume-fähig bleiben
- die exportierten `train_raw.jsonl` / `verify_raw.jsonl` unverändert auf NAS schreiben
- mit bestehenden Master-/Worker-Configs kompatibel bleiben

## Nicht-Ziele

- kein Rollout in den aktuell laufenden Produktionslauf
- kein Umbau der Phase-10-Artefaktverträge
- keine Verlagerung großer finaler JSONL-Artefakte nach MySQL
- kein zweiter Scheduler oder neuer Secret-Pfad

## Architekturvertrag

### 1. MySQL als Resume- und Dedup-Ledger

Ein neuer Label-Ledger in MySQL hält pro Label-Namespace:

- FEN-Hash
- FEN
- Split (`train` / `verify`)
- Sample-Metadaten
- Label-Status (`reserved` / `labeled`)
- gewählten Move

Der Namespace entspricht logisch dem bisherigen `work_dir`.

Damit bleibt der Lauf resume-fähig, ohne lokale SQLite-Datei.

### 2. NAS bleibt Exportziel

Große finale Artefakte bleiben auf NAS:

- `progress.json`
- `train_raw.jsonl`
- `verify_raw.jsonl`
- `summary.json`

MySQL hält den laufenden Ledger-Zustand; die exportierten Rohkorpora bleiben Dateiartefakte.

### 3. Builder und Worker nutzen denselben DB-Vertrag

`build_unique_stockfish_pgn_corpus.py` soll den Ledger direkt aus denselben
`EK_MYSQL_*`-Umgebungsvariablen lesen wie `ek_ctl.py`, `ek_worker.py` und `ek_master.py`.

Der Worker reicht keine DB-Passwörter als CLI-Argumente weiter, damit keine Secrets
in Prozesslisten oder Attempt-Details auftauchen.

### 4. Kein lokaler Hot-State mehr erforderlich

Der Worker darf für den Label-Pfad keinen SQLite-, LMDB- oder vergleichbaren
lokalen Datenbankzustand mehr voraussetzen.

Lokaler Scratch bleibt für Logs und andere Task-Arten erlaubt, aber nicht mehr
als Pflichtbestandteil des Unique-Corpus-Labelings.

## Implementierungsschritte

1. ExecPlan dokumentieren.
2. MySQL-Ledger-Tabellen und Schema-Erweiterung definieren.
3. MySQL-Ledger-Helper mit persistentem DB-Handle implementieren.
4. `build_unique_stockfish_pgn_corpus.py` auf den Ledger umstellen.
5. Export-Snapshot auf MySQL-Ledger statt SQLite umstellen.
6. Worker-Label-Handler von `state_dir`/SQLite entkoppeln.
7. `build_unique_stockfish_dataset_pipeline.py` auf denselben Ledger-Vertrag bringen.
8. Regressionstests für Builder, Worker und ggf. DB-Helfer ergänzen.
9. Doku aktualisieren.

## Tests

- Builder-Unit-Tests mit Fake-Ledger:
  - dedup
  - disjunkte Splits
  - `complete_at_eof`
  - `run_max_games`
- Worker-Tests:
  - Label-Task schreibt nur NAS-Artefakte
  - kein SQLite-Artefakt im Workdir
- Orchestrator-Tests:
  - bestehende Label-/Master-Pfade bleiben kompatibel
- `ruff check python`

## Exit-Kriterien

- `label_pgn_corpus` funktioniert ohne SQLite
- `export_unique_corpus_snapshot()` exportiert aus MySQL-Ledger auf NAS
- Worker-Summaries enthalten keinen SQLite-Pfad mehr
- bestehende Master-/Controller-Verträge bleiben kompatibel
- Tests und Doku sind aktualisiert

## Risiken

- MySQL-Last steigt während großer Label-Läufe deutlich
- der Ledger speichert mehr Zwischenzustand in MySQL als bisher geplant
- bei sehr großen Läufen kann ein späterer Sharding-/Batching-Ausbau nötig werden
