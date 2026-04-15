# Verteiltes Dauertraining für EngineKonzept: MySQL-Control-Plane, NAS-Artefakte und Implementierungsplan

Diese Datei ersetzt den früheren Filesystem-Queue-Plan vollständig. Sie ist die alleinige Quelle der Wahrheit für den Aufbau eines verteilten Dauertrainings rund um LAPv2.

Der Kernwechsel lautet:

- MySQL ist die Control Plane.
- Das NAS unter `/srv/schach/engine_training/...` ist der Artefakt- und Datenspeicher.
- Jeder Worker nutzt nur lokalen Scratch für Hot State, temporäre Datenbanken und Zwischenzustände.

## Implementierungsstand vom 2026-04-10

Der Stand im Repository ist nicht mehr rein planerisch. Die erste lauffähige
Control-Plane-Schicht ist jetzt implementiert.

Bereits umgesetzt:

- MySQL-Schema für Campaigns, Models, Tasks, Task-Attempts, Workers, Artefakte und Arena-Matches
- Python-Orchestrator unter `python/train/orchestrator/`
- CLI-Einstiege `python/scripts/ek_ctl.py`, `python/scripts/ek_worker.py` und `python/scripts/ek_master.py`
- CherryPy-basierter HTTP-Steuerpfad für `ek_master.py`:
  - JSON-API für Runtime, Spec und Ad-hoc-Submissions
  - lokale Statusseite mit denselben Stellgrößen
  - derselbe `OrchestratorMaster`, keine zweite Scheduler-Logik
- Lease- und Heartbeat-Handling
- Task-Claiming mit Requeue abgelaufener Leases
- Master-Prozess für langlebige Lineages:
  - Label- und Phase-10-Campaigns submitten
  - Generationen bewerten
  - historische Nicht-`LAPv1`-Arme und `vice` so lange in der Arena halten, bis sie sicher geschlagen sind
  - `stockfish18` parallel dazu als Skill-Leiter hochstufen, sobald die aktuelle Stufe sicher geschlagen ist
  - Modelle akzeptieren oder verwerfen
  - Folgegenerationen mit Warm-Start materialisieren
  - Selfplay- und Arena-Partien als PGN-Feedback exportieren, über `label_pgn_corpus` neu labeln und in Folgegenerationen zurückmischen
- DAG-Planung für einen Phase-10-Lauf:
  - Materialisierung
  - Workflow-Prepare
  - Workflow-Chunk-Tasks
  - Workflow-Finalize
  - Training
  - optionales verteiltes Pre-Verify-Selfplay nur für das getrackte LAPv2
  - Selfplay-Prepare
  - Selfplay-Shard-Tasks mit Opening-Suite
  - Selfplay-Finalize
  - Verify
  - Arena-Prepare
  - Arena-Match-Tasks
  - Arena-Finalize
  - Top-Level-Campaign-Finalize
- dedizierter `label_pgn_corpus`-Task für resumables PGN/Stockfish-Labeling
- verteilbarer Arena-Pfad über einzelne Matchup-Tasks statt nur monolithischer Gesamtausführung
- verteilbarer Pre-Verify-Selfplay-Pfad über einzelne Session-Shards
- Workflow-Builder-Unterstützung für splitweise Ausführung über `--only-split`
- Worker-seitige Thread-Drossel für verteilte Inferenz-/Workflow-Tasks über `--distributed-task-threads`
- Lineage-Bootstrap für Generation 1 auf bereits fertigen Seed-Dataset-/Workflow-Artefakten über `bootstrap_generation_from_seed_artifacts=true` plus generiertes `reuse_existing_artifacts=true`
- Campaign-Status-Update direkt beim Task-Start:
  - lange `train_lapv1`-Tasks lassen die Campaign nicht mehr optisch auf `queued`
- MySQL-gesteuerter Idle-Pfad für Pi-Worker:
  - sharded `label_pgn_corpus_idle_slice`
  - Merge der shardweisen Raw-Corpora
  - Materialisierung der Phase-5-Tiers
  - Workflow-Prepare/Chunk/Finalize
  - finale `LAPv2 phase10`-Artefakte auf dem NAS, wie sie das Training konsumiert
- lineage-weites Trainingsdaten-Ledger in MySQL:
  - pro `FEN` und Split wird nur die Wiederverwendungs-Metadaten gepflegt
  - laufende Generationen schreiben dafür kompakte Hash-/Counter-Upserts statt voller Sample-Metadaten
  - der schwere erste Backfill ist einmalig; spätere Generationen erhöhen nur die Zähler der neu trainierten Positionen
- robuste Master-Reconcile-Schicht gegen transiente MySQL-Verbindungsabbrüche:
  - Fehler wie `MySQL server has gone away` im Usage-Ledger-/Backfill-Pfad beenden den Master nicht mehr sofort
  - der Master loggt den transienten Fehler, wartet kurz und setzt die Reconcile-Schleife fort
- Worker-seitige Retry-Schicht für kurzzeitig unsichtbare NAS-/`/srv`-Config-Dateien:
  - Phase-10-Campaign-Configs, Train-Configs und Arena-Specs werden bei `FileNotFoundError` kurz erneut gelesen
  - damit scheitern shardbare Tasks nicht mehr sofort an kurzzeitiger Dateisichtbarkeit
- neuer Stamm-Start ist jetzt flexibler:
  - `seed_warm_start_checkpoint` erlaubt Generation 1 direkt mit einem bereits akzeptierten Netz zu starten
  - `seed_raw_dirs` erlaubt zusätzliche fertige Raw-Snapshots in Generation 1 einzumischen
  - `use_all_available_labeled_positions=true` übernimmt die gesamte aktuell verfügbare gelabelte Kandidatenmenge statt nur die vorige Generationsgröße
- schaltbare Master-Spec-Einträge über `enabled`:
  - `label_jobs`
  - `idle_phase10_jobs`
  - `lineages`
  - damit lassen sich künftige Läufe über denselben API-/UI-Pfad pausieren oder eingrenzen

Bereits verifiziert:

- lokale Python-Tests für Controller-DAG, Workflow-Chunking und Arena-Matchup-Pfad
- lokale Ruff-Prüfung des Python-Baums
- erfolgreiche Initialisierung der MariaDB-Control-Plane auf `10.42.42.42`
- erfolgreiche Submission der realen Campaign
  `phase10_lapv2_stage2_native_arena_all_sources_v1`
- echter lokaler MySQL-Smoke-Run mit `2` und danach `4` parallelen Workern unter `/srv/engine_training_temp/...`
- echter Master-Smoke-Run mit einem dedizierten `label`-Worker und zwei vollständig durchgelaufenen Phase-10-Generationen
- echter ultrakurzer lokaler Komplett-Smoke-Run unter
  `/srv/engine_training_temp/mysql_master_finalization_smoke_20260415_run1`:
  - bootstrapped `train -> selfplay -> verify -> arena -> phase10_finalize`
  - danach automatisches Feedback-Labeling
  - danach erfolgreicher Trainingsnutzungs-Backfill im Master
  - Endzustand `all_terminal=true`, `status=completed`, `latest_recorded_generation=1`

Bewusst noch nicht ausgeführt:

- keine Worker-Ausführung gegen das NAS
- keine Materialisierung auf `/srv/schach`
- kein Training, Verify oder Arena-Lauf mit echten NAS-Artefakten

Das ist absichtlich so, weil das NAS in dieser Arbeitsphase nicht benutzt werden durfte.
Der aktuelle Stopppunkt ist daher:

- Code und DB-Control-Plane sind bereit
- die erste echte Campaign ist in MySQL angelegt
- der Data-Plane-Lauf beginnt erst, wenn NAS-Zugriffe wieder erlaubt sind

## Aktueller Repo-Stand und vorhandene Bausteine

Der aktuelle Repo-Stand ist für diesen Umbau günstig, weil die zentralen Domänenbausteine bereits existieren:

- LAPv2 ist als echter Modellpfad vorhanden und nicht nur eine Idee.
- Ein nativer Phase-10-Lauf ist vorbereitet:
  - `python/configs/phase10_lapv2_stage2_native_all_sources_v1.json`
  - `python/configs/phase10_lapv2_stage2_native_arena_all_sources_v1.json`
- Die Datenpfade liegen bereits NAS-zentriert unter `/srv/schach/engine_training/phase10/...`.
- Der Workflow-Build ist bereits chunked und shardbar:
  - `python/scripts/build_phase10_lapv1_workflow.py`
  - `--train-chunk-start/--end`
  - `--validation-chunk-start/--end`
  - `--verify-chunk-start/--end`
  - `--skip-finalize`
  - `--finalize-only`
- Arena ist bereits artefaktorientiert und schreibt `progress.json` sowie `summary.json`:
  - `python/train/eval/arena.py`
- Die Phase-9-Evolution zeigt, wie iteratives Training, Verify und Arena als wiederholbarer Prozess zusammenspielen:
  - `python/train/eval/evolution_campaign.py`
- Das PGN/Stockfish-Corpus ist bereits resumable, aber heute noch lokal-zentriert:
  - `python/scripts/build_unique_stockfish_pgn_corpus.py`
  - nutzt jetzt einen MySQL-Label-Ledger statt SQLite
  - `work_dir` ist das NAS-Exportziel für `progress.json`, `train_raw.jsonl`, `verify_raw.jsonl`
  - Resume- und FEN-Dedup-Zustand liegt im selben MySQL-System wie die Control Plane

Wichtig ist auch der aktuelle negative Befund:

- Der heutige Phase-10-Runner ist noch monolithisch:
  - `python/scripts/run_phase10_lapv1_stage1_arena_campaign.py`
- Er verkettet Materialisierung, Workflow, Training, Verify und Arena in einem Prozess.
- Genau dieser Ablauf soll künftig durch eine verteilte Orchestrierung ersetzt werden, ohne die Artefaktverträge zu zerstören.

## Zielbild

Das System wird in drei Ebenen getrennt.

### 1. Control Plane = MySQL

Hier liegen nur kleine, transaktionale Zustände:

- Campaigns
- Models und Generationen
- Tasks
- Task-Dependencies
- Worker-Heartbeats
- Leases
- Artefakt-Metadaten
- Arena-Match-Indizes
- Promotion-Entscheidungen
- optionale globale Ledgers

### 2. Data Plane = NAS

Hier liegen alle großen Daten:

- `train_raw.jsonl`, `verify_raw.jsonl`
- Dataset-Roots
- Workflow-Chunks und finalisierte Workflow-Roots
- Checkpoints und Bundles
- Arena-Game-Records
- Selfplay-PGNs oder JSONL-Shards
- Logs
- `progress.json`
- `summary.json`

### 3. Execution Plane = Worker-Hosts

Jeder Worker hat:

- lokale SSD/NVMe oder lokales Dateisystem für Scratch
- lokale Logs und temporäre Dateiartefakte, falls einzelne Tasks sie brauchen
- Zugriff auf MySQL
- Zugriff auf das NAS
- explizite Capabilities:
  - `label`
  - `materialize`
  - `workflow`
  - `train`
  - `verify`
  - `arena`
  - `selfplay`
  - `aggregate`
  - optional zusätzlich für Hintergrund-Artefaktbau während Training:
    - `label_idle`
    - `materialize_idle`
    - `workflow_idle`
    - `aggregate_idle`

## Die wichtigste Regel

MySQL speichert niemals die großen Nutzdaten.

Nicht in MySQL:

- keine Checkpoints
- keine Bundles
- keine JSONL-Datensätze
- keine PGN-Streams
- keine Game-Records
- keine Workflow-Blobs

In MySQL stehen nur:

- Pfade
- Status
- Prioritäten
- Leases
- Checksummen
- Größen
- Owner
- Abhängigkeiten
- Kennzahlen
- kleine Ergebniszusammenfassungen

## Unified Task-API

Jede verteilte Task muss idempotent und artefaktorientiert sein.

Pflichtvertrag:

- Input besteht nur aus Pfaden und deterministischen Parametern.
- Output ist ein definierter Pfad oder ein definiertes Output-Directory.
- Jede erfolgreiche Task schreibt ein `summary.json`.
- Erfolg heißt:
  - Output vorhanden
  - `summary.json` vorhanden
  - Schema-/Versionsfeld plausibel
- Fehlgeschlagene Tasks dürfen erneut laufen.
- Jeder Attempt ist separat auditierbar.

Das ist wichtig, weil es zu den heutigen Repo-Mustern passt:

- `skip-existing`
- chunked finalize
- `summary.json`
- wiederaufnahmefähige Outputs

## HTTP-Steuerung des Masters

Für künftige Läufe kann der Master zusätzlich mit lokalem HTTP-Server betrieben werden:

```bash
python3 python/scripts/ek_master.py \
  --config /path/to/master_config.json \
  --http-host 127.0.0.1 \
  --http-port 8080
```

Dabei gilt ausdrücklich:

- die Fachlogik bleibt in `OrchestratorMaster`
- der HTTP-Server ist nur Transport- und UI-Schicht
- die aktive Spec bleibt die Datei auf dem NAS oder lokalen Dateisystem
- MySQL bleibt die Control Plane für Tasks, Worker, Leases und Status

Der aktuelle API-v1-Vertrag ist:

- `GET /api/v1/bootstrap`
- `GET /api/v1/runtime`
- `POST /api/v1/runtime/start|stop|pause|resume|reconcile|requeue_expired`
- `GET /api/v1/spec`
- `POST /api/v1/spec/patch`
- `POST /api/v1/spec/lineages/<name>`
- `POST /api/v1/spec/label_jobs/<name>`
- `POST /api/v1/spec/idle_phase10_jobs/<name>`
- `POST /api/v1/submit/phase10|label|idle_phase10`

Die Statusseite unter `/` zeigt:

- Runtime-Zustand
- letzte Master-`summary.json`
- aktuellen DB-Snapshot
- Poll-Intervall
- `enabled`-Flags und Kernparameter pro Lineage
- Raw-Spec als Patch-/Kontrollansicht

Wichtig für den Betrieb:

- der HTTP-Pfad ist für spätere Läufe gedacht
- er wird nicht mitten in einen schon laufenden Produktionslauf hinein ausgerollt
- bestehende CLI-Pfade (`--once`, `--until-terminal`, `run_forever`) bleiben unverändert nutzbar

## Datenfluss als dauerhafte Pipeline

Ein vollständiger Dauerbetrieb sieht so aus:

1. Labeling
2. Raw-Merge und Dedup
3. Materialisierung der Phase-5-Dataset-Tiers
4. Phase-10-Workflow-Build
5. Training
6. optional Selfplay des frisch trainierten LAP-Modells mit Opening-Suite
7. Verify
8. Arena
9. Promotion oder Archivierung
10. Rückfluss in neue Rohdaten

Wesentlich ist:

- Labeling, Workflow und Arena sind natürlich shardbar.
- Training und Verify sind exklusive Modelljobs.
- Promotion und Registry-Updates bleiben zentral.
- Selfplay ist ein zusätzlicher Datenzufluss, nicht der Scheduler selbst.
- Das neue Pre-Verify-Selfplay ist bewusst ein eigener Zwischenschritt zwischen `train` und `verify`.
- Selfplay- und Arena-Sessions werden generationweise als PGNs exportiert, vom `label`-Worker mit Stockfish gelabelt und vor der nächsten Generation in den Raw-Corpus zurückgeführt.
- Für künftige Läufe können Pi-Worker während eines aktiven `train_lapv1` zusätzlich einen niedrig priorisierten Hintergrundpfad abarbeiten:
  - `PGN -> shardweise Raw-Corpora -> Merge -> Materialize -> Workflow`
  - dieser Pfad baut echte `LAPv2 phase10`-Artefakte, nicht nur Rohdaten
  - sobald kein `train_lapv1` mehr geleast ist, claimen Worker keine `*_idle`-Tasks mehr und wechseln automatisch zurück auf `selfplay`, `verify`, `arena` und normale Aggregation
- Folgegenerationen können diese Idle-Daten jetzt direkt vor `materialize` mitverwenden:
  - der Master nimmt nicht nur fertige Idle-Campaign-Endartefakte, sondern bereits exportierte `label_shards/shard_*/train_raw.jsonl` und `verify_raw.jsonl`
  - frische, noch nie trainierte FENs werden immer zuerst gezogen
  - falls für die gewünschte Corpus-Größe Wiederholung nötig ist, werden die bislang am seltensten trainierten FENs bevorzugt
  - der Trainingsnutzungszustand pro Lineage liegt in MySQL und nicht auf dem NAS

## Reell getesteter Master-Lauf

Ein echter lokaler Testlauf unter `/srv/engine_training_temp/mysql_master_label_smoke_20260410_run1`
lief inzwischen komplett durch.

Ablauf:

1. `label_pgn_corpus`
2. `phase10_master:g0001`
3. Master-Auswertung von `g0001`
4. `phase10_master:g0002`
5. Master-Endzustand `all_terminal=true`

Reale Resultate:

- Label-Job:
  - `16` Train-Records
  - `4` Verify-Records
  - `5` gelesene Partien
  - `0` übersprungene Partien
- Generation `g0001`:
  - akzeptiert
  - `verify_top1=0.0`
  - `verify_top3=1.0`
  - bestes getracktes Arena-`score_rate=0.375`
- Generation `g0002`:
  - akzeptiert
  - `verify_top1=0.0`
  - `verify_top3=1.0`
  - bestes getracktes Arena-`score_rate=0.5`
- MySQL-Endstand:
  - `3` Campaigns
  - `65` Tasks
  - alle `succeeded`

Wichtige Artefakte:

- `/srv/engine_training_temp/mysql_master_label_smoke_20260410_run1/master/summary.json`
- `/srv/engine_training_temp/mysql_master_label_smoke_20260410_run1/label_work/summary.json`
- `/srv/engine_training_temp/mysql_master_label_smoke_20260410_run1/lineage/generation_0001/decision.json`
- `/srv/engine_training_temp/mysql_master_label_smoke_20260410_run1/lineage/generation_0002/decision.json`

## Worker-Topologie für Mehrkern-Hosts

Für echte Läufe ist die CPU-Strategie jetzt bewusst asymmetrisch:

- `train` läuft idealerweise auf genau einem starken Host mit vielen Threads.
- `selfplay`, `verify`, `workflow` und einzelne `arena_match`-Tasks werden besser über mehrere Worker verteilt als über einen Worker mit vielen Kernen.

Der Grund ist simpel:

- ein Orchestrator-Worker claimed immer nur einen Task gleichzeitig
- Parallelität entsteht daher primär über viele shardbare Tasks und mehrere Worker-Prozesse
- ein einzelner Worker mit vielen freien Kernen beschleunigt verteiltes Selfplay kaum, wenn jeder Shard nur seriell läuft
- mehrere Worker mit `--distributed-task-threads 1` verhindern Oversubscription in `torch`, BLAS und OpenMP deutlich robuster

Die empfohlene Praxis ist daher:

- auf dem stärksten Host ein Worker mit `train,aggregate` und ohne künstliche Trainingsdrossel
- auf Selfplay-/Verify-Hosts mehrere Worker mit `selfplay,verify,workflow,arena` und `--distributed-task-threads 1`
- keine Annahme, dass ein Selfplay-Subprozess automatisch CPU-fair bleibt; die Thread-Grenze wird explizit gesetzt

## Warum MySQL und nicht Filesystem-Queue

Der frühere Filesystem-Queue-Ansatz war als Minimalstart brauchbar, aber für echten Dauerbetrieb ist MySQL robuster.

Filesystem-Queues haben drei strukturelle Schwächen:

- Claiming und Lease-Verwaltung hängen an Dateisystemsemantik.
- Shared Filesystems machen atomische Annahmen schwieriger.
- Reconcile und Abhängigkeiten werden unnötig aufwendig.

MySQL löst genau diese Punkte sauberer:

- transaktionales Claiming
- saubere Leases
- eindeutige Task-States
- Abhängigkeiten als relationale Daten
- Heartbeats und Worker-Sicht zentral
- bessere Statusabfragen

Die frühere Sorge aus dem alten Plan bleibt aber als Designregel erhalten:

- NFS/NAS bleibt Artefaktspeicher, nicht Scheduler.
- Lokaler Scratch bleibt lokal.
- Große Daten werden nie über die Control Plane geschoben.

## Warum nicht sofort Celery oder Ray

Celery und Ray bleiben valide spätere Optionen, sind aber jetzt nicht die beste erste Stufe.

Warum nicht jetzt:

- zusätzlicher Stack
- neue Betriebsabhängigkeiten
- unnötiger Overhead, bevor die Repo-eigenen Task-Verträge sauber stehen

Warum MySQL jetzt genügt:

- die Domäne ist bereits stark artefaktorientiert
- MySQL deckt Control-Plane-Bedürfnisse direkt ab
- der Ausbau kann schrittweise erfolgen

Die Reihenfolge ist daher bewusst:

- zuerst MySQL-Control-Plane
- dann Worker-DAGs
- erst später optional Broker/Cluster-Framework, falls nötig

## SQL- und Datensicherheitsregel

Der PGN-Corpus-Builder nutzt jetzt denselben MySQL-Server wie die übrige Control Plane.

Daraus folgt:

- kein SQLite mehr im verteilten Label-Pfad
- Resume- und Dedup-Zustand für `label_pgn_corpus` liegt in MySQL
- exportierte Rohkorpora und Folgeartefakte bleiben auf dem NAS
- Secrets bleiben außerhalb des Repos und werden per Umgebung oder lokaler Env-Datei injiziert

## Observability, Wiederaufnahme und Debugbarkeit

Jede Task schreibt:

- `summary.json`
- stdout-Log
- stderr-Log oder kombiniertes Log
- Host- und Versionsmetadaten

Jede Task besitzt in der Control Plane:

- `state`
- `worker_id`
- `lease_until`
- `attempt_count`
- `result_json`
- Pfade zu Logs oder Resultaten

Das System muss aus zwei Quellen rekonstruierbar bleiben:

- MySQL
- NAS-Artefakte

Wenn ein Worker stirbt:

- Lease läuft aus
- Task wird requeued
- vorhandene Artefakte werden beim Retry berücksichtigt

## Konkrete MySQL-Struktur

### `campaigns`

Ein Eintrag pro großer Kampagne oder Dauerlauf.

Felder:

- `id`
- `name`
- `kind`
- `status`
- `config_path`
- `active_model_id`
- `created_at`
- `updated_at`

### `models`

Ein Eintrag pro geplantem oder trainiertem Netz.

Felder:

- `id`
- `campaign_id`
- `parent_model_id`
- `generation`
- `train_config_path`
- `agent_spec_path`
- `checkpoint_path`
- `bundle_path`
- `verify_json_path`
- `arena_summary_path`
- `status`
- `promotion_score`
- `created_at`

### `artifacts`

Zentrales Register für NAS-Artefakte.

Felder:

- `id`
- `kind`
- `path`
- `sha256`
- `size_bytes`
- `producer_task_id`
- `state`
- `metadata_json`
- `created_at`

### `tasks`

Kern der Orchestrierung.

Felder:

- `id`
- `campaign_id`
- `model_id`
- `task_type`
- `capability`
- `priority`
- `state`
- `payload_json`
- `result_json`
- `worker_id`
- `lease_until`
- `attempt_count`
- `max_attempts`
- `depends_on_count`
- `not_before`
- `created_at`
- `updated_at`

### `task_dependencies`

Für DAG-Beziehungen.

Felder:

- `task_id`
- `depends_on_task_id`

### `task_attempts`

Für Audit und Debug.

Felder:

- `id`
- `task_id`
- `worker_id`
- `started_at`
- `ended_at`
- `exit_code`
- `stdout_path`
- `stderr_path`
- `result_summary_path`

### `workers`

Felder:

- `id`
- `hostname`
- `capabilities_json`
- `scratch_root`
- `status`
- `last_heartbeat_at`
- `version`
- `metadata_json`

### `arena_matches`

Spezialtabelle für Arena-Jobs.

Felder:

- `id`
- `campaign_id`
- `model_a`
- `model_b`
- `opening_id`
- `color_assignment`
- `seed`
- `task_id`
- `result`
- `game_record_path`
- `finished_at`

### optional `position_ledger`

Nur später, wenn echte globale Online-Dedup nötig wird.

Felder:

- `fen_hash`
- `assigned_split`
- `first_artifact_id`
- `first_seen_at`

## Claiming und Worker-Kommunikation

Jeder Worker arbeitet ungefähr so:

1. Heartbeat schreiben
2. Transaktion öffnen
3. passende `queued`-Tasks claimen
4. Lease setzen
5. Commit
6. Task ausführen
7. Artefakte registrieren
8. Task auf `succeeded` oder `failed` setzen

Regeln:

- Worker kommunizieren untereinander nur indirekt über MySQL.
- Worker tauschen große Resultate nie über MySQL aus.
- Worker sehen gemeinsame Artefakte ausschließlich über das NAS.

## NAS-Layout

Empfohlener Root:

```text
/srv/schach/engine_training/
  control_exports/
  phase10/
    raw/
    datasets/
    workflows/
    models/
    arena/
    selfplay/
    logs/
    reports/
```

Kampagnenbezogen:

```text
phase10/
  campaigns/
    lapv2_native_all_sources_v2/
      raw/
        shards/
        merged/
      datasets/
        train/
        verify/
      workflow/
        chunks/
        final/
      models/
        model_000123/
          checkpoint.pt
          bundle/
          summary.json
      arena/
        jobs/
        games/
        summary.json
      selfplay/
      logs/
```

Der wichtige Punkt ist nicht das exakte Namensschema, sondern:

- Artefakte bleiben NAS-zentriert
- Pfade bleiben stabil
- große Daten bleiben dateibasiert

## Worker-Rollen

### 1. Label-Worker

Aufgabe:

- bekommt PGN-Dateiliste oder Selfplay-Shard
- arbeitet lokal
- schreibt `train_raw_shard_*.jsonl` und `verify_raw_shard_*.jsonl` aufs NAS
- meldet nur Pfade und Metadaten an MySQL

Wichtig:

- kein SQLite-Sonderpfad mehr
- Resume- und Dedup erfolgen im MySQL-Ledger des jeweiligen Label-Namespace
- exportierte Shards werden weiter als NAS-Artefakte behandelt

### 2. Materialize-Worker

Aufgabe:

- nimmt einen gemergten Raw-Snapshot
- ruft die vorhandene Materialisierung auf
- schreibt Dataset-Roots aufs NAS
- registriert `dataset_dir`-Artefakte

### 3. Workflow-Worker

Aufgabe:

- übernimmt genau einen Chunk
- arbeitet mit `--skip-finalize`
- schreibt Chunk-Artefakte aufs NAS
- Finalize erfolgt in separater Task

Das ist direkt kompatibel zum heutigen Phase-10-Chunk-Modell.

### 4. Train-Worker

Aufgabe:

- claimt exklusiv einen Trainingsjob
- liest Workflow-Artefakte vom NAS
- trainiert lokal
- schreibt Checkpoint, Bundle und Summary aufs NAS
- registriert Pfade und Kernmetriken in MySQL

### 5. Verify-Worker

Aufgabe:

- liest Checkpoint und Verify-Split
- schreibt Verify-JSON aufs NAS
- meldet Kennzahlen an MySQL

### 6. Arena-Worker

Aufgabe:

- übernimmt Match-Jobs
- spielt definierte Partien
- schreibt Game-Records aufs NAS
- meldet nur Ergebnis-Metadaten an MySQL

Die zentrale Aggregation erzeugt Standings, Matrix und Summary.

### 7. Selfplay-Worker

Aufgabe:

- lädt freigegebenen Agentenstand
- erzeugt Selfplay-Partien
- schreibt PGN- oder JSONL-Shards aufs NAS
- meldet Volumen, Seeds und Pfade an MySQL

### 8. Aggregator-Worker

Optional als eigener Typ oder Master-Funktion.

Aufgabe:

- merged Shards
- finalisiert Workflow-Roots
- aggregiert Arena-Resultate
- schreibt Reports und Promotion-Inputs

## Kommunikation vs. Daten

Die harte Trennlinie lautet:

- MySQL ist Kommunikations- und Steuerungsebene.
- Das NAS ist Artefakt- und Datenspeicher.
- Lokales Scratch ist nur Worker-intern.

Daraus folgen drei Regeln:

- keine großen Daten in MySQL
- keine Scheduler-Logik über NAS-Dateirennen
- keine shared scratch states zwischen Hosts

## Überarbeiteter Implementierungsplan für Codex-CLI

### Etappe 1: MySQL-Control-Plane

Neue Module:

- `python/train/orchestrator/db.py`
- `python/train/orchestrator/models.py`
- `python/train/orchestrator/lease.py`
- `python/train/orchestrator/controller.py`
- `python/train/orchestrator/worker.py`

Neue CLIs:

- `python/scripts/ek_worker.py`
- `python/scripts/ek_ctl.py`

Pflichten:

- versionierte Task-Schemata
- Worker-Registrierung
- Claiming und Lease-Timeouts
- Requeue
- Heartbeats
- Statusanzeige

Akzeptanz:

- konkurrierende Worker claimen ohne Doppelausführung
- abgelaufene Leases werden korrekt requeued
- `ek_ctl status` zeigt Campaigns, Tasks und Worker

### Etappe 2: Artifact-Contract standardisieren

Neue Typen:

- `ArtifactRef`
- `TaskResult`
- `ResultSummary`

Pflichten:

- jede Task liefert Pfade plus Metadaten
- jedes Output-Directory enthält `summary.json`
- Artefakte werden in MySQL registriert

Akzeptanz:

- der Controller kann Task-Zustände aus DB plus Pfaden rekonstruieren

### Etappe 3: Verteiltes Workflow-Chunking

Ziel:

- Phase-10-Workflow-Chunks werden als einzelne Tasks modelliert
- Finalize wird als separater Task modelliert

Pflichten:

- ein Chunk = eine Task
- Finalize hängt von allen Chunk-Tasks ab
- Output-Layout bleibt kompatibel zum heutigen Workflow-Root

Akzeptanz:

- `lapv2_workflow_all_sources_v1`-artige Builds können auf mehreren Hosts erzeugt und deterministisch finalisiert werden

### Etappe 4: Verteilte Arena

Ziel:

- `run_selfplay_arena_distributed(...)`
- Match-Expansion zentral
- Game- oder Match-Batches als Tasks

Pflichten:

- gleiche Standings- und Summary-Verträge wie heute
- nur die Ausführung wird verteilt

Akzeptanz:

- ein Arena-Lauf verteilt sich über mehrere Hosts
- das Resultat bleibt kompatibel zum heutigen `summary.json`

### Etappe 5: Verteiltes Training und Verify

Ziel:

- Training und Verify werden exklusive MySQL-Jobs

Pflichten:

- ein Trainingsjob pro Modell
- Verify hängt vom Training ab
- Resultate als NAS-Artefakte, Status in MySQL

Akzeptanz:

- der heutige native LAPv2-Lauf kann als DAG statt monolithisch ausgeführt werden

### Etappe 6: Verteiltes Labeling

Ziel:

- shard-basiertes PGN-Labeling mit MySQL-Ledger statt SQLite

Pflichten:

- `train_raw_shard_*`
- `verify_raw_shard_*`
- Merge-Task mit fen-hash-Dedup
- Verify-Precedence über Train
- deterministische Split-Zuordnung

Akzeptanz:

- mehrere Hosts können parallel labeln
- der Merge erzeugt deterministische, disjunkte Resultate

### Etappe 7: Model Registry und Promotion

Ziel:

- Modelllebenszyklus zentral verwalten

Pflichten:

- Parent/Child-Beziehungen
- Verify- und Arena-Referenzen
- Promotion-Regeln
- Active/Challenger/Archived

Akzeptanz:

- neue Kandidaten können systematisch trainiert, verifiziert, verglichen und promoted werden

### Etappe 8: Continuous Campaign Service

Ziel:

- Dauerbetrieb über Mindestfüllstände und Regelkreise

Pflichten:

- Rohdaten-Füllstände überwachen
- Workflow-Vorräte überwachen
- Trainings- und Arena-Backlogs erzeugen
- Selfplay optional einspannen

Akzeptanz:

- das System erzeugt dauerhaft neue Modelle ohne manuelles Starten jedes Einzelschritts

## Migrationspfad ohne Big Bang

Die Umstellung soll LAPv2 nicht blockieren. Deshalb erfolgt sie in vier Stufen.

### Phase A: Arena zuerst verteilen

- Training und Verify bleiben zunächst beim bestehenden Runner
- nur Arena wird durch den verteilten Arena-Runner ersetzt

Nutzen:

- sofortiger Skalierungsgewinn
- geringes Risiko

### Phase B: Workflow verteilen

- Phase-10-Workflow wird über Chunk-Tasks verteilt
- Training und Verify bleiben noch lokal oder exklusiv monolithisch

Nutzen:

- der teuerste Datenvorbereitungsblock wird host-übergreifend

### Phase C: Distributed Campaign Runner

- neuer Campaign-Runner ersetzt den monolithischen Runner
- gleiche Configs
- gleiche Output-Verträge
- andere Ausführungsform

### Phase D: Continuous Mode

- Model Registry
- automatische Kandidaten
- Promotion
- Selfplay-Rückfluss

## Direkter Bezug zum aktuellen nativen LAPv2-Lauf

Der aktuelle native LAPv2-Lauf bleibt die erste reale Zielkampagne für diese Architektur.

Das heißt:

- dieselben Config-Pfade bleiben gültig
- dieselben NAS-Roots bleiben gültig
- dieselben Artefaktverträge bleiben gültig
- nur die Orchestrierung wird ersetzt

Wichtig:

- kein Big-Bang-Neuschreiben der Trainingslogik
- der neue Orchestrator ersetzt die Ausführungsform, nicht die Modellidee

## Risiken und offene Punkte

- Environment Drift zwischen Hosts bleibt ein reales Risiko.
- Jeder Worker braucht kompatible Python-, Torch-, Chess- und Engine-Binaries.
- Shared NAS bleibt ein potenzieller Engpass für große Artefakte.
- MySQL darf nicht mit Payloads missbraucht werden.
- Verteiltes Labeling braucht von Anfang an klare Dedup- und Split-Regeln.
- Finalize- und Promotion-Schritte müssen idempotent bleiben.

## Klare Empfehlung

Die richtige Version des Plans ist:

- MySQL als Control Plane
- NAS als Artefakt-Store
- lokale SSD als Scratch und Hot State

Und daraus folgt praktisch:

- keine Filesystem-Queue mehr
- kein SQLite-Sonderpfad für Labeling
- große finale Artefakte bleiben auf dem NAS, aber der resumable Label-Ledger liegt bewusst in MySQL
- Arena, Workflow, Labeling, Selfplay, Training und Verify als DB-gesteuerte Jobs
- bestehende `/srv/schach/engine_training/...`-Roots bleiben die Wahrheit für große Daten
- der heutige native LAPv2-Lauf wird als erste Campaign auf den neuen Orchestrator gehoben, nicht neu erfunden
