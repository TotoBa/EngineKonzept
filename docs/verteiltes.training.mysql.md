# Verteiltes Dauertraining für EngineKonzept: MySQL-Control-Plane, NAS-Artefakte und Implementierungsplan

Diese Datei ersetzt den früheren Filesystem-Queue-Plan vollständig. Sie ist die alleinige Quelle der Wahrheit für den Aufbau eines verteilten Dauertrainings rund um LAPv2.

Der Kernwechsel lautet:

- MySQL ist die Control Plane.
- Das NAS unter `/srv/schach/engine_training/...` ist der Artefakt- und Datenspeicher.
- Jeder Worker nutzt nur lokalen Scratch für Hot State, temporäre Datenbanken und Zwischenzustände.

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
  - nutzt `corpus.sqlite3`
  - dokumentiert implizit, dass dessen Work-Dir lokal bleiben sollte

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
- lokale temporäre SQLite/LMDB/JSONL-Arbeitszustände
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

## Datenfluss als dauerhafte Pipeline

Ein vollständiger Dauerbetrieb sieht so aus:

1. Labeling
2. Raw-Merge und Dedup
3. Materialisierung der Phase-5-Dataset-Tiers
4. Phase-10-Workflow-Build
5. Training
6. Verify
7. Arena
8. Promotion oder Archivierung
9. optional Selfplay
10. Rückfluss in neue Rohdaten

Wesentlich ist:

- Labeling, Workflow und Arena sind natürlich shardbar.
- Training und Verify sind exklusive Modelljobs.
- Promotion und Registry-Updates bleiben zentral.
- Selfplay ist ein zusätzlicher Datenzufluss, nicht der Scheduler selbst.

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

Der bestehende PGN-Corpus-Builder nutzt heute SQLite. Das ist lokal gut, aber kein Zielmodell für Multi-Host-Schreiber auf einem Shared Mount.

Daraus folgt:

- kein Shared-SQLite als globaler Koordinator
- lokales SQLite pro Worker-Task ist erlaubt
- globaler Zustand liegt in MySQL
- verteiltes Labeling arbeitet mit NAS-Shards plus Merge

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

- kein globales Shared-SQLite
- lokale Resume-Stati erlaubt
- globale Dedup erfolgt im Merge-Schritt

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

- shard-basiertes PGN-Labeling statt globaler Shared-SQLite

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
- kein Shared-SQLite für globale Koordination
- keine großen Daten in MySQL
- Arena, Workflow, Labeling, Selfplay, Training und Verify als DB-gesteuerte Jobs
- bestehende `/srv/schach/engine_training/...`-Roots bleiben die Wahrheit für große Daten
- der heutige native LAPv2-Lauf wird als erste Campaign auf den neuen Orchestrator gehoben, nicht neu erfunden
