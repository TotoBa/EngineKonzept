# ExecPlan: HTTP-Steuerung für den Orchestrator-Master

## Status

Stand `2026-04-11`:

- ExecPlan dokumentiert
- Runtime-Wrapper implementiert
- CherryPy-API implementiert
- Statusseite implementiert
- nicht auf den aktuell laufenden Produktionslauf ausgerollt

## Ziel

Der bestehende MySQL-basierte `ek_master.py` soll zusätzlich als lokaler
HTTP-Kontrollpunkt laufen können.

Die neue Schicht muss:

- denselben `OrchestratorMaster` und dieselbe Master-Spec-Datei verwenden wie der CLI-Pfad
- per API steuerbar sein
- eine einfache Statusseite mit denselben Stellgrößen anbieten
- keine neue parallele Orchestrierungslogik einführen
- den laufenden Produktionslauf unberührt lassen; die Funktion gilt erst für spätere Läufe

## Nicht-Ziele

- kein Ersatz der MySQL-Control-Plane
- keine Browser-Heavy-UI oder externer Frontend-Build
- keine direkte Worker-Prozessverwaltung über HTTP in v1
- keine Auth-/Multi-Tenant-Schicht in v1
- kein Rollout in den aktuell laufenden Generation-1-Run

## Architekturvertrag

### 1. Eine Runtime-Schicht über dem Master

Ein neuer Runtime-Wrapper kapselt:

- den aktuellen `OrchestratorMaster`
- den periodischen Reconcile-Loop
- Pause/Resume/Stop/Start des Loops
- den letzten erfolgreichen Summary-Snapshot
- den letzten Fehler
- den aktuellen Runtime-Status

Wichtig:

- `OrchestratorMaster.reconcile_once()` bleibt die einzige fachliche Reconcile-Operation
- die Runtime serialisiert konkurrierende Reconcile-Aufrufe
- die Runtime bleibt prozesslokal; der dauerhafte Zustand liegt weiter in MySQL plus Spec-Datei

### 2. Steuerung über dieselbe Master-Spec

Alle einstellbaren fachlichen Parameter bleiben in der Master-Spec.

Die HTTP-Schicht darf:

- die Spec lesen
- validierte Änderungen in die Spec zurückschreiben
- die im Speicher aktive Spec atomar ersetzen

Damit bleibt der Steuerpfad einheitlich:

- CLI liest die Spec
- HTTP liest und schreibt die Spec
- der Master reconciled immer gegen die gerade aktive Spec

### 3. CherryPy als dünne Transport- und UI-Schicht

CherryPy wird nur für drei Dinge verwendet:

- JSON-API
- Statusseite
- Prozess-Hosting des HTTP-Servers

Geschäftslogik bleibt außerhalb von CherryPy.

## API v1

### Runtime-Endpunkte

- `GET /api/v1/runtime`
  - aktueller Runtime-Zustand
- `POST /api/v1/runtime/start`
  - Hintergrund-Loop starten
- `POST /api/v1/runtime/stop`
  - Hintergrund-Loop anhalten, HTTP bleibt aktiv
- `POST /api/v1/runtime/pause`
  - Loop pausieren
- `POST /api/v1/runtime/resume`
  - Pause aufheben
- `POST /api/v1/runtime/reconcile`
  - sofort einen Reconcile-Zyklus ausführen
- `POST /api/v1/runtime/requeue-expired`
  - abgelaufene Leases requeueen

### Read-API

- `GET /api/v1/summary`
  - letzter Master-Summary-Snapshot
- `GET /api/v1/status?limit=N`
  - DB-Status-Snapshot wie `ek_ctl status`
- `GET /api/v1/spec`
  - aktive Master-Spec
- `GET /api/v1/bootstrap?limit=N`
  - Runtime + Summary + Spec + DB-Status in einer Antwort für die Statusseite

### Write-API

- `POST /api/v1/spec/patch`
  - rekursiver Merge-Patch auf die Master-Spec
- `POST /api/v1/spec/replace`
  - vollständige Spec ersetzen
- `POST /api/v1/spec/lineages/<name>`
  - gezielte Änderung einer Lineage über benannte Felder
- `POST /api/v1/spec/label-jobs/<name>`
  - gezielte Änderung eines Label-Jobs
- `POST /api/v1/spec/idle-phase10-jobs/<name>`
  - gezielte Änderung eines Idle-Artifact-Jobs

### Submit-API

Für ad-hoc Starts, die heute schon über `ek_ctl.py` existieren:

- `POST /api/v1/submit/phase10`
- `POST /api/v1/submit/label`
- `POST /api/v1/submit/idle-phase10`

Diese Endpunkte sprechen direkt mit `OrchestratorController`.

## UI v1

Die Statusseite unter `/` bleibt absichtlich einfach:

- Runtime-Status
- letzte Summary
- DB-Snapshot
- Buttons für Start/Stop/Pause/Resume/Reconcile/Requeue
- editierbare Poll-Interval- und Enabled-Flags
- editierbare Kernwerte pro Lineage:
  - `enabled`
  - `max_generations`
  - `on_accept`
  - `on_reject`
  - `promotion_thresholds`
- Rohansicht der aktiven Spec

Die Seite nutzt ausschließlich die API und hat keine Sonderlogik.

## Datenmodell-Erweiterungen

Für UI/API-Steuerung braucht die Master-Spec explizite `enabled`-Flags auf:

- `LabelPgnCorpusJobSpec`
- `IdlePhase10ArtifactJobSpec`
- `Phase10LineageSpec`

Deaktivierte Einträge werden vom Master als `disabled` gemeldet und nicht reconciled.

## Implementierungsschritte

1. Plan-Dokument anlegen und Vertrag festziehen.
2. `enabled`-Flags und Spec-Ersetzung im Master implementieren.
3. Runtime-Wrapper mit Threading und Snapshot-API bauen.
4. CherryPy-API auf Runtime und Controller aufsetzen.
5. einfache HTML-Statusseite ergänzen.
6. CLI `ek_master.py` um HTTP-Betriebsmodus erweitern.
7. Tests für Runtime, Spec-Patching und HTTP-Smoke-Pfade ergänzen.
8. Doku und Betriebsanleitung aktualisieren.

## Tests

- Unit-Tests für:
  - deaktivierte Jobs/Lineages
  - Spec-Patching und Spec-Ersatz
  - Runtime-Start/Pause/Resume/Stop
- API-Smoke-Tests für:
  - `bootstrap`
  - `runtime/reconcile`
  - `spec`-Update
- CLI-Help-Test für `ek_master.py`

## Exit-Kriterien

- `ek_master.py` kann wie bisher ohne HTTP laufen
- `ek_master.py` kann zusätzlich mit CherryPy-Server laufen
- die Statusseite zeigt Master-, Spec- und DB-Zustand an
- die wichtigsten Laufparameter lassen sich per API und UI ändern
- alle Änderungen bleiben kompatibel mit dem bestehenden MySQL/NAS-Vertrag

## Risiken

- CherryPy bringt einen neuen optionalen Runtime-Stack mit
- konkurrierende Reconcile-Aufrufe müssen strikt serialisiert bleiben
- Array-Patches in JSON-Specs müssen klar definiert sein, damit UI und API dieselben Regeln haben
