# Netzwerkanforderungen und geeignete Architekturen für TotoBa/EngineKonzept

## Executive Summary

Das private Repository **TotoBa/EngineKonzept** ist aktuell (Stand: Phase‑5‑Snapshot) **stark auf deterministische, lokal reproduzierbare Workflows** ausgelegt: Der Rust‑Runtime‑Pfad exponiert als einziges externes Protokoll **UCI** (über **stdin/stdout**, also *kein* Netzwerk) und die Python‑Dataset‑Pipeline nutzt eine **Rust‑„Dataset‑Oracle“** ebenfalls über **stdin/stdout** (per `subprocess`, standardmäßig `cargo run ... dataset-oracle`). fileciteturn10file0L1-L1 fileciteturn23file0L1-L1 fileciteturn41file0L1-L1  
Im Repo ist **keine** Netzwerk‑Implementierung (TCP/UDP/HTTP/gRPC/QUIC) vorhanden; selbst die „Planner“‑ und „Selfplay“‑Crates sind derzeit Platzhalter. fileciteturn29file0L1-L1 fileciteturn31file0L1-L1

Aus den Projekt‑„Guardrails“ lässt sich dennoch ein klares **Netzwerk‑Designziel** ableiten: Wenn verteilte Komponenten (Selfplay, Labeling/Oracle‑Farm, Inference‑Serving) später nötig werden, müssen sie **Architektur‑Disziplin (Rust‑Runtime vs. Python‑Training), Determinismus/Checkbarkeit und testbare Schnittstellen** bewahren. fileciteturn11file0L1-L1 fileciteturn16file0L1-L1 fileciteturn15file0L1-L1  

Empfehlung in drei Stufen (robust gegen aktuell unbekannte Deployment‑Randbedingungen):  
Erstens eine **Local‑First IPC‑Architektur** (Unix Domain Socket oder Named Pipe) als Drop‑in‑Ersatz für `subprocess`‑Spawns der Dataset‑Oracle (niedriges Risiko, minimaler Dependency‑Zuwachs). Zweitens optional ein **gRPC‑basiertes Service‑Interface** (Protobuf, TLS/mTLS), wenn Komponenten wirklich über Hosts hinweg skaliert werden müssen. Drittens für massenhaft asynchrone Workloads (Selfplay/Batch‑Labeling/Eval) ein **Message‑Broker‑Ansatz** (z. B. NATS JetStream) mit klaren Datenverträgen und reproduzierbarer Artefakt‑Versionierung. citeturn8search3turn4search5turn17search1turn5search0turn18search0turn18search3

Die größten „Unknowns“ sind aktuell **Skalierungs- und Echtzeitannahmen** (UCI‑Zeitbudget vs. Remote‑Inference‑Latenz, Cluster‑/Kubernetes‑Ziel, Sicherheits‑/Trust‑Boundary). Diese sind im Bericht explizit als offene Anforderungen aufgeführt und in den Architektur‑Optionen als Entscheidungshebel genutzt.

## Repositorium-Analyse

EngineKonzept verfolgt explizit eine **nicht‑klassische** Engine‑Architektur („latent adversarial planning“) mit einem Ziel‑Runtime‑Pfad `position -> encoder -> legality/policy proposer -> latent dynamics -> opponent module -> recurrent planner -> WDL + move selection -> UCI output`. fileciteturn10file0L1-L1 fileciteturn11file0L1-L1  
Wesentlich ist dabei die **Sprach- und Verantwortungsgrenze**: *Runtime/Protokolle in Rust*, *Datasets/Training/Experimente in Python*. fileciteturn10file0L1-L1 fileciteturn11file0L1-L1 fileciteturn16file0L1-L1

### Explizite Schnittstellen, Protokolle und Datenflüsse im Repo

Der einzige „externe“ Runtime‑Eingang ist **UCI**, derzeit als minimaler Stdio‑Loop implementiert. `engine-app` liest zeilenweise, parst UCI‑Commands und schreibt Responses auf stdout; `go` liefert deterministisch einen legalen Stub‑Zug. fileciteturn23file0L1-L1  
Das Protokollmodell (`uci-protocol`) ist ein **reiner Parser/Formatter** ohne Netzwerkbezug. fileciteturn25file0L1-L1

Für die Dataset‑Pipeline existiert eine zweite, implizit „protokollartige“ Schnittstelle: Das Rust‑Binary **`dataset-oracle`** verarbeitet **JSON‑Lines über stdin** und gibt je Input‑Record eine JSON‑Zeile aus. fileciteturn33file0L1-L1 fileciteturn34file0L1-L1  
In Python wird diese Oracle standardmäßig via `subprocess.run(...)` mit `cargo run --quiet -p tools --bin dataset-oracle` aufgerufen; ein Environment‑Override `ENGINEKONZEPT_DATASET_ORACLE` erlaubt ein alternatives Kommando. Dieser Hook ist zentral, weil er später **ohne Breaking Change** auf IPC/Netzwerk‑Clients umgebogen werden kann. fileciteturn41file0L1-L1

Die Phasen‑Dokumentation bestätigt, dass UCI „die einzige erforderliche externe Protokolloberfläche“ ist und dass Phase‑4/5 Artefakte und Durchsatzmessungen primär offline/experimentell sind. fileciteturn12file0L1-L1 fileciteturn45file0L1-L1

### Performance- und Qualitätsleitplanken, die Netzwerktechnik beeinflussen

Im Repo sind die wichtigsten „Constraints“ weniger Netzwerk‑spezifisch, aber stark netzwerk‑relevant, weil sie die zulässige Architekturform definieren:

Determinismus/Testbarkeit/Reproduzierbarkeit sind zentrale Designregeln („Every phase must leave the tree buildable, testable, and documented.“). fileciteturn11file0L1-L1 fileciteturn10file0L1-L1  
Label‑Semantik muss am **exakten Rust‑Rules‑Kernel** hängen: Python soll Labels **nicht** nachimplementieren, sondern Records an die Oracle geben. fileciteturn15file0L1-L1 fileciteturn41file0L1-L1  
Die Workspace-/Dependency‑Struktur ist aktuell absichtlich **schlank** (keine Async‑Runtime, kein HTTP‑Stack), was gegen „zu früh“ eingeführte Netzwerk‑Frameworks spricht, solange Anforderungen unklar sind. fileciteturn19file0L1-L1

### Findings zu „Pla“-Implementierung, Security Notes und Tests

Eine konkrete Implementierung eines „Planner“/„PLA“ (im Sinne von latentem Planer) ist **noch nicht vorhanden**; die Crate `planner` ist ein Placeholder. fileciteturn29file0L1-L1  
Ebenso ist `selfplay` aktuell Placeholder. fileciteturn31file0L1-L1

Explizite **Security Notes** bezogen auf Netzwerk (TLS, AuthN/Z) existieren im Repo nicht, weil keine Netzwerkoberfläche implementiert ist. Das Repo hat jedoch eine klare **Sicherheits‑/Korrektheits‑Prämisse**: keine illegalen Züge über UCI, symbolische Regeln bleiben Autorität. fileciteturn11file0L1-L1 fileciteturn12file0L1-L1  

Tests sind stark auf Korrektheit/Determinismus ausgelegt: z. B. UCI‑Session Smoke‑Tests im Rust‑Runtime‑Loop. fileciteturn23file0L1-L1  
Diese Testkultur sollte bei jeder Netzwerk‑Einführung fortgeführt werden (Contract‑Tests, deterministische Fixtures, Regression).

## Abgeleitete Netzwerkanforderungen

### Explizite Anforderungen aus dem Repo

UCI ist die **einzige** geforderte externe Protokolloberfläche der Engine (aktuell über stdio). Das impliziert: Netzwerk darf UCI nicht ersetzen, sondern höchstens ergänzen (z. B. Remote‑Kontrolle/Telemetry). fileciteturn12file0L1-L1 fileciteturn23file0L1-L1  
Strikte Trennung Rust‑Runtime vs. Python‑Training: Netzwerk‑Architektur muss diese Boundary respektieren (z. B. „Inference‑Serving“ als separater Dienst ohne UCI‑Logik). fileciteturn11file0L1-L1  
Die Dataset‑Oracle ist eine „Authority“ für Labels; jede verteilte Architektur muss diese Rolle bewahren und die Semantik versionieren/testen. fileciteturn15file0L1-L1

### Implizite Anforderungen aus Codepfaden und Tooling

Die existierende Oracle‑Integration ist **batch‑orientiert** (alle Records als JSONL‑String, round‑trip über Child‑Process). Ein Netz‑ oder IPC‑Design sollte daher **Streaming und/oder Batching** unterstützen (für Durchsatz), ohne dabei deterministische Zuordnung Input↔Output zu verlieren (Python prüft `len(outputs) == len(records)`). fileciteturn41file0L1-L1  
„Dependency‑Footprint“ ist implizit kritisch: Der Rust‑Workspace kommt aktuell ohne Tokio/HTTP‑Stacks aus. Ein späterer Netz‑Stack sollte daher nur eingeführt werden, wenn klarer Nutzen besteht und idealerweise modular isoliert wird (z. B. neue Crate `net-api`). fileciteturn19file0L1-L1

### Fehlende oder mehrdeutige Anforderungen

Die folgenden Punkte sind im Repo **nicht spezifiziert** und müssen als „unknown“ behandelt werden (sie sind die wichtigsten Architektur‑Schalter):

- Ziel‑Deployment: Desktop‑Binary, Server‑Daemon, Kubernetes‑Cluster, HPC‑Nodes, CI‑Only? (nicht angegeben). fileciteturn10file0L1-L1  
- Erwartete Skalierung: Anzahl paralleler Selfplay‑Games, Dataset‑Records pro Sekunde, gleichzeitige Clients. (nicht angegeben). fileciteturn12file0L1-L1  
- Echtzeit‑Budget: zulässige Latenz für Move‑Selection unter UCI‑Zeitkontrolle vs. Remote‑Inference. (noch nicht implementiert; `go`‑Zeitmanagement ist explizit Non‑Goal). fileciteturn46file0L1-L1  
- Trust‑Boundary/Sicherheitsmodell: läuft alles „localhost“, oder muss über unsichere Netze (LAN/WAN) abgesichert werden? (keine Security‑Spez).  
- Betriebsanforderungen: Rolling updates, Backward compatibility von Schemas, Zertifikats-/Key‑Management, Observability‑Backend (Prometheus/OTel/Log‑Stack).  
- Datenpersistenz & Artefakt‑Kanon: Wo liegen Datasets/Model‑Bundles in verteilten Setups (S3/NFS/Artifact store)? (im Repo nur als Verzeichnisstruktur, nicht als Betriebsmodell beschrieben). fileciteturn10file0L1-L1  

## Architektur-Optionen und Bewertung

### Bausteine, die alle Optionen abdecken müssen

**Transport/Protokolle.** Klassisch TCP (zuverlässig), UDP (niedrige Latenz, aber unzuverlässig), QUIC (über UDP, mit Streams + schnellerem Handshake und Mobility). QUIC ist als Standard in RFC 9000 definiert und nutzt TLS zur Absicherung (RFC 9001). citeturn3search1turn16search2  
gRPC basiert konzeptuell auf Service‑Definitionen (typisch Protobuf) und hat standardisierte Auth‑Mechanismen (TLS/mTLS). citeturn8search3turn17search1

**Serialisierung.**  
Protobuf (proto3) bietet eine IDL‑Spezifikation für Messages/Services. citeturn5search1  
CBOR (RFC 8949) ist eine kompakte „JSON‑ähnliche“ Binärrepräsentation. citeturn1search2  
MessagePack ist ebenfalls ein kompaktes, mehrsprachiges Binärformat. citeturn6search1  

**Krypto/TLS.** TLS 1.3 ist in RFC 8446 spezifiziert; für deutsche Empfehlungen zur sicheren TLS‑Verwendung ist BSI TR‑02102‑2 relevant. citeturn2search1turn3search3  
In Rust ist `rustls` ein verbreiteter TLS‑Stack (TLS 1.2/1.3). citeturn4search4

**Service Discovery/Load Balancing/Resilience.**  
In Rust/Tokio‑Ökosystemen ist Tower ein gängiger Middleware‑Baukasten mit Modulen für Retry/Timeout/Load‑Balancing und auch Discovery. citeturn17search2  
Als klassische externe Discovery‑Komponente ist Consul dokumentiert. citeturn21search6

**NAT‑Traversal (nur falls P2P nötig).**  
STUN (RFC 8489), ICE (RFC 8445) und TURN (RFC 8656) sind Standardbausteine für UDP‑basierte NAT‑Traversal‑Lösungen. citeturn7search1turn7search3turn7search2  
Wenn EngineKonzept später P2P‑Selfplay „über das Internet“ braucht, werden diese Themen relevant; aktuell ist das reine Spekulation.

**Mesh vs. Client‑Server vs. P2P.**  
Service‑Mesh‑Ansätze (z. B. Linkerd/Istio) liefern mTLS und Traffic‑Management „plattformseitig“. Linkerd betont automatisches mTLS für meshed TCP‑Traffic. citeturn21search0 Istio beschreibt Traffic‑Management/TLS‑Konfiguration im Mesh. citeturn21search1turn21search5  
P2P ist möglich, erhöht aber Komplexität (NAT‑Traversal, Sicherheitsmodell, Debugging).

### Vergleichstabelle der Top‑4 Kandidaten

Bewertungsskala: **hoch / mittel / niedrig** (relativ zueinander, da Scale‑Ziele unbekannt sind).

| Kandidat-Architektur | Security | Latenz | Durchsatz | Skalierbarkeit | Fehlertoleranz | Implementierung Rust/Python | Dependency-Footprint | Testing/Observability | Maintainability |
|---|---|---|---|---|---|---|---|---|---|
| Local IPC (UDS) + (CBOR/MsgPack/JSONL) | mittel (OS‑Boundary; optional TLS unnötig) | hoch (sehr niedrig) | hoch | niedrig–mittel (Host‑gebunden) | mittel | hoch (einfach) | hoch (klein) | mittel (eigene Tracing nötig) | hoch |
| gRPC (HTTP/2) + Protobuf + TLS/mTLS | hoch (TLS/mTLS Standardpfad) citeturn17search1turn2search1turn3search3 | mittel–hoch | hoch | hoch | hoch (LB/Retry/Timeout möglich) citeturn17search2 | mittel | mittel | hoch (grpc tooling + OTel) citeturn15search0turn12search0 | hoch |
| QUIC‑RPC (quinn/aioquic) + TLS 1.3 | hoch (QUIC+TLS) citeturn3search1turn16search2 | hoch (0‑RTT/1‑RTT möglich) | hoch | hoch | mittel–hoch | niedrig–mittel (komplex) citeturn6search0turn0search2 | mittel | mittel | mittel |
| Broker/Queue (NATS JetStream) für Workloads | hoch (TLS/Auth möglich) citeturn18search4turn18search1 | niedrig–mittel (nicht für Echtzeit‑RPC) | hoch (Ingestion/Replay) citeturn5search0 | hoch | hoch (Persistenz/Replay/Replication) citeturn5search0 | mittel | mittel | hoch (Monitoring + OTel‑Integration möglich) | mittel–hoch |

Interpretation: Für EngineKonzept ist **Local IPC** der beste „Low‑Regret“‑Schritt (ersetzt subprocess‑Overhead, passt zur heutigen Architekturkultur). **gRPC** ist der beste „Skalierungs‑Upgrade‑Pfad“, wenn Oracle/Inference/Selfplay über Hosts verteilt werden müssen. **NATS JetStream** ist besonders geeignet für *asynchrone* Mass‑Workloads (Selfplay/Eval/Batch‑Labeling). QUIC ist attraktiv, wenn **Netzwerkbedingungen schwierig** sind (mobil, NAT, Paketverlust), ist aber der **riskanteste** Implementationspfad.

## Empfohlene Zielarchitekturen und Implementierung

### Empfehlung eins: Local‑First Oracle‑Daemon über Unix Domain Socket

**Rationale.** Diese Architektur adressiert das *heutige* Engpass‑Muster („Oracle via `cargo run` pro Batch“) ohne verteilte Komplexität. Sie ist kompatibel zur bestehenden Oracle‑Abstraktion (`ENGINEKONZEPT_DATASET_ORACLE`) und bewahrt deterministische Semantik (Input‑Records → Output‑Records). fileciteturn41file0L1-L1  

**Trade‑offs.** Host‑gebunden; keine horizontale Skalierung über Maschinen. Security ist primär OS‑basiert (Socket‑Dateirechte), nicht TLS‑basiert.

**Konkrete Umsetzungsschritte (Rust/Python).**  
Erstens einen neuen Binary‑Entry Point `dataset-oracle-daemon` (im `tools`‑Crate oder eigenem `net-tools`‑Crate) implementieren, der auf einer UDS‑Adresse lauscht und Requests verarbeitet, aber intern dieselbe `label_dataset_input`‑Logik nutzt. fileciteturn34file0L1-L1  
Zweitens ein stabiles Framing definieren (z. B. length‑prefix + CBOR oder MsgPack). CBOR ist RFC‑standardisiert. citeturn1search2  
Drittens Python‑seitig `label_records_with_oracle(...)` erweitern: Wenn `ENGINEKONZEPT_DATASET_ORACLE` z. B. mit `unix://` beginnt, wird ein Socket‑Client verwendet; falls nicht, bleibt der bestehende subprocess‑Pfad erhalten (Backwards kompatibel). fileciteturn41file0L1-L1  
Viertens deterministische Contract‑Tests: Fixture‑Input JSONL → identische Outputs über subprocess‑Oracle vs. Daemon‑Oracle (Golden Files).  

**Bibliotheken/Crates/Pakete.**  
Für Rust kann das UDS‑Listening zunächst ohne Tokio über `std::os::unix::net::UnixListener` umgesetzt werden (minimale Dependencies). Wenn Async‑Streaming/Parallelität nötig wird, ist Tokio der de‑facto‑Standard‑Runtime. citeturn9search0  
Für Serialisierung: `serde_cbor` (CBOR) citeturn10search0 oder MessagePack (Rust: `rmp-serde`, Python: `msgpack` auf PyPI). citeturn9search4turn6search1

**Aufwand / Risiko.** Aufwand: **niedrig–mittel** (1–2 Wochen inkl. Tests/Benchmarks). Risiko: **niedrig** (lokal, klare Fallback‑Strategie).

### Empfehlung zwei: gRPC‑Services für Oracle und optional Inference

**Rationale.** gRPC bietet eine sehr stabile Cross‑Language‑RPC‑Basis (Rust/Python), klare Service‑Contracts (Protobuf) und standardisierte Auth‑Mechanismen (TLS/mTLS). citeturn8search3turn17search1turn5search1  
Diese Option ist besonders sinnvoll, wenn EngineKonzept perspektivisch **Oracle‑Farm**, **Selfplay‑Koordination** oder **Remote‑Inference** braucht.

**Trade‑offs.** Höherer Footprint (Codegen/Protobuf, Netzwerkstack, Zertifikate), mehr Betriebsaufwand (Deployments, Discovery, LB). Dafür sehr gut test‑ und observability‑fähig (grpcurl, OTel).

**Konkrete Umsetzungsschritte.**  
Erstens Protobuf‑Schema definieren (proto3): `OracleService` mit Streaming‑RPC, z. B. `rpc Label(stream OracleInput) returns (stream OracleOutput);` (Streaming passt zu Batch‑Datenflüssen). citeturn5search1turn8search3  
Zweitens Rust‑Server mit `tonic` + `prost` erstellen; `tonic` ist gRPC over HTTP/2. citeturn4search5turn14search0turn8search2  
Drittens TLS‑Pfad: in Rust `rustls` nutzen und TLS‑Konfiguration an BSI TR‑02102‑2 ausrichten (Cipher Suites/Min‑Version/Cert‑Handling). citeturn4search4turn3search3  
Viertens Python‑Client/Server Stubs via `grpcio` und `grpcio-tools` (Quickstart dokumentiert Installation). citeturn8search0  
Fünftens Resilience: Timeouts/Retry/Load‑Balancing über Tower‑Middleware (oder gRPC‑Client‑LB je nach Umgebung). citeturn17search2  
Sechstens Observability: Rust‑seitig `tracing` für strukturierte Traces und optional OpenTelemetry Export; Python‑seitig OTel Instrumentation. citeturn14search2turn15search1turn12search0  
Siebtens Test‑Tooling: `grpcurl` für Debug/Contract‑Tests (Schema‑Browsing, JSON‑Payloads). citeturn15search0  

**Bibliotheken/Crates/Pakete.**  
Rust: `tokio` (Runtime) citeturn9search0, `tonic` citeturn4search5, `prost` citeturn14search0, `rustls` citeturn4search4, `tower` citeturn17search2, `tracing` citeturn14search2.  
Python: `grpcio`, `grpcio-tools` citeturn8search0, optional `opentelemetry-*` citeturn12search0.

**Aufwand / Risiko.** Aufwand: **mittel–hoch** (3–6 Wochen inkl. PKI/Deploy‑Story). Risiko: **mittel** (Betrieb, Cert‑Handling, Schnittstellen‑Versionierung).

### Empfehlung drei: Asynchrones Work‑Distribution‑System für Selfplay/Batch‑Jobs über NATS JetStream

**Rationale.** Selfplay/Batch‑Labeling sind typischerweise **asynchron und massiv parallel**. Ein Broker mit Persistenz/Replay reduziert Kopplung zwischen Producer/Consumer und erhöht Fehlertoleranz. JetStream beschreibt Persistenz, Replay und HA‑Konfigurationen. citeturn5search0turn18search3  

**Trade‑offs.** Nicht optimal für harte Echtzeit‑RPC (z. B. „Inference pro Zug“), da Latenzen und Semantik (at‑least‑once/exactly‑once) anders sind.

**Konkrete Umsetzungsschritte.**  
Erstens NATS Server + JetStream aktivieren (betriebsseitig). citeturn5search0  
Zweitens Subject‑Design: z. B. `enginekonzept.jobs.selfplay`, `enginekonzept.jobs.label`, `enginekonzept.results.*`.  
Drittens Message‑Schema festlegen (Protobuf/CBOR/MessagePack; CBOR ist RFC‑standardisiert). citeturn1search2turn5search1  
Viertens Worker‑Implementierungen: Rust‑Worker mit `async-nats` citeturn18search0 und Python‑Worker mit `nats-py` citeturn18search7.  
Fünftens Security: TLS + Auth (NATS Doku beschreibt TLS‑Konfiguration und Mutual‑TLS; außerdem Auth‑Mechanismen). citeturn18search1turn18search2turn18search4  

**Aufwand / Risiko.** Aufwand: **mittel** (2–4 Wochen, abhängig von Ops‑Setup). Risiko: **mittel** (Operationalisierung, Exactly‑Once‑Illusionen, Replay‑Logik).

### Netzwerk-Topologie und Dataflow-Diagramme

Aktueller Ist‑Stand (lokal, ohne Netzwerk):

```mermaid
graph TD
  GUI[Schach-GUI / Manager] -->|UCI über stdin| Engine[engine-app (Rust)]
  Engine --> Rules[rules/position (Rust)]
  Engine --> UCI[uci-protocol (Rust)]

  Py[Python Dataset/Training] -->|subprocess: cargo run dataset-oracle| Oracle[dataset-oracle (Rust tools)]
  Oracle --> Rules
  Py -->|torch.export + metadata| Bundle[Model-Bundle auf Disk]
  Engine -->|lädt metadata.json| InferenceLoader[inference crate (Rust)]
```

Begründung aus Repo: UCI‑Loop läuft über stdio fileciteturn23file0L1-L1, die Oracle wird via `subprocess.run` gestartet fileciteturn41file0L1-L1, und Inference‑Loader validiert Bundle‑Metadaten fileciteturn27file0L1-L1.

Zielbild (skalierbar, optional hybrid: IPC + gRPC + Broker):

```mermaid
graph LR
  subgraph Runtime
    GUI[UCI GUI] -->|UCI stdio| Engine[Engine (Rust)]
    Engine -->|optional gRPC mTLS| Infer[Inference Service]
  end

  subgraph DataPipeline
    Py[Python Builder/Trainer] -->|IPC (UDS) oder gRPC stream| OracleSvc[Oracle Service (Rust)]
    Workers[Selfplay/Label Workers] <--> Broker[NATS JetStream]
  end

  OracleSvc --> Artifacts[(Artifact Store: datasets/models)]
  Workers --> Artifacts
  Py --> Artifacts
```

TLS/mTLS‑Empfehlung: TLS 1.3 (RFC 8446) citeturn2search1 und BSI TR‑02102‑2 als deutsche Leitlinie für TLS‑Einsatz citeturn3search3.

## Follow-up Plan und Zeitplan

### Meilensteine, Deliverables, Tests und Benchmarks

**Milestone A: Anforderungen explizit machen (Woche 1)**  
Deliverables: ein „Network/Distribution ADR“ (Decision Record) mit Antworten auf die Unknowns (Deployment‑Ziel, Scale, Trust‑Boundary, Latenzbudget). Ausgangsbasis sind die Repo‑Leitplanken (Rust/Python‑Split, UCI‑Only‑Surface, deterministische Phasen). fileciteturn11file0L1-L1 fileciteturn12file0L1-L1  
Tests: reine Dokumenten‑Review‑Gate (kein Code).  
Security‑Review: initiales Threat‑Model (lokal vs LAN/WAN; mTLS ja/nein; Secrets‑Handling).

**Milestone B: Local‑IPC Oracle Daemon (Woche 2–3)**  
Deliverables: `dataset-oracle-daemon` + Python‑Clientpfad (Feature‑Flag über `ENGINEKONZEPT_DATASET_ORACLE`), plus Compatibility‑Mode (subprocess bleibt). fileciteturn41file0L1-L1  
Tests: Golden‑Fixture‑Tests (Outputs identisch zu subprocess‑Oracle), Fuzz/Robustness‑Tests für Framing/Parser (z. B. „malformed frames“).  
Benchmarks: Durchsatz „records/s“ und p95‑Latenz pro Record (subprocess vs. daemon), CPU‑Zeit pro 10k Records.

**Milestone C: gRPC‑Prototyp (optional, Woche 4–5)**  
Deliverables: `.proto`‑Schema, `tonic`‑Server + `grpcio`‑Client, minimaler mTLS‑Pfad, Reflection oder Descriptor‑Distribution für Tooling (`grpcurl`). citeturn4search5turn8search0turn15search0turn17search1  
Tests: Contract‑Tests (proto‑Compatibility), Integration‑Tests mit `grpcurl`‑Skripten, Fault‑Injection (Timeout/Retry über Tower). citeturn17search2  
Security‑Review: TLS‑Konfiguration gegen BSI TR‑02102‑2 prüfen (Min‑Version, Cipher, Cert‑Rotation). citeturn3search3

**Milestone D: Broker‑basierte Work‑Distribution (optional, Woche 6–7)**  
Deliverables: NATS JetStream Stream/Consumer‑Definitionen, Worker‑Skeletons (Rust `async-nats`, Python `nats-py`), Idempotency‑Schlüssel/Job‑IDs, Artefakt‑Persistenzkonzept. citeturn5search0turn18search0turn18search7  
Tests: End‑to‑End Replay‑Tests (JetStream Replay), Chaos‑Tests (Worker kill/restart), Exactly‑Once‑Semantik *explizit* testen und dokumentieren (JetStream kann „exactly once“ unterstützen, aber Applikationslogik muss es korrekt nutzen). citeturn5search0  
Security‑Review: NATS TLS/Auth aktivieren, mTLS optional. citeturn18search1turn18search2turn18search4

**Milestone E: Observability & Hardening (Woche 8)**  
Deliverables: standardisierte Traces/Logs/Metrics: Rust `tracing` + optional OpenTelemetry Export; Python OTel Instrumentation für Dataset/Service‑Clients; Dashboards/Runbooks. citeturn14search2turn15search1turn12search0  
Benchmarks: Lasttests (Throughput/Latency) unter realistischen Batch‑Größen, sowie Ressourcenprofile (CPU/RAM).  
Ergebnis: „Go/No‑Go“ für verteilte Architektur (gRPC/Broker) basierend auf gemessenen Anforderungen, nicht Spekulation.

### Vorgeschlagener Zeitplan

Woche 1: Requirements/ADR + Threat‑Model + Messplan.  
Woche 2–3: Local‑IPC Oracle Daemon + Backwards‑Kompatibilität + Benchmarks.  
Woche 4–5: gRPC‑Prototyp (falls Multi‑Host absehbar) inkl. mTLS‑Baseline.  
Woche 6–7: NATS JetStream Work‑Distribution (falls Selfplay/Batch‑Farm geplant).  
Woche 8: Observability‑Härtung + Security Review + Performance‑Regression‑Gates.

### Hinweis zur Mesh-/P2P‑Option

Service Mesh (Linkerd/Istio) kann später sinnvoll sein, wenn EngineKonzept auf Kubernetes läuft und **mTLS, Traffic Shaping und Telemetrie** „plattformseitig“ gewünscht sind. Linkerd bewirbt automatisches mTLS für meshed TCP‑Traffic, Istio dokumentiert detaillierte TLS‑Konfigurationspfade. citeturn21search0turn21search5  
P2P‑Netze (für Selfplay über NAT/WAN) wären technisch möglich, erfordern aber in der Praxis ICE/STUN/TURN‑Bausteine und ein deutlich komplexeres Sicherheitsmodell. citeturn7search3turn7search1turn7search2  
Da diese Anforderungen im Repo **nicht** formuliert sind, sollten Mesh/P2P erst nach Milestone A (Requirements) konkretisiert werden.

