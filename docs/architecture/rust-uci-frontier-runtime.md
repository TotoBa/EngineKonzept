# ExecPlan: Rust-UCI-Runtime mit Bounded Frontier Deliberation

## Status

Stand `2026-04-16`:

- ExecPlan dokumentiert
- Implementierung läuft in diesem Arbeitsgang an
- aktueller Rust-Runtime-Pfad lädt nur den exportierten symbolischen Proposer-Bundle-Vertrag

## Ziel

Der Rust-UCI-Pfad soll von der aktuellen deterministischen Stub-Auswahl zu
einem planner-driven `go` weiterentwickelt werden, ohne die Architekturgrenzen
aus `AGENTS.md` zu verletzen.

Der erste reale Schritt ist bewusst eng geschnitten:

- Rust lädt einen tatsächlich Rust-ausführbaren Bundle-Vertrag
- `go` nutzt keine lexikographische Stub-Wahl mehr, sondern einen bounded
  deliberation planner über exakte legale Kandidaten
- der Planner emittiert Runtime-Diagnostik und bleibt sichtbar verschieden von
  klassischer Suche
- der neue Planner bildet die Brücke in Richtung der stärksten aktuellen
  LAPv2-Runtime-Idee: mehr adaptive innere Deliberation statt Baum-Suche

## Nicht-Ziele

- kein Alpha-Beta-, Negamax-, PVS-, TT- oder Quiescence-Pfad
- kein MCTS oder PUCT als eigentlicher `bestmove`-Selektor
- kein stiller Python-Fallback im Rust-UCI-Lauf
- keine direkte Rust-Ausführung vollständiger LAPv1/LAPv2-Checkpoints in
  diesem Schritt; dafür fehlt heute noch der passende Exportvertrag
- kein Eingriff in den laufenden Trainingscluster

## Architekturvertrag

### 1. Erster echter Rust-Runtime-Pfad bleibt beim loadbaren Bundle-Vertrag

Heute kann Rust den offiziellen exportierten symbolischen Proposer-Bundlepfad
wirklich laden und ausführen:

- `metadata.json`
- `proposer.pt2`
- `symbolic_runtime.bin`

Darauf setzt der erste planner-driven Rust-UCI-Schritt auf. Das gewählte Netz
für den ersten vollständigen Rust-Pfad ist deshalb der aktuelle loadbare
symbolische Runtime-Bundle-Typ, nicht ein LAPv1/LAPv2-Checkpoint.

Das ist kein inhaltlicher Kurswechsel, sondern ein ehrlicher
Implementierungs-Schnitt: nur Verträge, die Rust heute wirklich ausführen kann,
dürfen der erste shipped UCI-Runtime-Pfad werden.

### 2. Bounded Frontier Deliberation statt Stub-Bestmove

Der neue Planner arbeitet ausschließlich auf dem exakten legalen Kandidatensatz
des aktuellen Brettzustands.

Er bekommt:

- legal move set
- Policy-Scores des geladenen Bundles
- symbolische Candidate-Features

Er darf:

- eine kleine Top-K-Frontier bilden
- mehrere bounded innere Schritte ausführen
- Kandidaten revisiten
- taktischen Druck und Unsicherheit sichtbar machen
- Rollback-/Revisit-artige Priorisierung im latenten Sinn approximieren

Er darf nicht:

- einen Brettbaum expandieren
- Rollouts oder Besuchszähler über Knoten führen
- Positionskinder rekursiv als Suchbaum durchsuchen

### 3. MCTS-inspirierte Form, aber kein MCTS

Die Frontier-Deliberation darf sich an MCTS-artigen Ideen orientieren:

- begrenzte Frontier statt Vollkandidatenmenge
- adaptive innere Schritte
- Unsicherheits-gesteuerte Revisit-Logik
- lokale Konkurrenz zwischen Kandidaten

Sie bleibt trotzdem AGENTS-konform:

- keine Baumstatistik
- keine UCT/PUCT
- kein exakter Suchbaum
- kein klassischer Search-Backup

### 4. Der nächste große Exportvertrag bleibt explizit separat

Vollwertige Rust-LAPv1/LAPv2-Runtime braucht später einen eigenen
Exportvertrag, der mindestens transportiert:

- Root-Encoder / trunk
- Deliberation-Cell
- Sharpness-/Halting-Pfad
- Value-/Policy-Readout
- optionale Opponent-/Dynamics-Komponenten

Dieser Schritt bereitet die UCI- und Planner-Oberfläche dafür vor, ersetzt den
späteren Exportvertrag aber nicht.

## Frontier-Design v1

### Gewählte Runtime-Semantik

Die erste Runtime-Stufe verwendet:

- exakte legale Kandidaten
- Bundle-Policy-Score als Root-Prior
- bounded Frontier von `top_k`
- adaptive Schrittzahl zwischen `min_inner_steps` und `max_inner_steps`
- per-step Revisit-/Exploration-Bonus mit schneller Abnahme
- taktischen Druck aus vorhandenen Candidate-Features:
  - capture
  - promotion
  - gives check
  - attacked-from / attacked-to
  - captured minor/major

### Adaptive Halting-Idee

Die Runtime stoppt früh, wenn:

- der Score-Vorsprung des Leaders klar genug ist
- taktischer Druck niedrig ist
- die Frontier über mehrere Schritte stabil bleibt

Sie läuft tiefer, wenn:

- die Top-Kandidaten eng beieinander liegen
- hoher taktischer Druck sichtbar ist
- Revisit-Scores noch spürbar schwanken

### Zielbezug zu `auto4`

Diese Frontier-Logik ist bewusst die Runtime-Brücke zur derzeit interessantesten
LAPv2-Hypothese:

- stärkere innere Deliberation kann nützlich sein
- aber sie muss bounded, adaptiv und beobachtbar bleiben

Der neue Rust-Pfad macht diese Steuergrößen erstmals UCI-seitig sichtbar, statt
Deliberation nur indirekt aus Arena-Ergebnissen zu erraten.

## Implementierungsschritte

1. ExecPlan dokumentieren.
2. `planner`-Crate von Placeholder auf echte Frontier-Datentypen und
   Deliberation-Logik umstellen.
3. `engine-app` auf planner-driven `go` umstellen.
4. Runtime-Konfiguration über klare Env-/Go-Budget-Grenzen ergänzen.
5. UCI-Diagnostik über `info string` ausgeben.
6. Dokumentation in `docs/architecture/uci.md`, `docs/architecture/overview.md`
   und `README.md` nachziehen.
7. Rust-Tests für Planner-Logik, Searchmoves-Filter und UCI-Smoke ergänzen.

## Tests

- `cargo fmt --all`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test --workspace`

Zusätzliche neue Regressionstests:

- Frontier respektiert `searchmoves`
- adaptive Halting reduziert Schritte bei klarem Leader
- taktischer Druck kann den Planner zu mehr Frontier-Arbeit zwingen
- `go` emittiert einen legalen `bestmove`
- Debug-/Diagnostikpfad zeigt Planner-Zusammenfassung

## Exit-Kriterien

- Rust-UCI kann mit geladenem Bundle legal vollständige Partien spielen
- `go` ist planner-driven und nicht mehr nur lexikographischer Stub
- keine klassische Suchlogik wurde eingeführt
- Frontier-/Budget-Diagnostik ist sichtbar
- Doku grenzt den Pfad klar gegen MCTS und klassische Suche ab

## Risiken

- die erste Frontier-Runtime bleibt noch schwächer als der Python-LAPv1/LAPv2-Pfad,
  weil der Bundle-Vertrag enger ist
- zu aggressive Heuristiken könnten nur scheinbar „deliberieren“, ohne reale
  Spielstärke zu gewinnen
- ohne späteren LAPv1/LAPv2-Exportvertrag bleibt Rust zunächst auf den
  schmaleren Runtime-Bundle-Typ begrenzt
