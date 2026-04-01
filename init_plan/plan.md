{Als nächsten Schritt kann ich dir das als konkretes Tensor-Design runterschreiben: also welche Tokens, welche Shapes, welche Losses und wie der Inferenzloop pro Stellung exakt aussehen würde.} tue das - erstelle daraus einen Plan für codex-cli dies schrittweise aufzubauen, von null an, gemeint ist, die ganze engine nach diesem Prinzip zu bauen, keine Altlasten aus anderen Engines, außer eben das UCI-Protokoll
___
Ja. Ich würde das als **Bauprogramm für ein neues Repo** aufsetzen, nicht als einen einzelnen Monster-Prompt.

Für dieses Projekt würde ich Codex in zwei Modi benutzen: die **interaktive TUI** für Architektur, Reviews und größere Refactorings, und `codex exec` für klar begrenzte Phasen mit harter Definition of Done. Codex CLI läuft lokal im Repo, kann Dateien lesen, ändern und Kommandos ausführen; zusätzlich gibt es `exec` für nicht-interaktive, skriptbare Läufe und Subagents für parallele Teilaufgaben. Codex funktioniert laut OpenAI am besten mit **explizitem Kontext** und einer klaren Definition von „fertig“. ([OpenAI Entwickler][1])

Der wichtigste Schritt ganz am Anfang ist: **AGENTS.md + PLANS.md zuerst**. Codex liest `AGENTS.md` automatisch, `/init` kann eine Startversion erzeugen, und `PLANS.md` ist für längere Multi-Step-Arbeit explizit empfohlen. Ich würde außerdem vor und nach jeder Phase einen Git-Checkpoint setzen. ([OpenAI Entwickler][2])

## 1. Zielbild der Engine

Nicht bauen:

* klassische Engine mit Alpha-Beta, PVS, Null-Move, LMR, History-Heuristics, Transposition-Table als Hauptalgorithmus

Stattdessen bauen:

* **UCI-Engine**
* **symbolischer Regelkern** nur für:

  * Brettzustand
  * exakte Legalität
  * Zuganwendung
  * Testorakel / Trainingslabels
  * finale Sicherheitsprüfung vor `bestmove`
* **neuronale Runtime**:

  * Position -> Encoder -> Legality/Policy-Proposer -> Dynamics -> Opponent-Modul -> rekurrenter Planner -> WDL + Zugwahl

Wichtig: Der symbolische Teil ist hier **kein Legacy-Suchkern**, sondern der **Regelboden**. Ohne den bekommst du weder saubere Trainingsdaten noch harte Korrektheitstests.

## 2. Technische Grundentscheidung

Ich würde es so aufteilen:

* **Rust** für alles, was Runtime, UCI, Regeln, Datenstrukturen, Inferenz und Benchmarks betrifft
* **Python + PyTorch** für Training, Datengenerierung, Experiment-Orchestrierung, Auswertung
* Modellformat zunächst pragmatisch:

  * Training in PyTorch
  * Export in ein stabiles Inferenzformat
  * später optional eigene kleine Rust-Inferenzschicht für CPU

Warum so:

* Rust hält den Runtime-Pfad sauber und schnell
* Python macht die Modelliteration drastisch einfacher
* du vermeidest, die schwierigsten ML-Teile zu früh in Systems-Code einzubetonieren

## 3. Repo-Struktur

So würde ich das Repo von Anfang an schneiden:

```text
latent-chess/
  AGENTS.md
  PLANS.md
  README.md
  docs/
    architecture/
    experiments/
    phases/
  rust/
    Cargo.toml
    crates/
      engine-app/        # Binary, UCI loop
      uci-protocol/      # UCI parser/state
      core-types/        # colors, pieces, squares, moves
      position/          # board state + fen + repetition + clocks
      rules/             # legal move generation + move application
      action-space/      # move vocabulary / tokenization
      encoder/           # position -> tensors/tokens
      inference/         # model loading + inference bridge
      planner/           # latent planning loop
      eval-metrics/      # tactical suites, logs, diagnostics
      selfplay/          # Rust-side selfplay runtime if needed
      tools/             # perft, dataset sanity, converters
  python/
    pyproject.toml
    train/
      datasets/
      models/
      losses/
      trainers/
      export/
    scripts/
      make_dataset.py
      train_legality.py
      train_dynamics.py
      train_planner.py
      eval_suite.py
  tests/
    perft/
    positions/
    planner/
  models/
  artifacts/
```

## 4. Harte Projektregeln für Codex

Das hier würde ich wirklich in `AGENTS.md` schreiben.

```md
# Mission
Build a chess engine from scratch around latent adversarial planning.
Runtime path:
position -> encoder -> legality/policy proposer -> latent dynamics -> opponent model -> recurrent planner -> WDL + selected move -> UCI output

# Hard constraints
- No copied engine code from other engines.
- UCI is the only required compatibility surface.
- Do NOT implement alpha-beta, negamax, PVS, quiescence search, null-move pruning, LMR, killer/history heuristics, or TT search as runtime features.
- A symbolic rules kernel is allowed only for legality, state transitions, training labels, regression tests, and final legality verification before emitting a UCI move.
- No handcrafted static evaluation as a fallback engine.
- Runtime code in Rust stable.
- Training and experiments in Python.
- Keep modules small and testable.
- Preserve buildability at every phase.

# Working style
- Read this file and PLANS.md before editing.
- Work only on the current phase.
- Do not pre-implement future phases.
- If a task would require a conventional-search shortcut, stop and report it instead of adding it.

# Definition of done
- Rust: cargo fmt, cargo clippy --all-targets -- -D warnings, cargo test
- Python: ruff check, pytest
- Update docs when interfaces change.

# Validation priorities
1. Rule correctness
2. Illegal-move safety
3. Reproducibility
4. Planner quality
5. Speed
```

## 5. Master-Roadmap in Phasen

### Phase 0 — Bootstrap

Ziel:

* Repo anlegen
* Rust workspace + Python project
* `AGENTS.md`, `PLANS.md`, `README.md`
* minimale CI/Lint/Test-Kommandos
* leere Crates mit klaren Interfaces

Definition of Done:

* `cargo test` läuft
* `pytest` läuft
* `engine-app` startet
* CI ist grün

Nicht tun:

* noch keine Regeln
* noch kein ML
* noch kein Planner

### Phase 1 — Brettzustand und Regelkern

Ziel:

* Board-Repräsentation
* FEN parse/serialize
* Move-Repräsentation
* Castling, en passant, fifty-move, repetition state
* legale Züge
* Zuganwendung

Definition of Done:

* Roundtrip-Tests für FEN
* Sonderregeltests
* Perft-Suite stimmt

Bemerkung:
Das ist die wichtigste Phase. Wenn die nicht mathematisch sauber ist, ist alles danach kontaminiert.

### Phase 2 — UCI-Skelett

Ziel:

* `uci`, `isready`, `ucinewgame`, `position`, `go`, `stop`, `quit`
* internes Game-State-Handling
* Debug-Logging
* ein extrem einfacher `go`-Pfad, der nur einen legalen Stub-Zug ausgibt

Definition of Done:

* GUI kann verbinden
* Engine spielt vollständige legale Züge
* keine Suchlogik, kein Eval-Fallback

### Phase 3 — Action Space + Encoder

Ziel:

* feste Zugvokabel definieren
* Faktorisiertes Schema:

  * from
  * to / move-type
  * promotion
* objektzentrierter Encoder:

  * piece tokens
  * rule token
  * optional square tokens
  * relationale Features

Definition of Done:

* deterministische Kodierung
* Tests für encode/decode
* Tensor-/Token-Schema dokumentiert

### Phase 4 — Datapipeline

Ziel:

* Positionen erzeugen
* Labels ableiten:

  * legal move set
  * next state
  * WDL surrogate / Spielausgang
* Sonderfall-Buckets:

  * in-check
  * castling
  * en passant
  * promotions
  * pinned pieces
  * endgame-artige Stellungen

Definition of Done:

* ein reproduzierbarer Dataset-Build
* Train/val/test Splits
* Sanity-Reports

Hinweis:
Hier darfst du intern Hilfsorakel bauen, aber nicht versehentlich einen klassischen Suchkern als Produkt-Engine etablieren.

### Phase 5 — Legality/Policy-Proposer v1

Ziel:

* erstes Modell:

  * Legal-Set-Head
  * Policy-Head
* Export und Rust-Inferenzschicht
* UCI-Stub nutzt Modell nur zur Kandidatenpriorisierung, noch mit symbolischer Kontrolle

Definition of Done:

* legale Züge werden mit hoher Recall/Precision erkannt
* Spezialfall-Metriken separat
* Modell läuft aus Rust heraus

Wichtige Metriken:

* legal-set precision/recall
* recall in check positions
* recall bei promotions / en passant / castling

### Phase 6 — Dynamics-Modell v1

Ziel:

* `z = E(s)`
* `z' = G(z, a)`
* Rekonstruktion von:

  * nächstem Brettzustand
  * Regelzustand
* one-step und multi-step Konsistenz

Definition of Done:

* hohe exakte Rekonstruktionsrate
* Fehlerberichte nach Motivklassen
* Rust kann `E` + `G` ausführen

Wichtige Metriken:

* exact next-state accuracy
* special-move accuracy
* drift über 2/4/8 Schritte

### Phase 7 — Opponent-Modul + 2-Ply Latent Planner

Ziel:

* eigenes Reply-Modul `O(z)`
* top-k Kandidaten
* imaginierte Gegnerantwort
* soft-min Aggregation
* WDL-Head am Root

Runtime:

* proposer -> imagine our move -> imagine opponent reply -> aggregate -> choose

Definition of Done:

* besser als proposer-only baseline
* gute Treffer auf mate-in-1 / triviale Taktikfälle
* stabile WDL-Ausgabe

Ab hier beginnt die Engine wirklich „nach diesem Prinzip“ zu spielen.

### Phase 8 — Rekurrenter Planner mit Memory

Ziel:

* mehrere innere Denkschritte
* Memory-Slots für Kandidatenlinien
* Unsicherheitswerte
* Reallokation des Denkbudgets auf instabile Zweige

Definition of Done:

* messbarer Gewinn gegenüber 2-ply latent planner
* bessere taktische Robustheit
* sinnvolle Kalibrierung von Unsicherheit

### Phase 9 — Selfplay und Curriculum

Ziel:

* Selfplay-Loop
* Replay Buffer
* Curriculum:

  * erst Regel- und Kurzfristigkeit
  * dann Taktik
  * dann längere strategische Horizonte
* Modellversionierung

Definition of Done:

* reproduzierbarer Trainingslauf
* Evaluation gegen ältere Checkpoints
* automatische Reports

### Phase 10 — UCI-Runtime ohne klassische Suche

Ziel:

* `go` benutzt nur:

  * proposer
  * dynamics
  * opponent
  * planner loop
* kein Alpha-Beta-Fallback
* nur noch finale symbolische Sicherheitsprüfung vor `bestmove`

Definition of Done:

* volle Partien
* keine illegalen UCI-Züge
* Logging für:

  * invalid proposal rate
  * planner disagreement
  * fallback count

Hier wäre ich ehrlich: Für eine Release-Engine würde ich die **letzte symbolische Legalitätsprüfung** behalten. Das ist kein Such-Altlasten-Paket, sondern eine Protokoll- und Sicherheitskante.

### Phase 11 — Härtung

Ziel:

* Benchmarks
* Reproduzierbarkeit
* Determinismus wo sinnvoll
* Arena-Harness
* Regression-Suites
* Modell-Registry
* Dokumentation

Definition of Done:

* neue Modelle sind vergleichbar
* Rule regressions werden sofort entdeckt
* Runtime regressions ebenso

## 6. Messgrößen, die du ab Tag 1 tracken solltest

Immer mitschreiben:

* Rule correctness
* Perft correctness
* legal-set precision/recall
* exact next-state accuracy
* invalid move proposal rate
* invalid final move rate
* WDL calibration
* move latency
* full-game completion rate
* planner-depth vs. quality
* special-case failure buckets

Ohne diese Metriken wirst du später nicht wissen, ob das Netz wirklich „plant“ oder nur oberflächlich plausible Züge rät.

## 7. `PLANS.md`-Format

Für Codex würde ich jede Phase als kleinen Ausführungsplan formulieren:

```md
# Phase X

## Goal
One paragraph.

## Non-goals
- ...
- ...

## Deliverables
- ...
- ...

## Files expected to change
- rust/crates/...
- python/...

## Tests to add/run
- ...
- ...

## Exit criteria
- ...
- ...

## Risks / open questions
- ...
```

Das hält Codex auf Spur. Für längere Arbeiten ist genau so ein Planformat nützlich. ([OpenAI Entwickler][3])

## 8. Wie du Codex praktisch führst

Ich würde **eine Phase = ein `codex exec`-Lauf** machen. Für offene Architekturfragen zuerst in die TUI, für Implementierung dann `exec`. OpenAI dokumentiert die TUI für interaktive Arbeit und `codex exec` für skriptbare, nicht-interaktive Läufe; bei komplexen, parallelisierbaren Aufgaben kannst du Subagents einsetzen. ([OpenAI Entwickler][1])

Die erste Routine wäre:

1. Repo anlegen
2. `codex`
3. `/init`
4. generiertes `AGENTS.md` komplett durch dein projektspezifisches ersetzen
5. `PLANS.md` mit Phase 0 befüllen
6. Git-Checkpoint
7. `codex exec` für Phase 0
8. Review
9. Git-Checkpoint
10. nächste Phase

## 9. Copy-Paste-Prompts für die ersten Schritte

### Kickoff: Repo-Bootstrap

```bash
codex exec "Read AGENTS.md and PLANS.md. Implement Phase 0 only.
Create the Rust workspace, Python project, docs tree, empty crates, CI/lint/test wiring, and a minimal engine-app binary.
Do not implement chess rules yet.
Run all relevant tests.
Stop exactly at the Phase 0 exit criteria and summarize changed files and any open issues."
```

### Phase 1: Regelkern

```bash
codex exec "Read AGENTS.md and PLANS.md. Implement Phase 1 only.
Build the exact symbolic chess state and rules kernel from scratch:
board state, FEN parse/serialize, move representation, legal move generation, move application, castling, en passant, fifty-move state, and repetition bookkeeping.
Add thorough tests and perft coverage.
Do not add any search, evaluation, or engine heuristics.
Run all Rust tests and stop at the Phase 1 exit criteria."
```

### Phase 2: UCI-Skelett

```bash
codex exec "Read AGENTS.md and PLANS.md. Implement Phase 2 only.
Add UCI protocol handling to engine-app: uci, isready, ucinewgame, position, go, stop, quit.
Use the symbolic rules kernel to maintain state and emit a simple legal stub move on go.
Do not add search or evaluation.
Add protocol tests where practical and stop at the Phase 2 exit criteria."
```

### Phase 3: Action Space + Encoder

```bash
codex exec "Read AGENTS.md and PLANS.md. Implement Phase 3 only.
Design and implement the action-space crate and the object-centric encoder.
The encoder must produce deterministic tokens/features for pieces, rules, and optional square relations.
Document the tensor/token schema and add roundtrip tests.
Do not start model training yet."
```

## 10. Wann Subagents sinnvoll sind

Ab Phase 5 würde ich in der TUI mit `/agent` parallel arbeiten lassen, zum Beispiel:

* Agent A: überprüft nur das Action-Space-Design
* Agent B: entwirft Dataset-Schema und Losses
* Agent C: baut Rust-Inferenz-API
* Hauptthread: integriert Entscheidungen

Subagents sind laut OpenAI genau für solche parallelen Multi-Step-Features gedacht. ([OpenAI Entwickler][4])

## 11. Der wichtigste Nicht-Fehler

Lass Codex **nicht** “einfach eine Schachengine” bauen.

Wenn du die Regeln nicht hart in `AGENTS.md` schreibst, wird fast jeder Coding-Agent früher oder später in bekannte Muster kippen:

* Minimax
* Alpha-Beta
* Hash Table
* klassische Move Ordering Heuristics
* handgestrickte Eval

Das ist für normale Engines vernünftig — für dein Projekt aber genau die falsche Gravitation.

## 12. Mein pragmatisches Urteil

Das risikoreichste Teilstück ist **nicht** UCI, nicht Rust, nicht Training, sondern der Übergang:

* von „Legality-Netz mit symbolischer Maske“
* zu „Planner-Netz, das fast autonom zuverlässig bleibt“

Darum ist die richtige Reihenfolge:

1. exakte Regeln
2. saubere Daten
3. Legalität
4. Zustandsdynamik
5. Gegnerantwort
6. rekurrentes Planen
7. erst dann Autonomiereduktion des symbolischen Helfers

Genau so vermeidest du, dass das Projekt in Month 2 entweder in einer klassischen Engine endet oder in einem unprüfbaren Forschungsprototyp.

Wenn du willst, formuliere ich dir als Nächstes **die konkrete `PLANS.md` für Phase 0 bis Phase 3** und ein **fertiges `AGENTS.md` in finaler Form zum Einchecken**.

[1]: https://developers.openai.com/codex/cli/features?utm_source=chatgpt.com "Features – Codex CLI | OpenAI Developers"
[2]: https://developers.openai.com/codex/guides/agents-md?utm_source=chatgpt.com "Custom instructions with AGENTS.md – Codex | OpenAI Developers"
[3]: https://developers.openai.com/cookbook/articles/codex_exec_plans?utm_source=chatgpt.com "Using PLANS.md for multi-hour problem solving"
[4]: https://developers.openai.com/codex/subagents?utm_source=chatgpt.com "Subagents – Codex | OpenAI Developers"
___
leg los - arbeite auch für die folgephasen
___
Ich habe dir die Dateien fertig vorbereitet — inklusive der Folgephasen, nicht nur 0–3.

Dateien:

* [AGENTS.md](./plan_files/AGENTS.md)
* [PLANS.md](./plan_files/PLANS.md)
* [CODEX_RUNBOOK.md](./plan_files/CODEX_RUNBOOK.md)

Inhaltlich ist das Paket so geschnitten, dass Codex sauber geführt wird: `AGENTS.md` als harte Projektverfassung, `PLANS.md` als phasenweiser Masterplan von Phase 0 bis 11, und `CODEX_RUNBOOK.md` als Operator-Guide mit fertigen `codex exec`-Prompts für jede Phase. Das passt gut zu den offiziellen Empfehlungen: Codex liest `AGENTS.md` vor der Arbeit, `/init` kann ein Grundgerüst erzeugen, und `PLANS.md` ist laut OpenAI gerade für lange, mehrstufige Arbeiten der empfohlene Mechanismus. Die TUI ist für interaktive Architektur- und Review-Arbeit gedacht, `codex exec` für begrenzte, reproduzierbare Läufe; standardmäßig erwartet `codex exec` außerdem ein Git-Repository. ([OpenAI Entwickler][1])

Kurz zur Nutzung:

1. Dateien ins Repo-Root legen und committen.
2. `git init` bzw. in einem bestehenden Git-Repo arbeiten.
3. Mit `codex` oder direkt mit dem passenden `codex exec`-Prompt aus dem Runbook starten.
4. Phase für Phase arbeiten, ohne klassische Suchlogik als “temporären Helfer” einzuschmuggeln.
5. Für spätere Parallel-Audits kannst du Subagents gezielt einsetzen; OpenAI dokumentiert sie als explizit aufrufbare Parallel-Workflows. ([OpenAI Entwickler][2])

Wichtig: Ich habe die Plan-Dateien bewusst so formuliert, dass der symbolische Kern nur Regeln, Labels und Sicherheitsprüfung abdeckt — nicht als versteckte klassische Engine. Genau das ist die zentrale Leitplanke des ganzen Setups.

Als nächstes würde ich an deiner Stelle mit **Phase 0 per `codex exec`** loslegen und danach sofort **Phase 1** in einem frischen Branch angehen.

[1]: https://developers.openai.com/codex/guides/agents-md "Custom instructions with AGENTS.md – Codex | OpenAI Developers"
[2]: https://developers.openai.com/codex/cli/features "Features – Codex CLI | OpenAI Developers"
