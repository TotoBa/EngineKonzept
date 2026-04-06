# Architekturplan: LAPv1 — Latent Adversarial Planner v1

**Ziel:** Eine einheitliche, handlungsfähige Arm-Architektur, die alle bestehenden Planner-Arme ersetzt und die empirisch stärkste Richtung (rekurrente Deliberation) mit reichhaltigerer symbolischer Eingabe, Piece-Intentions und größerer Netz-Kapazität kombiniert — ohne klassische Suche und ohne die Architekturgrenzen aus `AGENTS.md` zu verletzen.

---

## 1. Was der lange Lauf uns gesagt hat

Aus `phase9-evolution-round03-vice-v1-summary-2026-04-05.md`:

- **Klarer Gewinner:** `planner_recurrent_expanded_v1` — finales `top1=0.819858`, `MRR=0.891312`, interne Arena `15.0/24` (`0.625`). Einziger Arm, der sowohl Verify- als auch interne Arena-Tabelle anführt. Delta über `10` Runden: `top1 +0.0128`, `MRR +0.0057`.
- **Stabiler Feed-Forward-Baseline:** `planner_set_v6_expanded_v1` — `top1=0.809220`.
- **MoE-Bild gemischt:** `moe_v1` degradiert unter Selfplay-Korrektur (`0.817 → 0.800`), `moe_v2` arena-robust, verify-schwach.
- **`vice_v2` bleibt weit vorn:** `23.5/24` gegen die gesamte Lernfamilie. Externer Rung nicht übersprungen — Kalibrationsbenchmark, kein Promotion-Gate.
- **Korrektur-Volumen war ausreichend:** `6.5k–7.1k` Mistake-Rows pro trainierbarem Arm über 10 Runden.

**Konsequenz:** Die Rekurrenz-Achse trägt. Aufwand-nach-Bedarf (Komplexitäts-geroutetes MoE) trägt noch nicht. Die nächste Architektur sollte Rekurrenz zum Zentrum machen und die MoE/Set-Familie als Spezialformen integrieren, nicht als parallele Armzoo weiterführen.

---

## 2. Leitprinzipien

Abgeleitet aus `AGENTS.md`, `docs/architecture/overview.md` und dem Experimentbefund:

1. **Exakte Legalität bleibt symbolisch.** Zuggenerator, Repetition, 50-Züge-Regel kommen aus dem Rust-`rules`-Crate. Keine gelernte Legalität.
2. **Keine klassische Suche zur Laufzeit.** Kein Alpha-Beta, keine Transpositionstabelle als Primärplaner, kein Null-Move, keine Killer-Heuristik. Die innere Schleife ist **bounded learned recurrent deliberation**, kein Baum.
3. **Bounded Compute.** Die innere Schleife hat eine harte Obergrenze an Iterationen (`max_inner_steps`, z.B. `8`), ein Lernsignal für Früh-Abbruch, und keine unbounded Rekursion.
4. **Ein Arm, viele Köpfe.** Statt neun paralleler Arme ein **unifizierter Stack** mit mehreren spezialisierten Köpfen, die gemeinsam trainiert werden.
5. **Eingabe ist reich, nicht nur flach.** Graph-artige Felder-Reachability, Piece-Intentions und Angriffszähler sind erstklassige Eingabestrukturen, keine nachträglich angehängten Features.
6. **Alles ist trace-bar.** Jede innere Iteration emittiert ein inspizierbares Update (Router-Wahl, Unsicherheit, gewählter Kandidat, PV-Fortschreibung). UCI-`info`-Ausgaben speisen sich direkt aus diesen Traces.
7. **Netze dürfen groß sein.** Zielgrößen `~100MB` für Value-Netz und Policy-Netz (wie vom Benutzer vorgegeben), deutlich größer als heute. CPU-Inferenz bleibt tragfähig, weil pro Zug nur eine bounded Anzahl Forward-Passes nötig ist.

---

## 3. Zielarchitektur im Überblick

```text
┌──────────────────────────────────────────────────────────────────────┐
│                      SYMBOLIC INPUT GATE (Rust)                      │
│  • exakte legale Züge           • castling/ep/rep state              │
│  • own_attacks[64], opp_attacks[64] (Angriffs-Zähler je Feld)        │
│  • reachability_graph: pro Feld die Figuren, die es erreichen können │
│  • Zugriffe auf eigenen/gegnerischen König (Schachpfade)             │
│  • global tactical flags (in_check, single_legal, promotion_near)    │
└────────────────────┬─────────────────────────────────────────────────┘
                     │ StateContextV1 (neu) + CandidateContextV2 (existiert)
                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│             PIECE-INTENTION ENCODER  (learned, ~20MB)                │
│  • pro Figur: Intention-Vektor (will_capture, is_threatened,         │
│    is_blocking_own, role_stability, king_danger_contribution)        │
│  • König: special_will (safety_distance, escape_routes, shelter)     │
│  • Cross-Attention zwischen Figuren über reachability_graph          │
└────────────────────┬─────────────────────────────────────────────────┘
                     │ piece_tokens + intentions + graph
                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│                 STATE EMBEDDER (learned, ~40MB)                      │
│  relational transformer über (piece-intentions ⊕ square-tokens)      │
│  output: z_root (latenter Zustand) + uncertainty_root                │
└────────────────────┬─────────────────────────────────────────────────┘
                     │ z_root
          ┌──────────┼───────────────┬────────────────────┐
          ▼          ▼               ▼                    ▼
    ┌──────────┐ ┌────────┐    ┌────────────┐     ┌──────────────┐
    │  VALUE   │ │SHARPNES│    │  POLICY    │     │  OPPONENT    │
    │  HEAD    │ │  HEAD  │    │  HEAD      │     │  HEAD        │
    │ ~100MB   │ │ small  │    │  ~100MB    │     │  ~30MB       │
    │ WDL+cp+σ │ │ [0..1] │    │ prior über │     │ reply dist + │
    │          │ │        │    │ legal cand │     │ pressure + σ │
    └──────────┘ └────────┘    └────────────┘     └──────────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             │
┌──────────────────────────────────────────────────┴───────────────────┐
│          BOUNDED RECURRENT DELIBERATION LOOP  (≤ max_inner_steps)    │
│                                                                      │
│  Memory M_t  (planner_memory_slots)                                  │
│  PV_scratch  (latente imaginierte Ply-Reihenfolge, nicht Game-PV)    │
│                                                                      │
│  ┌─ step t ────────────────────────────────────────────────────┐    │
│  │ 1. Score top-K Kandidaten via Policy+Value+Opponent          │    │
│  │ 2. Wähle 1..K zu verfeinernde Kandidaten (learned selector)  │    │
│  │ 3. Imaginiere Folgezustand (bounded, via latent transition)  │    │
│  │ 4. Query Opponent-Head auf diesem latenten Folgezustand      │    │
│  │ 5. Update candidate scores, z_root-Annex, Memory M_t         │    │
│  │ 6. Prüfe Abbruch:                                            │    │
│  │    • single_legal_move → sofort raus (hart kodiert)          │    │
│  │    • sharpness < q_threshold AND step ≥ 2 → raus             │    │
│  │    • top1_value_stable_for_k_steps → raus                    │    │
│  │    • step == max_inner_steps → raus                          │    │
│  │ 7. emit trace (UCI info: depth=t, seldepth, score, pv, nps)  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  → wenn sharp und budget: weiter iterieren                           │
│  → wenn Kandidat-Haken erkannt: PV_scratch auf letzten stabilen      │
│    Punkt zurückrollen (kein Board-Undo, nur Latent-Rollback)         │
└────────────────────┬─────────────────────────────────────────────────┘
                     │ refined_policy + final_value + σ
                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        OUTPUT GATE (Rust)                            │
│  • finaler Kandidat aus refined_policy (argmax mit Tie-Break)        │
│  • symbolische Legalitäts-Verifikation (hart)                        │
│  • UCI bestmove                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. Komponenten im Detail

### 4.1 Symbolic Input Gate → `StateContextV1`

**Wo:** neues Modul in `rust/crates/encoder` + Python-Spiegelung in `python/train/datasets/contracts.py`. Erweitert `POSITION_FEATURE_SIZE` nicht direkt, sondern definiert eine separate, versionierte State-Context-Struktur.

**Felder pro Feld (`64 × ...`):**
- `occupant_code` (existiert)
- `own_attackers_count` (0..16)
- `opponent_attackers_count` (0..16)
- `reaches_own_king` (bool)
- `reaches_opp_king` (bool)
- `is_pinned_axis` (2 bit: orthogonal/diagonal/none)
- `x-ray_attackers_count` (sliders hinter blocker)

**Global (`~12` Zahlen):**
- `in_check`, `gives_double_check_possible`
- `own_king_attackers_count`, `opp_king_attackers_count`
- `own_king_escape_squares`, `opp_king_escape_squares`
- `material_phase` (0..1 normalisiert)
- `single_legal_move`
- `legal_move_count_normalized`

**Reachability-Graph (sparse, `~80` Kanten):** Pro Feld Adjazenzlisten für die Figuren, die es erreichen können (Springer direkt, Läufer/Turm/Dame über Ray-Casting mit Blocker-Berücksichtigung, Bauern inkl. Schlagzug, König direkt). Wird als Edge-Liste in drei Tensoren ausgegeben: `src_square[]`, `dst_square[]`, `piece_type[]`.

**Warum das reicht:** Angriffspfade und Blocker-Dynamik sind die entscheidenden symbolischen Features, die ein Netz sonst erst lernen müsste. Der `rules`-Crate erzeugt sie ohnehin beim `legal_moves`-Filter — wir exportieren, was schon berechnet wird.

**Harte Grenze:** `StateContextV1` enthält **keine** Material-Heuristik, keinen Piece-Value-Table, keine handgeschriebene Evaluation. Nur exakte geometrische und rule-state Größen.

---

### 4.2 Piece-Intention Encoder

**Wo:** `python/train/models/intention_encoder.py` (neu).

**Input:** Piece-Tokens (32 Stück, wie bereits) + pro Figur die zugehörigen Zeilen aus dem Reachability-Graph + dem Angriffszähler des Ursprungsfelds.

**Per-Piece Intention-Vektor (~64 Dim):**
- `wants_to_capture_target` (Attention-Scores über erreichbare besetzte Gegnerfelder)
- `is_attacked_count_encoded` / `defenders_count_encoded`
- `blocks_own_line` (kreuzt sich mit eigenen Slider-Rays)
- `role_stability` (wahrscheinlichkeit, dass Figur kurzfristig wegmuss)
- `king_danger_contribution` (Feinddruck am eigenen König)
- `piece_type_embedding`

**King-Special-Will (~64 Dim):**
- `safety_distance_to_own_pawns`
- `escape_ray_count`
- `attacker_proximity`
- `castled_flag`, `castling_rights_bitmask`

**Architektur:** Kleiner Transformer mit `4` Layern, `128` Hidden-Dim, cross-attention über `reachability_graph`-Kanten. Parametercount `~20MB`.

**Gemeinsames Training:** Als Hilfskopf produziert der Encoder `piece_role_labels` (unsupervised, bootstrapped aus selbstkonsistenten Stockfish-Trace-Daten). Reines Auxiliary-Loss — die Haupt-Gradienten kommen aus Policy/Value downstream.

---

### 4.3 State Embedder

**Wo:** `python/train/models/state_embedder.py` (neu).

**Input:** Piece-Intentions (`32×hidden`) ⊕ Square-Intentions (`64×hidden`) ⊕ globale Rule-State-Features.

**Architektur:** Relational Transformer, `6` Layer, `hidden=256`, `8` Heads. Self-Attention gemischt mit edge-masked Attention entlang des Reachability-Graphen. Output: `z_root ∈ ℝ^{state_dim}` (`state_dim=512`) + Unsicherheit `σ_root`.

**Parametercount:** `~40MB`.

**Exportpfad:** `torch.export` + Rust-Metadata-Validierung wie bereits in `rust/crates/inference/src/lib.rs` etabliert.

---

### 4.4 Die vier Heads

| Head | Input | Output | Größe | Zweck |
|------|-------|--------|-------|-------|
| **Value** | `z_root`, (optional) `M_t` | WDL (3-class) + cp-Skalar + `σ_v` | ~100MB | NNUE-ähnlicher Endwert, wie vom Benutzer vorgegeben |
| **Sharpness** | `z_root` | scalar ∈ [0,1] | ~1MB | Lernsignal "weiter iterieren?" — Quiescence als Lernaufgabe |
| **Policy** | `z_root`, `CandidateContextV2` pro legalem Zug | prior über exakte legale Kandidaten | ~100MB | großer Kandidaten-Scorer wie vom Benutzer vorgegeben |
| **Opponent** | `z_root` + imaginierter Folge-Latent, reply-candidates | reply-dist + pressure + `σ_o` | ~30MB | existierende `OpponentHeadV1`-Erweiterung |

**Warum groß:** Stockfish-NNUE ist `~100MB` — ein Value-Netz dieser Größe auf CPU läuft in wenigen Millisekunden. Der Policy-Head ist größer als üblich, weil er reichere symbolische Kandidaten-Kontexte konsumiert und weniger innere Iterationen kompensiert. Beide müssen **sehr viel mehr trainiert werden** als die aktuellen kleinen Arme — das ist ein expliziter Kompromiss.

**Auxiliary Losses zum Training:**
- Value: WDL-CE + MSE auf cp-Target + calibration-KL
- Sharpness: BCE auf `|top1_cp − top2_cp| < 20cp` Label
- Policy: CE über Teacher-Top-1 + KL über Teacher-Policy + margin-loss + rank-loss (wie `set_v6`-Familie)
- Opponent: reply-CE + pressure-MSE + uncertainty-calibration

---

### 4.5 Bounded Recurrent Deliberation Loop

**Wo:** `python/train/models/deliberation.py` (neu), Rust-Laufzeit-Zwilling in `rust/crates/planner` (bisher Placeholder).

**Zustand pro Schritt `t`:**
- `z_t`: latenter Zustand (angereichert über Iterationen)
- `M_t`: Memory-Slots (`memory_slots=16`)
- `C_t`: Kandidaten-Scores (Vektor über legale Kandidaten)
- `PV_scratch_t`: imaginierte Top-Linien-Latents (Liste von Latents, nicht Positions)
- `history_t`: bisher verworfene Kandidaten mit Gründen

**Schritt-Update:**
```text
1. selector(z_t, C_t, σ_t)  →  top-K Kandidaten-Indices zu verfeinern
2. für jeden ausgewählten Kandidaten c_i:
     a. apply_latent_transition(z_t, action_i)  →  z'_{t,i}
     b. opponent_head(z'_{t,i}, legal_replies_i)  →  reply_dist_i, pressure_i
     c. aggregate_over_replies(...)  →  refined_score_i
3. C_{t+1} = update(C_t, refined_scores)
4. z_{t+1}, M_{t+1} = recurrent_cell(z_t, M_t, C_{t+1}, aggregated_reply_signal)
5. sharpness_{t+1} = sharpness_head(z_{t+1})
6. abbruch prüfen
7. emit trace-record
```

**Abbruchbedingungen (harte Gates):**
- `single_legal_move` → sofort raus, `t=0` (hart kodiert)
- `t == max_inner_steps` (z.B. `8`) → raus (hart kodiert)
- `sharpness_t < q_threshold` UND `t ≥ min_inner_steps` (z.B. `2`) → raus (lernbar über `q_threshold`)
- `top1_index unchanged for k=3 steps` UND `value stable` → raus (lernbar)

**Rollback-Mechanismus (User-Idee "zurück zum letzten guten Punkt"):**
- Nach jedem Schritt Snapshot von `(z_t, C_t, top1_value_t)` in Ringpuffer (Größe 4).
- Wenn `step t+1` die `top1_value` um `> rollback_threshold` verschlechtert, wird der Snapshot von `t` wiederhergestellt und `C_t` markiert den verworfenen Kandidaten in `history_t`.
- Kein Board-Undo, nur Latent-Restore. Der echte Board-Zustand bleibt unberührt, weil die gesamte Deliberation im latenten Raum stattfindet.

**Warum das keine klassische Suche ist:**
- Kein Baum mit Verzweigung. Pro Schritt wird eine bounded Anzahl Latent-Transitions ausgeführt, nicht rekursiv expandiert.
- Keine Tiefe im klassischen Sinne. `max_inner_steps` ist eine feste Konstante.
- Keine Transpositionstabelle.
- Keine Alpha-Beta-Bounds.
- Alle Entscheidungen (außer den drei harten Gates oben) werden von gelernten Heads getroffen.
- Der "Rollback" ist ein Gradient-freundlicher Latent-Restore, kein Search-Undo.

---

### 4.6 Trace, PV, UCI-Info

**Wo:** `python/train/eval/deliberation_trace.py` (neu) + Rust-Seite im zukünftigen `engine-app`-Planer-Pfad.

**Pro Schritt emittiert:**
```json
{
  "step": 3,
  "selected_candidates": [12, 7, 3],
  "top1_action_index": 12,
  "top1_value_cp": 34,
  "sharpness": 0.71,
  "uncertainty": 0.22,
  "router_choice": "tactical_expert",
  "pv_scratch_uci": ["e2e4", "e7e5", "g1f3"],
  "rollback_fired": false
}
```

**UCI-Ausgabe während `go`:**
```text
info depth 3 seldepth 3 score cp 34 nodes 147 nps 1400 pv e2e4 e7e5 g1f3
```

`depth` und `seldepth` sind hier **nicht** klassische Suchtiefen — sie zählen innere Deliberation-Steps und die längste imaginierte PV_scratch-Linie. Das wird in der Dokumentation explizit abgegrenzt, um Verwechslungen mit klassischer Suche zu vermeiden.

---

## 5. Ablaufplan (Runtime)

```text
UCI "go"
  │
  ▼
current_position (Rust)
  │
  ▼
build_state_context_v1(position)              [Rust, ~exact, cached pro ply]
  ├─ legal_moves
  ├─ attack_maps
  ├─ reachability_graph
  ├─ global_flags
  └─ candidate_context_v2 für jeden legalen Zug
  │
  ▼
IF single_legal_move:
    emit "info depth 0 ..."
    emit "bestmove <the_move>"
    RETURN                                    [hart kodiert, AGENTS.md-konform]
  │
  ▼
piece_intention_encoder(state_context)        [~20MB forward]
  │
  ▼
state_embedder(intentions, square_tokens)     [~40MB forward]
  → z_0, σ_0
  │
  ▼
initialize M_0, C_0 via policy_head(z_0, candidate_context_v2)
  │
  ▼
FOR t in 0..max_inner_steps:
    │
    ▼
    sharpness_t = sharpness_head(z_t)
    │
    IF t >= min_inner_steps AND sharpness_t < q_threshold:
        BREAK
    │
    selected = candidate_selector(z_t, C_t, σ_t, top_k=3)
    │
    FOR each cand in selected:
        z'_cand = latent_transition(z_t, cand.action)
        reply_signal = opponent_head(z'_cand, cand.legal_replies)
        refined_score_cand = aggregate(z'_cand, reply_signal)
    │
    C_{t+1}, z_{t+1}, M_{t+1} = recurrent_update(z_t, M_t, C_t, refined_scores)
    │
    IF rollback_detected(C_t → C_{t+1}):
        (z_{t+1}, M_{t+1}, C_{t+1}) = snapshot_restore(t-1)
        history.add(rejected=top1_before, reason="value_regression")
    │
    emit_trace(t, ...)
    │
    IF top1_stable_for(k=3):
        BREAK
  │
  ▼
final_value = value_head(z_T, M_T)
final_policy = softmax(C_T)
best_action = argmax(final_policy) [tie-break deterministisch]
  │
  ▼
verify_legality(best_action) [Rust, hart]
  │
  ▼
emit "bestmove <uci>"
```

---

## 6. Trainingsstrategie

Das Training erfolgt **mehrstufig, gemeinsam, mit Curriculum**:

### Stufe T0 — Feature-Authority-Tests
- Goldene Vektortests (Rust ↔ Python) für `StateContextV1` sichern.

### Stufe T1 — Vortraining (offline, bootstrapping)
- **Ziele:** `piece_intention_encoder`, `state_embedder`, `value_head`, `policy_head` initialisieren.
- **Daten:** bestehende Stockfish-Teacher-Korpora (`10k + 122k + 400k`), bestehende `search_teacher_*.jsonl`, `search_traces_*.jsonl`, `search_curriculum_*.jsonl`.
- **Loss:** gewichtete Summe aus Value-WDL + cp-MSE + Policy-CE + Policy-KL + margin + rank + auxiliary piece-role.
- **Deliberation aus:** `max_inner_steps=0` in dieser Stufe. Nur die statischen Heads lernen zuerst.
- **Dauer-Ziel:** `20–30` Epochen auf CPU/GPU, bis Value-MAE plateauet.

### Stufe T2 — Deliberation-On
- **Ziele:** `recurrent_cell`, `sharpness_head`, `candidate_selector`, `rollback_mechanism` einlernen.
- **Daten:** gleiche Korpora plus disagreements.
- **Loss:** zusätzlich sharpness-BCE (Label: Teacher top1-top2-Gap) + monotonicity-loss (top1 soll über Steps nicht-monoton verschlechtern ohne Rollback).
- **max_inner_steps:** `2 → 4 → 8` curriculum-gerampt.

### Stufe T3 — Opponent-Integration
- **Ziele:** `opponent_head` mit Deliberation koppeln, Training adversarisch.
- **Daten:** `phase7`-Reply-Korpora + Selfplay-Replay.

### Stufe T4 — Selfplay-Retrain-Loop
- **Wie bisher:** Arena-Spec lädt LAPv1-Agent, Stockfish18 reviewed, Korrekturen fließen in warm-start Retrain.
- **Partner:** `vice_v2` bleibt externer Benchmark, `planner_recurrent_expanded_v1` bleibt interne Regressions-Referenz.

### Stufe T5 — Distillation (optional, später)
- **Train rich, serve simple:** reichere Trainings-Variante mit mehr Inner-Steps, inferenz-kleinere Variante über Distillation.

---

## 7. Migration — Was passiert mit den alten Armen?

| Arm | Entscheidung | Begründung |
|-----|--------------|-----------|
| `planner_recurrent_expanded_v1` | **KEEP** als Regressions-Referenz | Aktuell stärkster Arm, Benchmark für LAPv1-Fortschritt |
| `planner_set_v6_expanded_v1` | **KEEP** als FF-Baseline | Stabilster Non-Recurrent |
| `planner_moe_v2_expanded_v1` | **KEEP** experimental | Arena-robust, könnte später als Expert-Mix in LAPv1 einfließen |
| `planner_set_v2_expanded_v1` | **DEPRECATE** | Historische Baseline, ersetzbar |
| `planner_set_v2_wide_expanded_v1` | **REMOVE** | Wide brachte nichts |
| `planner_set_v5_expanded_v1` | **REMOVE** | Schlechteste Verify |
| `planner_set_v6_margin_expanded_v1` | **REMOVE** | Degradiert unter Selfplay |
| `planner_set_v6_rank_expanded_v1` | **REMOVE** | Kein Vorsprung mehr |
| `planner_moe_v1_expanded_v1` | **REMOVE** | Instabil, letzter Platz Arena |
| `planner_active_expanded_v2` | **KEEP** als static benchmark | Aktuelle Promotion-Spitze |
| `vice_v2` | **KEEP** als external | Kalibrationsgate |
| `symbolic_root_v1` | **KEEP** als minimal benchmark | Sanity-Floor |

**Migrations-Gate:** LAPv1 wird zur neuen `active`-Promotion, sobald:
1. Verify `top1 ≥ 0.825` (über `recurrent_v1` finale `0.8199`)
2. Interne Arena `score_rate ≥ 0.65` gegen alle Keep-Arme
3. Gegen `vice_v2` mindestens `2.0/24` (ein Remis-Signal, das der beste heutige Arm schon erreicht hat, plus ein echter Sieg oder Remis als Schwelle)

---

## 8. Codex-CLI Arbeitsplan

**Konventionen:**
- Jeder Task ist ein eigener Branch, ein eigener PR-Kandidat, ein eigener Commit.
- Jeder Task endet mit: `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`, `ruff check python`, `python -m pytest`.
- Kein Task startet Training. Konfigs und Skripte werden nur vorbereitet.
- Keine bestehende Config, kein bestehender Checkpoint, kein laufender Prozess wird berührt.
- Jeder Task endet mit einer Zusammenfassung: *was geändert, was getestet, welche Risiken offen, was wäre der nächste Schritt*.

---

### T01 — Architektur-Dokument committen
**Branch:** `lapv1/docs-architecture`

```text
codex exec "Read AGENTS.md and PLANS.md thoroughly.
Create docs/architecture/lapv1-overview.md with the full LAPv1
architecture description. Copy the section 'Zielarchitektur im
Überblick' plus the component details and flow plan from this plan
document into docs/architecture/lapv1-overview.md.

Add a cross-link from docs/architecture/overview.md to the new file
under a new section 'LAPv1 target architecture'.

Do NOT modify AGENTS.md, PLANS.md, or any existing phase docs.
Summarize: what changed, tests run, open risks, next step."
```

---

### T02 — `StateContextV1` Contract (Python side)
**Branch:** `lapv1/state-context-v1-python`

```text
codex exec "Read AGENTS.md, docs/architecture/lapv1-overview.md,
docs/architecture/contracts.md, and python/train/datasets/contracts.py.

Add StateContextV1 as a new versioned contract in
python/train/datasets/contracts.py:

1. Define STATE_CONTEXT_V1_FEATURE_ORDER with per-square fields
   (own_attackers_count, opp_attackers_count, reaches_own_king,
   reaches_opp_king, pin_axis_orthogonal, pin_axis_diagonal,
   xray_attackers_count) and global fields (in_check,
   own_king_attackers_count, opp_king_attackers_count,
   own_king_escape_squares, opp_king_escape_squares,
   material_phase, single_legal_move, legal_move_count_normalized,
   has_legal_castle, has_legal_en_passant, has_legal_promotion).

2. Export state_context_v1_feature_spec() that returns the
   versioned schema dict (analogous to
   symbolic_candidate_context_v2_feature_spec).

3. Add a pure-Python reference builder
   build_state_context_v1(dataset_example) that uses the
   existing python-chess oracle path to compute all fields.

4. Add reachability-graph output as three aligned lists:
   edge_src_square[], edge_dst_square[], edge_piece_type[].

5. Add tests in python/tests/test_state_context_v1.py covering:
   - feature_order has no duplicates
   - startpos produces deterministic values
   - single_legal_move triggers on a curated mate-in-one position
   - reachability_graph contains expected edges for startpos

Do NOT touch Rust code in this task.
Do NOT modify existing contracts.

Summarize: what changed, tests run, open risks, next step."
```

---

### T03 — `StateContextV1` Contract (Rust side, golden-vector tested)
**Branch:** `lapv1/state-context-v1-rust`

```text
codex exec "Read AGENTS.md, docs/architecture/lapv1-overview.md,
rust/crates/encoder/src/lib.rs, rust/crates/inference/src/lib.rs.

1. In rust/crates/encoder, add a new module state_context.rs that
   mirrors StateContextV1 from the Python side exactly. Use the
   existing attack-map helpers from rust/crates/rules where possible.

2. Expose the struct as StateContextV1 with fields matching the
   Python feature_order EXACTLY. Provide:
   - StateContextV1::from_position(position: &Position) -> Self
   - StateContextV1::reachability_edges(&self) -> Vec<ReachabilityEdge>

3. In rust/crates/inference, add a loader expectation for
   'state_context_version': 1 in future bundle metadata
   (forward-compatible, no behavior change yet).

4. Create artifacts/golden/state_context_v1_golden.json with
   10 curated FENs and their expected StateContextV1 output.
   Source of truth is the Python reference builder from T02.

5. Add python/scripts/check_state_context_golden.py that runs
   both the Python and Rust paths (via a small CLI exposed in
   rust/crates/tools) and compares against the golden file.
   Exit code 0 on exact match, 1 on drift.

6. Add Rust tests in rust/crates/encoder using the same 10 FENs.

Do NOT modify existing encoder output shapes.
Do NOT modify existing proposer metadata.

Summarize: what changed, tests run, open risks, next step."
```

---

### T04 — Piece-Intention Encoder (model only, no training)
**Branch:** `lapv1/intention-encoder-model`

```text
codex exec "Read AGENTS.md, docs/architecture/lapv1-overview.md,
python/train/models/planner.py, python/train/models/proposer.py.

Create python/train/models/intention_encoder.py with:

1. PieceIntentionEncoder(nn.Module):
   - inputs: piece_tokens (batch, 32, 3), state_context_v1_global,
     reachability_edges (batch, edge_count, 3)
   - output: piece_intentions (batch, 32, intention_dim=64)
   - architecture: piece_type embedding + square embedding + attack
     count embedding, followed by 4 transformer layers with
     edge-masked multi-head attention (heads=4, hidden=128)
   - padding pieces (beyond 32 real pieces) are masked out

2. KingSpecialHead(nn.Module):
   - takes the king rows from piece_intentions plus global features
   - outputs: own_king_will (batch, 64), opp_king_will (batch, 64)

3. Config dataclass IntentionEncoderConfig in
   python/train/config.py (separate section, not merged with
   PlannerModelConfig).

4. Tests in python/tests/test_intention_encoder.py:
   - forward pass shape is correct
   - masked padding pieces produce zero attention
   - deterministic on fixed seed
   - backward pass populates gradients
   - approximate parameter count matches ~20MB target
     (tolerance ±30%)

Do NOT wire it into the planner yet.
Do NOT run training.

Summarize: what changed, tests run, open risks, next step."
```

---

### T05 — State Embedder (model only, no training)
**Branch:** `lapv1/state-embedder-model`

```text
codex exec "Read AGENTS.md, docs/architecture/lapv1-overview.md,
python/train/models/intention_encoder.py (from T04).

Create python/train/models/state_embedder.py with:

1. RelationalStateEmbedder(nn.Module):
   - inputs: piece_intentions (batch, 32, intention_dim),
     square_tokens (batch, 64, square_dim),
     global_features (batch, global_dim),
     reachability_edges
   - output: z_root (batch, state_dim=512), sigma_root (batch, 1)
   - architecture: 6 transformer layers, hidden=256, heads=8,
     mixed self-attention + edge-masked attention.
   - pools over all tokens with learned attention pooling into z_root

2. Config StateEmbedderConfig in python/train/config.py.

3. Tests in python/tests/test_state_embedder.py:
   - forward pass shape is (batch, 512)
   - sigma_root is positive
   - gradient flows through all layers
   - approximate parameter count matches ~40MB target

Do NOT wire it into the planner yet.

Summarize: what changed, tests run, open risks, next step."
```

---

### T06 — Value + Sharpness Heads (model only)
**Branch:** `lapv1/value-sharpness-heads`

```text
codex exec "Read AGENTS.md, docs/architecture/lapv1-overview.md,
python/train/models/state_embedder.py (from T05).

Create python/train/models/value_head.py:

1. ValueHead(nn.Module):
   - input: z_root (batch, 512), optional memory (batch, M, 256)
   - outputs: wdl_logits (batch, 3), cp_score (batch, 1),
     sigma_value (batch, 1)
   - architecture: 4-layer MLP with width=1024, hidden=1024,
     targeting ~100MB parameter count (parameterize width if
     needed to hit target)

2. SharpnessHead(nn.Module):
   - input: z_root (batch, 512)
   - output: scalar in [0,1] via sigmoid
   - small 2-layer MLP (hidden=128)

3. Config ValueHeadConfig, SharpnessHeadConfig in config.py.

4. Tests in python/tests/test_value_sharpness_heads.py:
   - output shapes correct
   - wdl_logits softmax sums to 1
   - sharpness in [0,1]
   - parameter counts within tolerance of targets

Do NOT wire yet.

Summarize: what changed, tests run, open risks, next step."
```

---

### T07 — Large Policy Head (model only)
**Branch:** `lapv1/policy-head-large`

```text
codex exec "Read AGENTS.md, docs/architecture/lapv1-overview.md,
python/train/models/planner.py (existing set_v6 for reference).

Create python/train/models/policy_head_large.py:

1. LargePolicyHead(nn.Module):
   - inputs: z_root (batch, 512),
     candidate_context_v2 (batch, num_candidates, cand_feat_dim),
     action_embedding_indices (batch, num_candidates),
     candidate_mask (batch, num_candidates)
   - output: candidate_logits (batch, num_candidates)
   - architecture: wide candidate-conditioned MLP, 1024→1024→512→1,
     plus cross-attention between z_root and candidate tokens
     (4 layers, 8 heads), targeting ~100MB
   - pad-aware: masked candidates score -inf

2. Tests in python/tests/test_policy_head_large.py:
   - output shape matches candidate count
   - masked candidates produce -inf logits
   - gradient flows
   - parameter count ~100MB within tolerance

Do NOT wire yet. Do NOT modify the existing planner.

Summarize: what changed, tests run, open risks, next step."
```

---

### T08 — Bounded Deliberation Loop (core)
**Branch:** `lapv1/deliberation-loop`

```text
codex exec "Read AGENTS.md (section on forbidden runtime mechanisms),
docs/architecture/lapv1-overview.md, the recurrent_v1 arm in
python/train/models/planner.py for reference.

Create python/train/models/deliberation.py with:

1. DeliberationCell(nn.Module):
   - inputs: z_t, M_t, C_t, refined_reply_signals
   - outputs: z_{t+1}, M_{t+1}, C_{t+1}
   - architecture: GRU-like gated recurrent update over
     state + memory, plus candidate-score residual update

2. CandidateSelector(nn.Module):
   - input: z_t, C_t, sigma_t
   - output: indices of top-K candidates to refine this step
     (K configurable, default 3)

3. LatentTransition(nn.Module):
   - small learned forward that takes (z_t, action_embedding)
     and returns z'_action — it is NOT a tree expansion, only
     a bounded forward projection used to query the opponent head

4. DeliberationLoop(nn.Module) orchestrating T04-T07 + T08.1-T08.3:
   - hard gate: single_legal_move → return immediately at t=0
   - hard gate: t == max_inner_steps → return
   - soft gate: sharpness < q_threshold AND t ≥ min_inner_steps
   - soft gate: top1 stable for k=3 steps
   - rollback: if top1_value regresses by > rollback_threshold,
     restore (z_{t-1}, M_{t-1}, C_{t-1}) from ring buffer of size 4
   - config knobs: max_inner_steps=8, min_inner_steps=2,
     q_threshold=0.3, rollback_threshold=40cp, top_k_refine=3

5. Emit DeliberationTrace dataclass with per-step records:
   step, selected_candidates, top1_action_index, top1_value_cp,
   sharpness, uncertainty, pv_scratch_uci, rollback_fired

6. Tests in python/tests/test_deliberation_loop.py:
   - single_legal_move early-exits at t=0
   - max_inner_steps is a hard cap
   - rollback fires on synthetic value regression
   - deterministic with fixed seed
   - trace length matches actual step count

The loop MUST NOT:
- expand recursively
- maintain a transposition table
- compute alpha-beta bounds
- implement quiescence search

The loop IS:
- a bounded recurrent refinement of candidate scores
- allowed to invoke the opponent head per refined candidate
- allowed to emit UCI-style info records

Summarize: what changed, tests run, open risks, next step."
```

---

### T09 — LAPv1 Wrapper Model
**Branch:** `lapv1/model-wrapper`

```text
codex exec "Read AGENTS.md, docs/architecture/lapv1-overview.md,
and all modules created in T04-T08.

Create python/train/models/lapv1.py with:

1. LAPv1Model(nn.Module) that composes:
   intention_encoder → state_embedder →
   (value_head, sharpness_head, policy_head) →
   deliberation_loop (with opponent head from existing
   python/train/models/opponent.py)

2. forward() returns:
   - final_policy_logits
   - final_value (wdl + cp + sigma)
   - deliberation_trace
   - refined_top1_action_index

3. LAPv1Config assembling all subconfigs.

4. Config validator that accepts architecture='lapv1' in
   PlannerTrainConfig as a top-level alternative wiring, behind
   a new wrapper config layer (do not overload existing
   PlannerModelConfig beyond adding a union tag).

5. Tests in python/tests/test_lapv1_model.py:
   - end-to-end forward pass produces correct shapes
   - deliberation_trace length ≤ max_inner_steps
   - output is differentiable
   - total parameter count ~200-300MB within tolerance

Do NOT add trainer glue yet.

Summarize: what changed, tests run, open risks, next step."
```

---

### T10 — Trainer for LAPv1 (stage T1 only, static heads)
**Branch:** `lapv1/trainer-stage1`

```text
codex exec "Read AGENTS.md, docs/architecture/lapv1-overview.md,
python/train/trainers/planner.py for reference patterns.

Create python/train/trainers/lapv1.py supporting training stage T1
(deliberation OFF, max_inner_steps=0, only static heads learn):

1. train_lapv1(config: LAPv1TrainConfig) returning LAPv1TrainingRun
2. evaluate_lapv1_checkpoint(path) returning LAPv1Metrics
3. Loss composition:
   loss = value_wdl_weight * wdl_ce
        + value_cp_weight * cp_mse
        + sharpness_weight * sharpness_bce
        + policy_ce_weight * policy_ce
        + policy_kl_weight * policy_kl
        + policy_margin_weight * margin_loss
        + policy_rank_weight * rank_loss
        + intention_aux_weight * piece_role_aux_loss
4. Per-epoch flushed logging matching existing planner trainer.
5. Checkpoint save/load compatible with the export patterns in
   python/train/export/.

Add tests in python/tests/test_lapv1_trainer.py with a tiny
synthetic dataset (≤32 rows) that exercises one training step
and one evaluation pass on CPU.

Do NOT wire stage T2 (deliberation-on) yet.
Do NOT prepare a real training config.

Summarize: what changed, tests run, open risks, next step."
```

---

### T11 — Stage T1 Training Config (prepared, not run)
**Branch:** `lapv1/config-stage1`

```text
codex exec "Read AGENTS.md, docs/architecture/lapv1-overview.md,
python/configs/phase8_planner_corpus_suite_set_v6_expanded_v1.json
for reference scope.

Create python/configs/phase10_lapv1_stage1_10k_122k_v1.json:

1. Reuses the 10k + 122k filtered workflow data paths that
   produced the strongest set_v2 reference.
2. Sets architecture='lapv1', stage='T1', max_inner_steps=0.
3. Epochs=20, batch_size appropriate for ~200MB model on CPU.
4. Output dir: models/lapv1/stage1_10k_122k_v1 (new path).
5. Export bundle dir: models/lapv1/stage1_10k_122k_v1/bundle.

Create python/scripts/run_lapv1_stage1_first_eval.sh:
1. Validates the config loads
2. Prints the expected parameter count
3. Does NOT actually start training

Update python/configs/README.md with a new 'LAPv1 Configs' section
describing stage1 and the future stage2/T3/T4 configs.

Do NOT run training. Do NOT modify existing phase9 configs.

Summarize: what changed, tests run, open risks, next step."
```

---

### T12 — Stage T2: Deliberation-On Trainer Extension
**Branch:** `lapv1/trainer-stage2`

```text
codex exec "Read T08 deliberation loop, T10 trainer.

Extend train_lapv1 to support stage T2:
1. Add max_inner_steps curriculum: 2 → 4 → 8 scheduled over
   training epochs (configurable).
2. Add sharpness_target_loss (supervised on teacher top1-top2
   gap below/above 20cp).
3. Add deliberation_monotonicity_loss: penalize top1_value
   regressions across steps that were NOT rolled back.
4. Add rollback_statistics logging (rollbacks_per_epoch, mean step
   at rollback, rollback_hit_rate).

Add tests in python/tests/test_lapv1_trainer_stage2.py with a
tiny synthetic dataset that:
- runs 2 epochs with max_inner_steps=2
- verifies that monotonicity_loss is non-negative
- verifies rollback stats are logged

Do NOT prepare a real stage2 config yet.

Summarize: what changed, tests run, open risks, next step."
```

---

### T13 — Agent Spec + Runtime Glue
**Branch:** `lapv1/agent-spec`

```text
codex exec "Read AGENTS.md, python/train/eval/agent_spec.py,
python/train/eval/planner_runtime.py.

1. Extend SelfplayAgentSpec to support agent_kind='lapv1'
   with fields: lapv1_checkpoint, state_context_version,
   deliberation_max_inner_steps, deliberation_q_threshold.
   Keep schema backwards-compatible.

2. Create python/train/eval/lapv1_runtime.py with
   build_lapv1_runtime_from_spec(spec, repo_root) that loads the
   LAPv1 checkpoint and returns an object with select_move(example)
   matching the SelfplayAgent interface.

3. Create python/configs/phase10_agent_lapv1_stage1_v1.json as
   the first LAPv1 agent spec (points at a placeholder checkpoint
   path that will exist after stage1 training runs).

4. Add tests in python/tests/test_lapv1_runtime.py using a
   fresh, untrained LAPv1 model instance:
   - select_move returns a legal move
   - deliberation_trace is emitted
   - single-legal-move positions return at step 0

Do NOT run selfplay. Do NOT modify existing agent specs.

Summarize: what changed, tests run, open risks, next step."
```

---

### T14 — Arena-Config for LAPv1 vs best existing arms
**Branch:** `lapv1/arena-benchmark`

```text
codex exec "Read AGENTS.md, python/configs/phase9_arena_active_vs_vice_v1.json,
python/train/eval/arena.py.

Create python/configs/phase10_arena_lapv1_vs_baseline_v1.json:
1. Agent specs:
   - phase10_agent_lapv1_stage1_v1 (new, from T13)
   - phase9_agent_planner_recurrent_v1 (best existing trainable arm)
   - phase9_agent_planner_set_v6_expanded_v1 (FF baseline)
   - phase9_agent_planner_active_expanded_v2 (current promotion)
   - phase9_agent_uci_vice_v2 (external benchmark)
2. round_robin, swap_colors=true, games=1.
3. Same opening suite and max_plies adjudication as existing phase9.
4. parallel_workers=6.

Do NOT run the arena. Only prepare the config.

Summarize: what changed, tests run, open risks, next step."
```

---

### T15 — Documentation & PLANS.md Update
**Branch:** `lapv1/docs-and-plans`

```text
codex exec "Read AGENTS.md and PLANS.md.

1. Add a new phase section to PLANS.md:
   'Phase 10 — LAPv1 unified planner'
   with Goal, Non-goals, Deliverables, Exit criteria matching
   the LAPv1 architecture document.

2. Append to docs/architecture/overview.md a 'Phase 10 Status'
   section describing the LAPv1 target and pointing at
   docs/architecture/lapv1-overview.md.

3. Update README.md current-scope list to mention LAPv1
   scaffolding as prepared-but-untrained.

4. Create docs/experiments/lapv1-migration-plan-2026-04-05.md
   with the arm-by-arm migration table from the architecture
   document (KEEP/REMOVE/DEPRECATE).

Do NOT remove any existing arm configs or checkpoints yet.
Do NOT run any training or arena.

Summarize: what changed, tests run, open risks, next step."
```

---

## 9. Warum diese Architektur die AGENTS.md-Grenzen hält

| Frage | Antwort |
|-------|---------|
| Ist das Alpha-Beta? | Nein. Es gibt keine Tiefensuche, keine Baumexpansion, keine Bounds-Propagation. Die innere Schleife iteriert fixiert über bounded Deliberation-Steps im latenten Raum. |
| Ist das MCTS? | Nein. Keine Rollouts, keine UCT-Selektion, keine Besuch-Zähler. Kandidaten-Scores werden von einem rekurrenten gelernten Update verfeinert, nicht von Baumstatistik. |
| Ist das Quiescence Search? | Nein. Es gibt einen `sharpness_head`, der lernt, wann mehr Deliberation nötig ist. Kein Capture-only-Expand, kein Stand-Pat, kein klassisches Qsearch. |
| Ist das eine Transposition Table? | Nein. Kein Positions-Hash als Cache. Der Ringpuffer speichert Latent-Snapshots der *letzten 4 Schritte der aktuellen Deliberation*, nicht Positionen. |
| Ist das handgeschriebene Evaluation? | Nein. Alle numerischen Urteile kommen aus gelernten Netzen. Hart kodiert sind nur: exakte Legalität, `single_legal_move`-Kurzschluss, `max_inner_steps`-Cap. |
| Warum darf der "Rollback" existieren? | Er arbeitet rein im latenten Deliberation-Raum, nicht am Brett. Das Brett bewegt sich nur, wenn am Ende `bestmove` emittiert wird. Der Rollback ist semantisch identisch mit "rekurrente Zelle ignoriert das letzte Gradientensignal" und bricht AGENTS.md nicht. |

---

## 10. Offene Risiken

1. **Parametervolumen:** `~200-300MB` pro Inference-Call. Forward-Pass-Latenz auf CPU muss gemessen werden, bevor Stage T2 groß gefahren wird. Fallback: Heads verkleinern, `hidden_dim` reduzieren, später distillieren.
2. **Trainingskosten:** Ein `~100MB` Value-Netz braucht deutlich mehr Datenvolumen als die aktuellen kleinen Arme. Die bestehenden `10k+122k+400k`-Korpora reichen eventuell nicht — möglicherweise muss der Teacher-Output expandiert werden (z.B. `1M` Positionen über erweiterten Stockfish-Lauf).
3. **Sharpness-Label-Qualität:** Die Definition "Teacher top1-top2-Gap" ist ein Proxy. Wenn das Label verrauscht ist, zirkelt der Sharpness-Head gegen Gradienten-Rauschen. Mitigation: BCE mit Label-Smoothing, Calibration-Tracking über Trainingsepochen.
4. **Rollback-Stabilität:** Der Mechanismus könnte zu Oszillation führen (Rollback → gleicher Kandidat wieder → Rollback). Mitigation: `history_t` blockt bereits ausgewählte Kandidaten für mindestens `2` Steps, `rollback_threshold` wird über Curriculum geschärft.
5. **Inference-zu-Training-Drift:** Wenn `max_inner_steps` zur Inference-Zeit anders ist als zur Trainingszeit, driftet die gelernte Deliberation. Mitigation: Inference-Config wird aus dem Checkpoint gelesen, nicht aus dem Agent-Spec überschrieben.
6. **`vice_v2`-Gap:** Selbst der aktuell stärkste Arm verliert `23/24` gegen `vice`. LAPv1 muss nicht nur die interne Armfamilie schlagen, sondern auch echt auf `vice` Druck machen, damit der Architekturwechsel den Aufwand rechtfertigt.

---

## 11. Nächster Schritt nach codex-cli

Wenn die Tasks T01–T15 sauber durchlaufen sind:
1. Training Stage T1 auf `10k+122k` starten (siehe Config aus T11).
2. Konvergenz-Plot gegen `set_v6_expanded`-Baseline bei gleicher Epochenzahl.
3. Bei Erfolg: Stage T2 curriculum mit `max_inner_steps: 2→4→8`.
4. Arena-Vergleich gegen `recurrent_expanded_v1` (aus T14).
5. Bei `top1 ≥ 0.825` und Arena `score_rate ≥ 0.65`: Promotion zur neuen `active`-Referenz, Deprecation-PR der entfernten Arme.