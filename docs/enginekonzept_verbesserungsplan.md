# EngineKonzept — Projektstand, Verbesserungsanalyse & Codex-CLI Arbeitsplan

## 1. Projektstand-Zusammenfassung

### Wo steht das Projekt?

EngineKonzept befindet sich in **Phase 9** — der Selfplay- und Replay-Campaign-Phase. Die Phasen 0–8 (Scaffolding, Rules Kernel, UCI Shell, Action Space, Dataset Pipeline, Proposer, Dynamics, Opponent Head, Planner) sind abgeschlossen und materialisiert.

### Was läuft aktuell (Long-Run)?

Es läuft einer der beiden Phase-9 Long-Runs:

- **Replay Campaign** (`run_phase9_replay_campaign_longrun.sh`): Arena → Replay-Buffer → Planner-Retrain → nächste Arena-Runde
- **Fulltrain Arena Campaign** (`run_phase9_fulltrain_arena_longrun.sh`): Voller Planner-Family-Sweep über 10k+122k+400k, dann Arena-Vergleich

### Aktuelle Leistungsdaten

| Metrik | Wert | Interpretation |
|---|---|---|
| Bester Planner (held-out) | `set_v2_10k_122k_expanded`: top1=0.819, MRR=0.890 | Stärkstes Offline-Ranking |
| Arena-Leader | `set_v6_rank_expanded_v1`: score_rate=0.55 | Tentativ, kleine Stichprobe |
| Aktiver Arm (promoted) | `planner_active_expanded_v2` | Aus der set_v6-Familie |
| vs. VICE (externes Benchmark) | 0.5 / 20 (1 Draw, 19 Losses) | Noch weit unter einem echten Engine-Level |
| Dynamics exact accuracy | 0.0% | Kritisches Problem |
| Arena resolved ratio | 0.65 (mit taktischen Starts) | Verbessert von 0.20 (nur startpos) |
| Replay-Buffer Größe | 3640 rows, 60 Spiele | Erste brauchbare Selfplay-Daten |

### Zentrale Schwächen (priorisiert)

1. **Dynamics-Genauigkeit bei 0%** — Der Planner hat kein funktionierendes Vorausdenk-Substrat; er ist effektiv ein Pattern-Matcher auf Teacher-Labels
2. **Latent-State bringt nichts** — `set_v3` mit Latent-Input ist schwächer als `set_v2` ohne; der Dynamics-Kanal ist Rauschen
3. **400k-Tier schadet** — Mehr Daten ≠ besserer Planner; die Datenqualität der 400k-Partition drückt die Metrik
4. **Breite hilft nicht** — `set_v2_wide` ist schlechter als `set_v2`; das Bottleneck ist nicht Kapazität
5. **Recurrence noch schwach** — `recurrent_v1` schlechter als one-shot `set_v2`; Memory-Integration unreif
6. **Kein Runtime-Inference** — Rust-Crates `planner`, `selfplay`, `eval-metrics` sind Placeholder
7. **Cross-Language Schema-Drift** — Keine Golden-Tests zwischen Python-Features und Rust-Encoder

---

## 2. Verbesserungsmöglichkeiten

### A. Datenqualität & Teacher-Targets (höchster Hebel)

**Problem:** Der Planner lernt primär, Teacher-Rankings nachzuahmen. Die Qualität der Teacher-Labels bestimmt die Planner-Obergrenze.

**Verbesserungen:**
- Curriculum-bewusstes Training mit den vorhandenen `search_curriculum_*.jsonl`-Artefakten
- Gewichtung der Training-Samples nach Schwierigkeitsgrad (leicht → schwer ramping)
- Filtering der 400k-Partition: Nur Positionen mit klarem Teacher-Signal behalten
- Multi-Teacher-Ensemble: Stockfish bei verschiedenen Tiefen → Uncertainty-Labels

### B. Dynamics-Durchbruch (kritischster Block)

**Problem:** 0% exact next-state accuracy macht den gesamten latenten Planning-Pfad wertlos.

**Verbesserungen:**
- Hybrid Dynamics: Exakte symbolische Zustandsänderung + gelerntes Residual für nicht-triviale Features
- Piece-centric Dynamics: Pro-Figur statt globaler Zustandsvorhersage
- Curriculum auf Dynamics: Erst einfache Züge (Bauernzüge), dann komplexere

### C. Planner-Architektur (mittlerer Hebel)

**Problem:** MLP-Backbone + flat candidate scoring limitiert die Ausdrucksstärke.

**Verbesserungen:**
- Cross-Attention zwischen State-Embedding und Candidate-Set statt getrennter Projektion
- Pairwise-Interactions zwischen Kandidaten (wer blockiert wen, welche Züge sind komplementär)
- Gated Residual Connections im Candidate-MLP

### D. MoE-Experimentalarm (neuer Forschungsansatz)

**Konzept aus arch.ideas.md:** Router-DAG mit spezialisierten Experten, Complexity-Head für Easy/Hard-Kaskade, getypter Latent-Bus.

**Anwendung auf den Planner:**
- Positionstyp-Router (taktisch vs. positionell vs. Endspiel) wählt spezialisierte Candidate-Scorer
- Compute-Budget-Kaskade: Scout entscheidet, ob 1 oder 3 Deliberation Steps nötig sind
- Experten-Fusion mit Confidence-Gewichtung statt simpler Mittelung

### E. Selfplay-Effizienz (laufende Phase 9)

**Problem:** Viele Spiele enden durch `max_plies`, geringe Replay-Ausbeute.

**Verbesserungen:**
- Aggressivere Opening-Suite (stärker taktische Startstellungen)
- Dynamische Plies-Erweiterung bei unklaren Stellungen
- Temperature-Scheduling: Höhere Exploration in frühen Selfplay-Runden

---

## 3. Codex-CLI Arbeitsplan (10 Tasks)

> **Konvention:** Jeder Task ist unabhängig vom laufenden Long-Run ausführbar. Nichts berührt aktive Training-Configs, Checkpoints oder laufende Prozesse. Alle Tests auf dem separaten Host.

---

### Task 1: Curriculum-Weighted Sampler
**Branch:** `improvement/curriculum-sampler`
**Ziel:** Trainingsdaten nach Teacher-Difficulty gewichten

```
codex exec "Read AGENTS.md and PLANS.md.

Create a curriculum-weighted data sampler for planner training:

1. In python/train/datasets/planner_head.py, add a function
   compute_curriculum_weights(examples, strategy='linear_ramp')
   that assigns per-example weights based on:
   - teacher_root_value_cp spread (higher spread = harder = upweight later)
   - candidate count (more candidates = harder)
   - teacher agreement (low agreement = harder)

2. Add a CurriculumSampler class in python/train/datasets/curriculum.py
   that wraps a dataset and supports:
   - 'uniform' (current behavior)
   - 'linear_ramp' (easy-first, hard-later over epochs)
   - 'sqrt_ramp' (gentler curve)

3. Wire it into PlannerTrainConfig as optional 'curriculum' section.
   If absent, behavior is identical to current.

4. Add unit tests in python/tests/test_curriculum_sampler.py:
   - weights sum correctly
   - linear_ramp increases hard-example weight over epochs
   - uniform strategy matches current behavior exactly

5. Do NOT modify any existing configs or running processes.

Summarize: what changed, tests run, open risks, next step."
```

---

### Task 2: 400k-Tier Quality Filter
**Branch:** `improvement/400k-filter`
**Ziel:** Filtern der 400k-Partition nach Teacher-Signalstärke

```
codex exec "Read AGENTS.md and PLANS.md.

The 400k dataset tier hurts planner quality on the preferred 10k+122k slice.
Create a quality-filter pipeline:

1. In python/scripts/filter_400k_by_teacher_quality.py:
   - Load planner_head artifacts from the 400k tier
   - Filter out rows where:
     a) teacher_root_value_cp is NaN or extreme (>2000 cp)
     b) all candidate scores are within 5cp of each other (ambiguous)
     c) candidate count < 2 (trivial)
   - Write filtered artifacts in the same schema
   - Report statistics: kept/dropped/total, distribution shift

2. Add a config template:
   python/configs/phase8_planner_corpus_suite_set_v2_10k_122k_400k_filtered_v1.json
   pointing at filtered 400k + unmodified 10k + 122k

3. Add a unit test that the filter preserves schema exactly.
4. Do NOT run any training, only prepare the data artifacts.

Summarize: what changed, tests run, open risks, next step."
```

---

### Task 3: Hybrid Dynamics Module
**Branch:** `improvement/hybrid-dynamics`
**Ziel:** Symbolische Zustandsänderung + gelerntes Residual

```
codex exec "Read AGENTS.md and PLANS.md.
Read python/train/models/dynamics.py and docs/architecture/dynamics.md.

The current dynamics module has 0% exact next-state accuracy.
Create a hybrid dynamics approach:

1. In python/train/models/dynamics.py, add architecture='hybrid_v1':
   - Takes the current encoder features PLUS symbolic delta features
     (which pieces moved, captured, castled, en-passant changed)
   - Predicts only the RESIDUAL between symbolic next-state and
     actual next-state features, not the full state from scratch
   - The symbolic delta comes from the existing Rust oracle labels

2. Add the symbolic delta fields to the dynamics dataset contract
   in python/train/datasets/contracts.py as optional backward-compatible
   fields: 'symbolic_move_delta_features'

3. Add a test in python/tests/test_dynamics_hybrid.py:
   - Verify hybrid_v1 forward pass shape
   - Verify residual output is added to symbolic prediction
   - Verify backward-compatible: old data without delta fields still works

4. Do NOT change the dynamics training loop yet. Only add the model.

Summarize: what changed, tests run, open risks, next step."
```

---

### Task 4: Cross-Attention Candidate Scorer
**Branch:** `improvement/cross-attention-scorer`
**Ziel:** State-Candidate Cross-Attention statt getrennter MLP-Projektion

```
codex exec "Read AGENTS.md and PLANS.md.
Read python/train/models/planner.py carefully.

Add a new planner architecture 'set_v7' that replaces the current
separated state_backbone + candidate_projection + candidate_mlp
with cross-attention:

1. Keep state_backbone as-is (produces state embedding)
2. Replace candidate_projection with a multi-head cross-attention layer:
   - Query: candidate features (action embedding + symbolic features)
   - Key/Value: state embedding (broadcast to candidate set)
   - 4 heads, hidden_dim dimension
3. Keep candidate_mlp but feed it the attention output instead of
   the concatenated raw features
4. Keep root_value_head, root_gap_head, candidate_score_head unchanged
5. Register 'set_v7' in all config validation checks

Add tests:
- Forward pass shape matches set_v6 output contract exactly
- Gradient flows through attention layer
- Config with architecture='set_v7' loads and validates

Do NOT modify existing architectures. Do NOT run training.

Summarize: what changed, tests run, open risks, next step."
```

---

### Task 5: Pairwise Candidate Interaction Layer
**Branch:** `improvement/pairwise-candidates`
**Ziel:** Kandidaten können voneinander lernen

```
codex exec "Read AGENTS.md and PLANS.md.
Read python/train/models/planner.py.

Add a PairwiseCandidateInteraction module that can be optionally
inserted into set_v6 or set_v7:

1. In python/train/models/planner.py, add class PairwiseCandidateLayer:
   - Takes candidate embeddings (batch, num_candidates, hidden_dim)
   - Computes pairwise dot-product attention between candidates
   - Masks padded candidates via candidate_mask
   - Returns refined candidate embeddings (same shape)
   - Uses 2 heads, single layer

2. Add config flag 'enable_pairwise_candidates: bool = False'
   to PlannerModelConfig. When True, insert the layer between
   candidate_projection and candidate_mlp.

3. Add tests:
   - Pairwise layer preserves shape
   - Masked candidates get zero attention
   - When disabled, output is identical to current behavior

Do NOT modify existing default configs.

Summarize: what changed, tests run, open risks, next step."
```

---

### Task 6: MoE Planner Arm — Router + Experten (Experimentalarm)
**Branch:** `experiment/moe-planner-v1`
**Ziel:** Erster MoE-Planner nach dem Router-DAG-Muster aus arch.ideas.md

```
codex exec "Read AGENTS.md and PLANS.md.
Read docs/arch.ideas.md thoroughly — especially sections on
Router-DAG, Complexity Head, and Top-2 routing.
Read python/train/models/planner.py.

Create a new planner architecture 'moe_v1' that implements the
simplest viable MoE pattern for candidate scoring:

1. Add python/train/models/moe_planner.py with:

   a) PositionRouter(nn.Module):
      - Input: state embedding (hidden_dim)
      - Output: expert weights (num_experts) via softmax
      - Architecture: 2-layer MLP → softmax
      - Top-k selection with k=2 (configurable)
      - Load-balancing auxiliary loss (importance + load terms)

   b) CandidateExpert(nn.Module):
      - Same interface as current candidate_mlp
      - Each expert has its own weights
      - Input: candidate embedding features
      - Output: scalar score per candidate

   c) MoEPlannerHeadModel(nn.Module):
      - state_backbone: same as set_v6
      - candidate_projection: same as set_v6
      - router: PositionRouter with num_experts=4
      - experts: nn.ModuleList of 4 CandidateExperts
      - fusion: weighted sum of expert outputs using router weights
      - root_value_head, root_gap_head: same as set_v6
      - candidate_score_head: shared across fusion output
      - forward() returns same output contract as set_v6

2. Add MoEConfig dataclass:
   - num_experts: int = 4
   - top_k: int = 2
   - load_balance_weight: float = 0.01
   - expert_hidden_dim: int = 128

3. Wire 'moe_v1' into PlannerModelConfig validation.
   Add 'moe' optional section to PlannerTrainConfig.

4. Add auxiliary load-balancing loss to the planner trainer:
   - Only active when architecture='moe_v1'
   - Logged as separate metric: 'load_balance_loss'

5. Add metrics for expert utilization:
   - Per-expert activation frequency
   - Router entropy (high = balanced, low = collapsed)
   - Logged per epoch in training summary

6. Add tests in python/tests/test_moe_planner.py:
   - Forward pass produces same output shape as set_v6
   - Router weights sum to 1.0
   - Top-k selection works (only k experts have nonzero weight)
   - Load balance loss is finite and non-negative
   - Expert utilization metrics are correctly computed
   - Config validation accepts and rejects correctly

7. Add a template config:
   python/configs/phase9_planner_moe_v1_template.json
   with architecture='moe_v1', num_experts=4, top_k=2

Do NOT modify existing architectures or configs.
Do NOT run any training.

Summarize: what changed, tests run, open risks, next step."
```

---

### Task 7: MoE Complexity-Head Erweiterung
**Branch:** `experiment/moe-complexity-head`
**Ziel:** Easy/Hard-Kaskade — billiger Scout entscheidet über Compute-Budget

```
codex exec "Read AGENTS.md and PLANS.md.
Read docs/arch.ideas.md section on Aufwand-nach-Bedarf-Kaskade.
Read python/train/models/moe_planner.py (from Task 6).

Extend the MoE planner with a Complexity Head:

1. Add ComplexityHead(nn.Module) to moe_planner.py:
   - Input: state embedding
   - Output: complexity_score in [0, 1] via sigmoid
   - Architecture: small 2-layer MLP (hidden_dim // 4)

2. Add compute budget routing to MoEPlannerHeadModel:
   - If complexity_score < easy_threshold (default 0.3):
     → use only 1 expert (cheapest), 1 deliberation step
   - If complexity_score > hard_threshold (default 0.7):
     → use top_k experts, full deliberation steps
   - Between thresholds: use 2 experts, reduced deliberation
   - Thresholds are configurable in MoEConfig

3. Add complexity-aware training:
   - Teacher signal for complexity: positions where teacher
     top1 and top2 are close (< 20cp) are 'hard'
   - Positions where top1 is >> top2 (> 100cp) are 'easy'
   - Binary cross-entropy loss on complexity_score
   - Logged as 'complexity_loss'

4. Add metrics:
   - Fraction of positions routed easy/medium/hard
   - Average expert count per complexity tier
   - Compute savings estimate (easy positions * saved expert calls)

5. Add tests:
   - Complexity head produces valid [0,1] scores
   - Easy positions use fewer experts
   - Thresholds work correctly at boundaries
   - Backward compatible: when complexity head disabled,
     behavior matches vanilla moe_v1

Summarize: what changed, tests run, open risks, next step."
```

---

### Task 8: MoE Expert Specialization Analysis
**Branch:** `experiment/moe-expert-analysis`
**Ziel:** Tooling zur Analyse welcher Expert was lernt

```
codex exec "Read AGENTS.md and PLANS.md.

Create analysis tooling for the MoE planner:

1. python/scripts/analyze_moe_expert_specialization.py:
   - Load a trained moe_v1 checkpoint
   - Run inference on a labeled dataset
   - For each example, record:
     a) which experts were selected
     b) router weights
     c) position phase (opening/middlegame/endgame via piece count)
     d) position tactical level (checks, captures in legal moves)
     e) teacher difficulty (score spread)
   - Output a JSON report with:
     - Expert activation by game phase
     - Expert activation by tactical level
     - Expert activation by difficulty
     - Router entropy distribution
     - Expert agreement rate (how often do top-2 experts agree on ranking?)

2. python/scripts/visualize_moe_routing.py:
   - Read the JSON report
   - Generate matplotlib plots:
     a) Heatmap: expert × game_phase activation frequency
     b) Histogram: router entropy distribution
     c) Scatter: complexity_score vs actual difficulty
   - Save to artifacts/moe_analysis/

3. Add tests:
   - Analysis script runs on a minimal synthetic dataset
   - JSON report schema is valid
   - Visualization script produces files without error

Summarize: what changed, tests run, open risks, next step."
```

---

### Task 9: Cross-Language Golden Tests
**Branch:** `improvement/cross-language-golden-tests`
**Ziel:** Sicherstellen dass Python-Features und Rust-Encoder identisch sind

```
codex exec "Read AGENTS.md and PLANS.md.
Read docs/architecture/contracts.md.

Create cross-language golden tests:

1. In rust/crates/encoder/tests/golden_vectors.rs:
   - Define 10 canonical FEN positions covering:
     castling rights, en passant, promotions, checks, pins
   - For each position, compute the full 230-feature encoder output
   - Store expected feature vectors as literal arrays
   - Assert exact match

2. In python/tests/test_golden_vectors.py:
   - Load the same 10 FEN positions
   - Use the Python encoder path to compute features
   - Compare against the same expected vectors from step 1
   - Assert exact match (tolerance 0.0, these must be identical)

3. Create artifacts/golden/encoder_golden_v1.json:
   - The 10 FENs and their expected 230-dim feature vectors
   - Both test files read from this single source of truth

4. Add a CI-friendly script python/scripts/check_golden_vectors.py:
   - Runs both Python feature computation and compares to golden file
   - Exit code 0 if match, 1 if drift detected

5. Add tests that verify the golden file itself is well-formed.

Summarize: what changed, tests run, open risks, next step."
```

---

### Task 10: MoE Training Config + Erste Evaluierung vorbereiten
**Branch:** `experiment/moe-first-eval`
**Ziel:** Alles für den ersten MoE-Trainingslauf auf dem Test-Host vorbereiten

```
codex exec "Read AGENTS.md and PLANS.md.

Prepare the first MoE planner training run:

1. Create python/configs/phase9_planner_moe_v1_10k_122k_v1.json:
   - architecture: moe_v1
   - Same data paths as phase8_planner_corpus_suite_set_v2_10k_122k_expanded_v1
   - num_experts: 4, top_k: 2
   - load_balance_weight: 0.01
   - hidden_dim: 256, hidden_layers: 2
   - epochs: 12, same optimizer settings as set_v6
   - output_dir pointing to a new moe_v1 subdirectory

2. Create python/configs/phase9_agent_planner_moe_v1.json:
   - Agent spec template for selfplay with moe_v1 checkpoint
   - Same contract as existing agent specs

3. Create python/scripts/run_moe_v1_first_eval.sh:
   - Train moe_v1 on 10k+122k filtered data
   - Evaluate checkpoint on held-out validation
   - Compare metrics against set_v2 and set_v6 baselines
   - Print summary table: top1, MRR, expert_entropy, load_balance_loss
   - Save results to artifacts/moe_v1/first_eval_summary.json

4. Create a README section in docs/architecture/moe-planner.md:
   - Motivation (from arch.ideas.md)
   - Architecture diagram (text-based)
   - Expected behavior of router and experts
   - How to interpret expert utilization metrics
   - Relationship to existing planner arms

5. Do NOT run the training. Only prepare configs and scripts.

Summarize: what changed, tests run, open risks, next step."
```

---

## 4. Ausführungsreihenfolge

Die Tasks sind so geordnet, dass sie den Long-Run nicht stören und auf dem Test-Host parallelisiert werden können:

```
Woche 1 (Fundament):
  Task 1: Curriculum Sampler        — reine Python-Library, kein Training
  Task 2: 400k Quality Filter       — Data-Prep, kein Training
  Task 9: Cross-Language Golden Tests — Test-Infrastruktur

Woche 2 (Planner-Architektur):
  Task 3: Hybrid Dynamics            — neues Modell, kein Training
  Task 4: Cross-Attention Scorer     — set_v7, kein Training
  Task 5: Pairwise Candidates        — optionaler Layer

Woche 3 (MoE-Experimentalarm):
  Task 6: MoE Router + Experten      — Kernstück des MoE-Arms
  Task 7: Complexity Head             — Erweiterung
  Task 8: Expert Analysis Tooling    — Analyse-Pipeline

Woche 4 (Integration + Erste Evaluierung):
  Task 10: MoE Eval vorbereiten      — Configs + Scripts
  → Dann auf Test-Host: Erster moe_v1 Trainingslauf
  → Vergleich mit set_v2 und set_v6 Baselines
```

## 5. Erwartete Ergebnisse nach Abarbeitung

| Verbesserung | Erwarteter Impact | Risiko |
|---|---|---|
| Curriculum Sampler | +1-3% top1 durch bessere Sample-Gewichtung | Niedrig |
| 400k Filter | Verhindert Qualitätsverlust durch schlechte Daten | Niedrig |
| Hybrid Dynamics | Potentiell erster echter Durchbruch bei Next-State | Mittel |
| Cross-Attention Scorer (set_v7) | +1-2% top1 durch bessere State-Candidate-Interaktion | Mittel |
| Pairwise Candidates | Besser bei Positionen mit vielen ähnlichen Zügen | Mittel |
| MoE Planner (moe_v1) | Spezialisierung auf Positionstypen, +2-5% top1 möglich | Hoch (Forschung) |
| Complexity Head | Compute-Effizienz: 30-50% billigere einfache Positionen | Mittel |
| Golden Tests | Verhindert stille Cross-Language-Bugs | Niedrig |

## 6. MoE-Arm: Architektur-Skizze

```
Position
  │
  ▼
┌─────────────────┐
│ State Backbone   │ (gleich wie set_v6)
│ MLP: 230 → 256   │
└────────┬────────┘
         │ state_embedding
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐  ┌────────────┐
│Complexity│  │  Router     │
│  Head    │  │ MLP → Top-2 │
│ σ → [0,1]│  │  Softmax    │
└────┬────┘  └──┬──┬──┬──┘
     │          │  │  │  │
     │       w₁ │  w₂│  0  0  (Top-2 Selection)
     │          │    │
     │    ┌─────┘    └─────┐
     │    ▼                ▼
     │  ┌──────────┐  ┌──────────┐
     │  │ Expert 1  │  │ Expert 2  │
     │  │ Candidate │  │ Candidate │
     │  │  Scorer   │  │  Scorer   │
     │  └─────┬────┘  └─────┬────┘
     │        │              │
     │        ▼              ▼
     │   ┌────────────────────┐
     │   │  Weighted Fusion    │
     │   │  w₁·s₁ + w₂·s₂     │
     │   └─────────┬──────────┘
     │             │
     │    (complexity < 0.3? → nur Expert 1)
     │             │
     ▼             ▼
┌──────────────────────┐
│ Root Value + Gap Head │
│ Candidate Score Head  │
│ (gleicher Output-     │
│  Contract wie set_v6) │
└──────────────────────┘
```

## 7. Hinweise für die Ausführung

- **Alle Tasks auf dem Test-Host ausführen**, nicht auf dem Host mit dem Long-Run
- **Kein Task berührt laufende Prozesse, Configs oder Checkpoints**
- **Branch-Konvention:** `improvement/*` für Verbesserungen, `experiment/*` für MoE
- **Jeder Task ist ein einzelner `codex exec`-Aufruf** — Copy-Paste ready
- **Nach jedem Task:** `cargo test` und `python -m pytest python/tests/` auf dem Test-Host
- **Git-Rhythmus:** Ein Commit pro Task, Review-Gate vor Merge in main
