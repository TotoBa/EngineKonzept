# LAPv1 Overview

LAPv1 is the current target architecture for converging the planner family toward one bounded latent-adversarial stack instead of keeping a long-lived arm zoo.

It is intended to replace the separate Phase-8/9 planner variants only if it beats them empirically while staying inside the architectural boundaries from [AGENTS.md](/home/torsten/EngineKonzept/AGENTS.md):

- exact legality stays symbolic
- there is no alpha-beta or classical search runtime
- deliberation is bounded, learned, and traceable

## Target Architecture

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

## Component Details

### Symbolic Input Gate → `StateContextV1`

This layer lives in Rust and mirrors into Python as a versioned contract.

Per-square fields are intended to carry exact geometric and rules-derived facts such as:

- own and opponent attacker counts
- king-reach flags
- pin-axis indicators
- x-ray attacker counts

Global fields carry only exact symbolic state, for example:

- `in_check`
- king attacker counts
- king escape-square counts
- normalized legal-move count
- legal castle / en-passant / promotion availability

The same layer also emits a sparse reachability graph through aligned edge lists:

- `edge_src_square[]`
- `edge_dst_square[]`
- `edge_piece_type[]`

This layer must not include handcrafted evaluation. It exports exact structure, not chess heuristics.

### Piece-Intention Encoder

The piece-intention encoder is the first learned stage on top of `StateContextV1`.

The repository now includes a model-only Python reference implementation in
[intention_encoder.py](/home/torsten/EngineKonzept/python/train/models/intention_encoder.py)
plus isolated unit tests. It is intentionally not wired into the planner yet.

Its role is to turn exact piece, square, and reachability structure into per-piece intention vectors such as:

- capture pressure
- attacked / defended status
- blocking pressure on own lines
- role stability
- king-danger contribution

The king also gets a special intention path for:

- shelter
- escape geometry
- attacker proximity
- castling-state context

Architecturally, this is a small transformer over piece tokens with graph-aware attention over the reachability edges.

### State Embedder

The state embedder consumes:

- piece intentions
- square-side symbolic tokens
- global rule-state features
- the same reachability graph

It produces:

- `z_root`, the root latent state
- `sigma_root`, a root uncertainty scalar

The intended implementation is a relational transformer with mixed token self-attention and graph-masked attention.

The repository now also includes a model-only Python reference in
[state_embedder.py](/home/torsten/EngineKonzept/python/train/models/state_embedder.py).
It remains isolated from the planner until the later LAPv1 integration steps.

### Heads

LAPv1 uses one shared latent root with several specialized heads:

- `ValueHead`: WDL logits, cp score, and uncertainty
- `SharpnessHead`: a bounded scalar indicating whether more internal deliberation is likely worth it
- `LargePolicyHead`: scores only exact legal candidates through `CandidateContextV2`
- `OpponentHead`: predicts reply distribution, pressure, and uncertainty on imagined successor latents

The value and policy heads are intentionally large. The architecture prefers richer learned scoring over more classical search structure.

The repository now includes model-only Python references for the first two head
types in [value_head.py](/home/torsten/EngineKonzept/python/train/models/value_head.py).
They remain isolated from the planner until the later LAPv1 integration steps.

The large candidate-scoring policy path now also has a model-only reference in
[policy_head_large.py](/home/torsten/EngineKonzept/python/train/models/policy_head_large.py),
again still isolated from the planner.

### Bounded Recurrent Deliberation Loop

The deliberation loop is the central runtime mechanism.

It keeps:

- a latent root state `z_t`
- planner memory `M_t`
- candidate scores `C_t`
- a scratch imagined principal variation `PV_scratch_t`

At each bounded step it:

1. picks a small candidate subset to refine
2. applies a latent transition for each selected action
3. queries the opponent head on the imagined successor
4. updates candidate scores and planner memory
5. decides whether to stop, continue, or roll back to a previous latent snapshot

Hard boundaries:

- no recursive tree expansion
- no transposition table
- no alpha-beta bounds
- no quiescence search

This loop is allowed to be recurrent and adversarial, but it must remain bounded and traceable.

### Trace, PV, and UCI `info`

Every inner step should emit a trace record containing:

- selected candidates
- current top-1 action
- current top-1 value
- sharpness
- uncertainty
- scratch PV
- whether rollback fired

The future UCI `info depth ... pv ...` output is intended to reflect these learned deliberation steps, not classical search depths.

The repository now includes a model-only Python core loop in
[deliberation.py](/home/torsten/EngineKonzept/python/train/models/deliberation.py)
with deterministic rollback-focused unit tests. It remains isolated from the
runtime until the later LAPv1 wrapper and runtime steps.

The first full model-only wrapper now also exists in
[lapv1.py](/home/torsten/EngineKonzept/python/train/models/lapv1.py). It
composes the new encoder/embedder/head stack with bounded deliberation and the
existing opponent head, but still does not add trainer glue or runtime wiring.

The repository now also contains the first static-head stage-T1 trainer in
[lapv1.py](/home/torsten/EngineKonzept/python/train/trainers/lapv1.py). It
trains the wrapper with deliberation disabled on existing `planner_head`
artifacts by reconstructing LAPv1-side inputs from `fen` plus packed root
features, without yet enabling full deliberation-on training. The prepared CLI
entry points for that first run are now
[train_lapv1.py](/home/torsten/EngineKonzept/python/scripts/train_lapv1.py),
[eval_lapv1.py](/home/torsten/EngineKonzept/python/scripts/eval_lapv1.py), and
[run_lapv1_stage1_train.sh](/home/torsten/EngineKonzept/python/scripts/run_lapv1_stage1_train.sh).
The current Stage-T1 bootstrap path also keeps candidate masking finite and
clips gradients conservatively to reduce early large-model CPU-training
instability. The current bootstrap value path also uses a bounded `cp_score`
output plus robustly clipped root-value targets so rare mate/sentinel labels
cannot dominate the early regression loss. The same bootstrap path now also
clips raw teacher top1-top2 gap targets before they feed LAPv1 margin
supervision, so extreme mate-gap labels do not swamp the bounded policy losses.
The same trainer now also reads `planner_head` JSONL artifacts lazily via file
offset indexes instead of materializing the entire split into RAM up front. That
change was required once the all-unique Stage-T1 run moved past `675k` train
examples: the earlier eager path reached host OOM pressure before the first real
epoch could start.

For the next all-unique Stage-T1 bootstrap, the workflow now also materializes a
dedicated `lapv1_<split>.jsonl` artifact family. Those artifacts precompute the
LAPv1-side training inputs that previously had to be reconstructed inside the
trainer from `fen` plus packed root features:

- `piece_tokens`
- `square_tokens`
- `state_context_global`
- `reachability_*` sparse edge lists
- normalized teacher WDL / sharpness targets

That extra workflow layer keeps the Phase-10 contract explicit and removes the
most expensive per-example Python reconstruction from the training loop.

The original all-unique bootstrap config
[phase10_lapv1_stage1_all_unique_v1.json](/home/torsten/EngineKonzept/python/configs/phase10_lapv1_stage1_all_unique_v1.json)
remains as the larger historical reference. The preferred restart target is now
the smaller fast follow-up:

- [phase10_lapv1_stage1_fast_all_unique_v1.json](/home/torsten/EngineKonzept/python/configs/phase10_lapv1_stage1_fast_all_unique_v1.json)
- [phase10_agent_lapv1_stage1_fast_all_unique_v1.json](/home/torsten/EngineKonzept/python/configs/phase10_agent_lapv1_stage1_fast_all_unique_v1.json)
- [phase10_lapv1_stage1_fast_arena_all_unique_v1.json](/home/torsten/EngineKonzept/python/configs/phase10_lapv1_stage1_fast_arena_all_unique_v1.json)
- [run_phase10_lapv1_stage1_fast_arena_longrun.sh](/home/torsten/EngineKonzept/python/scripts/run_phase10_lapv1_stage1_fast_arena_longrun.sh)

That fast variant cuts the Stage-T1 bootstrap model from roughly `77.3M`
parameters (`~295 MB` FP32) to `19.8M` parameters (`~75.7 MB` FP32), raises the
batch size to `12`, and keeps the first all-unique LAPv1 benchmark within a
realistic CPU budget before the arena stage.

The same trainer now also supports a first stage-T2 extension with
deliberation enabled under a bounded `max_inner_steps` curriculum,
additional sharpness-target supervision on the emitted trace, a
deliberation-monotonicity auxiliary loss, and rollback statistics in the
epoch metrics. No real stage-T2 run config is prepared yet; the support is
kept trainer-local until the first dedicated LAPv1 follow-up config exists.

The first runtime-facing LAPv1 glue now also exists in
[lapv1_runtime.py](/home/torsten/EngineKonzept/python/train/eval/lapv1_runtime.py)
plus [phase10_agent_lapv1_stage1_v1.json](/home/torsten/EngineKonzept/python/configs/phase10_agent_lapv1_stage1_v1.json).
That path can already load an LAPv1 checkpoint, rebuild exact candidate inputs
from `StateContextV1` and `CandidateContextV2`, emit a bounded deliberation
trace, and choose a final legal move through the existing selfplay-agent
contract. It remains offline/runtime-glue only until the first real Stage-T1
arena bootstrap exists.

The next prepared bootstrap path is now the all-data Stage-T1 setup:

- [phase10_lapv1_stage1_all_unique_v1.json](/home/torsten/EngineKonzept/python/configs/phase10_lapv1_stage1_all_unique_v1.json)
- [phase10_agent_lapv1_stage1_all_unique_v1.json](/home/torsten/EngineKonzept/python/configs/phase10_agent_lapv1_stage1_all_unique_v1.json)
- [build_phase10_lapv1_workflow.py](/home/torsten/EngineKonzept/python/scripts/build_phase10_lapv1_workflow.py)
- [run_phase10_lapv1_stage1_arena_campaign.py](/home/torsten/EngineKonzept/python/scripts/run_phase10_lapv1_stage1_arena_campaign.py)
- [run_phase10_lapv1_stage1_arena_longrun.sh](/home/torsten/EngineKonzept/python/scripts/run_phase10_lapv1_stage1_arena_longrun.sh)

That path intentionally:

- reuses the strongest six current planner-family reference arms from the last completed `vice` run
- keeps those reference arms fixed
- trains only LAPv1 up front for `2` epochs on the new merged all-unique corpus
- then benchmarks LAPv1 in an 8-agent arena against those six arms plus `vice`
- builds the all-unique LAPv1 workflow in bounded chunks so the train tier can scale past the earlier single-process memory limit

The current all-unique raw source for that bootstrap is:

- [phase5_stockfish_all_unique_v1](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_all_unique_v1)

with the intake and selection rationale documented in
[phase10-all-unique-lapv1-prep-2026-04-05.md](/home/torsten/EngineKonzept/docs/experiments/phase10-all-unique-lapv1-prep-2026-04-05.md).
checkpoint is trained.

The first direct benchmark template is now also prepared in
[phase10_arena_lapv1_vs_baseline_v1.json](/home/torsten/EngineKonzept/python/configs/phase10_arena_lapv1_vs_baseline_v1.json).
It keeps the future LAPv1 Stage-T1 agent, the strongest kept planner baselines,
and `vice_v2` inside one seeded round-robin suite, but remains config-only until
the first trained LAPv1 checkpoint exists.

## Remaining Scale TODOs

The first all-unique memory pass exposed the next obvious scaling hotspots.
They are not blocking the current LAPv1 Stage-T1 resume, but they are the next
cleanup targets if later runs hit memory pressure again:

- [planner.py](/home/torsten/EngineKonzept/python/train/trainers/planner.py) still eagerly loads full `planner_head` corpora
- [evolution_campaign.py](/home/torsten/EngineKonzept/python/train/eval/evolution_campaign.py) still reads verify artifacts eagerly when assembling matrices
- [moe_analysis.py](/home/torsten/EngineKonzept/python/train/eval/moe_analysis.py) still assumes full in-memory `planner_head` access
- several JSONL dataset loaders still use eager `read_text(...).splitlines()` paths and should move to streaming readers when they become part of large-corpus jobs

## Runtime Flow

```text
UCI "go"
  │
  ▼
current_position (Rust)
  │
  ▼
build_state_context_v1(position)              [Rust, exact, cached pro ply]
  ├─ legal_moves
  ├─ attack_maps
  ├─ reachability_graph
  ├─ global_flags
  └─ candidate_context_v2 for each legal move
  │
  ▼
IF single_legal_move:
    emit "info depth 0 ..."
    emit "bestmove <the_move>"
    RETURN
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
best_action = argmax(final_policy) [deterministic tie-break]
  │
  ▼
verify_legality(best_action) [Rust, hard]
  │
  ▼
emit "bestmove <uci>"
```

## Why This Shape

LAPv1 tries to make the strongest current direction explicit:

- exact symbolic legality at the boundary
- one unified planner family instead of many loosely related arms
- recurrent bounded deliberation instead of wider static arm branching
- richer symbolic structure instead of flatter state summaries

It is intentionally a target architecture document, not a claim that the repository has already implemented the full stack.
