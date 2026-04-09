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

LAPv2 now also adds an optional shared-FT policy path in
[policy_head_nnue.py](/home/torsten/EngineKonzept/python/train/models/policy_head_nnue.py).
When `lapv2.nnue_policy` is enabled, root candidate logits come from
NNUE-style successor scoring on the same sparse FT used by the NNUE
value head instead of the dense `LargePolicyHead`.

The sharpness path can now also be phase-routed. With
`lapv2.sharpness_phase_moe`, the root sharpness scalar and the inner-loop
sharpness projector share the same hard phase routing as the other LAPv2
phase-aware modules while remaining bit-identical in the flag-off path.

The opponent reply path now also has an optional LAPv2 upgrade in
[opponent_readout.py](/home/torsten/EngineKonzept/python/train/models/opponent_readout.py).
With `lapv2.shared_opponent_readout`, the deliberation loop replaces the
separate dense opponent head with a lighter shared-backbone
`OpponentReadout` built around a move-conditioned `DeltaOperator`, while
preserving the legacy reply aggregation
`best_reply - 10 * pressure - 10 * uncertainty`.

On top of that readout path, `lapv2.distill_opponent` now enables an
optional training-only teacher loss. The trainer asks the deliberation
stack for per-step student and teacher reply targets and distills the
shared readout against the legacy opponent path, but ordinary runtime and
evaluation forwards keep using the same inference graph as before.

The current LAPv2 eval/runtime path can now also enable
`lapv2.accumulator_cache`. In that mode the NNUE policy scorer fixes the
phase expert once at the root, scores successor accumulators through the
incremental cache, and emits explicit cache hit/miss diagnostics. The
training path deliberately stays on the old vectorized scorer so step-12
does not change optimization behavior while the cache contract is still
being hardened.

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

The current implementation now treats halting and rollback as per-example
decisions inside a batch. Earlier prototypes applied those gates at batch scope,
which made large-batch T2 training distort the inner-loop statistics and the
effective update path.

Stage-T2 now also supports explicit training phases. The intended path is:

- `freeze_inner`: keep the root encoder/heads fixed and train only the inner-loop path
- `joint_finetune`: unfreeze the full wrapper for a short consolidation pass
- a phase may override its own train/validation artifacts
- a phase may mix multiple train artifacts with explicit weights and a fixed
  epoch example budget
- a phase may assign different learning-rate scales to different trainable
  parameter groups
- a phase may independently schedule `min_inner_steps` and `max_inner_steps`

This avoids conflating "the root got better" with "the inner step learned to use
its budget better".

The current inner-loop update is also residual over the root policy scores:

- `root_logits` remain the fixed reference inside one forward pass
- deliberation learns `delta_logits`
- final ranking is `root_logits + delta_logits`

This preserves the useful Stage-T1 root prior and lets additional inner steps
learn targeted corrections instead of repeatedly rewriting the full score vector.

Stage-T2 diagnostics now explicitly compare root vs final behavior:

- root `top1` / root `MRR`
- final `top1` / final `MRR`
- `top1_changed_rate`
- teacher-rank improvement/degradation rates
- root-incorrect improvement rate vs root-correct degradation rate
- per-example rollback rate
- mean executed inner steps and step histogram

That makes later UCI-side trace work much easier because the training summary now
already exposes whether deeper budgets actually helped.

Stage-T2 checkpoint selection is now also allowed to use a separate common
holdout that is fixed across phases. That is important once some phases validate
on hard subsets and others validate on the full corpus; otherwise `best_epoch`
quietly compares incomparable validation slices.

The first explicit LAPv2 warm-start export path now also exists. Instead of
bootstrapping a new LAPv2 run by loading a legacy checkpoint ad hoc at train
time, [build_lapv2_warm_start_checkpoint.py](/home/torsten/EngineKonzept/python/scripts/build_lapv2_warm_start_checkpoint.py)
can now materialize one proper init checkpoint in advance:

- phase-routed encoder/embedder/sharpness modules are expanded from the LAPv1 source
- the old shared trunk remains numerically identical
- `ft`, `value_head_nnue`, `policy_head_nnue`, and `opponent_readout` stay on
  fresh initialization for the actual LAPv2 adaptation

Stage-T2 now also uses an explicit improvement-over-root loss on positions where
the detached root policy is still wrong. Final logits and intermediate step
logits are penalized when they fail to beat the detached root cross-entropy by a
small margin. The intended effect is to train the residual deliberation path to
make real corrections instead of merely relearning the already-strong root
distribution.

The Stage-T2 trainer can now also apply an explicit phase load balancer.
When `stage2.phase_load_balance` is enabled, the trainer computes empirical
phase weights from the current batch and reweights the per-example root
value/policy/sharpness losses before the later shared-loss normalization.
That keeps the rarer phase buckets visible once the LAPv2 runs start mixing
hard subsets and broader full-corpus epochs.

The old single `nnue_phase_gate_steps` hook is now complemented by an
explicit two-stage T2 gate:

- `stage2.gate_stage_a_steps`: NNUE experts stay under mean-pull
- `stage2.gate_stage_b_steps`: the NNUE hook is released and the heads may
  diverge phase-specifically

The encoder/embedder phase routing stays active in both stages; the gate only
controls how quickly the NNUE heads are allowed to specialize.

For long CPU runs, LAPv1 also now emits explicit progress logs during the
previously quiet parts of the pipeline:

- lazy `lapv1_*` dataset indexing logs start/progress/done markers
- validation passes emit their own batch progress lines
- the Phase-10 campaign log reports the real configured LAPv1 stage (`T1` or
  `T2`) instead of a hard-coded `Stage1` label

On top of those progress lines, each completed epoch now records and prints
LAPv2-specific diagnostics that are directly useful for longer tuning runs:

- expert usage per phase bucket
- per-phase root value and policy branch losses
- FT expert drift from the shared mean
- cosine distance between NNUE value and policy adapters
- reply-consistency correlation when opponent distillation exposes both
  student and teacher reply tensors

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

For the cache audit path, the loop now also emits:

- selected candidate-slot tensors per step
- fixed phase indices per step when phase-routed LAPv2 modules are active

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
realistic CPU budget before the arena stage. The paired arena config now also
resolves its six legacy reference arms from the materialized `round_10`
`active_agent_specs` snapshot of the last completed `vice` evolution run and
uses two runtime LAPv1 variants from the same trained checkpoint (`inner0` and
`inner1`). It also switches to a `150`-opening Thor suite with global per-pair
opening assignment, so non-swapped arena games stay unique while color-swapped
rematches keep the same opening.

The same trainer now also supports a first stage-T2 extension with
deliberation enabled under a bounded `max_inner_steps` curriculum,
additional sharpness-target supervision on the emitted trace, a
deliberation-monotonicity auxiliary loss, and rollback statistics in the
epoch metrics. No real stage-T2 run config is prepared yet; the support is
kept trainer-local until the first dedicated LAPv1 follow-up config exists.

The first completed all-unique fast Stage-T1 arena run is documented in
[phase10-lapv1-stage1-fast-arena-summary-2026-04-06.md](/home/torsten/EngineKonzept/docs/experiments/phase10-lapv1-stage1-fast-arena-summary-2026-04-06.md).
Its key takeaway is narrow but important: `inner1` underperformed `inner0`, but
that comparison used the same Stage-T1 checkpoint trained only with
`max_inner_steps = 0`. The correct next comparison is therefore not another
runtime-only override, but a real Stage-T2 checkpoint evaluated at multiple
runtime inner-step caps.

The follow-up path now reflects that diagnosis directly:

- Stage-T2 warm-starts from the completed fast Stage-T1 checkpoint instead of
  relearning the shared encoder/state stack from scratch
- the trainer now carries an explicit per-step policy-supervision loss over the
  intermediate candidate-score tensors emitted by the deliberation loop
- runtime `deliberation_max_inner_steps` is treated as the hard inner-loop
  budget cap, while `q_threshold` plus top-1 stability remain the learned
  early-stop gates inside that budget

That means the intended comparison is now:

- `inner0`: same trained Stage-T2 checkpoint, budget cap `0`
- `inner1`: same trained Stage-T2 checkpoint, budget cap `1`
- `inner2`: same trained Stage-T2 checkpoint, budget cap `2`
- `auto4`: same trained Stage-T2 checkpoint, budget cap `4`, but the loop may
  stop earlier on its own

Future UCI-side status reporting for the inner loop remains a deliberate
follow-up, not part of the current repair. The runtime should later be able to
surface at least:

- current inner step
- current top-1 move
- sharpness / uncertainty
- rollback fired or not
- early-stop reason / budget exhaustion

Stage-T2 training now also emits explicit collapse warnings when a validation
pass falls back toward `root ~= final` despite a nonzero inner-step budget. That
warning is meant to catch the exact failure mode that showed up after the early
hard-curriculum gains were washed out by later full-data finetuning.

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
