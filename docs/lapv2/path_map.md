# LAPv2 Path Map

This file maps the logical LAPv2 plan vocabulary onto the current
repository paths. It is the execution anchor for the Rev-3 / Rev-3.1
LAPv2 documents.

| Logical Name | Path | Symbol | Purpose |
| --- | --- | --- | --- |
| IntentionEncoder | [python/train/models/intention_encoder.py](/home/torsten/EngineKonzept/python/train/models/intention_encoder.py) | `PieceIntentionEncoder` | Piece-token encoder and auxiliary intention probe source. |
| StateEmbedder | [python/train/models/state_embedder.py](/home/torsten/EngineKonzept/python/train/models/state_embedder.py) | `RelationalStateEmbedder` | Relational state trunk after the intention pass. |
| PolicyHead | [python/train/models/policy_head_large.py](/home/torsten/EngineKonzept/python/train/models/policy_head_large.py) | `LargePolicyHead` | Root candidate scorer for LAPv1. |
| ValueHead | [python/train/models/value_head.py](/home/torsten/EngineKonzept/python/train/models/value_head.py) | `ValueHead` | Scalar root value head. |
| SharpnessHead | [python/train/models/value_head.py](/home/torsten/EngineKonzept/python/train/models/value_head.py) | `SharpnessHead` | Sharpness / uncertainty head. |
| OpponentHead | [python/train/models/opponent.py](/home/torsten/EngineKonzept/python/train/models/opponent.py) | `OpponentHeadModel` | Opponent reply readout in the current LAPv1 runtime. |
| DeliberationLoop | [python/train/models/deliberation.py](/home/torsten/EngineKonzept/python/train/models/deliberation.py) | `DeliberationLoop` | Bounded inner-loop planning module. |
| LAPv1 Top-Level Model | [python/train/models/lapv1.py](/home/torsten/EngineKonzept/python/train/models/lapv1.py) | `LAPv1Model`, `LAPv1Config` | Wrapper that composes encoder, heads, and deliberation. |
| LAPv1 Artifact Loader | [python/train/datasets/lapv1_training.py](/home/torsten/EngineKonzept/python/train/datasets/lapv1_training.py) | `LAPv1TrainingExample`, `lapv1_training_example_from_planner_head` | Phase-10 JSONL artifact used by LAPv1 T1/T2. |
| Planner-Head Source Artifact | [python/train/datasets/planner_head.py](/home/torsten/EngineKonzept/python/train/datasets/planner_head.py) | `PlannerHeadExample` | Root-candidate training artifact that LAPv1 artifacts are built from. |
| Stage-T1 / Stage-T2 Trainer | [python/train/trainers/lapv1.py](/home/torsten/EngineKonzept/python/train/trainers/lapv1.py) | `train_lapv1`, `evaluate_lapv1_checkpoint` | Trainer, evaluator, selection logic, and curriculum orchestration. |
| Phase-10 Workflow Builder | [python/scripts/build_phase10_lapv1_workflow.py](/home/torsten/EngineKonzept/python/scripts/build_phase10_lapv1_workflow.py) | `main` | Builds the `planner_head_*` and `lapv1_*` workflow artifacts. |
| Hard-Subset Builder | [python/scripts/build_lapv1_hard_positions_dataset.py](/home/torsten/EngineKonzept/python/scripts/build_lapv1_hard_positions_dataset.py) | `main` | Produces the hard curriculum subset used by Stage-T2. |
| Arena Campaign Runner | [python/scripts/run_phase10_lapv1_stage1_arena_campaign.py](/home/torsten/EngineKonzept/python/scripts/run_phase10_lapv1_stage1_arena_campaign.py) | `run_phase10_lapv1_stage1_arena_campaign` | End-to-end Phase-10 bootstrap, verify, and arena driver. |
| Arena Core | [python/train/eval/arena.py](/home/torsten/EngineKonzept/python/train/eval/arena.py) | `SelfplayArenaSpec`, `run_selfplay_arena` | Generic arena harness used by Phase-9 and Phase-10 runs. |
| Initial FEN Suite Loader | [python/train/eval/initial_fens.py](/home/torsten/EngineKonzept/python/train/eval/initial_fens.py) | `SelfplayInitialFenSuite` | Opening-suite loader for deterministic arena starts. |
| Arena Matrix Builder | [python/train/eval/matrix.py](/home/torsten/EngineKonzept/python/train/eval/matrix.py) | `build_selfplay_arena_matrix` | Aggregates finished arena sessions into standings and pairwise matrices. |
| Move/Action Mapping | [python/train/datasets/opponent_head.py](/home/torsten/EngineKonzept/python/train/datasets/opponent_head.py) | `move_uci_for_action` | Maps flat action indices back to exact UCI moves for artifact enrichment. |
| Action Flattening Contract | [python/train/datasets/contracts.py](/home/torsten/EngineKonzept/python/train/datasets/contracts.py) | `flatten_action` | Canonical symbolic action-index contract used across artifacts. |
| Phase Feature Heuristic | [python/train/datasets/phase_features.py](/home/torsten/EngineKonzept/python/train/datasets/phase_features.py) | `phase_score`, `phase_index` | Hard 4-bucket phase heuristic introduced for LAPv2 artifact enrichment. |
| HalfKA Feature Builder | [python/train/datasets/nnue_features.py](/home/torsten/EngineKonzept/python/train/datasets/nnue_features.py) | `halfka_active_indices`, `halfka_index` | Sparse NNUE-style feature extraction for both king perspectives. |
| Move Delta Builder | [python/train/datasets/move_delta.py](/home/torsten/EngineKonzept/python/train/datasets/move_delta.py) | `halfka_delta`, `is_king_move`, `move_type_hash` | Candidate-local sparse feature delta annotations for LAPv2 artifacts. |
| Phase Router | [python/train/models/phase_router.py](/home/torsten/EngineKonzept/python/train/models/phase_router.py) | `PhaseRouter` | Hard pass-through router from precomputed `phase_index` tensors. |
| Phase MoE Wrapper | [python/train/models/phase_moe.py](/home/torsten/EngineKonzept/python/train/models/phase_moe.py) | `PhaseMoE` | Generic hard-routed expert wrapper for later phase-dependent modules. |
| Feature Transformer | [python/train/models/feature_transformer.py](/home/torsten/EngineKonzept/python/train/models/feature_transformer.py) | `FeatureTransformer` | EmbeddingBag-based HalfKA accumulator backbone for future NNUE heads. |
| Dual Accumulator | [python/train/models/dual_accumulator.py](/home/torsten/EngineKonzept/python/train/models/dual_accumulator.py) | `DualAccumulatorBuilder`, `IncrementalAccumulator` | White/black accumulator construction and sparse incremental updates. |
| NNUE Value Head | [python/train/models/value_head_nnue.py](/home/torsten/EngineKonzept/python/train/models/value_head_nnue.py) | `NNUEValueHead`, `ClippedReLU` | Single-phase FT-based LAPv2 value readout before phase-dependent value experts. |
| NNUE Policy Head | [python/train/models/policy_head_nnue.py](/home/torsten/EngineKonzept/python/train/models/policy_head_nnue.py) | `NNUEPolicyHead` | Shared-FT successor scorer for LAPv2 root policy logits. |

Notes:

- The Rev-3.1 document expects `docs/lapv2/plan_rev3.md`. The user-supplied
  Rev-3 source currently lives at
  [docs/LAPv2_Codex_Plan_rev3.md](/home/torsten/EngineKonzept/docs/LAPv2_Codex_Plan_rev3.md);
  a local bridge file is added alongside this map.
- The Rev-3.1 baseline-reproduction assumption is outdated in this repo state:
  the existing `v4` run already contains verify and arena outputs.
