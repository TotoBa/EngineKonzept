# Phase 7

## Goal

Introduce the first explicit opponent model over exact legal replies.

The architectural intent remains:

- exact legality stays symbolic
- exact successor states stay symbolic
- the opponent module predicts reply behavior
- the planner consumes opponent signals later

## Current repository state

The repository now has the first trained Phase-7 opponent model, but it is still below the symbolic reply-scorer baseline.

It now does have the first explicit Phase-7 preparation artifacts:

- versioned `OpponentHeadV1` dataset examples in [opponent_head.py](/home/torsten/EngineKonzept/python/train/datasets/opponent_head.py)
- reproducible builder in [build_opponent_head_dataset.py](/home/torsten/EngineKonzept/python/scripts/build_opponent_head_dataset.py)
- supporting offline workflow layers:
  - `search_teacher_<split>.jsonl`
  - `search_traces_<split>.jsonl`
  - `search_disagreements_<split>.jsonl`
  - `search_curriculum_<split>.jsonl`
- larger merged-unique workflow slices:
  - [summary.json](/home/torsten/EngineKonzept/artifacts/phase7/opponent_workflow_merged_unique_train_v1/summary.json)
  - [summary.json](/home/torsten/EngineKonzept/artifacts/phase7/opponent_workflow_merged_unique_validation_v1/summary.json)
  - [summary.json](/home/torsten/EngineKonzept/artifacts/phase7/opponent_workflow_merged_unique_verify_v1/summary.json)
- the symbolic baseline artifacts:
  - [README.md](/home/torsten/EngineKonzept/artifacts/phase7/README.md)
  - [opponent_head_verify_probe_v1.jsonl](/home/torsten/EngineKonzept/artifacts/phase7/opponent_head_verify_probe_v1.jsonl)
  - [opponent_symbolic_baseline_verify_probe_v1.json](/home/torsten/EngineKonzept/artifacts/phase7/opponent_symbolic_baseline_verify_probe_v1.json)
- and the first trained head:
  - [phase7_opponent_merged_unique_mlp_v1.json](/home/torsten/EngineKonzept/python/configs/phase7_opponent_merged_unique_mlp_v1.json)
  - [summary.json](/home/torsten/EngineKonzept/artifacts/phase7/opponent_merged_unique_mlp_v1/summary.json)
  - [opponent_merged_unique_mlp_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase7/opponent_merged_unique_mlp_v1_verify.json)
  - [opponent_merged_unique_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase7/opponent_merged_unique_compare_v1.json)

The repo also now has larger end-to-end workflow slices over the merged unique corpus:

- [summary.json](/home/torsten/EngineKonzept/artifacts/phase7/opponent_workflow_merged_unique_train_v1/summary.json)
- [summary.json](/home/torsten/EngineKonzept/artifacts/phase7/opponent_workflow_merged_unique_verify_v1/summary.json)

## What the first dataset does

For one root position it builds:

1. the teacher-chosen root move
2. the exact successor state after that move
3. the exact legal opponent replies from that successor state
4. symbolic reply features over those legal replies
5. a teacher best-reply target
6. a first pressure target
7. a first uncertainty target

That gives the repo a real, inspectable contract for opponent modeling without hiding the problem inside the planner.

## What it does not do yet

- no Rust runtime opponent inference
- no planner integration
- no full reply-distribution supervision beyond the current best-reply-focused v1 target

## Current baseline rule

The first comparison baseline for Phase 7 is explicitly symbolic:

1. exact apply our move
2. exact-generate opponent legal replies
3. reuse the current symbolic proposer as the opponent reply scorer

Any learned opponent head should beat that baseline before it is treated as real progress.

The current verify probe on `16` held-out examples scored:

- `reply_top1_accuracy=0.25`
- `reply_top3_accuracy=0.25`
- `teacher_reply_mean_reciprocal_rank=0.364583`

with `/usr/games/stockfish18` at `64` nodes.

The larger merged-verify workflow slice now gives a more useful baseline target:

- [opponent_symbolic_baseline_merged_unique_verify_v1.json](/home/torsten/EngineKonzept/artifacts/phase7/opponent_symbolic_baseline_merged_unique_verify_v1.json)
- `reply_top1_accuracy=0.3`
- `reply_top3_accuracy=0.4`
- `teacher_reply_mean_reciprocal_rank=0.419262`

The first trained `mlp_v1` opponent head on that same verify slice currently scores:

- `reply_top1_accuracy=0.066667`
- `reply_top3_accuracy=0.333333`
- `teacher_reply_mean_reciprocal_rank=0.272664`

So the first learned head is a real experimental reference, but it does not yet clear the symbolic baseline bar.

## Next pressure

The next useful Phase-7 steps are:

1. use the trained head as an experimental planner-facing reply model
2. keep the symbolic reply scorer as the Phase-7 default until a learned head beats it
3. grow the workflow corpus and supervision richness before treating Phase 7 as solved
