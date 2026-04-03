# Phase 7

## Goal

Introduce the first explicit opponent model over exact legal replies.

The architectural intent remains:

- exact legality stays symbolic
- exact successor states stay symbolic
- the opponent module predicts reply behavior
- the planner consumes opponent signals later

## Current repository state

The repository does not yet have a trained Phase-7 opponent model.

It now does have the first explicit Phase-7 preparation artifacts:

- versioned `OpponentHeadV1` dataset examples in [opponent_head.py](/home/torsten/EngineKonzept/python/train/datasets/opponent_head.py)
- reproducible builder in [build_opponent_head_dataset.py](/home/torsten/EngineKonzept/python/scripts/build_opponent_head_dataset.py)
- supporting offline workflow layers:
  - `search_teacher_<split>.jsonl`
  - `search_traces_<split>.jsonl`
  - `search_disagreements_<split>.jsonl`
  - `search_curriculum_<split>.jsonl`
- and the first exact symbolic baseline probe:
  - [README.md](/home/torsten/EngineKonzept/artifacts/phase7/README.md)
  - [opponent_head_verify_probe_v1.jsonl](/home/torsten/EngineKonzept/artifacts/phase7/opponent_head_verify_probe_v1.jsonl)
  - [opponent_symbolic_baseline_verify_probe_v1.json](/home/torsten/EngineKonzept/artifacts/phase7/opponent_symbolic_baseline_verify_probe_v1.json)

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

- no trained opponent head
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

## Next pressure

The next useful Phase-7 steps are:

1. scale the `OpponentHeadV1` workflow beyond the current verify probe
2. train the first explicit opponent head against the symbolic reply-scorer baseline
3. compare it against the documented baseline artifact before any planner integration
