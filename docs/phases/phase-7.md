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

## Next pressure

The next useful Phase-7 steps are:

1. materialize `opponent_head_<split>.jsonl` on a real corpus
2. implement the exact symbolic reply-scorer baseline
3. train the first explicit opponent head against that baseline
