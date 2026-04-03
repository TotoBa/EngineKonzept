# Opponent Architecture

This page defines the first explicit opponent-facing contract in EngineKonzept.

## Role

The opponent module exists to model what the adversary is likely to do after one exact chosen move.

It is not hidden inside the planner and it is not classical search.

The intended shape remains:

`root position -> chosen move -> exact successor state -> opponent head over exact legal replies`

## OpponentHeadV1

The first explicit contract should predict at least:

- a reply distribution over exact legal replies
- a threat or pressure signal
- an uncertainty signal

The repository now has the first dataset-level version of that contract:

- [opponent_head.py](/home/torsten/EngineKonzept/python/train/datasets/opponent_head.py)
- [build_opponent_head_dataset.py](/home/torsten/EngineKonzept/python/scripts/build_opponent_head_dataset.py)

It is still a dataset/workflow layer, not a trained Phase-7 model yet.

## Current Dataset Contract

`OpponentHeadV1` examples now carry:

- root packed features
- chosen root move as exact flat action index
- `TransitionContextV1` for that chosen move
- exact successor packed features
- exact legal reply candidates after the chosen move
- symbolic reply candidate features
- teacher best-reply supervision from search traces
- curriculum labels and priority from the offline workflow layer
- a first pressure target
- a first uncertainty target

This keeps the contract fully compatible with the repo’s symbolic authority:

- exact legal reply generation stays symbolic
- exact successor state stays symbolic
- search remains offline and teacher-only

## Baseline Rule

Before a learned opponent head counts as progress, compare it against the exact symbolic baseline:

1. exact apply our move
2. exact-generate opponent legal replies
3. reuse the current symbolic proposer as the opponent reply scorer

That baseline is the current minimum bar for Phase 7.

## Why This Contract

This contract is deliberately narrow and useful:

- it reuses the current symbolic proposer and dynamics contracts
- it exposes reply prediction as its own measurable problem
- it keeps planner code from absorbing opponent logic too early
- it gives offline workflows a clean place to attach search traces and curriculum buckets

## Current Limits

The current `OpponentHeadV1` dataset still uses simple proxies for some targets:

- reply supervision is currently best-reply focused, not a full reply-policy trace
- pressure is a simple exact reply-category signal
- uncertainty is currently curriculum-derived, not a learned calibrated posterior target

Those are acceptable for the first dataset contract, but they are not the final Phase-7 target design.
