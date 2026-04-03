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

The repo now has the first real probe of that baseline:

- [opponent_head_verify_probe_v1.jsonl](/home/torsten/EngineKonzept/artifacts/phase7/opponent_head_verify_probe_v1.jsonl)
- [opponent_symbolic_baseline_verify_probe_v1.json](/home/torsten/EngineKonzept/artifacts/phase7/opponent_symbolic_baseline_verify_probe_v1.json)

Current probe result on `16` verify examples with `/usr/games/stockfish18` at `64` nodes:

- `reply_top1_accuracy=0.25`
- `reply_top3_accuracy=0.25`
- `teacher_reply_mean_reciprocal_rank=0.364583`
- `teacher_reply_mean_probability=0.201552`

The repo now also has the first larger workflow slices over the merged unique corpus:

- [summary.json](/home/torsten/EngineKonzept/artifacts/phase7/opponent_workflow_merged_unique_train_v1/summary.json)
- [summary.json](/home/torsten/EngineKonzept/artifacts/phase7/opponent_workflow_merged_unique_verify_v1/summary.json)

These are still workflow artifacts, not trained opponent-head runs, but they show that the full offline stack now scales beyond the small probe:

- merged-train slice: `256` examples, `61` reply-supervised
- merged-verify slice: `128` examples, `30` reply-supervised
- disagreement rate stays high at about `0.85` to `0.87`

The larger verify slice also has a fresh symbolic baseline artifact:

- [opponent_symbolic_baseline_merged_unique_verify_v1.json](/home/torsten/EngineKonzept/artifacts/phase7/opponent_symbolic_baseline_merged_unique_verify_v1.json)

Current result on the `128`-example verify slice:

- `reply_top1_accuracy=0.3`
- `reply_top3_accuracy=0.4`
- `teacher_reply_mean_reciprocal_rank=0.419262`

## First Trained Heads

The repo now also has the first trained `OpponentHeadV1` model:

- config: [phase7_opponent_merged_unique_mlp_v1.json](/home/torsten/EngineKonzept/python/configs/phase7_opponent_merged_unique_mlp_v1.json)
- bundle: [merged_unique_mlp_v1](/home/torsten/EngineKonzept/models/opponent/merged_unique_mlp_v1)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase7/opponent_merged_unique_mlp_v1/summary.json)
- verify eval: [opponent_merged_unique_mlp_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase7/opponent_merged_unique_mlp_v1_verify.json)
- direct comparison: [opponent_merged_unique_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase7/opponent_merged_unique_compare_v1.json)

Current verify result on the `128`-example merged slice:

- `reply_top1_accuracy=0.066667`
- `reply_top3_accuracy=0.333333`
- `teacher_reply_mean_reciprocal_rank=0.272664`

This means the first learned opponent head is measurable and reproducible, but it is still below the symbolic baseline on reply ranking.

The next larger-corpus rerun now changes that status:

- workflow suite:
  - [summary.json](/home/torsten/EngineKonzept/artifacts/phase7/opponent_workflow_corpus_suite_v1/summary.json)
- config:
  - [phase7_opponent_corpus_suite_set_v2_v1.json](/home/torsten/EngineKonzept/python/configs/phase7_opponent_corpus_suite_set_v2_v1.json)
- bundle:
  - [corpus_suite_set_v2_v1](/home/torsten/EngineKonzept/models/opponent/corpus_suite_set_v2_v1)
- summary:
  - [summary.json](/home/torsten/EngineKonzept/artifacts/phase7/opponent_corpus_suite_set_v2_v1/summary.json)
- verify comparison:
  - [opponent_corpus_suite_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase7/opponent_corpus_suite_compare_v1.json)

Aggregate verify result over the current `10k`, `122k`, and `400k` workflow slices:

- symbolic baseline:
  - `reply_top1_accuracy=0.288952`
  - `reply_top3_accuracy=0.524079`
  - `teacher_reply_mean_reciprocal_rank=0.448373`
- learned `set_v2`:
  - `reply_top1_accuracy=0.368272`
  - `reply_top3_accuracy=0.603399`
  - `teacher_reply_mean_reciprocal_rank=0.521661`

This makes `corpus_suite_set_v2_v1` the first Phase-7 learned head that actually clears the symbolic baseline bar.

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
