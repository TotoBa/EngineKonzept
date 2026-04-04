# Architecture Overview

EngineKonzept is converging toward a latent-planning chess engine, not a classical search engine with neural scoring attached.

## Target Runtime Path

`position -> encoder -> legality/policy proposer -> latent dynamics -> opponent module -> recurrent planner -> WDL + move selection -> UCI output`

The current canonical roadmap for the learned stack is documented in [model-roadmap.md](/home/torsten/EngineKonzept/docs/architecture/model-roadmap.md).

## LAPv1 Target Architecture

The next explicit unification target for the planner family is documented in [lapv1-overview.md](/home/torsten/EngineKonzept/docs/architecture/lapv1-overview.md).

That document captures the intended shift from many separate planner arms toward one bounded recurrent latent-adversarial stack with:

- symbolic `StateContextV1`
- piece-intention encoding
- relational state embedding
- large value and policy heads
- bounded recurrent deliberation with trace emission

## Repository Boundaries

- `rust/` will hold runtime, exact rules, UCI, inference integration, planner runtime, and evaluation harnesses.
- `python/` will hold datasets, training, model export, and experiment code.
- `tests/` will hold shared fixtures and reference suites.
- `models/` and `artifacts/` will store generated outputs, not source code.
- local IPC and future distribution notes live in [distribution.md](/home/torsten/EngineKonzept/docs/architecture/distribution.md)
- the current Phase-6 dynamics design lives in [dynamics.md](/home/torsten/EngineKonzept/docs/architecture/dynamics.md)

## Phase 0 Constraints

This phase intentionally stops at scaffolding:

- no chess rules
- no UCI handling
- no planner code
- no model training
- no classical search fallback

The purpose of the current tree is to lock the module boundaries and validation workflow before any engine logic starts.

## Phase 1 Status

The repository now also includes the exact symbolic rules core under `core-types`, `position`, and `rules`.
That core is the correctness floor for later planner-oriented phases, not a conventional engine runtime.

## Phase 2 Status

The repository now additionally exposes a minimal UCI shell under `uci-protocol` and `engine-app`.
That shell can reconstruct exact positions and emit deterministic legal stub moves, but it still contains no search, evaluation, or planner logic.

## Phase 3 Status

The repository now additionally defines a factorized move vocabulary in `action-space` and a deterministic object-centric position encoder in `encoder`.
These layers make symbolic state consumable by later learned components without introducing inference, policy models, or planner logic yet.

## Phase 4 Status

The repository now additionally includes a Python dataset pipeline and a Rust dataset oracle backed by the exact rules kernel.
This adds reproducible example schemas, labels, split generation, and summary reporting without introducing training, inference, or selfplay yet.

## Phase 5 Status

The repository now additionally includes the first learned legality/policy proposer in Python, held-out proposer metrics, a `torch.export` bundle, a Rust-side bundle loader, and an offline PGN-to-Stockfish labeling path for larger policy datasets.
The runtime still does not execute learned inference yet, and no dynamics, opponent, planner, or classical search logic has been introduced.

Phase 5 now has two architecture families behind the same export contract:

- `mlp_v1`
- `multistream_v2`

The measured result so far is that structured inputs are promising but not yet enough by themselves. The current default remains the simpler MLP path, while the next preferred model step is a factorized proposer decoder rather than immediately moving to heavier expert-routing designs.

## Phase 6 Status

The repository now additionally includes the first action-conditioned latent-dynamics baseline in Python, lean dynamics split artifacts, held-out reconstruction and drift metrics, and a Rust-side bundle loader for exported dynamics metadata.

The larger merged-unique reruns now show a clearer direction inside the same Phase-6 contract:

- the large-corpus `structured_v3` rerun beats the old large `structured_v2_latent` baseline on both one-step and drift
- the large-corpus `structured_v6` rerun is now the best measured Phase-6 path so far

Exact packed next-state accuracy is still `0.0`, so the next pressure is still model quality rather than more plumbing, but the action-conditioned symbolic move-side contract now looks materially more promising than it did on the earlier `10k` corpus alone.

## Phase 7 Status

The repository now additionally includes the first explicit Phase-7 dataset contract plus a larger-corpus learned opponent head that beats the symbolic reply-scorer baseline on the current three-tier verify suite:

- exact successor-state generation for one chosen root move
- exact legal reply generation from that successor state
- symbolic reply-candidate features
- teacher best-reply supervision
- first pressure and uncertainty targets

The current preferred Phase-7 reference is now the larger-corpus `set_v2` arm over the `10k`, `122k`, and `400k` workflow tiers. Details live in [opponent.md](/home/torsten/EngineKonzept/docs/architecture/opponent.md) and [phase-7.md](/home/torsten/EngineKonzept/docs/phases/phase-7.md).

## Phase 8 Status

The repository now additionally includes the first trained bounded planner arm over the same `10k`, `122k`, and `400k` workflow tiers:

- bounded planner-head datasets derived from exact root candidates, exact successor states, and bounded opponent signals
- a trainable `set_v1` planner head in Python
- aggregate held-out comparison against root-only, symbolic-reply, and learned-reply bounded baselines

The current Phase-8 picture now has two useful references:

- full mixed-suite training reference:
  [phase8_planner_corpus_suite_set_v2_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v2_expanded_v1.json)
- preferred filtered validation reference:
  [phase8_planner_corpus_suite_set_v2_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v2_v1.json)

The planner contract also supports optional Phase-6 latent successor features, validated on a filtered `10k + 122k` slice in [planner_corpus_suite_two_tier_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_two_tier_compare_v1.json). The first direct latent-state arm (`set_v3`) underperforms `set_v2` there, so the contract extension is kept while the simpler `set_v2` planner remains preferred. The newer expanded-data filtered comparison in [planner_corpus_suite_expanded_two_tier_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_expanded_two_tier_compare_v1.json) shows that the `400k` tier helps the full mixed run, but does not yet beat the older two-tier `set_v2` reference on the preferred filtered slice.

The next rerun on stronger `10k + 122k`-only workflow material changes that filtered picture again: the new local rerun lifts the preferred filtered `set_v2` reference to `root_top1_accuracy=0.819336` and `MRR=0.889811`, which is now the best Phase-8 result on the user-preferred validation slice.
