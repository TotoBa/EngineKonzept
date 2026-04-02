# Model Contracts

This page defines the next explicit model-facing contracts that should stabilize before Phase 7 grows.

The current repo already has:

- exact symbolic legality
- a symbolic proposer over exact legal candidates
- a symbolic-action dynamics path that benefits from selected-move symbolic features

The next pressure is no longer just “another model variant.”
It is to make the proposer, dynamics, opponent, and planner interfaces explicit and versionable.

## Contract Discipline

The repo should evolve by versioned contracts, not by hidden feature drift.

That means:

- named feature orders
- versioned schemas
- explicit metadata in exported bundles
- cross-language validation between Python and Rust

The intended next contracts are:

1. `CandidateContextV2`
2. `TransitionContextV1`
3. `LatentStateV1`
4. `OpponentHeadV1`

## Feature Authority

One concrete risk is silent drift between:

- Python-side dataset/materialization code
- Rust-side runtime candidate-feature construction

The next strengthening step should be a single source of truth for symbolic feature schemas.

That can be implemented either by:

- generating Python and Rust constants from one schema file
or
- keeping the current duplicated definitions but adding golden cross-language tests that assert exact identity

Until that is done, feature evolution should be treated as high-risk.

## CandidateContextV2

This is the next root-candidate contract for proposer, offline search teachers, and later planner roots.

Status now:

- implemented as a versioned Python-side contract
- explicitly serialized in symbolic proposer artifacts
- intentionally not yet the default Rust runtime contract

That means the repository now has a real `CandidateContextV2` for datasets and offline workflows, while the shipped symbolic runtime still stays on the current `CandidateContextV1` until a matching runtime scorer is trained and exported.

It should keep:

- the current exact legal candidate discipline
- the current action key based on the existing factorized action vocabulary

It should add:

- promotion piece identity
- castle side identity
- full captured-piece type
- normalized move geometry
- clearer naming for existing pre-move attack-map features

It should remain:

- symbolic
- exact
- local
- cheap to compute

It should not drift into handcrafted static-eval fragments.

The current implemented `V2` fields are:

- promotion piece identity
- castle side identity
- full captured-piece type
- normalized move geometry
- renamed pre-move attack/defense fields

## TransitionContextV1

This is the next selected-action contract for dynamics and later opponent modeling.

It should contain:

- `CandidateContextV2`
- plus exact post-move tags from symbolic apply

Recommended first post-move tags:

- `opponent_in_check_after_move`
- `destination_attacked_after_move`
- `destination_defended_after_move`
- `halfmove_reset`
- castling-rights delta bits
- en-passant created or cleared

The important point is that dynamics should not be forced forever to reuse the proposer row unchanged.

## LatentStateV1

The most plausible planner-facing node state is dual-channel:

- `s_exact`: exact symbolic shadow state or exact encoded state
- `g_exact`: exact global-summary features
- `z`: learned latent state
- `u`: uncertainty/confidence summary

This keeps:

- legality
- exact candidate generation
- exact move application

under symbolic control, while letting learned modules carry:

- consequence representation
- opponent modeling
- planning memory
- WDL/value signals

## OpponentHeadV1

The first explicit opponent module should not be hidden inside the planner.

The minimum contract should predict:

- reply distribution over exact legal replies
- threat or pressure signal
- uncertainty

The first strong baseline should be:

1. exact apply our move
2. exact-generate opponent legal replies
3. reuse the current symbolic proposer as the opponent reply scorer

An explicit opponent head should only count as progress if it beats that baseline.

## Metrics Pressure

The current all-or-nothing `exact_next_feature_accuracy` is useful, but not enough by itself.

The next useful exactness metrics are:

- rule-token exactness
- occupancy exactness
- special-move exactness buckets
- `next_global_feature_l1`
- transition-tag accuracy

These are more informative for later opponent and planner work than one single global exactness number.

## Workflow Tie-In

These contracts are also the natural attachment points for offline alpha-beta and MCTS workflows:

- `CandidateContextV2` for proposer teacher labels
- `TransitionContextV1` for dynamics/opponent traces
- `LatentStateV1` for planner-facing supervision
- `OpponentHeadV1` for reply-policy and threat targets
