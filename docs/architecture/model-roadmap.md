# Model Roadmap

This document is the canonical architecture roadmap for the learned part of EngineKonzept.

It consolidates:

- the current measured Phase-5 proposer results
- the broader architectural direction from [arch.ideas.md](/home/torsten/EngineKonzept/docs/arch.ideas.md)
- the follow-up repo-specific ideas from `docs/followup.ideas.md`
- the phase boundaries in [PLANS.md](/home/torsten/EngineKonzept/PLANS.md)

The intent is to keep one clear answer to two questions:

1. what the project is trying to build next
2. what it is explicitly **not** prioritizing yet

## Current Decision Stack

The current preferred direction is:

1. keep the symbolic rules core exact and outside learned move selection
2. keep UCI/runtime in Rust and training/experiments in Python
3. improve inductive bias before adding heavy routing complexity
4. preserve externally checkable dataset, export, and evaluation contracts
5. treat Phase 5 as a proposer-quality and representation-learning phase, not a hidden planner phase
6. use classical search methods only as offline workflow tools unless the runtime architecture is explicitly changed by decision

That implies the following near-term sequence:

1. exact symbolic legality plus learned candidate scoring
2. stronger proposer candidate scoring
3. action-conditioned latent dynamics
4. explicit opponent modeling
5. bounded recurrent planning
6. only then heavier expert routing or option-style deliberation control

## What The Current Results Say

The current `10,240 / 2,048` Pi-labeled Phase-5 corpus has nine externally checkable proposer arms:

- `current_default`
- `h256`
- `policy_focus`
- `multistream_v2`
- `factorized_v3`
- `factorized_v4`
- `factorized_v5`
- `factorized_v6`
- `relational_v1`

See [stockfish_pgn_10k_seven_way_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_10k_seven_way_compare_v1.json).

The important findings are:

- `legality` is easier than `policy`
- widening the flat MLP helps legal-set F1 more than policy accuracy
- a pure policy-loss reweighting run did not improve held-out policy quality
- the first structured multi-stream proposer slightly improved validation legality over the default MLP, but it did not beat `h256`, did not improve policy, and was slower
- the first additive factorized decoder cut parameters drastically, but collapsed both legality and policy quality
- the first conditional factorized decoder became the best legal-set-F1 arm so far, but still did not take the policy lead from `current_default`
- the next policy-stronger conditional decoder regained much of the policy gap while still beating the older MLP baselines on legality
- the next pairwise-coupled decoder `factorized_v6` became the strongest legality arm so far by a clear margin
- the first relational typed-backbone plus stronger-decoder hybrid improved policy over the older factorized arms, but still did not take the policy lead from `current_default`
- explicit checkpoint selection now shows a real legality-vs-policy tradeoff inside `factorized_v5`
- the first symbolic-candidate proposer arm replaced learned legality with exact legal-move generation and decisively beat every learned-legality arm on the current `10k` corpus

That means:

- raw width is still usable
- structure helps somewhat
- but the strongest new signal is no longer "more of the same MLP"; it is "score exact legal candidates instead of learning legality"

## Architecture Decisions

## Phase 5

### Keep

- the exact Rust legality authority
- the current factorized move vocabulary
- the current Phase-5 dataset and evaluation harnesses

### Add next

- richer symbolic candidate features over the exact legal move set
- stronger candidate scorers over the current symbolic proposer contract
- downstream dynamics and planner modules that consume the same symbolic candidate set

### Do not add yet

- mixture-of-experts routing as the default proposer
- dynamic-depth or adaptive-compute controllers
- planner-time expert graphs
- anything that behaves like hidden search

### Why

The repo already has an object-centric encoder and a factorized move schema. The biggest Phase-5 mismatch was that the learned proposer spent capacity learning legality and scoring over the full flat `20480` action space instead of exact legal candidates.

The next best repo-local step is therefore:

- not "more routing"
- not "more learned legality"
- but "score exact legal candidates with richer symbolic move context"

The factorized decoder line now has three concrete lessons:

- additive factorization threw away too much coupling
- conditional factorization recovered that coupling and improved legality substantially
- policy-specific residual capacity recovered much of the lost policy signal while keeping the factorized structure

The remaining gap is now:

- best policy among learned-legality arms still belongs to `current_default`
- best legality now belongs to `factorized_v6`
- best legality/policy balance among the newer factorized arms still does not beat `current_default` on policy
- the typed-backbone `relational_v1` run is now the better policy result among the newer structured arms
- the new symbolic-candidate `symbolic_v1` arm beats all of them on the `10k` corpus and now carries the official runtime/export contract

Inside the learned-legality line there is still a useful method choice as well:

- `legality_first` selection keeps the strongest legal-set F1
- `balanced` selection lifts policy somewhat further, but gives up legality

## Phase 6

### Preferred Dynamics Shape

The dynamics model should be local and action-conditioned:

- input: encoded state plus action
- output: next latent state
- reconstruction and drift metrics remain explicit

The preferred design is a local updater, not a whole-board hallucination model:

- update affected piece and square state
- update the global rule state
- track special moves separately

### Current implementation status

The repository now has a first `v1` baseline for this phase:

- lean `dynamics_<split>.jsonl` artifacts
- action-conditioned latent transition training
- one-step reconstruction metrics
- short-horizon drift metrics
- exported `torch.export` bundle plus Rust metadata validation

The first result is useful, but still weak:

- held-out reconstruction error decreases materially
- exact packed next-state accuracy remains `0.0`
- multi-step drift is measurable but not yet good

That means the plumbing direction is now validated, while model quality is still open.

The next structured follow-up already clarified two Phase-6 decisions:

- separate piece/square/rule decoding is better than the flat `v1` decoder
- explicit held-out drift-slice checkpoint selection is worth keeping

The latest parallel follow-up clarified two more:

- auxiliary latent-consistency supervision is worth keeping and is now the current Phase-6 default
- pure local edit-target reconstruction is informative, but in its current form it is too drift-unstable to become the default

The next `structured_v3_v1` follow-up refined that picture:

- auxiliary delta supervision can improve one-step soft reconstruction without moving all the way to the unstable `edit_v1` contract
- but the current formulation still gives back some drift quality, so it remains experimental

The explicit drift-supervision `structured_v4_v1` follow-up clarified one more point:

- simply adding a short rollout loss on top of the current contract is not enough; this formulation is worse than `structured_v2_latent_v1` and `structured_v3_v1`

The next `structured_v5_v1` follow-up clarified the first direct proposer-to-dynamics contract question:

- feeding exact symbolic move-side features from the symbolic proposer contract helps one-step reconstruction
- but that alone is not enough to protect multi-step drift, so it remains experimental

The next `structured_v6_v1` follow-up refines the selected-action contract question again:

- the richer `TransitionContextV1` contract is now wired through a real experimental dynamics arm
- on the `10k` corpus it improves both one-step feature error and drift over `structured_v5_v1`
- but it is still not enough evidence to replace the current large-corpus `structured_v5` default without rerunning the same contract at scale

The larger merged-unique reruns refine that again:

- the large `structured_v3` rerun beats the old large `structured_v2_latent` baseline on both verify feature error and verify drift
- the large symbolic-action `structured_v5` rerun becomes the first strong large-corpus Phase-6 arm
- the large transition-context `structured_v6` rerun then edges out `structured_v5` on one-step reconstruction and improves drift materially on the same corpus

So the current preferred Phase-6 path is now the large-corpus transition-context dynamics line, not the earlier smaller-corpus `structured_v2_latent` reference and not the earlier large `structured_v5` default.

### Why

This fits the exactness constraints and the planned Phase-6 measurements better than a diffuse global reconstructor.

## Phase 7

### Preferred Opponent Model

The opponent model should be explicit and multi-output.

At minimum it should predict:

- a reply distribution
- a threat or pressure signal
- uncertainty

It should not be hidden inside the planner as an uninspectable intermediate.

### Why

The repo is specifically aiming for adversarial latent planning. An explicit opponent module is the cleanest way to preserve that architectural intent.

The repo now also has the first exact symbolic Phase-7 baseline probe:

- exact teacher root move from the offline workflow layer
- exact successor state
- exact legal reply set
- current symbolic proposer reused as the reply scorer

That gives Phase 7 a real minimum bar before any learned opponent head is treated as progress.

The repo now also has two concrete learned `OpponentHeadV1` references:

- the earlier merged-unique `mlp_v1` baseline
- the newer three-tier `set_v2` run over the `10k`, `122k`, and `400k` workflow suite

Current decision:

- keep the trained `mlp_v1` head as an explicit experimental reference
- use the larger-corpus `set_v2` head as the active Phase-7 default
- keep the symbolic reply scorer as the regression baseline

Why:

- the learned head is now measurable, planner-usable, and actually stronger than the symbolic baseline on the current multi-corpus holdout
- the symbolic reply scorer is still valuable as a stable sanity-check baseline, especially for future workflow and planner regressions

## Phase 8

### Preferred Planner Shape

The planner should be:

- bounded
- recurrent
- inspectable
- uncertainty-aware

It should operate over:

- proposer candidates
- imagined successor latents
- opponent signals
- planner memory slots

It should not become a disguised tree search.

The repo now has the first bounded opponent-aware planner baseline in exactly that spirit:

- root candidates still come from the symbolic proposer over exact legal moves
- successor states and reply candidates are still generated symbolically
- opponent scoring is plugged in as an explicit bounded two-ply aggregation term
- there is no alpha-beta, no tree expansion, and no runtime search fallback

Current status on the larger verify slice:

- root-only symbolic proposer: `root_top1_accuracy=0.148438`
- symbolic-reply bounded aggregation: `0.15625`
- learned-reply bounded aggregation: `0.15625`

The repo now also has the first trained bounded planner over the current three-tier workflow suite:

- workflow suite: `10k`, `122k`, `400k`
- planner config: [phase8_planner_corpus_suite_set_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v1.json)
- verify comparison: [planner_corpus_suite_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_compare_v1.json)

Aggregate held-out verify result over `1,410` examples:

- root-only bounded baseline: `root_top1_accuracy=0.153901`, `MRR=0.230615`
- symbolic-reply bounded baseline: `0.159574`, `MRR=0.232861`
- learned-reply bounded baseline: `0.142553`, `MRR=0.224232`
- trained planner `set_v1`: `0.788652`, `MRR=0.872636`

The richer-target follow-up now exists as well:

- planner config: [phase8_planner_corpus_suite_set_v2_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v2_v1.json)
- verify comparison: [planner_corpus_suite_compare_v2.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_compare_v2.json)
- expanded rerun config: [phase8_planner_corpus_suite_set_v2_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v2_expanded_v1.json)
- expanded filtered comparison: [planner_corpus_suite_expanded_two_tier_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_expanded_two_tier_compare_v1.json)

Aggregate held-out verify result:

- trained planner `set_v2`: `root_top1_accuracy=0.795035`, `MRR=0.875355`

So Phase 8 is now past pure bounded baselines and has a first refined planner line, not just a one-off trained reference.

The next contract test has now been run as well:

- filtered latent-state workflow: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_workflow_corpus_suite_latent_two_tier_v1/summary.json)
- latent-state config: [phase8_planner_corpus_suite_set_v3_two_tier_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v3_two_tier_v1.json)
- filtered comparison: [planner_corpus_suite_two_tier_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_two_tier_compare_v1.json)

Result on the filtered `10k + 122k` verify slice (`1,024` examples):

- `set_v2`: `root_top1_accuracy=0.80957`, `MRR=0.883382`
- latent `set_v3`: `root_top1_accuracy=0.708008`, `MRR=0.825521`

So the planner-facing latent-state channel is now implemented and reproducible, but the first direct concatenation path is not yet the right refinement.

The expanded-data planner reruns refine that again:

- full mixed-suite `set_v2_expanded` improves validation to `top1=0.813702`, `MRR=0.891489`
- but on the preferred filtered `10k + 122k` slice:
  - old `set_v2`: `top1=0.80957`, `MRR=0.883382`
  - `set_v2_expanded`: `top1=0.798828`, `MRR=0.87972`
  - `set_v2_wide_expanded`: `top1=0.790039`, `MRR=0.874837`
  - `set_v5_expanded`: `top1=0.798828`, `MRR=0.880534`
  - `set_v2_10k_122k_expanded`: `top1=0.819336`, `MRR=0.889811`

So the next Planner lever is not "more width". The new evidence says:

- stronger `10k + 122k` workflow material does help
- mixing the `400k` tier into planner training did not help the preferred filtered slice
- the next open question is now whether better latent-state integration or better teacher targets can move this stronger filtered `set_v2` reference again

## Deferred Architecture Ideas

The following ideas remain relevant and are intentionally being kept in view, but they are deferred until the dense single-path stack is stronger:

- sparse experts
- router DAGs
- option-style deliberation programs
- debate or dual-mind planners
- backward latent planning
- shared-workspace or block-routed expert systems

These ideas are interesting for later phases, but they are not the current bottleneck.

## Immediate Priorities

The next model experiments should now be ordered like this:

1. bring stronger Phase-6 latent-state information into the planner-facing contract
2. test whether better opponent uncertainty signals improve planner calibration more than raw reply accuracy alone
3. use alpha-beta/MCTS-supported offline workflows for richer opponent/planner targets without making them the runtime path
4. explore richer symbolic proposer candidate features only if downstream modules need them
5. only then decide whether the next gain should come from richer planner recurrence or better planner-state structure

The first three offline search-workflow layers are now in place:

- `search_teacher_<split>.jsonl`
- `search_traces_<split>.jsonl`
- `search_disagreements_<split>.jsonl`

The next curriculum layer is now in place as well:

- `search_curriculum_<split>.jsonl`

That means the next workflow pressure is no longer raw label generation alone. It is to use those artifacts for:

- curriculum buckets
- proposer failure clustering
- later opponent/planner supervision

## Current Contract Work

The next concrete contract definitions should be:

1. `CandidateContextV2`
   Exact legal candidate key plus richer, versioned symbolic candidate features.
2. `TransitionContextV1`
   Selected-action features for dynamics and opponent modules, including post-move exact tags.
3. `LatentStateV1`
   Dual-channel planner-facing state made of exact symbolic shadow state plus learned latent state and uncertainty.
4. `OpponentHeadV1`
   Explicit reply distribution, pressure/threat, and uncertainty outputs over exact legal replies.

The next implementation hygiene requirement should be:

5. a single symbolic-feature authority or golden cross-language tests so Python dataset materialization and Rust runtime feature generation cannot silently diverge

Current status:

- `CandidateContextV2` now exists as a versioned Python-side dataset/workflow contract
- `TransitionContextV1` now exists as a versioned Python-side transition artifact contract
- `structured_v6_v1` now consumes `TransitionContextV1` as the first real experimental model arm
- the first offline `search_teacher_<split>.jsonl` workflow now exists over exact legal candidates
- the second offline `search_traces_<split>.jsonl` workflow now exists over the same exact legal candidate set plus PV/reply trace data
- the first `OpponentHeadV1` dataset contract now exists over exact successor states and exact legal replies
- shipped Rust runtime bundles still validate and execute only the current `CandidateContextV1` scorer path
- current default large-corpus dynamics bundles still do not consume `TransitionContextV1`
- that split is intentional until the richer transition contract is rerun and wins on the larger corpus

## Success Criteria For The Next Step

The next proposer experiment should count as successful only if it improves at least one of these without breaking the current symbolic contracts:

- verify `policy_top1_accuracy`
- verify `legal_set_f1`
- training or inference throughput enough to justify the added structure

The next dynamics/opponent preparation step should count as successful only if it does at least one of these without breaking the symbolic runtime path:

- improves verify drift on top of large-corpus `structured_v5`
- clarifies the selected-action contract for later imagined rollouts
- introduces opponent/planner supervision targets without embedding runtime search
- beats the exact-reply symbolic proposer baseline once an explicit opponent head exists

If a new arm is slower and does not materially improve held-out policy or legality, it should remain experimental rather than become the default.
