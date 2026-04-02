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

The larger merged-unique reruns refine that again:

- the large `structured_v3` rerun beats the old large `structured_v2_latent` baseline on both verify feature error and verify drift
- the large symbolic-action `structured_v5` rerun becomes the best measured Phase-6 arm overall on the merged unique corpus

So the current preferred Phase-6 path is now the symbolic-action dynamics line on the larger corpus, not the earlier smaller-corpus `structured_v2_latent` reference.

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

The next model experiments should be ordered like this:

1. improve the Phase-6 dynamics model over the symbolic proposer candidate contract
2. define the first explicit opponent-head contract before planner work
3. define alpha-beta/MCTS-supported offline workflows for targets, benchmarking, and curriculum without making them the runtime path
4. explore richer symbolic proposer candidate features only if downstream modules need them
5. only then resume broader proposer exploration if Phase-6/7 pressure points point back at representation quality

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

## Success Criteria For The Next Step

The next proposer experiment should count as successful only if it improves at least one of these without breaking the current symbolic contracts:

- verify `policy_top1_accuracy`
- verify `legal_set_f1`
- training or inference throughput enough to justify the added structure

The next dynamics/opponent preparation step should count as successful only if it does at least one of these without breaking the symbolic runtime path:

- improves verify drift on top of large-corpus `structured_v5`
- clarifies the selected-action contract for later imagined rollouts
- introduces opponent/planner supervision targets without embedding runtime search

If a new arm is slower and does not materially improve held-out policy or legality, it should remain experimental rather than become the default.
