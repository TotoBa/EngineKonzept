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

That implies the following near-term sequence:

1. stronger proposer structure
2. factorized move decoding
3. action-conditioned latent dynamics
4. explicit opponent modeling
5. bounded recurrent planning
6. only then heavier expert routing or option-style deliberation control

## What The Current Results Say

The current `10,240 / 2,048` Pi-labeled Phase-5 corpus has seven externally checkable proposer arms:

- `current_default`
- `h256`
- `policy_focus`
- `multistream_v2`
- `factorized_v3`
- `factorized_v4`
- `factorized_v5`

See [stockfish_pgn_10k_seven_way_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_10k_seven_way_compare_v1.json).

The important findings are:

- `legality` is easier than `policy`
- widening the flat MLP helps legal-set F1 more than policy accuracy
- a pure policy-loss reweighting run did not improve held-out policy quality
- the first structured multi-stream proposer slightly improved validation legality over the default MLP, but it did not beat `h256`, did not improve policy, and was slower
- the first additive factorized decoder cut parameters drastically, but collapsed both legality and policy quality
- the first conditional factorized decoder became the best legal-set-F1 arm so far, but still did not take the policy lead from `current_default`
- the next policy-stronger conditional decoder regained much of the policy gap while still beating the older MLP baselines on legality
- explicit checkpoint selection now shows a real legality-vs-policy tradeoff inside `factorized_v5`

That means:

- raw width is still usable
- structure helps somewhat
- but the next likely win is not just "more of the same MLP"

## Architecture Decisions

## Phase 5

### Keep

- the exact Rust legality authority
- the current factorized move vocabulary
- the current exported-bundle contract
- the current Phase-5 dataset and evaluation harnesses

### Add next

- conditional factorized proposer decoding over the existing move schema
- more structured proposer backbones where they preserve the same export and evaluation contract

### Do not add yet

- mixture-of-experts routing as the default proposer
- dynamic-depth or adaptive-compute controllers
- planner-time expert graphs
- anything that behaves like hidden search

### Why

The repo already has an object-centric encoder and a factorized move schema. The biggest architectural mismatch is that the current proposer still collapses this into large flat `20480` heads.

The next best repo-local step is therefore:

- not "more routing"
- but "decode the existing move structure in a way that matches the action space"

The factorized decoder line now has three concrete lessons:

- additive factorization threw away too much coupling
- conditional factorization recovered that coupling and improved legality substantially
- policy-specific residual capacity recovered much of the lost policy signal while keeping the factorized structure

The remaining gap is now:

- best policy still belongs to `current_default`
- best legality still belongs to `factorized_v4`
- best legality/policy balance among the factorized arms currently belongs to `factorized_v5`

And there is now a separate method choice as well:

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

1. make the checkpoint-selection strategy a deliberate default decision
2. conditional factorized proposer decoder with stronger policy coupling
3. stronger relational proposer variant if the improved conditional decoder still does not close the policy gap
4. Phase-6 local latent dynamics prototype
5. explicit opponent-head design before recurrent planner work

## Success Criteria For The Next Step

The next proposer experiment should count as successful only if it improves at least one of these without breaking the current contracts:

- verify `policy_top1_accuracy`
- verify `legal_set_f1`
- training or inference throughput enough to justify the added structure

If a new arm is slower and does not materially improve held-out policy or legality, it should remain experimental rather than become the default.
