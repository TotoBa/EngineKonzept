# MoE Planner

## Motivation

The current bounded planner family is still dominated by one shared scorer path.

The experimental `moe_v1` line takes the Router-DAG direction from [arch.ideas.md](/home/torsten/EngineKonzept/docs/arch.ideas.md) and asks a narrower question:

- can the planner keep the same bounded candidate contract
- while routing different positions into different candidate-scoring experts
- without crossing the project boundary into classical search

This is still a planner head, not a search tree.

## Current Scope

`moe_v1` stays inside the existing bounded Phase-8 / Phase-9 contract:

- exact legal root candidates still come from the symbolic proposer
- successor states and reply summaries stay explicit and bounded
- the MoE arm only changes how the bounded root candidate slice is scored

It does not introduce:

- alpha-beta
- tree expansion
- transposition-table search
- a hidden classical fallback path

## Architecture

The current prepared `moe_v1` path is:

```text
root state features
  -> state backbone
  -> state embedding
      -> PositionRouter -> sparse Top-k expert weights
      -> optional ComplexityHead -> easy / medium / hard budget tier

candidate rows
  -> action embedding + symbolic candidate features + transition features + optional latent features
  -> candidate projection
  -> bounded candidate token set
  -> optional reduced/full refinement passes (complexity-aware)
  -> CandidateExpert[0..N-1]
  -> weighted fusion
  -> root candidate logits

shared heads
  -> root value head
  -> root gap head
  -> optional candidate score head
  -> optional candidate rank head
```

## Expected Behavior

The router should learn coarse specialization, for example:

- quieter positions versus forcing positions
- different game phases
- easy teacher-gapped positions versus ambiguous positions

The complexity head is a separate budget signal:

- easy positions should use fewer experts and fewer refinement passes
- medium positions should take an intermediate path
- hard positions should take the full configured Top-k route

This should lower compute on easy positions without changing the planner-head contract.

## Metrics

The current MoE trainer now logs:

- `load_balance_loss`
- `router_entropy`
- `expert_activation_frequencies`
- `complexity_loss`
- routed easy / medium / hard fractions
- per-tier average expert counts
- `compute_savings_estimate`

Interpretation:

- low `load_balance_loss` means the router is not collapsing into one expert
- very low `router_entropy` can indicate collapse or overconfident routing
- per-expert activation frequencies show whether some experts are unused
- high easy-route fraction plus nontrivial savings indicates budget routing is active
- `complexity_loss` only matters when the complexity head is enabled

## Analysis Tooling

The repo now includes:

- [analyze_moe_expert_specialization.py](/home/torsten/EngineKonzept/python/scripts/analyze_moe_expert_specialization.py)
- [visualize_moe_routing.py](/home/torsten/EngineKonzept/python/scripts/visualize_moe_routing.py)

Those tools report and visualize:

- expert activation by phase
- expert activation by tactical level
- expert activation by teacher difficulty bucket
- router-entropy distribution
- complexity score versus actual teacher-gap difficulty
- agreement between the two strongest routed experts

## Relationship To Existing Planner Arms

`moe_v1` is not a replacement for the current `set_v2` / `set_v6` family yet.

It is an experimental arm that should be judged against the same bounded held-out suite:

- first on `10k + 122k`
- then, only if promising, inside the same selfplay and replay loops as the active family

The current preferred non-MoE references remain the bounded `set_v2` / `set_v6` planner line until a trained MoE run beats them on both holdout quality and practical selfplay behavior.

## First Eval Result

The first real `10k + 122k` MoE run is now complete:

- config:
  [phase9_planner_moe_v1_10k_122k_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_planner_moe_v1_10k_122k_v1.json)
- compact report:
  [first_eval_summary.json](/home/torsten/EngineKonzept/artifacts/moe_v1/first_eval_summary.json)
- training summary:
  [summary.json](/srv/schach/engine_training/phase9/planner_moe_v1_10k_122k_v1/summary.json)

Held-out result:

- `moe_v1_10k_122k_v1`: `top1=0.794141`, `MRR=0.878695`
- reference `set_v2_10k_122k_expanded`: `top1=0.819336`, `MRR=0.889811`
- reference `set_v6_10k_122k_expanded`: `top1=0.817383`, `MRR=0.890625`

So the first trained MoE arm is currently below the established non-MoE planner line.

## What The Analysis Says

The offline routing analysis confirms that the first run did not fail because of a subtle tradeoff. The router mostly collapsed.

Reports:

- [moe_v1_10k_validation_analysis.json](/home/torsten/EngineKonzept/artifacts/moe_v1/moe_v1_10k_validation_analysis.json)
- [moe_v1_122k_validation_analysis.json](/home/torsten/EngineKonzept/artifacts/moe_v1/moe_v1_122k_validation_analysis.json)
- plots:
  [plots_10k](/home/torsten/EngineKonzept/artifacts/moe_v1/plots_10k)
  [plots_122k](/home/torsten/EngineKonzept/artifacts/moe_v1/plots_122k)

The important findings are:

- the router almost always selects the same Top-2 pair: experts `0` and `3`
- experts `1` and `2` are effectively unused on both validation tiers
- router entropy stays very low at about `0.087`
- load-balance loss stays high at about `0.498`
- the routing pattern barely changes across opening / middlegame / endgame, tactical level, or teacher-difficulty buckets

That means the current `moe_v1` issue is not "experts specialize but still underperform". It is "routing never really diversifies".

## Current Decision

The repo should keep `moe_v1` as an experimental branch, but not promote it.

The next MoE follow-up should be narrow and diagnostic:

- keep the same bounded planner contract
- enable the complexity head
- increase load-balance pressure
- rerun on the same `10k + 122k` suite before involving the wider selfplay loop
