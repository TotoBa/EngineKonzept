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
