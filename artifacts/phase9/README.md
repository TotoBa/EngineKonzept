# Phase 9 Artifacts

This directory contains the first small selfplay artifacts.

Current contents:

- first bounded selfplay probe:
  [selfplay_set_v2_probe_v1.json](/home/torsten/EngineKonzept/artifacts/phase9/selfplay_set_v2_probe_v1.json)
- first replay-buffer artifact:
  [replay_buffer_set_v2_probe_v1.jsonl](/home/torsten/EngineKonzept/artifacts/phase9/replay_buffer_set_v2_probe_v1.jsonl)
- first replay-buffer summary:
  [replay_buffer_set_v2_probe_v1.summary.json](/home/torsten/EngineKonzept/artifacts/phase9/replay_buffer_set_v2_probe_v1.summary.json)
- first arena summary:
  [summary.json](/home/torsten/EngineKonzept/artifacts/phase9/arena_active_probe_v1/summary.json)
- first curriculum/launch plan:
  [curriculum_active_experimental_expanded_v1.json](/home/torsten/EngineKonzept/artifacts/phase9/curriculum_active_experimental_expanded_v1.json)
- first expanded active-plus-experimental arena:
  [summary.json](/home/torsten/EngineKonzept/artifacts/phase9/arena_active_experimental_expanded_v1/summary.json)
- resolved expanded arena spec:
  [arena_spec.resolved.json](/home/torsten/EngineKonzept/artifacts/phase9/arena_active_experimental_expanded_v1/arena_spec.resolved.json)

Current probe summary:

- `1` game from `startpos`
- symbolic proposer
- learned Phase-7 opponent head `corpus_suite_set_v2_v1`
- bounded Phase-8 planner `set_v2_10k_122k_expanded`
- `8` legal plies
- termination reason: `max_plies`

This is a reproducible Phase-9 probe, not yet a curriculum dataset.

The new replay-buffer artifact is the first derived training-facing Phase-9 artifact. It flattens the exact selfplay session into one JSONL row per ply while keeping:

- exact FEN before and after the move
- selected move/action
- bounded planner-side diagnostic scalars
- final game outcome from the mover point of view

The first arena artifact proves the same session and agent contracts can already drive checkpoint-vs-checkpoint evaluation:

- ordered color-swapped round-robin
- versioned agent specs
- versioned arena suite spec
- per-matchup session JSON plus aggregate summary

The first curriculum/launch plan ties the next large run together:

- it explicitly requires the `10k`, `122k`, and `400k` tiers
- it lists the large planner reruns to materialize first
- it points at the post-400k active and experimental selfplay agent specs
- it points at the post-400k full active-plus-experimental arena suite

That expanded arena suite is now materialized as well:

- `30` ordered matchups
- `60` games
- `48` `max_plies` terminations
- `8` threefold-repetition draws
- `4` decisive checkmates

Current expanded arena standings by score rate:

- `planner_set_v6_rank_expanded_v1`: `0.55`
- `planner_set_v6_expanded_v1`: `0.50`
- `planner_set_v6_margin_expanded_v1`: `0.50`
- `planner_set_v2_expanded_v1`: `0.50`
- `symbolic_root_v1`: `0.50`
- `planner_recurrent_expanded_v1`: `0.45`

The current practical conclusion is:

- the full expanded planner family can already selfplay against each other under the versioned agent and arena contracts
- `planner_set_v6_rank_expanded_v1` is the first tentative arena leader
- the next useful step is to flatten this arena into replay data rather than immediately overreacting to a still-small selfplay sample
