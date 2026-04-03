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
