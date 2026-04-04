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
- curated expanded initial-position suite:
  [initial_fens_active_experimental_expanded_v1.json](/home/torsten/EngineKonzept/artifacts/phase9/initial_fens_active_experimental_expanded_v1.json)
- larger expanded curriculum plan:
  [curriculum_active_experimental_expanded_v2.json](/home/torsten/EngineKonzept/artifacts/phase9/curriculum_active_experimental_expanded_v2.json)
- larger expanded arena:
  [summary.json](/home/torsten/EngineKonzept/artifacts/phase9/arena_active_experimental_expanded_v2/summary.json)
- expanded arena comparison:
  [arena_active_experimental_expanded_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase9/arena_active_experimental_expanded_compare_v1.json)

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

That expanded replay-buffer follow-up is now materialized as well:

- replay artifact:
  [replay_buffer.jsonl](/home/torsten/EngineKonzept/artifacts/phase9/replay_buffer_active_experimental_expanded_v1/replay_buffer.jsonl)
- replay summary:
  [summary.json](/home/torsten/EngineKonzept/artifacts/phase9/replay_buffer_active_experimental_expanded_v1/summary.json)

Current expanded replay summary:

- `30` sessions
- `60` games
- `3640` replay rows
- `mean_considered_candidate_count=3.505`

This is the first large selfplay-derived training artifact for the active-plus-experimental expanded planner family.

The newer `v2` expanded arena is the preferred replay source now:

- it keeps the same active/experimental agent family
- it swaps `startpos`-only launching for a versioned curated FEN suite drawn from the `10k`, `122k`, and `400k` verify corpora
- it raises the expanded stage from `60` to `180` games
- it improves resolved-game yield from `12` to `117`
- it increases checkmates from `4` to `28`

Current practical conclusion:

- the arena contract is now flexible enough for future architecture changes because start-position selection is versioned separately from the agent/runtime contracts
- the `v2` expanded arena should feed the next replay-buffer-driven planner reruns

The next training-facing follow-up is now materialized too:

- replay supervision:
  - [planner_replay_train.jsonl](/home/torsten/EngineKonzept/artifacts/phase9/planner_replay_active_experimental_expanded_v1/planner_replay_train.jsonl)
  - [summary.json](/home/torsten/EngineKonzept/artifacts/phase9/planner_replay_active_experimental_expanded_v1/summary.json)
- replay planner-head artifact:
  - [planner_head_train.jsonl](/home/torsten/EngineKonzept/artifacts/phase9/planner_replay_head_active_experimental_expanded_v1/planner_head_train.jsonl)
  - [summary.json](/home/torsten/EngineKonzept/artifacts/phase9/planner_replay_head_active_experimental_expanded_v1/summary.json)

Current replay-training summary:

- `568` resolved replay rows after excluding unfinished `max_plies` positions
- `568` planner-head fine-tuning rows
- source agents remain mixed across the active and experimental expanded planner family

So Phase 9 now has a full path from:

- expanded arena
- to replay buffer
- to replay-derived planner supervision
- to replay-derived planner-head fine-tuning data

The stronger `v2` expanded arena now extends that path:

- replay buffer:
  [replay_buffer.jsonl](/home/torsten/EngineKonzept/artifacts/phase9/replay_buffer_active_experimental_expanded_v2/replay_buffer.jsonl)
- replay buffer summary:
  [summary.json](/home/torsten/EngineKonzept/artifacts/phase9/replay_buffer_active_experimental_expanded_v2/summary.json)
- replay supervision:
  [planner_replay_train.jsonl](/home/torsten/EngineKonzept/artifacts/phase9/planner_replay_active_experimental_expanded_v2/planner_replay_train.jsonl)
- replay supervision summary:
  [summary.json](/home/torsten/EngineKonzept/artifacts/phase9/planner_replay_active_experimental_expanded_v2/summary.json)
- replay planner-head artifact:
  [planner_head_train.jsonl](/home/torsten/EngineKonzept/artifacts/phase9/planner_replay_head_active_experimental_expanded_v2/planner_head_train.jsonl)
- replay planner-head summary:
  [summary.json](/home/torsten/EngineKonzept/artifacts/phase9/planner_replay_head_active_experimental_expanded_v2/summary.json)

Current `v2` replay-training summary:

- `12976` replay-buffer rows
- `6928` resolved replay supervision rows
- `6928` replay planner-head rows

This is now the preferred Phase-9 replay source for planner fine-tuning.

That stronger replay source has now also produced the first explicit expanded active promotion:

- promotion decision:
  [active_promotion_decision_v1.json](/home/torsten/EngineKonzept/artifacts/phase9/active_promotion_decision_v1.json)
- promoted active agent:
  [phase9_agent_planner_active_expanded_v2.json](/home/torsten/EngineKonzept/python/configs/phase9_agent_planner_active_expanded_v2.json)
- replay challenger kept live:
  [phase9_agent_planner_set_v6_replay_expanded_v2.json](/home/torsten/EngineKonzept/python/configs/phase9_agent_planner_set_v6_replay_expanded_v2.json)
- next replay-aware arena suite:
  [phase9_arena_active_experimental_replay_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_arena_active_experimental_replay_expanded_v1.json)

Promotion summary:

- previous active reference `set_v2_expanded`: `top1=0.797163`, `MRR=0.879433`, `mean_probability=0.693195`
- promoted `set_v6_margin_replay_expanded_v2`: `top1=0.813475`, `MRR=0.889894`, `mean_probability=0.725571`

So the active expanded selfplay slot is now no longer the old `set_v2_expanded` line.
It is the replay-promoted `set_v6_margin_replay_expanded_v2` line, but still through a versioned agent spec so future planner changes can replace it without changing the arena runner.

The promoted active expanded agent is also smoke-verified here:

- [selfplay_active_expanded_v2_probe_v1.json](/home/torsten/EngineKonzept/artifacts/phase9/selfplay_active_expanded_v2_probe_v1.json)
- `1` game
- `12` legal plies
- termination reason: `max_plies`

Pairwise selfplay summaries can now also be converted into a full matrix artifact with:

- [build_selfplay_arena_matrix.py](/home/torsten/EngineKonzept/python/scripts/build_selfplay_arena_matrix.py)

That matrix is row-agent-centric and aggregates both color directions, so it is the intended basis for later long-run selfplay campaign comparisons.

The repo now also has a versioned long-run campaign runner:

- [campaign.py](/home/torsten/EngineKonzept/python/train/eval/campaign.py)
- [run_phase9_replay_campaign.py](/home/torsten/EngineKonzept/python/scripts/run_phase9_replay_campaign.py)

It is intended to drive, in one reproducible pass:

- replay-aware arena materialization
- replay-buffer flattening
- planner replay supervision
- replay-head materialization
- replay-mirror planner reruns
- held-out planner verify comparison
- arena full-matrix export

The current startable long-run entry point is:

- config:
  [phase9_replay_campaign_active_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_replay_campaign_active_expanded_v1.json)
- replay-aware curriculum plan:
  [curriculum_active_experimental_replay_expanded_v1.json](/home/torsten/EngineKonzept/artifacts/phase9/curriculum_active_experimental_replay_expanded_v1.json)
- launcher:
  [run_phase9_replay_campaign_longrun.sh](/home/torsten/EngineKonzept/python/scripts/run_phase9_replay_campaign_longrun.sh)

That path is intended to produce:

- a replay-aware expanded arena summary
- a replay-aware expanded arena full matrix
- replay-buffer and replay-head artifacts
- replay-mirror planner reruns
- a planner verify matrix against the current active reference

For tiny smoke runs, the intended override is:

```bash
python/scripts/run_phase9_replay_campaign_longrun.sh --output-root .tmp/phase9_replay_campaign_smoke_v1 --games-per-matchup 2 --max-plies 24 --max-replay-examples 128 --max-replay-head-examples 32 --include-unfinished-replay --run planner_set_v6_margin_replay_campaign_v1
```

The underlying arena contract now also supports optional `max_plies` adjudication:

- engine path, bounded engine limit, neutral threshold, and bounded extra-plies budget live on the arena spec
- the intended default judge is `/usr/games/stockfish18`
- inside `[-0.3, +0.3]` pawns the game is extended
- outside that neutral band the position is adjudicated instead of ending as unresolved `max_plies`

That keeps long-run replay campaigns reproducible while reducing low-signal unfinished games.

The supporting initial-position tooling is now broader as well:

- [build_selfplay_opening_fen_suite.py](/home/torsten/EngineKonzept/python/scripts/build_selfplay_opening_fen_suite.py) can derive opening starts from TSV opening books such as `../Thor_CE/openings`
- [merge_selfplay_initial_fen_suites.py](/home/torsten/EngineKonzept/python/scripts/merge_selfplay_initial_fen_suites.py) can combine curated dataset suites and opening suites into one deduped arena-start suite
- arena specs can use those merged starts together with `parallel_workers` for larger adjudicated comparison runs under one arena master process
