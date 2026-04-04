# Phase 9

## Goal

Build the first exact selfplay loop around the current proposer, opponent, and planner contracts.

The phase boundary stays strict:

- exact legality remains symbolic
- there is still no classical search runtime
- selfplay is driven by the same bounded learned stack already used in offline evaluation

## Current repository state

The repository now has the first small selfplay loop in Python:

- [planner_runtime.py](/home/torsten/EngineKonzept/python/train/eval/planner_runtime.py)
- [agent_spec.py](/home/torsten/EngineKonzept/python/train/eval/agent_spec.py)
- [arena.py](/home/torsten/EngineKonzept/python/train/eval/arena.py)
- [curriculum.py](/home/torsten/EngineKonzept/python/train/eval/curriculum.py)
- [selfplay.py](/home/torsten/EngineKonzept/python/train/eval/selfplay.py)
- [run_selfplay.py](/home/torsten/EngineKonzept/python/scripts/run_selfplay.py)
- [build_replay_buffer.py](/home/torsten/EngineKonzept/python/scripts/build_replay_buffer.py)
- [build_curriculum_stage_replay_buffer.py](/home/torsten/EngineKonzept/python/scripts/build_curriculum_stage_replay_buffer.py)
- [run_selfplay_arena.py](/home/torsten/EngineKonzept/python/scripts/run_selfplay_arena.py)
- [run_selfplay_curriculum_stage.py](/home/torsten/EngineKonzept/python/scripts/run_selfplay_curriculum_stage.py)
- [build_selfplay_curriculum_plan.py](/home/torsten/EngineKonzept/python/scripts/build_selfplay_curriculum_plan.py)
- [README.md](/home/torsten/EngineKonzept/artifacts/phase9/README.md)
- [selfplay_set_v2_probe_v1.json](/home/torsten/EngineKonzept/artifacts/phase9/selfplay_set_v2_probe_v1.json)
- [replay_buffer_set_v2_probe_v1.jsonl](/home/torsten/EngineKonzept/artifacts/phase9/replay_buffer_set_v2_probe_v1.jsonl)
- [summary.json](/home/torsten/EngineKonzept/artifacts/phase9/arena_active_probe_v1/summary.json)

This first implementation is intentionally small and contract-first:

1. exact current positions are labeled through the Rust dataset oracle
2. exact legal root candidates are scored by the symbolic proposer
3. optional bounded planner refinement reuses the current planner-head contract
4. successor positions are exact-applied symbolically
5. termination remains exact via checkmate, stalemate, threefold repetition, and fifty-move checks

## What it does

- supports proposer-only play
- supports bounded planner-guided play over the same exact root candidate set
- supports optional learned opponent and dynamics checkpoints when the chosen planner contract needs them
- supports versioned JSON agent specs so future arm changes do not require another bespoke CLI layer
- supports different white and black agents for later checkpoint-vs-checkpoint work
- supports offline-only external UCI-engine opponents through the same exact move/output contract, so arena benchmarking can compare the learned stack against third-party engines without changing the runtime architecture
- supports master-process arena parallelism via `parallel_workers`, so one game thread can run per CPU while the arena orchestration stays in one Python process
- writes reproducible JSON session artifacts
- can flatten finished sessions into replay-buffer rows for later training and curriculum use

## What it does not do yet

- no Rust selfplay runtime yet
- no recurrent planner-memory training loop on top of selfplay data yet

The replay-buffer, arena, and curriculum-plan layers now exist, but there is still no Rust selfplay runtime and no replay-buffer-driven planner retraining loop yet.

## Why this shape

The first selfplay loop is deliberately narrow so later architecture changes do not require another rewrite.

The stable part is the agent contract:

- exact position in
- exact legal move out
- optional bounded planner signals inside

That means later changes can swap:

- proposer family
- opponent family
- planner backbone
- latent contract
- or an offline-only external UCI benchmark opponent

without changing the session runner itself.

The versioned launch data around that contract is now also reusable:

- curated dataset-derived initial-position suites
- curated opening-derived initial-position suites
- merged suites that combine both without changing the arena or campaign runner

## First probe

The first materialized probe keeps the loop deliberately small:

- proposer: [stockfish_pgn_symbolic_v1_v1/checkpoint.pt](/home/torsten/EngineKonzept/models/proposer/stockfish_pgn_symbolic_v1_v1/checkpoint.pt)
- opponent: [corpus_suite_set_v2_v1/checkpoint.pt](/home/torsten/EngineKonzept/models/opponent/corpus_suite_set_v2_v1/checkpoint.pt)
- planner: [corpus_suite_set_v2_10k_122k_expanded_v1/checkpoint.pt](/home/torsten/EngineKonzept/models/planner/corpus_suite_set_v2_10k_122k_expanded_v1/checkpoint.pt)
- starting position: `startpos`
- games: `1`
- max plies: `8`

Observed result:

- `8` legal plies were produced
- termination reason: `max_plies`
- the loop stayed entirely within the exact symbolic legality contract

The first replay-buffer follow-up is now also materialized:

- agent specs:
  - [phase9_agent_symbolic_root_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_agent_symbolic_root_v1.json)
  - [phase9_agent_planner_set_v2_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_agent_planner_set_v2_v1.json)
- replay artifact:
  - [replay_buffer_set_v2_probe_v1.jsonl](/home/torsten/EngineKonzept/artifacts/phase9/replay_buffer_set_v2_probe_v1.jsonl)
- replay summary:
  - [replay_buffer_set_v2_probe_v1.summary.json](/home/torsten/EngineKonzept/artifacts/phase9/replay_buffer_set_v2_probe_v1.summary.json)

The first arena follow-up is now also materialized:

- additional agent specs:
  - [phase9_agent_planner_set_v6_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_agent_planner_set_v6_v1.json)
  - [phase9_agent_planner_set_v6_margin_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_agent_planner_set_v6_margin_v1.json)
  - [phase9_agent_planner_set_v6_rank_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_agent_planner_set_v6_rank_v1.json)
  - [phase9_agent_planner_recurrent_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_agent_planner_recurrent_v1.json)
- arena specs:
  - [phase9_arena_active_probe_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_arena_active_probe_v1.json)
  - [phase9_arena_active_experimental_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_arena_active_experimental_v1.json)
- arena summary:
  - [summary.json](/home/torsten/EngineKonzept/artifacts/phase9/arena_active_probe_v1/summary.json)

Observed arena probe result:

- `2` games
- ordered color-swapped round-robin between `planner_set_v2_v1` and `symbolic_root_v1`
- both games stayed legal and terminated by `max_plies`

The first curriculum/launch-plan follow-up is now also materialized:

- expanded planner configs:
  - [phase8_planner_corpus_suite_set_v6_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v6_expanded_v1.json)
  - [phase8_planner_corpus_suite_set_v6_margin_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v6_margin_expanded_v1.json)
  - [phase8_planner_corpus_suite_set_v6_rank_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v6_rank_expanded_v1.json)
  - [phase8_planner_corpus_suite_recurrent_v1_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_recurrent_v1_expanded_v1.json)
- expanded selfplay agent specs:
  - [phase9_agent_planner_set_v2_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_agent_planner_set_v2_expanded_v1.json)
  - [phase9_agent_planner_set_v6_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_agent_planner_set_v6_expanded_v1.json)
  - [phase9_agent_planner_set_v6_margin_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_agent_planner_set_v6_margin_expanded_v1.json)
  - [phase9_agent_planner_set_v6_rank_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_agent_planner_set_v6_rank_expanded_v1.json)
  - [phase9_agent_planner_recurrent_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_agent_planner_recurrent_expanded_v1.json)
- expanded arena suite:
  - [phase9_arena_active_experimental_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_arena_active_experimental_expanded_v1.json)
- launch plan:
  - [curriculum_active_experimental_expanded_v1.json](/home/torsten/EngineKonzept/artifacts/phase9/curriculum_active_experimental_expanded_v1.json)

That launch plan already encodes the intended next large step:

- required tiers: `pgn_10k`, `merged_unique_122k`, `unique_pi_400k`
- planner reruns to materialize before selfplay:
  - active `set_v2_expanded`
  - experimental `set_v6_expanded`
  - experimental `set_v6_margin_expanded`
  - experimental `set_v6_rank_expanded`
  - experimental `recurrent_v1_expanded`
- then the full active-plus-experimental selfplay arena over the expanded bundle set

That prerequisite planner step is now complete as well:

- current expanded suite summary: [planner_active_experimental_expanded_v1_summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_active_experimental_expanded_v1_summary.json)
- current expanded suite comparison: [planner_active_experimental_expanded_v1_compare.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_active_experimental_expanded_v1_compare.json)

Current full expanded leaderboard before arena promotion:

- `set_v6_expanded`: best `top1` on the full `10k + 122k + 400k` verify suite
- `set_v6_rank_expanded`: best `MRR` on the same suite
- `set_v6_margin_expanded`: effectively tied with the `MRR` leader and best `top3`
- `set_v2_expanded`: still useful as the older active launch reference, but no longer the strongest full expanded rerun

So the next Phase-9 step is no longer "prepare" the expanded planner family, but run the arena over the now-materialized expanded planner family and use that result to decide active-vs-experimental promotion.

That means Phase 9 now has:

- a stable agent-spec contract
- a stable session contract
- a stable replay-buffer contract
- a stable arena-suite contract
- a stable curriculum/launch-plan contract

before replay-buffer-driven retraining is added.

The first expanded active-plus-experimental arena stage is now also materialized directly from the curriculum plan:

- expanded arena summary:
  [summary.json](/home/torsten/EngineKonzept/artifacts/phase9/arena_active_experimental_expanded_v1/summary.json)
- resolved arena spec:
  [arena_spec.resolved.json](/home/torsten/EngineKonzept/artifacts/phase9/arena_active_experimental_expanded_v1/arena_spec.resolved.json)

That stage runs the currently materialized expanded planner family:

- `symbolic_root_v1`
- `planner_set_v2_expanded_v1`
- `planner_set_v6_expanded_v1`
- `planner_set_v6_margin_expanded_v1`
- `planner_set_v6_rank_expanded_v1`
- `planner_recurrent_expanded_v1`

Observed result:

- `30` ordered color-swapped matchups
- `60` games total
- termination counts:
  - `max_plies=48`
  - `threefold_repetition=8`
  - `checkmate=4`
- current arena score leader:
  - `planner_set_v6_rank_expanded_v1`
  - `score=11.0 / 20`
  - `score_rate=0.55`

Important interpretation:

- this is the first real selfplay-facing comparison over the full expanded planner family
- `set_v6_rank_expanded_v1` is the current tentative arena leader
- but the sample is still small and many games terminate by `max_plies`
- so this is enough to drive the next replay-buffer step, not yet enough to declare a final long-run runtime promotion

That replay-buffer follow-up is now also materialized directly from the curriculum stage:

- replay buffer:
  [replay_buffer.jsonl](/home/torsten/EngineKonzept/artifacts/phase9/replay_buffer_active_experimental_expanded_v1/replay_buffer.jsonl)
- replay summary:
  [summary.json](/home/torsten/EngineKonzept/artifacts/phase9/replay_buffer_active_experimental_expanded_v1/summary.json)

Observed result:

- `30` arena sessions flattened
- `60` exact selfplay games represented
- `3640` replay rows
- `mean_considered_candidate_count=3.505`
- stable session-prefixed `game_id` / `sample_id` values, so multi-session replay rows remain unique

This is the first large selfplay-derived training artifact for the active-plus-experimental expanded planner family.

That replay artifact now also feeds the first replay-driven planner retraining path:

- replay supervision:
  - [planner_replay_train.jsonl](/home/torsten/EngineKonzept/artifacts/phase9/planner_replay_active_experimental_expanded_v1/planner_replay_train.jsonl)
  - [summary.json](/home/torsten/EngineKonzept/artifacts/phase9/planner_replay_active_experimental_expanded_v1/summary.json)
- replay planner-head artifact:
  - [planner_head_train.jsonl](/home/torsten/EngineKonzept/artifacts/phase9/planner_replay_head_active_experimental_expanded_v1/planner_head_train.jsonl)
  - [summary.json](/home/torsten/EngineKonzept/artifacts/phase9/planner_replay_head_active_experimental_expanded_v1/summary.json)
- replay-retrain config:
  - [phase8_planner_corpus_suite_set_v6_rank_replay_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v6_rank_replay_expanded_v1.json)

Observed replay-data result:

- `568` resolved replay supervision rows after dropping unfinished `max_plies` positions
- same `568` rows materialized into planner-head replay fine-tuning examples
- replay source remains exact and planner-contract compatible: exact FEN -> symbolic candidates -> bounded planner head

Observed replay-retrain result on the full expanded verify suite:

- `set_v6_rank_expanded`: `top1=0.808511`, `MRR=0.887234`
- `set_v6_rank_replay_expanded`: `top1=0.807801`, `MRR=0.886525`
- but `top3` improves: `0.965957 -> 0.968794`

So Phase 9 now has not only replay-buffer creation, but also the first replay-buffer-driven planner retraining loop. The first run is useful and measurable, but not yet strong enough to replace the current expanded planner leaders.

The next expanded arena rerun is now also materialized with a versioned curated initial-position suite instead of `startpos` only:

- initial-position suite:
  [initial_fens_active_experimental_expanded_v1.json](/home/torsten/EngineKonzept/artifacts/phase9/initial_fens_active_experimental_expanded_v1.json)
- suite summary:
  [initial_fens_active_experimental_expanded_v1.summary.json](/home/torsten/EngineKonzept/artifacts/phase9/initial_fens_active_experimental_expanded_v1.summary.json)
- larger curriculum plan:
  [curriculum_active_experimental_expanded_v2.json](/home/torsten/EngineKonzept/artifacts/phase9/curriculum_active_experimental_expanded_v2.json)
- larger expanded arena summary:
  [summary.json](/home/torsten/EngineKonzept/artifacts/phase9/arena_active_experimental_expanded_v2/summary.json)
- comparison:
  [arena_active_experimental_expanded_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase9/arena_active_experimental_expanded_compare_v1.json)

Observed difference versus the smaller `startpos`-only expanded stage:

- games: `60 -> 180`
- resolved games: `12 -> 117`
- resolved ratio: `0.20 -> 0.65`
- checkmates: `4 -> 28`
- average move count: `60.667 -> 72.089`

Why this matters:

- the arena is now larger without becoming architecture-specific
- it remains fully exact: same legal-move contract, same agent specs, same bounded planner stack
- the replay yield is materially better because the new initial positions start from already tactical midgames rather than only `startpos`

The important boundary is unchanged:

- no classical-search runtime was added
- the only change is the versioned selfplay initial-position contract and a larger curriculum stage that consumes it

That stronger replay source has now also produced the first explicit expanded-arm promotion:

- decision artifact:
  [active_promotion_decision_v1.json](/home/torsten/EngineKonzept/artifacts/phase9/active_promotion_decision_v1.json)
- promoted active spec:
  [phase9_agent_planner_active_expanded_v2.json](/home/torsten/EngineKonzept/python/configs/phase9_agent_planner_active_expanded_v2.json)
- retained replay challenger:
  [phase9_agent_planner_set_v6_replay_expanded_v2.json](/home/torsten/EngineKonzept/python/configs/phase9_agent_planner_set_v6_replay_expanded_v2.json)
- next replay-aware arena suite:
  [phase9_arena_active_experimental_replay_expanded_v2.json](/home/torsten/EngineKonzept/python/configs/phase9_arena_active_experimental_replay_expanded_v2.json)

Promotion rule:

- primary metric: held-out `root_top1_accuracy`
- tie-breakers:
  - `teacher_root_mean_reciprocal_rank`
  - `teacher_root_mean_probability`

Observed promotion deltas versus the older expanded active reference `set_v2_expanded`:

- `root_top1_accuracy`: `0.797163 -> 0.813475`
- `teacher_root_mean_reciprocal_rank`: `0.879433 -> 0.889894`
- `teacher_root_mean_probability`: `0.693195 -> 0.725571`

That means the first replay-aware active promotion is now explicit, versioned, and still architecture-flexible:

- the active slot is now a versioned agent spec, not a hard-coded planner name
- the replay mirror remains available as a separate experimental challenger
- the next arena suite can swap in future planner arms without another runner rewrite

The promoted active expanded agent is also smoke-verified in direct selfplay:

- [selfplay_active_expanded_v2_probe_v1.json](/home/torsten/EngineKonzept/artifacts/phase9/selfplay_active_expanded_v2_probe_v1.json)
- `1` game
- `12` legal plies
- termination reason: `max_plies`

The arena side now also has a reusable full-matrix export path:

- builder:
  [build_selfplay_arena_matrix.py](/home/torsten/EngineKonzept/python/scripts/build_selfplay_arena_matrix.py)
- helper:
  [matrix.py](/home/torsten/EngineKonzept/python/train/eval/matrix.py)

This keeps pairwise selfplay analysis separate from the campaign runner itself, so future agent or planner changes can still reuse the same row-vs-column arena summary contract.

The next contract layer now also exists as a versioned long-run replay campaign:

- helper:
  [campaign.py](/home/torsten/EngineKonzept/python/train/eval/campaign.py)
- runner:
  [run_phase9_replay_campaign.py](/home/torsten/EngineKonzept/python/scripts/run_phase9_replay_campaign.py)

That runner is designed around:

- one versioned curriculum stage
- one derived replay source
- one replay-head artifact
- several replay-mirror planner reruns
- one held-out planner verify matrix

So later architecture changes should only need:

- new agent specs
- new base planner configs
- updated campaign manifests

rather than another orchestration rewrite.

The repo now also has the first directly startable long-run entry point for that campaign:

- campaign config:
  [phase9_replay_campaign_active_expanded_v2.json](/home/torsten/EngineKonzept/python/configs/phase9_replay_campaign_active_expanded_v2.json)
- replay-aware curriculum plan:
  [curriculum_active_experimental_replay_expanded_v2.json](/home/torsten/EngineKonzept/artifacts/phase9/curriculum_active_experimental_replay_expanded_v2.json)
- launcher:
  [run_phase9_replay_campaign_longrun.sh](/home/torsten/EngineKonzept/python/scripts/run_phase9_replay_campaign_longrun.sh)

That is the preferred one-command entry point for a future large run:

```bash
python/scripts/run_phase9_replay_campaign_longrun.sh
```

Useful override examples:

```bash
python/scripts/run_phase9_replay_campaign_longrun.sh --output-root /srv/schach/engine_training/phase9/replay_campaign_debug --games-per-matchup 1 --max-plies 16 --include-unfinished-replay --run planner_set_v6_margin_replay_campaign_v2
```

Phase 9 now also supports optional engine adjudication exactly at the `max_plies` boundary.
That path is intended to reduce unresolved `max_plies` endings without turning runtime into a classical search engine:

- the selfplay/arena loop still selects moves only through the symbolic proposer and learned planner stack
- `/usr/games/stockfish18` is only consulted after a game hits the configured `max_plies` limit
- if the white-POV evaluation stays inside `[-0.3, +0.3]` pawns, play is extended by a bounded number of extra plies
- if the position is outside that neutral window, the game is adjudicated instead of ending as unresolved `*`
- the preferred replay-aware arena suite now also samples its opening positions pseudo-randomly with a fixed `opening_selection_seed`, so the same opening is replayed under swapped colors while repeated runs remain deterministic
- the preferred replay-aware arena suite runs under one master arena process with `parallel_workers=6`, rather than spawning separate arena controller sessions

The contract is versioned on the arena spec via `max_plies_adjudication`, so later architecture changes can reuse or replace the adjudicator without rewriting the arena/campaign runners.

Arena specs now also carry `parallel_workers`.
That is explicitly an offline orchestration knob, not a model or engine-contract change.
The intended shape is one arena master process with several concurrent games, not several independent arena Python processes.

The first direct replay-campaign challenger comparison is now also materialized:

- replay-campaign challenger agent specs:
  - [phase9_agent_planner_set_v6_margin_replay_campaign_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_agent_planner_set_v6_margin_replay_campaign_v1.json)
  - [phase9_agent_planner_set_v6_replay_campaign_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_agent_planner_set_v6_replay_campaign_v1.json)
- adjudicated arena specs:
  - [phase9_arena_active_replay_campaign_adjudicated_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_arena_active_replay_campaign_adjudicated_v1.json)
  - [phase9_arena_active_replay_campaign_adjudicated_v2.json](/home/torsten/EngineKonzept/python/configs/phase9_arena_active_replay_campaign_adjudicated_v2.json)
- opening-derived suite:
  [initial_fens_thor_openings_v1.json](/home/torsten/EngineKonzept/artifacts/phase9/initial_fens_thor_openings_v1.json)
- mixed adjudicated suite:
  [initial_fens_active_replay_campaign_adjudicated_v2.json](/home/torsten/EngineKonzept/artifacts/phase9/initial_fens_active_replay_campaign_adjudicated_v2.json)
- direct arena summary:
  [arena_active_replay_campaign_adjudicated_v2.summary.json](/home/torsten/EngineKonzept/artifacts/phase9/arena_active_replay_campaign_adjudicated_v2.summary.json)
- direct arena matrix:
  [arena_active_replay_campaign_adjudicated_v2.matrix.json](/home/torsten/EngineKonzept/artifacts/phase9/arena_active_replay_campaign_adjudicated_v2.matrix.json)

Important interpretation:

- the original `v1` direct comparison over `startpos` only was too color-biased to drive promotion
- the broader `v2` rerun uses `14` curated starts: `4` harder dataset positions plus `10` opening-derived positions from `../Thor_CE/openings`
- `parallel_workers=6` now means one arena master process with six concurrent games
- under that broader direct comparison the current active arm remains the clear leader:
  - `planner_active_expanded_v2`: `score_rate=0.580357`
  - `planner_set_v6_margin_replay_campaign_v1`: `0.482143`
  - `planner_set_v6_replay_campaign_v1`: `0.4375`
- held-out verify still likes `planner_set_v6_margin_replay_campaign_v1` slightly better, but the direct broader arena does not confirm a promotion

So the current active expanded arm stays unchanged for now.

That decision is versioned here:

- [active_promotion_decision_v2.json](/home/torsten/EngineKonzept/artifacts/phase9/active_promotion_decision_v2.json)
