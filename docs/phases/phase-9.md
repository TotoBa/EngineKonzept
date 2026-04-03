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
- [run_selfplay_arena.py](/home/torsten/EngineKonzept/python/scripts/run_selfplay_arena.py)
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

without changing the session runner itself.

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

That means Phase 9 now has:

- a stable agent-spec contract
- a stable session contract
- a stable replay-buffer contract
- a stable arena-suite contract
- a stable curriculum/launch-plan contract

before replay-buffer-driven retraining is added.
