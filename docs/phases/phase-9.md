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
- [selfplay.py](/home/torsten/EngineKonzept/python/train/eval/selfplay.py)
- [run_selfplay.py](/home/torsten/EngineKonzept/python/scripts/run_selfplay.py)
- [README.md](/home/torsten/EngineKonzept/artifacts/phase9/README.md)
- [selfplay_set_v2_probe_v1.json](/home/torsten/EngineKonzept/artifacts/phase9/selfplay_set_v2_probe_v1.json)

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
- supports different white and black agents for later checkpoint-vs-checkpoint work
- writes reproducible JSON session artifacts

## What it does not do yet

- no replay buffer
- no curriculum scheduler
- no checkpoint arena
- no Rust selfplay runtime yet
- no recurrent planner-memory training loop on top of selfplay data yet

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

So Phase 9 is now real code with a real artifact, but still only a first reproducible probe.
