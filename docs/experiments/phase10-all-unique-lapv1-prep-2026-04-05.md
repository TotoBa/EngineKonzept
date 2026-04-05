# Phase 10 All-Unique LAPv1 Prep

This note records the reproducible setup work for the next LAPv1 Stage-T1 arena run over the largest currently available Phase-5 corpus family.

## Pi snapshot intake

The Raspberry Pi labeler at `10.42.0.20` was still running the older unique-corpus pipeline under:

- `/home/toto/git/tmp/phase5_stockfish_unique_pi_1m_v1`

It was stopped and snapshotted via a SQLite backup before local export.

Snapshot state at intake:

- `games_seen=30372`
- `labeled train=740557`
- `labeled verify=677`

The local imported snapshot now lives under:

- [phase5_stockfish_unique_pi_1m_snapshot_v1](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_unique_pi_1m_snapshot_v1)

with:

- [train_raw.jsonl](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_unique_pi_1m_snapshot_v1/train_raw.jsonl)
- [verify_raw.jsonl](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_unique_pi_1m_snapshot_v1/verify_raw.jsonl)
- [progress.json](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_unique_pi_1m_snapshot_v1/progress.json)

## New merged raw corpus

The following raw corpora were merged with later-source priority and hard verify-over-train precedence:

- [phase5_stockfish_merged_unique_v1](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_merged_unique_v1)
- [phase5_stockfish_unique_pi_400k_v1](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_unique_pi_400k_v1)
- [phase5_stockfish_unique_pi_1m_snapshot_v1](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_unique_pi_1m_snapshot_v1)

The merged raw output now lives under:

- [phase5_stockfish_all_unique_v1](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_all_unique_v1)

Current merged raw counts:

- train: `750429`
- verify: `2741`
- verify/train overlap: `0`

Selection summary:

- [selection_summary.json](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_all_unique_v1/selection_summary.json)

This merged raw tier is the new source of truth for the next LAPv1 all-data bootstrap.

## Arena reference family

The next arena run is not another full arm zoo. It uses:

- the new LAPv1 Stage-T1 checkpoint
- the best six current planner-family arms from the last completed `vice` run
- `vice_v2` as the external benchmark

The selection is dynamic and derived from the final internal arena standings in:

- [/srv/schach/engine_training/phase9/evolution_round03_vice_v1/final/arena/summary.json](/srv/schach/engine_training/phase9/evolution_round03_vice_v1/final/arena/summary.json)

with verify tie-breaks from:

- [/srv/schach/engine_training/phase9/evolution_round03_vice_v1/final/planner_verify_matrix.json](/srv/schach/engine_training/phase9/evolution_round03_vice_v1/final/planner_verify_matrix.json)

At prep time that resolves to:

1. `planner_recurrent_expanded_v1`
2. `planner_set_v2_expanded_v1`
3. `planner_moe_v2_expanded_v1`
4. `planner_set_v6_replay_expanded_v2`
5. `planner_set_v6_expanded_v1`
6. `planner_set_v6_rank_expanded_v1`

## New long-run path

The new long-run entry point is:

- [run_phase10_lapv1_stage1_arena_longrun.sh](/home/torsten/EngineKonzept/python/scripts/run_phase10_lapv1_stage1_arena_longrun.sh)

Backed by:

- [phase10_lapv1_stage1_arena_all_unique_v1.json](/home/torsten/EngineKonzept/python/configs/phase10_lapv1_stage1_arena_all_unique_v1.json)
- [run_phase10_lapv1_stage1_arena_campaign.py](/home/torsten/EngineKonzept/python/scripts/run_phase10_lapv1_stage1_arena_campaign.py)

Its stages are:

1. materialize the exact Phase-5 dataset tier from the new merged raw corpus
2. build a full LAPv1 planner-head workflow over that tier
3. train LAPv1 Stage-T1 for `2` epochs
4. evaluate the held-out LAPv1 verify slice
5. run one 8-agent arena with:
   - LAPv1
   - the dynamic top-6 planner arms from the last `vice` run
   - `vice_v2`
6. write arena summary plus matrix under one output root

Status visibility is now explicit in every long stage:

- Phase-5 materialization logs chunk progress
- workflow building logs teacher-analysis progress
- LAPv1 training logs mid-epoch batch progress
- arena writes `progress.json`

## Intent

This prep keeps the architectural boundary intact:

- exact legality stays symbolic
- LAPv1 is trained once up front
- legacy arms are reused as fixed references
- no classical search is introduced into the learned runtime path
