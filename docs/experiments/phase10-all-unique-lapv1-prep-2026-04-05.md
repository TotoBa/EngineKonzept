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

The arena no longer relies on the tracked repo agent specs for those reference
arms. It now prefers the materialized active specs from:

- [/srv/schach/engine_training/phase9/evolution_round03_vice_v1/iterations/round_10/active_agent_specs](/srv/schach/engine_training/phase9/evolution_round03_vice_v1/iterations/round_10/active_agent_specs)

so the resumed Phase-10 arena uses the exact checkpoint paths that actually
survived the last completed evolution run.

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
5. run one 9-agent arena with:
   - LAPv1 with `deliberation_max_inner_steps=0`
   - LAPv1 with `deliberation_max_inner_steps=1`
   - the dynamic top-6 planner arms from the last `vice` run
   - `vice_v2`
6. write arena summary plus matrix under one output root

The fast arena restart also no longer reuses the small `14`-entry mixed opening
suite. It now points at a larger Thor-derived opening pool:

- [initial_fens_thor_openings_150_v1.json](/home/torsten/EngineKonzept/artifacts/phase10/initial_fens_thor_openings_150_v1.json)

with `150` seeded openings. Round-robin opening selection is assigned globally
per unordered pair, so every non-swapped game in the `9`-agent arena gets a
unique starting position while the color-swapped rematch keeps the same opening.

Status visibility is now explicit in every long stage:

- Phase-5 materialization logs chunk progress
- workflow building logs teacher-analysis progress
- LAPv1 training logs mid-epoch batch progress
- arena writes `progress.json`

The workflow build is also explicitly chunked. The all-unique train split is sliced into bounded workflow/planner-head chunks and merged afterwards, so the large teacher-analysis pass no longer needs to keep the entire split in memory at once.

That chunked path replaced the earlier monolithic attempt after the first full all-unique workflow build was killed by the host during the train-split teacher pass.

The follow-up LAPv1 Stage-T1 resume also no longer preloads the complete
`planner_head_train.jsonl` into RAM. The trainer now builds a lightweight file
offset index and prepares examples per batch on demand, which removes the
earlier `~24 GiB RSS` spike before the first real epoch.

The next restart no longer trains directly from `planner_head_<split>.jsonl`
either. The Phase-10 workflow now materializes dedicated `lapv1_<split>.jsonl`
artifacts that precompute the LAPv1-side piece tokens, square tokens,
state-context globals, reachability edges, and bounded teacher targets once
during workflow build. The trainer then consumes those precomputed LAPv1
artifacts directly.

The preferred restart target is therefore no longer the large
[phase10_lapv1_stage1_all_unique_v1.json](/home/torsten/EngineKonzept/python/configs/phase10_lapv1_stage1_all_unique_v1.json)
reference. It is now the smaller fast variant:

- [phase10_lapv1_stage1_fast_all_unique_v1.json](/home/torsten/EngineKonzept/python/configs/phase10_lapv1_stage1_fast_all_unique_v1.json)
- [phase10_agent_lapv1_stage1_fast_all_unique_v1.json](/home/torsten/EngineKonzept/python/configs/phase10_agent_lapv1_stage1_fast_all_unique_v1.json)
- [phase10_lapv1_stage1_fast_arena_all_unique_v1.json](/home/torsten/EngineKonzept/python/configs/phase10_lapv1_stage1_fast_arena_all_unique_v1.json)
- [run_phase10_lapv1_stage1_fast_arena_longrun.sh](/home/torsten/EngineKonzept/python/scripts/run_phase10_lapv1_stage1_fast_arena_longrun.sh)

That restart path keeps the same all-unique corpus and reference-arm arena, but
changes three things deliberately:

- dedicated precomputed `lapv1_*` training artifacts
- smaller `lapv1_stage1_fast_all_unique_v1` model (`19.8M` params instead of `77.3M`)
- `batch_size=12` for materially better CPU throughput
- two runtime LAPv1 arena variants from the same trained checkpoint (`inner0` and `inner1`)

The completed Stage-T1 arena then clarified the next repair path:

- `inner1` was weaker than `inner0`
- but `inner1` was only a runtime override on a checkpoint trained with
  `max_inner_steps=0`
- the next meaningful comparison is therefore a real Stage-T2 checkpoint, not
  another Stage-T1 override run

That follow-up is now prepared through:

- [phase10_lapv1_stage2_fast_all_unique_v1.json](/home/torsten/EngineKonzept/python/configs/phase10_lapv1_stage2_fast_all_unique_v1.json)
- [phase10_agent_lapv1_stage2_fast_all_unique_v1.json](/home/torsten/EngineKonzept/python/configs/phase10_agent_lapv1_stage2_fast_all_unique_v1.json)
- [phase10_lapv1_stage2_fast_arena_all_unique_v1.json](/home/torsten/EngineKonzept/python/configs/phase10_lapv1_stage2_fast_arena_all_unique_v1.json)
- [run_phase10_lapv1_stage2_fast_arena_longrun.sh](/home/torsten/EngineKonzept/python/scripts/run_phase10_lapv1_stage2_fast_arena_longrun.sh)

The key deltas over Stage-T1 are:

- warm-start from the completed fast Stage-T1 checkpoint
- real `stage='T2'` training with schedule `1 -> 2 -> 4`
- explicit intermediate step-policy supervision for the inner loop
- much larger warm-start batch size on this host (`256` instead of `12`)
  because the current fast trainer stayed near `~1.2-1.7 GiB RSS` at batches
  `12-48` and still left ample headroom on a `23 GiB` machine
- frequent logging for this run (`log_interval_batches=24`) so the long T2
  warm-start remains observable without paying full per-batch logging overhead
- four LAPv1 runtime variants from the same trained checkpoint:
  `inner0`, `inner1`, `inner2`, `auto4`

`auto4` should be read as:

- hard runtime budget cap `4`
- but early stop is still learned through sharpness and top-1 stability

Future UCI-facing status updates for the inner loop remain a documented follow-up
task, not part of this prep note.

Remaining scale TODOs observed during this prep/resume cycle:

- planner-family trainers outside LAPv1 still load `planner_head` artifacts eagerly
- MoE analysis still assumes in-memory `planner_head` access
- verify/matrix tooling in some later campaigns should be converted to streaming readers if they move onto larger corpora
- if later Phase-10 or Phase-11 paths reuse large `planner_head` or `lapv1`
  artifacts, keep them on lazy readers instead of regressing to eager JSONL
  materialization

## Intent

This prep keeps the architectural boundary intact:

- exact legality stays symbolic
- LAPv1 is trained once up front
- legacy arms are reused as fixed references
- no classical search is introduced into the learned runtime path
