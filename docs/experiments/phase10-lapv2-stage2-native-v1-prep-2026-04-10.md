# Phase-10 LAPv2 Stage-T2 Native v1 Preparation

This note captures the first full native LAPv2 run prepared after the
completed compatibility-style `phase10_lapv2_stage2_arena_all_unique_v1`
campaign.

## Goal

Run the first LAPv2 campaign that is native in both senses:

- the model checkpoint is already LAPv2, not LAPv1
- the workflow artifacts come from the dedicated all-sources NAS/Pi
  conversion path instead of the older `lapv1_workflow_all_unique_v1`
  compatibility root

## Data path

Native sources:

- merged raw root:
  [/srv/schach/engine_training/phase10/lapv2_raw_merged_all_sources_v1](/srv/schach/engine_training/phase10/lapv2_raw_merged_all_sources_v1)
- materialized train root:
  [/srv/schach/engine_training/phase10/lapv2_dataset_all_sources_train_v1](/srv/schach/engine_training/phase10/lapv2_dataset_all_sources_train_v1)
- materialized verify root:
  [/srv/schach/engine_training/phase10/lapv2_dataset_all_sources_verify_v1](/srv/schach/engine_training/phase10/lapv2_dataset_all_sources_verify_v1)
- workflow root:
  [/srv/schach/engine_training/phase10/lapv2_workflow_all_sources_v1](/srv/schach/engine_training/phase10/lapv2_workflow_all_sources_v1)

Important note:

- The workflow builder still emits compatibility filenames such as
  `lapv1_train.jsonl`, `lapv1_validation.jsonl`, and `lapv1_test.jsonl`.
- For this run those files are treated as native LAPv2 artifacts because
  the content contract is what matters, not the basename.

## Data integrity

The merged all-sources corpus currently resolves to:

- `train_records=750429`
- `verify_records=2741`
- `verify_train_overlap=0`

The materialized split overlap checks are all clean:

- `train ∩ validation = 0`
- `train ∩ verify_test = 0`
- `validation ∩ verify_test = 0`

## Run structure

Warm start:

- source checkpoint:
  [checkpoint.pt](/home/torsten/EngineKonzept/models/lapv2/stage2_all_unique_v1/bundle/checkpoint.pt)

Training:

- same stable LAPv2 T2 schedule family as the completed `v1` run
- same `4`-epoch shape
- same `batch_size=256`
- native workflow root swapped in everywhere

Teacher workflow quality mode:

- `MultiPV=8`
- `train depth=7`
- `validation depth=8`
- `verify depth=8`
- `3` parallel workflow workers on the Pi
- full native workflow rebuild from scratch once the upgraded builder is in place

Distributed rebuild mode:

- the workflow builder can now run split-specific chunk shards
- intended deployment is `2` local shards plus `1` Pi shard
- worker shards use `--skip-finalize`
- one final local `--finalize-only` pass merges all completed chunk artifacts

Arena:

- `lapv2_inner0`
- `lapv2_inner1`
- `lapv2_auto4`
- `planner_recurrent_expanded_v1`
- `planner_set_v6_replay_expanded_v2`
- `planner_set_v6_expanded_v1`
- `planner_set_v6_rank_expanded_v1`
- `vice_v2`

This keeps the comparison focused on the strongest current LAPv2
budgets and the strongest current non-LAPv1 references.

## Prepared artifacts

- Train config:
  [phase10_lapv2_stage2_native_all_sources_v1.json](/home/torsten/EngineKonzept/python/configs/phase10_lapv2_stage2_native_all_sources_v1.json)
- Base runtime spec:
  [phase10_agent_lapv2_stage2_native_all_sources_v1.json](/home/torsten/EngineKonzept/python/configs/phase10_agent_lapv2_stage2_native_all_sources_v1.json)
- Arena campaign:
  [phase10_lapv2_stage2_native_arena_all_sources_v1.json](/home/torsten/EngineKonzept/python/configs/phase10_lapv2_stage2_native_arena_all_sources_v1.json)
- Launcher:
  [run_phase10_lapv2_stage2_native_arena_v1_longrun.sh](/home/torsten/EngineKonzept/python/scripts/run_phase10_lapv2_stage2_native_arena_v1_longrun.sh)

## Operational status

The Pi-native workflow conversion currently resumes in the background
under:

- log:
  [/srv/schach/engine_training/phase10/lapv2_artifact_build_all_sources_v1.console.log](/srv/schach/engine_training/phase10/lapv2_artifact_build_all_sources_v1.console.log)

Two operational blockers were handled on the way to this run shape:

- `torch` had to be installed on the Pi so the disagreement workflow
  builder could run at all
- the `planner_head -> lapv1_training` converter had to be fixed so it
  reconstructs moves from real flat action indices instead of assuming a
  local `[0, 0, ordinal]` fallback encoding

That second fix was verified locally against the exact failing
`chunk_0001_00000000` planner-head artifact before the Pi workflow was
resumed again.

The run should only be started once the workflow root has written its
final `summary.json`.
