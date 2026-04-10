# Phase-10 LAPv2 Stage-T2 v1 Summary

This note records the completed `phase10_lapv2_stage2_arena_all_unique_v1`
run and the decisions taken from it.

## Scope

This run was the first end-to-end LAPv2 campaign on top of the completed
`stage2_fast_all_unique_v4` LAPv1 checkpoint.

Important limitation:

- The run still consumed the older compatibility workflow under
  `/srv/schach/engine_training/phase10/lapv1_workflow_all_unique_v1`.
- That means the run exercised the full LAPv2 model stack, but not yet
  the fully native LAPv2 artifact contract produced from the NAS/Pi
  all-sources conversion path.

## Key results

### Stage-T2 training / selection holdout

Best epoch from
[summary.json](/home/torsten/EngineKonzept/models/lapv2/stage2_all_unique_v1/summary.json):

- `best_epoch=4`
- `selection_top1=0.698506`
- `selection_mrr=0.820796`
- `reply_consistency=0.399872`
- `mean_inner_steps_executed=2.059073`

Across epochs the run improved steadily:

- epoch 1 `selection_top1=0.550937`
- epoch 2 `selection_top1=0.662420`
- epoch 3 `selection_top1=0.677678`
- epoch 4 `selection_top1=0.698506`

### Final verify holdout

From
[lapv1_verify.json](/srv/schach/engine_training/phase10/lapv2_stage2_arena_all_unique_v1/lapv1_verify.json):

- `root_top1_accuracy=0.649763`
- `teacher_root_mean_reciprocal_rank=0.787699`
- `top1_changed_rate=0.835826`
- `root_incorrect_improvement_rate=0.863539`
- `root_correct_degraded_rate=0.574163`
- `reply_consistency=0.226036`
- `mean_inner_steps_executed=1.875958`

Interpretation:

- LAPv2 is clearly active. The inner loop is not collapsing to the root.
- The correction mechanism is still too aggressive. It improves many
  root mistakes, but it also degrades too many already-correct root
  decisions.

### Arena

From
[arena/summary.json](/srv/schach/engine_training/phase10/lapv2_stage2_arena_all_unique_v1/arena/summary.json):

- `vice_v2`: `24.0 / 24` (`1.000`)
- `planner_recurrent_expanded_v1`: `15.5 / 24` (`0.646`)
- `planner_set_v6_replay_expanded_v2`: `14.5 / 24` (`0.604`)
- `planner_set_v6_expanded_v1`: `14.5 / 24` (`0.604`)
- `planner_set_v6_rank_expanded_v1`: `13.5 / 24` (`0.562`)
- `lapv2_inner1`: `10.5 / 24` (`0.438`)
- `lapv1_v4_inner0`: `10.5 / 24` (`0.438`)
- `lapv2_inner0`: `10.0 / 24` (`0.417`)
- `lapv1_v4_inner2`: `10.0 / 24` (`0.417`)
- `lapv1_v4_inner1`: `9.0 / 24` (`0.375`)
- `lapv2_auto4`: `8.5 / 24` (`0.354`)
- `lapv1_v4_auto4`: `8.5 / 24` (`0.354`)
- `lapv2_inner2`: `7.0 / 24` (`0.292`)

Within the LAPv2 family:

- `inner1` was the strongest runtime budget
- `inner0` was the best control
- `auto4` was viable but not better than `inner1`
- `inner2` was the weakest and is not worth carrying into the next
  comparison run

Against the retired LAPv1 budgets:

- `lapv2_inner1` outscored `lapv1_v4_inner1` and `lapv1_v4_auto4`
- `lapv2_inner0` roughly matched the strongest old LAPv1 controls
- The new family did not yet beat the best non-LAPv1 planner arms

## Conclusions

1. LAPv2 is now the active research line.
   LAPv1 should not receive further benchmark slots in the next Phase-10
   run.
2. The next LAPv2 arena should keep only:
   - `inner0`
   - `inner1`
   - `auto4`
3. The best non-LAPv1 references remain:
   - `planner_recurrent_expanded_v1`
   - `planner_set_v6_replay_expanded_v2`
   - `planner_set_v6_expanded_v1`
   - `planner_set_v6_rank_expanded_v1`
   - `vice_v2`
4. The next major gain is no longer another LAPv1 comparison.
   It is the switch to the fully native LAPv2 artifact path built from
   the NAS/Pi all-sources corpus.

## Measures taken after the run

- The native all-sources raw sources were merged under
  `/srv/schach/engine_training/phase10/lapv2_raw_merged_all_sources_v1`.
- Materialized train/validation/test splits were built under:
  - `/srv/schach/engine_training/phase10/lapv2_dataset_all_sources_train_v1`
  - `/srv/schach/engine_training/phase10/lapv2_dataset_all_sources_verify_v1`
- Split disjointness was checked explicitly:
  - `train ∩ validation = 0`
  - `train ∩ verify_test = 0`
  - `validation ∩ verify_test = 0`
- The Pi workflow builder was resumed with `torch` installed so the
  native LAPv2 workflow can complete on the existing NAS dataset root.

## Next run shape

The next prepared run is the first full native LAPv2 run:

- warm-start from
  [checkpoint.pt](/home/torsten/EngineKonzept/models/lapv2/stage2_all_unique_v1/bundle/checkpoint.pt)
- train on the native all-sources workflow root
- benchmark only LAPv2 budgets plus the four strongest planner
  references and `vice_v2`
- no LAPv1 baselines in the arena
