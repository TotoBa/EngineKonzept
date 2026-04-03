# Phase 8

## Goal

Introduce the first bounded planner over:

- exact proposer candidates
- exact successor states
- explicit opponent signals

without crossing the project boundary into classical search.

## Current repository state

The repository now has the first materialized trained bounded planner run over the current multi-corpus workflow suite.

It first gained the planner-facing offline baselines:

- [eval_planner_baseline.py](/home/torsten/EngineKonzept/python/scripts/eval_planner_baseline.py)
- [planner.py](/home/torsten/EngineKonzept/python/train/eval/planner.py)
- [planner_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase7/planner_compare_v1.json)

That baseline is explicitly bounded and symbolic-contract-aware:

1. score exact legal root candidates with the symbolic proposer
2. exact-apply a bounded top-k root slice
3. exact-generate successor reply candidates
4. score replies with either:
   - no opponent head
   - the symbolic reply scorer
   - the first learned opponent head
5. aggregate a bounded pessimistic root score

## What it is not

- no alpha-beta
- no tree search
- no transposition table
- no runtime planner module

It is an offline evaluation baseline that proves the proposer/opponent contracts can already be composed into a first planner-like decision layer.

The repo now also has the first trainable planner-arm stack:

- [build_planner_head_dataset.py](/home/torsten/EngineKonzept/python/scripts/build_planner_head_dataset.py)
- [build_phase8_workflow_suite.py](/home/torsten/EngineKonzept/python/scripts/build_phase8_workflow_suite.py)
- [eval_planner_suite_baseline.py](/home/torsten/EngineKonzept/python/scripts/eval_planner_suite_baseline.py)
- [compare_planner_suite_runs.py](/home/torsten/EngineKonzept/python/scripts/compare_planner_suite_runs.py)
- [planner_head.py](/home/torsten/EngineKonzept/python/train/datasets/planner_head.py)
- [planner.py](/home/torsten/EngineKonzept/python/train/models/planner.py)
- [planner.py](/home/torsten/EngineKonzept/python/train/trainers/planner.py)
- [train_planner.py](/home/torsten/EngineKonzept/python/scripts/train_planner.py)
- [eval_planner.py](/home/torsten/EngineKonzept/python/scripts/eval_planner.py)

That trainable arm keeps the project boundary:

1. exact root candidates still come from the symbolic proposer contract
2. successor states still come from exact move application plus oracle relabeling
3. opponent-side signals still come from a bounded reply module
4. the learned planner only scores a bounded candidate set; it does not become hidden tree search

The first materialized planner reference was:

- config: [phase8_planner_corpus_suite_set_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v1.json)
- workflow suite: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_workflow_corpus_suite_v1/summary.json)
- bundle: [corpus_suite_set_v1](/home/torsten/EngineKonzept/models/planner/corpus_suite_set_v1)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v1/summary.json)
- verify: [planner_corpus_suite_set_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v1_verify.json)
- comparison: [planner_corpus_suite_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_compare_v1.json)

The first preferred planner reference was:

- config: [phase8_planner_corpus_suite_set_v2_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v2_v1.json)
- bundle: [corpus_suite_set_v2_v1](/home/torsten/EngineKonzept/models/planner/corpus_suite_set_v2_v1)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v2_v1/summary.json)
- verify: [planner_corpus_suite_set_v2_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v2_v1_verify.json)
- comparison: [planner_corpus_suite_compare_v2.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_compare_v2.json)

The repo now also has an expanded-data rerun:

- config: [phase8_planner_corpus_suite_set_v2_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v2_expanded_v1.json)
- repo-copied summary: [planner_corpus_suite_set_v2_expanded_v1_summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v2_expanded_v1_summary.json)
- repo-copied verify: [planner_corpus_suite_set_v2_expanded_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v2_expanded_v1_verify.json)
- filtered expanded comparison: [planner_corpus_suite_expanded_two_tier_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_expanded_two_tier_compare_v1.json)

The repo now also has a filtered latent-state validation slice over just the `10k` and `122k` tiers:

- workflow suite: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_workflow_corpus_suite_latent_two_tier_v1/summary.json)
- experimental config: [phase8_planner_corpus_suite_set_v3_two_tier_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v3_two_tier_v1.json)
- experimental bundle: [corpus_suite_set_v3_two_tier_v1](/home/torsten/EngineKonzept/models/planner/corpus_suite_set_v3_two_tier_v1)
- experimental summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v3_two_tier_v1/summary.json)
- filtered comparison: [planner_corpus_suite_two_tier_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_two_tier_compare_v1.json)

## Current result

Planner workflow suite:

- `10k` tier
- merged unique `122k` tier
- imported unique `400k` tier

Training summary:

- `best_epoch=5`
- validation `root_top1_accuracy=0.799107`
- validation `root_top3_accuracy=0.967634`
- validation `teacher_root_mean_reciprocal_rank=0.880952`

Aggregate held-out verify result over `1,410` planner examples:

- root-only bounded baseline:
  - `root_top1_accuracy=0.153901`
  - `teacher_root_mean_reciprocal_rank=0.230615`
- symbolic-reply bounded baseline:
  - `root_top1_accuracy=0.159574`
  - `teacher_root_mean_reciprocal_rank=0.232861`
- learned-reply bounded baseline:
  - `root_top1_accuracy=0.142553`
  - `teacher_root_mean_reciprocal_rank=0.224232`
- first trained planner `set_v1`:
  - `root_top1_accuracy=0.788652`
  - `root_top3_accuracy=0.958156`
  - `teacher_root_mean_reciprocal_rank=0.872636`
  - `teacher_root_mean_probability=0.616233`
- current planner `set_v2`:
  - `root_top1_accuracy=0.795035`
  - `root_top3_accuracy=0.968085`
  - `teacher_root_mean_reciprocal_rank=0.875355`
  - `teacher_root_mean_probability=0.685788`
  - `root_value_mae_cp=90.521303`
  - `root_gap_mae_cp=264.01746`

That means the repository is now past pure planner baselines. The richer-target `set_v2` arm stays comfortably above all bounded hand-aggregation baselines and improves modestly over the first `set_v1` planner on the same multi-corpus verify suite. The later expanded-data rerun improves the full mixed training and validation picture further.

## Latent-state validation slice

To validate planner-state changes without waiting on the `400k` tier, the repo now also has a filtered `10k + 122k` planner workflow slice with explicit Phase-6 latent successor features.

Filtered verify result over `1,024` held-out planner examples:

- root-only bounded baseline:
  - `root_top1_accuracy=0.151367`
  - `teacher_root_mean_reciprocal_rank=0.219482`
- symbolic-reply bounded baseline:
  - `root_top1_accuracy=0.158203`
  - `teacher_root_mean_reciprocal_rank=0.222819`
- learned-reply bounded baseline:
  - `root_top1_accuracy=0.135742`
  - `teacher_root_mean_reciprocal_rank=0.2111`
- reference planner `set_v2` on the same filtered slice:
  - `root_top1_accuracy=0.80957`
  - `root_top3_accuracy=0.975586`
  - `teacher_root_mean_reciprocal_rank=0.883382`
- latent-state planner `set_v3`:
  - `root_top1_accuracy=0.708008`
  - `root_top3_accuracy=0.933594`
  - `teacher_root_mean_reciprocal_rank=0.825521`

That is a clear negative result for the first latent-state planner arm: `PlannerHeadV1` can now carry Phase-6 latent successor vectors, but the first direct `set_v3` integration loses clearly to `set_v2`.

The repo now also has a filtered comparison of the expanded-data planner reruns on the same preferred `10k + 122k` slice:

- prior two-tier `set_v2`: `root_top1_accuracy=0.80957`, `MRR=0.883382`
- expanded-data `set_v2`: `root_top1_accuracy=0.798828`, `MRR=0.87972`
- expanded-data `set_v2_wide`: `root_top1_accuracy=0.790039`, `MRR=0.874837`
- expanded-data `set_v5`: `root_top1_accuracy=0.798828`, `MRR=0.880534`
- `10k + 122k`-only expanded `set_v2`: `root_top1_accuracy=0.819336`, `MRR=0.889811`

The next follow-up now adds a faster latent rerun on top of that stronger filtered workflow:

- latent workflow suite: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_workflow_corpus_suite_latent_10k_122k_expanded_v1/summary.json)
- latent config: [phase8_planner_corpus_suite_set_v3_10k_122k_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v3_10k_122k_expanded_v1.json)
- latent summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v3_10k_122k_expanded_v1/summary.json)
- latent verify: [planner_corpus_suite_set_v3_10k_122k_expanded_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v3_10k_122k_expanded_v1_verify.json)
- latent comparison: [planner_corpus_suite_latent_two_tier_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_latent_two_tier_compare_v1.json)

Result on the same filtered `10k + 122k` verify slice:

- `set_v2_10k_122k_expanded`: `root_top1_accuracy=0.819336`, `MRR=0.889811`
- latent `set_v3_10k_122k_expanded`: `root_top1_accuracy=0.797852`, `MRR=0.880778`
- latent `set_v3_10k_122k_expanded` does recover `root_top3_accuracy=0.973633`
- but it still loses the more important `top1` and `MRR` comparison to the stronger filtered `set_v2`

This is the current important Phase-8 conclusion:

- more mixed three-tier data helps the global training and validation story
- but on the preferred `10k + 122k` slice, the actual win comes from stronger `10k + 122k` workflow material without the `400k` tier mixed into planner training
- latent planner-head materialization is now reproducible directly from existing planner-head artifacts, so future latent reruns no longer require rebuilding the whole workflow from Phase 7
- even with that stronger filtered workflow, the current direct latent-state planner path is still not the best planner arm
- increasing width does not help
- `set_v5` re-enters the conversation on the filtered slice, but not strongly enough to replace the new `10k + 122k`-only `set_v2` rerun

## Next pressure

The next useful Phase-8 steps are now:

1. re-test planner-facing latent-state or contract upgrades on top of the new `10k + 122k`-only `set_v2` reference
2. focus on a different latent integration path or richer planner targets rather than another direct `set_v3`-style concatenation rerun
3. only then revisit richer bounded recurrence over the same exact candidate slice

The first richer-target follow-up is now prepared at the artifact-contract level:

- `PlannerHeadV1` rows can now carry restricted `teacher_candidate_scores_cp` aligned to the bounded candidate slice
- `PlannerHeadV1` rows can now also carry clipped `teacher_candidate_score_delta_targets_cp` relative to `teacher_root_value_cp`
- `PlannerHeadV1` rows can now also carry `teacher_rank_bucket_version=1` and discrete `teacher_candidate_rank_bucket_targets`
- the field is backward-compatible and remains optional for older artifacts
- this keeps the current workflow semantics intact while making the next score-aux planner arm possible without another contract break

That richer-target arm is now also prepared at the model/config level:

- planner architecture `set_v6` keeps the current `set_v2` bounded candidate-scoring backbone
- it adds an auxiliary candidate-score regression head over the same restricted root slice
- the first intended rerun stays on the preferred filtered expanded `10k + 122k` suite rather than widening back out to the `400k` tier

That rerun has now been executed on the filtered expanded `10k + 122k` suite:

- workflow summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_workflow_corpus_suite_score_10k_122k_expanded_v1/summary.json)
- config: [phase8_planner_corpus_suite_set_v6_10k_122k_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v6_10k_122k_expanded_v1.json)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v6_10k_122k_expanded_v1/summary.json)
- verify: [planner_corpus_suite_set_v6_10k_122k_expanded_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v6_10k_122k_expanded_v1_verify.json)
- comparison: [planner_corpus_suite_score_two_tier_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_score_two_tier_compare_v1.json)

Result on the same preferred held-out slice:

- `set_v2_10k_122k_expanded`: `root_top1_accuracy=0.819336`, `root_top3_accuracy=0.960938`, `MRR=0.889811`
- `set_v6_10k_122k_expanded`: `root_top1_accuracy=0.817383`, `root_top3_accuracy=0.964844`, `MRR=0.890625`

So the richer score-target arm is real and competitive:

- it improves `MRR`, `teacher_root_mean_probability`, and `top3`
- it gives back a small amount of `top1`
- `set_v2_10k_122k_expanded` therefore remains the preferred Phase-8 reference
- `set_v6_10k_122k_expanded` stays as the first useful score-aux experimental arm

The next narrower follow-up is now prepared as well:

- keep the same `set_v6` backbone
- keep the filtered `10k + 122k` suite
- reduce raw bounded score-regression pressure
- add explicit `top1-vs-top2/top3` margin supervision over the bounded candidate slice

That margin rerun has now also been executed on the same filtered suite:

- workflow summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_workflow_corpus_suite_margin_10k_122k_expanded_v1/summary.json)
- config: [phase8_planner_corpus_suite_set_v6_margin_10k_122k_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v6_margin_10k_122k_expanded_v1.json)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v6_margin_10k_122k_expanded_v1/summary.json)
- verify: [planner_corpus_suite_set_v6_margin_10k_122k_expanded_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v6_margin_10k_122k_expanded_v1_verify.json)
- comparison: [planner_corpus_suite_margin_two_tier_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_margin_two_tier_compare_v1.json)

Held-out result on the same `10k + 122k` slice:

- `set_v2_10k_122k_expanded`: `top1=0.819336`, `top3=0.960938`, `MRR=0.889811`
- `set_v6_10k_122k_expanded`: `top1=0.817383`, `top3=0.964844`, `MRR=0.890625`
- `set_v6_margin_10k_122k_expanded`: `top1=0.8125`, `top3=0.96582`, `MRR=0.889079`

Interpretation:

- the margin arm sharply stabilizes the score-target side of the problem:
  - `teacher_score_loss`: `78.46875 -> 0.277589`
  - `teacher_score_mae_cp`: `242.310951 -> 55.99959`
- but it does not translate into a planner-quality gain on the most important held-out metrics
- `set_v2_10k_122k_expanded` remains the preferred Phase-8 reference
- `set_v6_margin_10k_122k_expanded` stays as a useful negative/diagnostic result, not a new default

The next same-backbone follow-up is now prepared as well:

- `set_v6` can now also add an optional discrete rank-bucket head over the same bounded candidate slice
- that head uses the new `PlannerHeadV1` bucket targets instead of another raw centipawn-style auxiliary regression
- config: [phase8_planner_corpus_suite_set_v6_rank_10k_122k_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v6_rank_10k_122k_expanded_v1.json)

That rank-bucket rerun has now also been executed on the same filtered expanded `10k + 122k` suite:

- workflow summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_workflow_corpus_suite_rank_10k_122k_expanded_v1/summary.json)
- config: [phase8_planner_corpus_suite_set_v6_rank_10k_122k_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v6_rank_10k_122k_expanded_v1.json)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v6_rank_10k_122k_expanded_v1/summary.json)
- verify: [planner_corpus_suite_set_v6_rank_10k_122k_expanded_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v6_rank_10k_122k_expanded_v1_verify.json)
- comparison: [planner_corpus_suite_rank_two_tier_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_rank_two_tier_compare_v1.json)

Held-out result on the same `10k + 122k` slice:

- `set_v2_10k_122k_expanded`: `top1=0.819336`, `top3=0.960938`, `MRR=0.889811`
- `set_v6_rank_10k_122k_expanded`: `top1=0.8125`, `top3=0.956055`, `MRR=0.884684`

Interpretation:

- the discrete rank-bucket contract is now real, reproducible, and measurable
- but this first rank-bucket arm is weaker than both `set_v2` and the earlier score-aux `set_v6`
- `set_v2_10k_122k_expanded` therefore remains the preferred Phase-8 reference

The first bounded recurrent follow-up is now prepared at the model/config level as well:

- architecture: `recurrent_v1`
- keeps the same `PlannerHeadV1` data contract and the same bounded candidate slice
- adds explicit `memory_slots` and `deliberation_steps`
- refines candidate tokens through a small recurrent memory loop instead of widening the same one-shot scorer again
- config: [phase8_planner_corpus_suite_recurrent_v1_10k_122k_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_recurrent_v1_10k_122k_expanded_v1.json)

That recurrent rerun has now also been executed on the same preferred filtered expanded `10k + 122k` suite:

- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_recurrent_v1_10k_122k_expanded_v1/summary.json)
- verify: [planner_corpus_suite_recurrent_v1_10k_122k_expanded_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_recurrent_v1_10k_122k_expanded_v1_verify.json)
- comparison: [planner_corpus_suite_recurrent_two_tier_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_recurrent_two_tier_compare_v1.json)

Held-out result on the same `10k + 122k` slice:

- `set_v2_10k_122k_expanded`: `top1=0.819336`, `top3=0.960938`, `MRR=0.889811`
- `recurrent_v1_10k_122k_expanded`: `top1=0.805664`, `top3=0.962891`, `MRR=0.885742`

Interpretation:

- the first recurrent planner arm is now materially implemented, measured, and reproducible
- it keeps the existing planner-head contract intact, so future recurrence changes do not require another workflow-schema break
- but it does not beat the current filtered `set_v2` reference on the main held-out metrics
- so recurrence is now available as infrastructure for the next phase, not yet the preferred bounded Phase-8 model
