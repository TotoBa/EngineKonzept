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

The next full three-tier reruns now use a single current workflow contract:

- builder: [build_phase8_fulltargets_expanded_workflow.py](/home/torsten/EngineKonzept/python/scripts/build_phase8_fulltargets_expanded_workflow.py)
- suite orchestrator: [materialize_phase8_expanded_suite.py](/home/torsten/EngineKonzept/python/scripts/materialize_phase8_expanded_suite.py)
- repo-copied summary: [planner_workflow_fulltargets_expanded_v2_summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_workflow_fulltargets_expanded_v2_summary.json)

That contract exists so all future expanded planner arms can consume the same planner-head schema:

- bounded `teacher_candidate_scores_cp`
- clipped `teacher_candidate_score_delta_targets_cp`
- discrete `teacher_candidate_rank_bucket_targets`

This avoids rebuilding separate expanded workflow roots per experimental head and keeps the launch path flexible for later architecture changes without changing the underlying workflow semantics again.

Those full three-tier reruns are now fully materialized as well:

- suite summary: [planner_active_experimental_expanded_v1_summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_active_experimental_expanded_v1_summary.json)
- suite comparison: [planner_active_experimental_expanded_v1_compare.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_active_experimental_expanded_v1_compare.json)
- expanded `set_v6` summary: [planner_corpus_suite_set_v6_expanded_v1_summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v6_expanded_v1_summary.json)
- expanded `set_v6` verify: [planner_corpus_suite_set_v6_expanded_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v6_expanded_v1_verify.json)
- expanded `set_v6_margin` summary: [planner_corpus_suite_set_v6_margin_expanded_v1_summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v6_margin_expanded_v1_summary.json)
- expanded `set_v6_margin` verify: [planner_corpus_suite_set_v6_margin_expanded_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v6_margin_expanded_v1_verify.json)
- expanded `set_v6_rank` summary: [planner_corpus_suite_set_v6_rank_expanded_v1_summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v6_rank_expanded_v1_summary.json)
- expanded `set_v6_rank` verify: [planner_corpus_suite_set_v6_rank_expanded_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v6_rank_expanded_v1_verify.json)
- expanded `recurrent_v1` summary: [planner_corpus_suite_recurrent_v1_expanded_v1_summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_recurrent_v1_expanded_v1_summary.json)
- expanded `recurrent_v1` verify: [planner_corpus_suite_recurrent_v1_expanded_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_recurrent_v1_expanded_v1_verify.json)

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

The newer full three-tier expanded reruns over the current full-target workflow contract now add one more important conclusion on top of the filtered `10k + 122k` picture:

- `set_v2_expanded`: `top1=0.797163`, `top3=0.965957`, `MRR=0.879433`
- `set_v6_expanded`: `top1=0.810638`, `top3=0.964539`, `MRR=0.885816`
- `set_v6_margin_expanded`: `top1=0.80922`, `top3=0.970922`, `MRR=0.887175`
- `set_v6_rank_expanded`: `top1=0.808511`, `top3=0.965957`, `MRR=0.887234`
- `recurrent_v1_expanded`: `top1=0.804965`, `top3=0.964539`, `MRR=0.884279`

Interpretation:

- on the full `10k + 122k + 400k` verify suite, all newly rerun experimental arms now beat the older expanded `set_v2` rerun
- `set_v6_expanded` is the current best `top1` arm on that full expanded suite
- `set_v6_rank_expanded` is the current best `MRR` arm on that full expanded suite, with `set_v6_margin_expanded` effectively tied
- this does not overturn the filtered `10k + 122k` conclusion, where `set_v2_10k_122k_expanded` remains the preferred reference
- but it does mean the full expanded Phase-8 stack is now materially stronger than the earlier `set_v2_expanded` launch assumption

The first replay-driven planner retraining follow-up now also exists on top of that expanded stack:

- replay-retrain config: [phase8_planner_corpus_suite_set_v6_rank_replay_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v6_rank_replay_expanded_v1.json)
- summary: [planner_corpus_suite_set_v6_rank_replay_expanded_v1_summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v6_rank_replay_expanded_v1_summary.json)
- verify: [planner_corpus_suite_set_v6_rank_replay_expanded_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v6_rank_replay_expanded_v1_verify.json)
- comparison: [planner_corpus_suite_set_v6_rank_replay_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v6_rank_replay_compare_v1.json)

Held-out result versus the arena-leading reference:

- `set_v6_rank_expanded`: `top1=0.808511`, `top3=0.965957`, `MRR=0.887234`, `teacher_root_mean_probability=0.699544`
- `set_v6_rank_replay_expanded`: `top1=0.807801`, `top3=0.968794`, `MRR=0.886525`, `teacher_root_mean_probability=0.716682`

Interpretation:

- the replay-driven retraining path is now real, versioned, and warm-start-capable
- the first replay fine-tune improves `top3` and teacher probability on the full expanded holdout
- but it gives back a small amount of `top1` and `MRR`
- so `set_v6_rank_replay_expanded_v1` is a useful new experimental arm, not a new Phase-8 default

The larger `Phase 9` replay source now extends that picture materially:

- replay buffer summary:
  [summary.json](/home/torsten/EngineKonzept/artifacts/phase9/replay_buffer_active_experimental_expanded_v2/summary.json)
- replay supervision summary:
  [summary.json](/home/torsten/EngineKonzept/artifacts/phase9/planner_replay_active_experimental_expanded_v2/summary.json)
- replay planner-head summary:
  [summary.json](/home/torsten/EngineKonzept/artifacts/phase9/planner_replay_head_active_experimental_expanded_v2/summary.json)

Observed replay scale-up:

- replay rows: `3640 -> 12976`
- resolved replay supervision rows: `568 -> 6928`
- replay planner-head rows: `568 -> 6928`

That stronger replay source now feeds two replay-only warm-start mirrors:

- `set_v6` replay mirror:
  - config: [phase8_planner_corpus_suite_set_v6_replay_expanded_v2.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v6_replay_expanded_v2.json)
  - summary: [planner_corpus_suite_set_v6_replay_expanded_v2_summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v6_replay_expanded_v2_summary.json)
  - verify: [planner_corpus_suite_set_v6_replay_expanded_v2_verify.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v6_replay_expanded_v2_verify.json)
- `set_v6_margin` replay mirror:
  - config: [phase8_planner_corpus_suite_set_v6_margin_replay_expanded_v2.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v6_margin_replay_expanded_v2.json)
  - summary: [planner_corpus_suite_set_v6_margin_replay_expanded_v2_summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v6_margin_replay_expanded_v2_summary.json)
  - verify: [planner_corpus_suite_set_v6_margin_replay_expanded_v2_verify.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v6_margin_replay_expanded_v2_verify.json)
- mirror comparison:
  [planner_corpus_suite_replay_mirror_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_replay_mirror_compare_v1.json)

Held-out result versus the corresponding expanded baselines:

- `set_v6_expanded`: `top1=0.810638`, `MRR=0.885816`
- `set_v6_replay_expanded_v2`: `top1=0.812766`, `MRR=0.888061`
- `set_v6_margin_expanded`: `top1=0.80922`, `MRR=0.887175`
- `set_v6_margin_replay_expanded_v2`: `top1=0.813475`, `MRR=0.889894`

So the important new conclusion is:

- replay-only warm-starting on the larger `Phase 9` arena now improves both mirrored arms over their non-replay expanded baselines
- `set_v6_margin_replay_expanded_v2` is the current strongest replay-driven full-expanded arm

The Phase-8 training interface now also supports an optional curriculum-aware sampler for planner-head training:

- the canonical per-example weighting logic lives in [planner_head.py](/home/torsten/EngineKonzept/python/train/datasets/planner_head.py)
- the sampling strategies live in [curriculum.py](/home/torsten/EngineKonzept/python/train/datasets/curriculum.py)
- supported strategies are `uniform`, `linear_ramp`, and `sqrt_ramp`
- existing configs remain unchanged when the optional `curriculum` block is omitted

The next data-prep-only follow-up is now also versioned:

- [filter_400k_by_teacher_quality.py](/home/torsten/EngineKonzept/python/scripts/filter_400k_by_teacher_quality.py) filters `400k` planner-head artifacts by teacher-signal quality without changing the schema
- it drops rows with NaN or extreme root values, ambiguous bounded candidate scores, or trivial one-candidate slices
- [phase8_planner_corpus_suite_set_v2_10k_122k_400k_filtered_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v2_10k_122k_400k_filtered_v1.json) is the prepared rerun template for `10k + 122k + filtered 400k`
- this step deliberately stops at artifact preparation; no new planner training result is attached to it yet

The next model-only planner arm is also prepared now:

- `set_v7` in [planner.py](/home/torsten/EngineKonzept/python/train/models/planner.py)
- it keeps the bounded candidate/output contract from `set_v6`
- but replaces the first candidate mixing step with candidate-to-state cross-attention
- this step is intentionally architecture-only so far; no training result is attached to it yet

There is now also an optional candidate-refinement flag for the same planner family:

- `enable_pairwise_candidates` in [config.py](/home/torsten/EngineKonzept/python/train/config.py)
- `PairwiseCandidateLayer` in [planner.py](/home/torsten/EngineKonzept/python/train/models/planner.py)
- it applies a small masked self-attention pass over the bounded candidate set between the current projection stage and the candidate scorer
- when the flag is omitted or `false`, planner behavior stays unchanged
