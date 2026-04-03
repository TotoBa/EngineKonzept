# Phase 8 Artifacts

This directory contains the first multi-corpus planner-workflow and trained planner artifacts.

Current contents:

- workflow suite: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_workflow_corpus_suite_v1/summary.json)
- filtered latent workflow suite: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_workflow_corpus_suite_latent_two_tier_v1/summary.json)
- first trained planner summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v1/summary.json)
- current trained planner summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v2_v1/summary.json)
- expanded-data planner summary: [planner_corpus_suite_set_v2_expanded_v1_summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v2_expanded_v1_summary.json)
- latent-state planner summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v3_two_tier_v1/summary.json)
- first trained planner verify eval: [planner_corpus_suite_set_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v1_verify.json)
- current trained planner verify eval: [planner_corpus_suite_set_v2_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v2_v1_verify.json)
- expanded-data full verify eval: [planner_corpus_suite_set_v2_expanded_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v2_expanded_v1_verify.json)
- filtered `set_v2` verify eval: [planner_corpus_suite_set_v2_two_tier_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v2_two_tier_v1_verify.json)
- filtered expanded `set_v2` verify eval: [planner_corpus_suite_set_v2_expanded_two_tier_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v2_expanded_two_tier_v1_verify.json)
- filtered expanded `set_v2_wide` verify eval: [planner_corpus_suite_set_v2_wide_expanded_two_tier_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v2_wide_expanded_two_tier_v1_verify.json)
- filtered expanded `set_v5` verify eval: [planner_corpus_suite_set_v5_expanded_two_tier_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v5_expanded_two_tier_v1_verify.json)
- latent-state planner verify eval: [planner_corpus_suite_set_v3_two_tier_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v3_two_tier_v1_verify.json)
- baseline comparisons:
  - [planner_root_only_corpus_suite_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_root_only_corpus_suite_v1.json)
  - [planner_symbolic_reply_corpus_suite_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_symbolic_reply_corpus_suite_v1.json)
  - [planner_learned_reply_corpus_suite_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_learned_reply_corpus_suite_v1.json)
  - [planner_corpus_suite_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_compare_v1.json)
  - [planner_corpus_suite_compare_v2.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_compare_v2.json)
  - [planner_root_only_corpus_suite_two_tier_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_root_only_corpus_suite_two_tier_v1.json)
  - [planner_symbolic_reply_corpus_suite_two_tier_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_symbolic_reply_corpus_suite_two_tier_v1.json)
  - [planner_learned_reply_corpus_suite_two_tier_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_learned_reply_corpus_suite_two_tier_v1.json)
  - [planner_corpus_suite_two_tier_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_two_tier_compare_v1.json)
  - [planner_root_only_corpus_suite_expanded_two_tier_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_root_only_corpus_suite_expanded_two_tier_v1.json)
  - [planner_symbolic_reply_corpus_suite_expanded_two_tier_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_symbolic_reply_corpus_suite_expanded_two_tier_v1.json)
  - [planner_learned_reply_corpus_suite_expanded_two_tier_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_learned_reply_corpus_suite_expanded_two_tier_v1.json)
  - [planner_corpus_suite_expanded_two_tier_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_expanded_two_tier_compare_v1.json)

Current aggregate verify result over `1,410` held-out planner examples:

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
- current planner `set_v2`:
  - `root_top1_accuracy=0.795035`
  - `root_top3_accuracy=0.968085`
  - `teacher_root_mean_reciprocal_rank=0.875355`
  - `teacher_root_mean_probability=0.685788`

These are bounded offline planner artifacts, not runtime search.

Filtered latent-state verify slice over `1,024` held-out planner examples:

- root-only bounded baseline: `0.151367`, `MRR=0.219482`
- symbolic-reply bounded baseline: `0.158203`, `MRR=0.222819`
- learned-reply bounded baseline: `0.135742`, `MRR=0.2111`
- reference planner `set_v2`: `0.80957`, `MRR=0.883382`
- latent-state planner `set_v3`: `0.708008`, `MRR=0.825521`

So the latent planner workflow is now materialized and reproducible, but the first direct latent-state planner arm is still weaker than the current `set_v2` reference.

Expanded filtered `10k + 122k` comparison:

- prior two-tier `set_v2`: `0.80957`, `MRR=0.883382`
- expanded `set_v2`: `0.798828`, `MRR=0.87972`
- expanded `set_v2_wide`: `0.790039`, `MRR=0.874837`
- expanded `set_v5`: `0.798828`, `MRR=0.880534`

So the expanded three-tier rerun helps the mixed full-suite training picture, but the preferred filtered slice still prefers the older two-tier `set_v2` reference.
