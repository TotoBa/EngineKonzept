# Phase 8 Artifacts

This directory contains the first multi-corpus planner-workflow and trained planner artifacts.

Current contents:

- workflow suite: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_workflow_corpus_suite_v1/summary.json)
- first trained planner summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v1/summary.json)
- current trained planner summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v2_v1/summary.json)
- first trained planner verify eval: [planner_corpus_suite_set_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v1_verify.json)
- current trained planner verify eval: [planner_corpus_suite_set_v2_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v2_v1_verify.json)
- baseline comparisons:
  - [planner_root_only_corpus_suite_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_root_only_corpus_suite_v1.json)
  - [planner_symbolic_reply_corpus_suite_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_symbolic_reply_corpus_suite_v1.json)
  - [planner_learned_reply_corpus_suite_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_learned_reply_corpus_suite_v1.json)
  - [planner_corpus_suite_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_compare_v1.json)
  - [planner_corpus_suite_compare_v2.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_compare_v2.json)

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
