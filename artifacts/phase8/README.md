# Phase 8 Artifacts

This directory contains the first multi-corpus planner-workflow and trained planner artifacts.

Current contents:

- workflow suite: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_workflow_corpus_suite_v1/summary.json)
- trained planner summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v1/summary.json)
- trained planner verify eval: [planner_corpus_suite_set_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v1_verify.json)
- baseline comparisons:
  - [planner_root_only_corpus_suite_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_root_only_corpus_suite_v1.json)
  - [planner_symbolic_reply_corpus_suite_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_symbolic_reply_corpus_suite_v1.json)
  - [planner_learned_reply_corpus_suite_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_learned_reply_corpus_suite_v1.json)
  - [planner_corpus_suite_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_compare_v1.json)

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
- trained planner `set_v1`:
  - `root_top1_accuracy=0.788652`
  - `root_top3_accuracy=0.958156`
  - `teacher_root_mean_reciprocal_rank=0.872636`

These are bounded offline planner artifacts, not runtime search.
