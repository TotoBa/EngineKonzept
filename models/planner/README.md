# Planner Bundles

This directory contains trained Phase-8 planner bundles.

Current contents:

- [corpus_suite_set_v1](/home/torsten/EngineKonzept/models/planner/corpus_suite_set_v1)
- [corpus_suite_set_v2_v1](/home/torsten/EngineKonzept/models/planner/corpus_suite_set_v2_v1)
- [corpus_suite_set_v3_two_tier_v1](/home/torsten/EngineKonzept/models/planner/corpus_suite_set_v3_two_tier_v1)

Current reference:

- config: [phase8_planner_corpus_suite_set_v2_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v2_v1.json)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v2_v1/summary.json)
- verify eval: [planner_corpus_suite_set_v2_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v2_v1_verify.json)
- comparison: [planner_corpus_suite_compare_v2.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_compare_v2.json)

Expanded-suite references now also exist under the external training root and are tracked via copied summaries and verify reports in [artifacts/phase8/README.md](/home/torsten/EngineKonzept/artifacts/phase8/README.md):

- `set_v6_expanded` is the current full-suite `top1` leader
- `set_v6_rank_expanded` is the current full-suite `MRR` leader
- `set_v6_margin_expanded` is effectively tied on `MRR` and strongest on `top3`

Those expanded checkpoints live under `/srv/schach/engine_training/models/planner/*_expanded_v1` and are consumed by the Phase-9 expanded agent specs.

`corpus_suite_set_v1` remains the first trained bounded planner reference in the repo.
`corpus_suite_set_v2_v1` is the current preferred richer-target follow-up.
`corpus_suite_set_v3_two_tier_v1` is the first explicit latent-state planner arm over the filtered `10k + 122k` workflow slice; it keeps the new planner contract alive, but it is not preferred because it underperforms `set_v2` on that slice.
