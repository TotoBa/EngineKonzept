# Opponent Bundles

This directory contains trained Phase-7 opponent-head bundles.

Current contents:

- [merged_unique_mlp_v1](/home/torsten/EngineKonzept/models/opponent/merged_unique_mlp_v1)
- [corpus_suite_set_v2_v1](/home/torsten/EngineKonzept/models/opponent/corpus_suite_set_v2_v1)

Current reference:

- config: [phase7_opponent_corpus_suite_set_v2_v1.json](/home/torsten/EngineKonzept/python/configs/phase7_opponent_corpus_suite_set_v2_v1.json)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase7/opponent_corpus_suite_set_v2_v1/summary.json)
- verify eval: [opponent_corpus_suite_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase7/opponent_corpus_suite_compare_v1.json)

This is the current preferred learned `OpponentHeadV1` reference in the repo.

It is also the first learned Phase-7 arm that beats the symbolic reply-scorer baseline on the current three-tier verify suite.
