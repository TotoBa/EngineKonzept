# Opponent Bundles

This directory contains trained Phase-7 opponent-head bundles.

Current contents:

- [merged_unique_mlp_v1](/home/torsten/EngineKonzept/models/opponent/merged_unique_mlp_v1)

Current reference:

- config: [phase7_opponent_merged_unique_mlp_v1.json](/home/torsten/EngineKonzept/python/configs/phase7_opponent_merged_unique_mlp_v1.json)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase7/opponent_merged_unique_mlp_v1/summary.json)
- verify eval: [opponent_merged_unique_mlp_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase7/opponent_merged_unique_mlp_v1_verify.json)

This is the first explicit learned `OpponentHeadV1` reference in the repo.

It is not yet the preferred Phase-7 reply model. The symbolic reply-scorer baseline remains stronger on the current larger verify slice.
