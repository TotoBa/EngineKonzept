# Phase 7 Artifacts

This directory contains the first real Phase-7 baseline probe for EngineKonzept.

Current contents:

- [search_teacher_verify_probe_v1.jsonl](/home/torsten/EngineKonzept/artifacts/phase7/search_teacher_verify_probe_v1.jsonl)
- [search_traces_verify_probe_v1.jsonl](/home/torsten/EngineKonzept/artifacts/phase7/search_traces_verify_probe_v1.jsonl)
- [search_disagreements_verify_probe_v1.jsonl](/home/torsten/EngineKonzept/artifacts/phase7/search_disagreements_verify_probe_v1.jsonl)
- [search_curriculum_verify_probe_v1.jsonl](/home/torsten/EngineKonzept/artifacts/phase7/search_curriculum_verify_probe_v1.jsonl)
- [opponent_head_verify_probe_v1.jsonl](/home/torsten/EngineKonzept/artifacts/phase7/opponent_head_verify_probe_v1.jsonl)
- [opponent_symbolic_baseline_verify_probe_v1.json](/home/torsten/EngineKonzept/artifacts/phase7/opponent_symbolic_baseline_verify_probe_v1.json)

Probe setup:

- source dataset: [phase5_stockfish_pgn_verify_pi_10k_v1](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_pgn_verify_pi_10k_v1)
- split: `test`
- examples: `16`
- teacher: `/usr/games/stockfish18`
- teacher nodes: `64`
- teacher multipv: `8`
- proposer checkpoint: [checkpoint.pt](/home/torsten/EngineKonzept/models/proposer/stockfish_pgn_symbolic_v1_v1/checkpoint.pt)

Current baseline result:

- `reply_top1_accuracy=0.25`
- `reply_top3_accuracy=0.25`
- `teacher_reply_mean_reciprocal_rank=0.364583`
- `teacher_reply_mean_probability=0.201552`

This is a probe artifact, not yet a large-corpus Phase-7 benchmark.
