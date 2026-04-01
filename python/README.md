# EngineKonzept Python Project

This directory hosts the future dataset, training, export, and experiment code for EngineKonzept.

The repository now includes the Phase-4 dataset pipeline:

- raw position ingestion from edge-case files, FEN line files, EPD suites, and JSONL
- exact-rule labeling via the Rust dataset oracle
- deterministic train/validation/test splitting
- JSONL dataset artifact writing and summary reporting

Training, model definition, and planner learning still belong to later phases.
