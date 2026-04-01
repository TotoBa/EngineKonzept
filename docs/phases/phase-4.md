# Phase 4

## Goal

Build the dataset and label pipeline that turns exact symbolic positions into supervised examples for later legality, policy, dynamics, and planning work.

## Deliverables in this repository state

- structured raw-record schemas under `python/train/datasets`
- exact-rule dataset labels via the Rust `dataset-oracle`
- deterministic train/validation/test split assignment
- JSONL artifact writing and summary reporting
- architecture documentation for source formats and label semantics

## Non-goals still preserved

- no selfplay
- no learned proposer
- no latent dynamics model
- no planner logic
- no classical search fallback
