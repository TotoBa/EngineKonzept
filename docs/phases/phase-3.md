# Phase 3

## Goal

Define the canonical move vocabulary and the first deterministic position encoder for later learned components.

## Deliverables in this repository state

- factorized move representation in `action-space`
- encode/decode utilities between symbolic moves and model-facing action tuples
- deterministic object-centric position encoding in `encoder`
- documented tensor and token semantics
- deterministic tests for move roundtrips and fixed-FEN encodings

## Non-goals still preserved

- no inference
- no legality/policy model
- no planner logic
- no classical search fallback
