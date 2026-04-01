# Phase 2

## Goal

Add a clean UCI shell that can reconstruct exact positions and emit deterministic legal stub moves.

## Deliverables in this repository state

- `uci-protocol` command parser and response types
- `engine-app` session state and stdio loop
- `position` reconstruction from `position ... moves`
- deterministic legal `bestmove` selection for `go`
- parser and session smoke tests

## Non-goals still preserved

- no search
- no evaluation
- no neural inference
- no time-management logic beyond parsing `go` arguments

