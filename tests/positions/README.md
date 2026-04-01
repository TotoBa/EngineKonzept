# Position Fixtures

Reserved for rule-edge-case positions, FEN fixtures, and regression scenarios.

`edge_cases.txt` holds unlabeled rule fixtures for Phase 4 dataset smoke tests.
`policy_seed.jsonl` holds a small reproducible Phase 5 seed set with `selected_move_uci` labels so the proposer policy head can be exercised end-to-end.

Larger PGN-derived policy corpora are generated into `artifacts/datasets/` rather than stored here directly, because they are produced offline from external PGN collections plus Stockfish 18 labeling.
