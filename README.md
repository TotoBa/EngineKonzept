# EngineKonzept

EngineKonzept is a new chess-engine repository built around latent adversarial planning.
It is intentionally **not** a conventional search engine with a neural add-on.

The target runtime path is:

`position -> encoder -> legality/policy proposer -> latent dynamics -> opponent module -> recurrent planner -> WDL + move selection -> UCI output`

## Current Scope

The repository now covers Phase 8 foundations:

- root project rules and execution plans
- Rust workspace boundaries and placeholder future crates
- Python training-project boundaries and placeholder modules
- exact symbolic chess primitives, position state, FEN support, legal move generation, move application, and perft coverage
- a minimal UCI shell with exact legal move generation and symbolic proposer scoring
- a factorized action space for model-facing move IO
- a deterministic object-centric position encoder
- a Python dataset pipeline with exact-rule labels, deterministic splits, and summary reporting
- a first PyTorch legality/policy proposer with config-driven training, held-out metrics, and measured CPU throughput
- bounded PGN ingestion with offline Stockfish 18 labeling for policy-supervised runs
- a `torch.export` proposer bundle plus Rust-side metadata loading/validation
- a local-first Unix-domain-socket oracle daemon for faster reproducible dataset builds
- a first action-conditioned latent dynamics baseline with held-out reconstruction and drift metrics
- the first explicit Phase-7 dataset contract for opponent-reply supervision
- the first trained Phase-7 opponent-head baseline
- the first trained bounded planner arm over a multi-corpus workflow suite
- the first small exact selfplay loop over proposer/opponent/planner contracts
- CI, lint, and test wiring
- architecture and phase documentation

It still does **not** implement:

- full planner-driven runtime inference
- any search or evaluation runtime
- any classical engine/search machinery

## Phase 5 Snapshot

The current Phase-5 stack is intentionally narrow but externally checkable:

- training data can come from exact-rule fixtures, labeled JSONL seeds, or streamed PGN files
- legality, action encoding, and next-state generation still go through the Rust rules oracle
- policy labels for larger runs are generated offline with `/usr/games/stockfish18`
- the current 10k reference corpus was labeled on a Raspberry Pi host and evaluated locally
- the current default symbolic bundle now also carries a native Rust runtime weight file for candidate scoring

Current proposer shape:

- input: `230` packed encoder features
- backbone: `230 -> hidden -> hidden` MLP with ReLU and optional dropout
- heads: two flat `hidden -> 20480` heads for legality and policy

There are now two proposer families:

- `multistream_v2`: unpacks the same `230` features back into typed piece/square/rule streams, applies light cross-attention, then returns to the same flat legality/policy heads
- `factorized_v6`: keeps the stronger conditional factorized legality path and adds explicit policy-side `from-to` coupling plus a low-rank residual
- `relational_v1`: combines the typed multi-stream backbone with the stronger factorized legality/policy heads
- `symbolic_v1`: replaces learned legality with exact legal-move generation and learns only a scorer over legal candidates plus symbolic move features

Reference model sizes:

- `hidden_dim=128`: `5,329,920` parameters
- `hidden_dim=192`: `7,986,688` parameters
- `hidden_dim=256`: `10,651,648` parameters

Reference Phase-5 experiments on the `10,240` train / `2,048` verify Pi-labeled corpus:

- current default entry point: [phase5_stockfish_pgn_symbolic_v1_v1.json](/home/torsten/EngineKonzept/python/configs/phase5_stockfish_pgn_symbolic_v1_v1.json)
- materialized current-default bundle: [stockfish_pgn_symbolic_v1_v1](/home/torsten/EngineKonzept/models/proposer/stockfish_pgn_symbolic_v1_v1)
- materialized current-default summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_symbolic_v1_v1/summary.json)
- best speed/quality trade-off so far: [phase5_stockfish_pgn_pi_10k_bs128_v1.json](/home/torsten/EngineKonzept/python/configs/phase5_stockfish_pgn_pi_10k_bs128_v1.json)
- best verify legal-set F1 so far: [phase5_stockfish_pgn_factorized_v6_v1.json](/home/torsten/EngineKonzept/python/configs/phase5_stockfish_pgn_factorized_v6_v1.json)
- current three-way comparison: [stockfish_pgn_10k_three_way_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_10k_three_way_compare_v1.json)
- current four-way comparison including the structured multi-stream arm: [stockfish_pgn_10k_four_way_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_10k_four_way_compare_v1.json)
- current five-way comparison including the first factorized decoder arm: [stockfish_pgn_10k_five_way_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_10k_five_way_compare_v1.json)
- current six-way comparison including the conditional factorized decoder arm: [stockfish_pgn_10k_six_way_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_10k_six_way_compare_v1.json)
- current seven-way comparison including the policy-stronger conditional decoder arm: [stockfish_pgn_10k_seven_way_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_10k_seven_way_compare_v1.json)
- current nine-way comparison including `factorized_v6` and `relational_v1`: [stockfish_pgn_10k_nine_way_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_10k_nine_way_compare_v1.json)
- current symbolic comparison against the previous learned-legality default and `factorized_v6`: [stockfish_pgn_symbolic_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_symbolic_compare_v1.json)
- direct checkpoint-selection comparison for `factorized_v5`: [stockfish_pgn_factorized_v5_selection_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_factorized_v5_selection_compare_v1.json)
- comparison summary: [stockfish_pgn_pi_10k_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_pi_10k_compare_v1.json)
- oracle transport benchmark: [oracle_transport_bench_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_transport_bench_v1.json)

For dataset naming, treat [phase5_stockfish_pgn_train_pi_10k_v1](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_pgn_train_pi_10k_v1) and [phase5_stockfish_pgn_verify_pi_10k_v1](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_pgn_verify_pi_10k_v1) as the smallest current fully materialized symbolic/dynamics-ready Phase-5 tier. The next larger tier is [phase5_stockfish_merged_unique_train_v1](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_merged_unique_train_v1) plus [phase5_stockfish_merged_unique_verify_v1](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_merged_unique_verify_v1), and the next imported unique-corpus tier is [phase5_stockfish_unique_pi_400k_train_v1](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_unique_pi_400k_train_v1) plus [phase5_stockfish_unique_pi_400k_verify_v1](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_unique_pi_400k_verify_v1). The tracked tier manifest is [phase5_current_corpus_suite_v1.json](/home/torsten/EngineKonzept/artifacts/datasets/phase5_current_corpus_suite_v1.json). The smaller [phase5_stockfish_pgn_train_v1](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_pgn_train_v1), [phase5_stockfish_pgn_verify_v1](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_pgn_verify_v1), [phase5_stockfish_pgn_train_pi_v1](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_pgn_train_pi_v1), and [phase5_stockfish_pgn_verify_pi_v1](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_pgn_verify_pi_v1) remain available as early small-baseline artifacts and regression fixtures.

For the same categorization on configs, model bundles, and Phase-5 summaries, see:

- [python/configs/README.md](/home/torsten/EngineKonzept/python/configs/README.md)
- [models/proposer/README.md](/home/torsten/EngineKonzept/models/proposer/README.md)
- [artifacts/phase5/README.md](/home/torsten/EngineKonzept/artifacts/phase5/README.md)
- [model-roadmap.md](/home/torsten/EngineKonzept/docs/architecture/model-roadmap.md)
- [contracts.md](/home/torsten/EngineKonzept/docs/architecture/contracts.md)
- [search-workflows.md](/home/torsten/EngineKonzept/docs/architecture/search-workflows.md)
- [opponent.md](/home/torsten/EngineKonzept/docs/architecture/opponent.md)
- [gpt-pro-review-brief.md](/home/torsten/EngineKonzept/docs/experiments/gpt-pro-review-brief.md)

The first architecture-extension notes beyond the flat MLP live in [docs/arch.ideas.md](/home/torsten/EngineKonzept/docs/arch.ideas.md). The current implementation applies only the low-risk part of that direction so far: typed multi-stream fusion before considering any heavier routing or expert machinery.

The newer proposer comparison now extends beyond the first three factorized decoder baselines. `factorized_v6` is the current best learned-legality arm on the `10k` corpus by a clear margin, while `relational_v1` improves policy over the earlier factorized runs without taking the policy lead from the old learned-legality default. There is also an explicit checkpoint-selection comparison for `factorized_v5`, showing the expected tradeoff between legality-first and balanced selection. The `symbolic_v1` arm goes one step further and removes learned legality entirely in favor of exact legal-candidate generation; on the current `10k` corpus it is the strongest proposer result so far, and it now carries the official proposer export/runtime contract.

The current `engine-app` binary will use that symbolic proposer bundle automatically when it can find [stockfish_pgn_symbolic_v1_v1](/home/torsten/EngineKonzept/models/proposer/stockfish_pgn_symbolic_v1_v1). Override the bundle location with `ENGINEKONZEPT_PROPOSER_BUNDLE=/path/to/bundle` when needed.

## Phase 6 Snapshot

The proposer is now accepted as a temporary frontier, and the repository includes a checkable latent-dynamics baseline plus larger-corpus Phase-6 follow-ups:

- current Phase-6 config: [phase6_dynamics_merged_unique_structured_v6_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_merged_unique_structured_v6_v1.json)
- current Phase-6 bundle: [dynamics_merged_unique_structured_v6_v1](/home/torsten/EngineKonzept/models/dynamics/dynamics_merged_unique_structured_v6_v1)
- current Phase-6 summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_merged_unique_structured_v6_v1/summary.json)
- current Phase-6 verify eval: [dynamics_merged_unique_structured_v6_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_merged_unique_structured_v6_v1_verify.json)
- direct large-corpus comparison: [dynamics_merged_unique_compare_v2.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_merged_unique_compare_v2.json)
- architecture note: [dynamics.md](/home/torsten/EngineKonzept/docs/architecture/dynamics.md)
- phase note: [phase-6.md](/home/torsten/EngineKonzept/docs/phases/phase-6.md)

The first `v1` run establishes the exact Phase-6 plumbing:

- lean `dynamics_<split>.jsonl` artifacts
- action-conditioned latent transition training
- `torch.export` + Rust metadata validation
- one-step reconstruction and multi-step drift metrics

The current model family is still weak in the exact sense:

- validation and verify feature-reconstruction errors decrease into a stable range
- exact packed next-state accuracy remains `0.0`
- multi-step drift is measurable, but not yet good

The first structured follow-up is now also materialized:

- config: [phase6_dynamics_structured_v2_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_structured_v2_v1.json)
- bundle: [structured_v2_v1](/home/torsten/EngineKonzept/models/dynamics/structured_v2_v1)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v2_v1/summary.json)
- verify: [dynamics_structured_v2_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v2_v1_verify.json)

The explicit drift-selection follow-up established the first useful Phase-6 reference:

- verify `feature_l1_error`: `1.433716 -> 1.425823`
- verify `drift_feature_l1_error`: `1.595053 -> 1.557198`

The smaller-corpus latent-consistency follow-up kept that drift-aware structure and improved both main verify soft metrics again:

- verify `feature_l1_error`: `1.425823 -> 1.425074`
- verify `drift_feature_l1_error`: `1.557198 -> 1.429654`

The next `structured_v3_v1` follow-up adds a delta auxiliary head on top of the same latent-stable path:

- config: [phase6_dynamics_structured_v3_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_structured_v3_v1.json)
- bundle: [structured_v3_v1](/home/torsten/EngineKonzept/models/dynamics/structured_v3_v1)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v3_v1/summary.json)
- verify: [dynamics_structured_v3_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v3_v1_verify.json)

On the smaller `10k` corpus it improves one-step verify reconstruction again, but gives back a little drift quality:

- verify `feature_l1_error`: `1.425074 -> 1.353977`
- verify `drift_feature_l1_error`: `1.429654 -> 1.47778`

The next `structured_v4_v1` follow-up keeps the same `structured_v4` decoder family but adds explicit short-horizon drift supervision during training:

- config: [phase6_dynamics_structured_v4_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_structured_v4_v1.json)
- bundle: [structured_v4_v1](/home/torsten/EngineKonzept/models/dynamics/structured_v4_v1)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v4_v1/summary.json)
- verify: [dynamics_structured_v4_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v4_v1_verify.json)

It does not beat the current default or `structured_v3_v1`:

- verify `feature_l1_error`: `1.425074 -> 1.611914`
- verify `drift_feature_l1_error`: `1.429654 -> 1.49735`

The parallel `edit_v1` arm is also materialized as an experimental counterexample:

- config: [phase6_dynamics_edit_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_edit_v1.json)
- bundle: [edit_v1](/home/torsten/EngineKonzept/models/dynamics/edit_v1)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_edit_v1/summary.json)
- verify: [dynamics_edit_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_edit_v1_verify.json)

It wins one-step reconstruction strongly, but collapses multi-step drift and therefore remains experimental only.

The next `structured_v5_v1` arm connects Phase 6 directly to the symbolic proposer contract by feeding the selected move's exact symbolic candidate features into the transition path:

- config: [phase6_dynamics_structured_v5_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_structured_v5_v1.json)
- bundle: [structured_v5_v1](/home/torsten/EngineKonzept/models/dynamics/structured_v5_v1)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v5_v1/summary.json)
- verify: [dynamics_structured_v5_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v5_v1_verify.json)

On the smaller `10k` corpus it improves one-step held-out reconstruction over the current default, but not drift:

- verify `feature_l1_error`: `1.425074 -> 1.404499`
- verify `drift_feature_l1_error`: `1.429654 -> 1.556962`

The large `merged_unique` reruns change that decision on the `110,570 / 12,286 / 2,169` corpus:

- large `structured_v2_latent_v1`: verify `feature_l1_error=1.067843`, `drift_feature_l1_error=6.305117`
- large `structured_v3_v1`: verify `feature_l1_error=1.02784`, `drift_feature_l1_error=6.18409`
- large `structured_v5_v1`: verify `feature_l1_error=0.924808`, `drift_feature_l1_error=1.548861`
- large `structured_v6_v1`: verify `feature_l1_error=0.923791`, `drift_feature_l1_error=1.464848`

That makes `dynamics_merged_unique_structured_v6_v1` the new preferred Phase-6 reference. The richer `TransitionContextV1` contract now carries through at scale: it edges out the large symbolic-action `structured_v5` run on one-step reconstruction and improves drift materially on the same corpus.

Exact next-state accuracy still remains `0.0`.

## Phase 7 Status

The repository now has the first larger-corpus learned Phase-7 opponent head that beats the symbolic reply-scorer baseline across the current three-tier verify suite:

- architecture note: [opponent.md](/home/torsten/EngineKonzept/docs/architecture/opponent.md)
- phase note: [phase-7.md](/home/torsten/EngineKonzept/docs/phases/phase-7.md)
- workflow artifacts: [README.md](/home/torsten/EngineKonzept/artifacts/phase7/README.md)
- workflow suite: [summary.json](/home/torsten/EngineKonzept/artifacts/phase7/opponent_workflow_corpus_suite_v1/summary.json)
- opponent config: [phase7_opponent_corpus_suite_set_v2_v1.json](/home/torsten/EngineKonzept/python/configs/phase7_opponent_corpus_suite_set_v2_v1.json)
- trained bundle: [corpus_suite_set_v2_v1](/home/torsten/EngineKonzept/models/opponent/corpus_suite_set_v2_v1)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase7/opponent_corpus_suite_set_v2_v1/summary.json)
- verify comparison: [opponent_corpus_suite_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase7/opponent_corpus_suite_compare_v1.json)

The `OpponentHeadV1` workflow now scales over the current three corpus tiers and derives:

- the teacher-chosen root move
- the exact successor state after that move
- the exact legal opponent replies
- symbolic reply features
- teacher best-reply supervision
- first pressure and uncertainty targets

That is the intended bridge between the offline search-workflow layer and an explicit Phase-7 opponent model.

Current larger merged-unique verify result:

- symbolic baseline:
  - `reply_top1_accuracy=0.3`
  - `reply_top3_accuracy=0.4`
  - `teacher_reply_mean_reciprocal_rank=0.419262`
- first trained `mlp_v1` opponent head:
  - `reply_top1_accuracy=0.066667`
  - `reply_top3_accuracy=0.333333`
  - `teacher_reply_mean_reciprocal_rank=0.272664`

Aggregate verify result over the `10k`, `122k`, and `400k` workflow slices:

- symbolic baseline:
  - `reply_top1_accuracy=0.288952`
  - `reply_top3_accuracy=0.524079`
  - `teacher_reply_mean_reciprocal_rank=0.448373`
- learned `set_v2` head:
  - `reply_top1_accuracy=0.368272`
  - `reply_top3_accuracy=0.603399`
  - `teacher_reply_mean_reciprocal_rank=0.521661`

That moves the repo past the old Phase-7 bar: the learned opponent head now beats the symbolic reply scorer on the current multi-corpus holdout.

## Phase 8 Snapshot

Phase 8 now has two useful reference views:

- a full three-tier workflow view over `10k`, `122k`, and `400k`
- a preferred filtered validation view over `10k + 122k`

Full three-tier planner references:

- workflow suite: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_workflow_corpus_suite_v1/summary.json)
- first materialized planner config: [phase8_planner_corpus_suite_set_v2_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v2_v1.json)
- expanded-data planner config: [phase8_planner_corpus_suite_set_v2_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v2_expanded_v1.json)
- repo-copied expanded summary: [planner_corpus_suite_set_v2_expanded_v1_summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v2_expanded_v1_summary.json)
- repo-copied expanded verify eval: [planner_corpus_suite_set_v2_expanded_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v2_expanded_v1_verify.json)
- original direct comparison: [planner_corpus_suite_compare_v2.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_compare_v2.json)

Aggregate verify result over `1,410` held-out planner examples:

- root-only bounded baseline:
  - `root_top1_accuracy=0.153901`
  - `teacher_root_mean_reciprocal_rank=0.230615`
- symbolic-reply bounded baseline:
  - `root_top1_accuracy=0.159574`
  - `teacher_root_mean_reciprocal_rank=0.232861`
- learned-reply bounded baseline:
  - `root_top1_accuracy=0.142553`
  - `teacher_root_mean_reciprocal_rank=0.224232`
- first trained planner `set_v1`:
  - `root_top1_accuracy=0.788652`
  - `root_top3_accuracy=0.958156`
  - `teacher_root_mean_reciprocal_rank=0.872636`
- first materialized planner `set_v2`:
  - `root_top1_accuracy=0.795035`
  - `root_top3_accuracy=0.968085`
  - `teacher_root_mean_reciprocal_rank=0.875355`
  - `teacher_root_mean_probability=0.685788`
- expanded-data `set_v2` validation reference:
  - `best validation root_top1_accuracy=0.813702`
  - `best validation teacher_root_mean_reciprocal_rank=0.891489`
  - held-out per-tier verify stays in the `0.7927 .. 0.8047` top-1 band

There is now also a filtered `10k + 122k` validation slice for planner-facing Phase-6 and planner-architecture checks:

- first latent workflow suite: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_workflow_corpus_suite_latent_two_tier_v1/summary.json)
- faster expanded latent workflow suite: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_workflow_corpus_suite_latent_10k_122k_expanded_v1/summary.json)
- first latent-state config: [phase8_planner_corpus_suite_set_v3_two_tier_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v3_two_tier_v1.json)
- stronger latent-state config: [phase8_planner_corpus_suite_set_v3_10k_122k_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v3_10k_122k_expanded_v1.json)
- first latent comparison: [planner_corpus_suite_two_tier_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_two_tier_compare_v1.json)
- stronger latent comparison: [planner_corpus_suite_latent_two_tier_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_latent_two_tier_compare_v1.json)
- expanded-data comparison: [planner_corpus_suite_expanded_two_tier_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_expanded_two_tier_compare_v1.json)

Result on `1,024` held-out planner examples:

- reference planner `set_v2`: `root_top1_accuracy=0.80957`, `MRR=0.883382`
- latent-state planner `set_v3`: `root_top1_accuracy=0.708008`, `MRR=0.825521`
- expanded-data `set_v2`: `root_top1_accuracy=0.798828`, `MRR=0.87972`
- expanded-data `set_v2_wide`: `root_top1_accuracy=0.790039`, `MRR=0.874837`
- expanded-data `set_v5`: `root_top1_accuracy=0.798828`, `MRR=0.880534`
- `10k + 122k`-only expanded `set_v2`: `root_top1_accuracy=0.819336`, `MRR=0.889811`
- `10k + 122k`-only expanded latent `set_v3`: `root_top1_accuracy=0.797852`, `MRR=0.880778`
- `10k + 122k`-only expanded score-aux `set_v6`: `root_top1_accuracy=0.817383`, `MRR=0.890625`
- `10k + 122k`-only expanded score+margin `set_v6`: `root_top1_accuracy=0.8125`, `MRR=0.889079`
- `10k + 122k`-only expanded recurrent `recurrent_v1`: `root_top1_accuracy=0.805664`, `MRR=0.885742`

The important current conclusion is:

- more mixed workflow data helps the three-tier training/validation picture
- but the real gain on the preferred `10k + 122k` validation slice came only after removing the `400k` tier again during planner training
- faster latent materialization from existing planner-head artifacts is now reproducible and cheap enough to rerun
- but the planner-facing latent channel still does not beat the current filtered `set_v2` reference, even on the stronger `10k + 122k` workflow material
- richer teacher candidate scores are a real signal, but the first score-aux arm still gives back a little `top1` and therefore also does not replace the current filtered `set_v2` reference
- adding explicit `top1-vs-top2/top3` margin supervision stabilizes the score-target losses dramatically, but still does not move the held-out planner ranking above the current filtered `set_v2` reference
- replacing part of that bounded score shaping with discrete `top1 / top2-top3 / tail` rank buckets also does not beat the filtered `set_v2` reference on held-out `top1` or `MRR`
- the first bounded recurrent planner arm is now real and reusable, but it also remains below the filtered `set_v2` reference on held-out `top1` and `MRR`
- wider `set_v2` does not help
- `set_v5` becomes competitive again on the filtered slice, but still does not beat the new `10k + 122k`-only expanded `set_v2`

These are still bounded offline planner artifacts, not runtime search, but Phase 8 is now far enough along to separate data-scale effects from real planner-contract or architecture effects.

## Phase 9 Snapshot

The repository now also has the first small exact selfplay loop:

- runtime helpers: [planner_runtime.py](/home/torsten/EngineKonzept/python/train/eval/planner_runtime.py), [agent_spec.py](/home/torsten/EngineKonzept/python/train/eval/agent_spec.py), [arena.py](/home/torsten/EngineKonzept/python/train/eval/arena.py), [selfplay.py](/home/torsten/EngineKonzept/python/train/eval/selfplay.py)
- script entry points: [run_selfplay.py](/home/torsten/EngineKonzept/python/scripts/run_selfplay.py), [build_replay_buffer.py](/home/torsten/EngineKonzept/python/scripts/build_replay_buffer.py), [run_selfplay_arena.py](/home/torsten/EngineKonzept/python/scripts/run_selfplay_arena.py)
- phase note: [phase-9.md](/home/torsten/EngineKonzept/docs/phases/phase-9.md)
- first probe artifact: [selfplay_set_v2_probe_v1.json](/home/torsten/EngineKonzept/artifacts/phase9/selfplay_set_v2_probe_v1.json)
- first replay artifact: [replay_buffer_set_v2_probe_v1.jsonl](/home/torsten/EngineKonzept/artifacts/phase9/replay_buffer_set_v2_probe_v1.jsonl)
- first arena artifact: [summary.json](/home/torsten/EngineKonzept/artifacts/phase9/arena_active_probe_v1/summary.json)

Current first probe:

- `1` game from `startpos`
- symbolic proposer + learned `OpponentHeadV1` + bounded planner `set_v2_10k_122k_expanded`
- `8` legal plies
- termination reason: `max_plies`

That means Phase 9 now has a real small probe, a real replay-buffer contract, and the first versioned checkpoint arena. Curriculum scheduling is the next layer.

## Repository Layout

```text
.
|-- AGENTS.md
|-- PLANS.md
|-- CODEX_RUNBOOK.md
|-- README.md
|-- docs/
|   |-- architecture/
|   |-- experiments/
|   `-- phases/
|-- rust/
|   |-- Cargo.toml
|   `-- crates/
|-- python/
|   |-- pyproject.toml
|   |-- train/
|   |-- scripts/
|   `-- tests/
|-- tests/
|   |-- perft/
|   |-- planner/
|   `-- positions/
|-- models/
`-- artifacts/
```

## Validation Commands

From the repository root:

```bash
make check
```

Or run the individual commands directly:

```bash
cd rust && cargo fmt --all --check
cd rust && cargo clippy --workspace --all-targets --all-features -- -D warnings
cd rust && cargo test --workspace
python3 -m ruff check python
python3 -m pytest python/tests
```

Representative Phase-5 commands:

```bash
TMPDIR=/srv/schach/tmp .venv/bin/python python/scripts/build_stockfish_pgn_dataset.py --help
TMPDIR=/srv/schach/tmp .venv/bin/python python/scripts/train_legality.py --config python/configs/phase5_stockfish_pgn_symbolic_v1_v1.json
TMPDIR=/srv/schach/tmp .venv/bin/python python/scripts/eval_suite.py --checkpoint models/proposer/stockfish_pgn_symbolic_v1_v1/checkpoint.pt --dataset-path artifacts/datasets/phase5_stockfish_pgn_verify_pi_10k_v1 --split test
cargo run --quiet -p tools --bin dataset-oracle-daemon --socket /tmp/enginekonzept-oracle.sock
ENGINEKONZEPT_DATASET_ORACLE=unix:///tmp/enginekonzept-oracle.sock TMPDIR=/srv/schach/tmp .venv/bin/python python/scripts/build_stockfish_pgn_dataset.py --help
TMPDIR=/srv/schach/tmp .venv/bin/python python/scripts/benchmark_dataset_oracle.py --input tests/positions/edge_cases.txt --source-format edge-cases --records 10000
TMPDIR=/srv/schach/tmp cargo run --quiet -p tools --bin dataset-oracle-profile < /srv/schach/tmp/oracle-e2e/train_raw_2k.jsonl
TMPDIR=/srv/schach/tmp .venv/bin/python python/scripts/materialize_dynamics_artifacts.py --dataset-dir artifacts/datasets/phase5_stockfish_pgn_train_pi_10k_v1
TMPDIR=/srv/schach/tmp .venv/bin/python python/scripts/train_dynamics.py --config python/configs/phase6_dynamics_v1.json
TMPDIR=/srv/schach/tmp .venv/bin/python python/scripts/eval_dynamics.py --checkpoint models/dynamics/v1/checkpoint.pt --dataset-path artifacts/datasets/phase5_stockfish_pgn_verify_pi_10k_v1 --split test --drift-horizon 2
```

## Guardrails

- No alpha-beta, negamax, PVS, quiescence, TT-search, null-move pruning, LMR, or heuristic fallback engine.
- The symbolic chess core is allowed later only for exact rules, labels, tests, and final move safety checks.

If classical search methods are introduced for experiments, use them only under the boundary defined in [search-workflows.md](/home/torsten/EngineKonzept/docs/architecture/search-workflows.md): benchmark, teacher, analysis, and curriculum workflows are acceptable; runtime move selection is not.
- The current exact rules core is intentionally isolated from any runtime search logic.
- The current UCI shell is protocol-only and must not accrete classical engine behavior.
- The current action space and encoder are deterministic schema layers, not learned components.
- Runtime and protocol code belong in Rust.
- Training, datasets, and experiment code belong in Python.
- Every phase must leave the tree buildable, testable, and documented.

See [AGENTS.md](/home/torsten/EngineKonzept/AGENTS.md), [PLANS.md](/home/torsten/EngineKonzept/PLANS.md), and [CODEX_RUNBOOK.md](/home/torsten/EngineKonzept/CODEX_RUNBOOK.md) for the binding project rules.
