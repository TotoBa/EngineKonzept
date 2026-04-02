# Dataset Architecture

Phase 4 adds a reproducible dataset pipeline that keeps label generation anchored to the exact symbolic rules kernel.

## Raw Source Formats

The Python builder currently accepts four source formats:

- `edge-cases`: `name|fen` lines such as [edge_cases.txt](/home/torsten/EngineKonzept/tests/positions/edge_cases.txt)
- `fen-lines`: one FEN per line, optionally `name|fen`
- `epd`: synthetic suites using the first four EPD fields, normalized to full FEN with `0 1`
- `jsonl`: structured records with optional `selected_move_uci`, `result`, `source`, and `metadata`

For small reproducible Phase-5 policy runs, the repository also carries a labeled JSONL seed set at [policy_seed.jsonl](/home/torsten/EngineKonzept/tests/positions/policy_seed.jsonl).

## Exact Oracle Boundary

Labels are not reimplemented in Python.

Instead, Python sends raw records to the Rust `dataset-oracle` tool, which reuses:

- `position` for exact FEN parsing and state tracking
- `rules` for legal move generation, checks, and next-state application
- `action-space` for factorized move labels
- `encoder` for deterministic model-facing position tokens

This keeps dataset labels aligned with the runtime legality authority.

For policy-supervised experiments, Python may also derive `selected_move_uci` labels from external analysis engines during offline dataset generation. The current Phase-5 utility supports bounded PGN sampling with Stockfish 18 while still routing legality, action encoding, and next-state generation back through the exact Rust oracle.

The current larger reference corpus was produced by streaming PGNs on a separate Raspberry Pi host and labeling candidate positions there with `/usr/games/stockfish18` at a fixed `1500`-node budget per position. This keeps the label semantics reproducible across machines while moving the slowest offline work off the main development host.

The Python oracle client now also supports a local Unix-domain-socket daemon in addition to the original subprocess mode. Both transports preserve the same newline-delimited JSON request/response contract and the same Rust labeling logic.

## Example Schema

Each emitted example includes:

- raw position identity: `sample_id`, `source`, `fen`, `side_to_move`
- optional selected action supervision:
  - `selected_move_uci`
  - `selected_action_encoding`
  - `next_fen`
- legality labels:
  - `legal_moves`
  - `legal_action_encodings`
- deterministic position encoding:
  - `position_encoding.piece_tokens`
  - `position_encoding.square_tokens`
  - `position_encoding.rule_token`
- WDL target when available:
  - from supplied game `result`, mapped relative to side to move
  - or from immediate terminal surrogate on checkmate/stalemate positions
- tactical annotations:
  - `in_check`
  - `is_checkmate`
  - `is_stalemate`
  - `has_legal_en_passant`
  - `has_legal_castle`
  - `has_legal_promotion`
  - `is_low_material_endgame`
  - selected-move tactical flags when a selected move is present

## Splits and Reporting

Splits are assigned deterministically from `sample_id` and a user-provided seed.

The builder writes:

- `dataset.jsonl`
- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`
- `summary.json`

When a build is invoked with `--write-proposer-artifacts`, it also writes lean proposer-ready split files:

- `proposer_train.jsonl`
- `proposer_validation.jsonl`
- `proposer_test.jsonl`

These files contain:

- `sample_id`
- `split`
- packed `feature_vector`
- flattened `legal_action_indices`
- optional flattened `selected_action_index`

The full dataset artifacts remain the canonical review/debug path. The lean proposer files are an optional training convenience path for the Phase-5 proposer only. They keep the existing dataset semantics but avoid reparsing full `DatasetExample` payloads when the trainer only needs packed features plus legality/policy supervision.

When using PGN/Stockfish labeling with `--raw-output-dir`, the builder also writes:

- `train_raw.jsonl`
- `verify_raw.jsonl`
- `selection_summary.json`

The dataset build scripts now also expose offline throughput knobs for the Rust oracle path:

- `--oracle-workers`: number of concurrent oracle calls
- `--oracle-batch-size`: records per oracle call before splitting into multiple batches

These affect only offline dataset generation. They do not change label semantics or any runtime engine path.

The proposer trainer now prefers `proposer_<split>.jsonl` when those files are present in a dataset directory. If they are absent, it falls back to the existing full split artifacts and packs features on load, so older datasets remain valid.

When `oracle_workers > 1` and `oracle_batch_size == 0`, the builder now auto-splits the workload into roughly one batch per worker instead of falling back to a single giant batch. On a 2000-record build with `4` workers, that reduced wall-clock time from about `1.394s` to about `1.046s`, or about `1.33x` faster, with identical output digests. That measurement is stored in [oracle_auto_batch_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_auto_batch_v1.json).

For larger offline builds, the repository now also includes a reproducible sweep runner:

- `python/scripts/benchmark_dataset_build.py`

It benchmarks multiple `(workers, batch_size)` schedules against the same raw dataset, writes full artifacts for each config, and verifies that the emitted `dataset.jsonl` / split files remain identical across the sweep. The first 10k reference run is stored in [oracle_build_sweep_10k_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_build_sweep_10k_v1.json).

Both benchmark helpers now also record basic runtime metadata directly in their JSON output:

- `hostname`
- `platform`
- `python_version`
- `cpu_count`

That keeps cross-host comparisons externally checkable without requiring a separate roll-up step to explain where a measurement came from.

That 10k run currently shows:

- `w4_b500`: about `4.934s`
- `auto_w4`: about `5.204s`
- `w2_auto`: about `6.371s`
- `w1_single`: about `6.810s`

Repeated pairwise reruns are now also stored in:

- [oracle_pair_10k_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_pair_10k_v1.json)
- [oracle_pair_20k_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_pair_20k_v1.json)

Those pairwise runs show `w4_b500` ahead of the old auto point on both sizes:

- 10k: about `6.222s` vs. `6.378s`
- 20k: about `10.643s` vs. `13.336s`

Because that advantage now survives repeated pairwise runs, the default auto heuristic caps the per-batch size at `500` when `oracle_workers > 1`.

A follow-up host check against `raspberrypi` is stored in:

- [oracle_pair_10k_pi_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_pair_10k_pi_v1.json)
- [oracle_pair_20k_pi_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_pair_20k_pi_v1.json)
- [oracle_host_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_host_compare_v1.json)
- [oracle_pair_50k_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_pair_50k_v1.json)
- [oracle_pair_50k_pi_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_pair_50k_pi_v1.json)
- [oracle_host_compare_v2.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_host_compare_v2.json)

That roll-up is now generated reproducibly by:

- `python/scripts/compare_dataset_build_benchmarks.py`

That comparison is mainly a regression check for the new default on the slower host. Because `auto_w4` already resolves to `effective_batch_size = 500` under the new heuristic, the Pi run is not a second old-vs-new comparison; it is a confirmation that the capped default remains viable off the local workstation. On the 20k run, `w4_b500` still lands slightly ahead (`29.226s` vs. `31.176s`), while the 10k difference is small enough to treat as noise on that host.

The newer 50k sweep keeps that conclusion intact on both hosts:

- local `50k`: `w4_b500` at about `26.547s`, `auto_w4` at about `26.577s`, `w2_auto` at about `29.535s`
- `raspberrypi` `50k`: `auto_w4` at about `76.293s`, `w4_b500` at about `76.330s`, `w2_auto` at about `88.787s`

So the current evidence is now stable across `10k`, `20k`, and `50k`:

- keeping `4` workers is still clearly better than dropping to `2`
- the `500` cap remains the right default guardrail
- once the auto heuristic resolves to the same `500`-sized batches, explicit `w4_b500` and `auto_w4` are effectively the same schedule and should be treated as measurement noise rather than separate operating points

After the move-label-path optimization in [oracle_pair_50k_encode_v4.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_pair_50k_encode_v4.json), the same 50k host comparison was rerun on `raspberrypi` and stored in [oracle_pair_50k_pi_v4.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_pair_50k_pi_v4.json). The corresponding regenerated roll-up is [oracle_host_compare_v4.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_host_compare_v4.json).

That follow-up keeps the same conclusion while confirming that the `v4` move-label-path gain is not just a workstation artifact:

- local `50k` after `v4`: `auto_w4` at about `23.149s`, `w4_b500` at about `23.861s`
- `raspberrypi` `50k` after `v4`: `auto_w4` at about `74.679s`, `w4_b500` at about `76.729s`

Both hosts now resolve `auto_w4` to the same effective `500`-record batch size at `50k`, and both still produce identical output digests across the two schedule labels.

When the Python wrapper uses the one-shot subprocess oracle path, it now prefers a prebuilt local binary at `rust/target/debug/dataset-oracle` before falling back to `cargo run`. On a warmed 250-record one-shot benchmark, that reduced wall-clock time from about `0.117s` to about `0.076s`, or about `1.53x` faster, with identical output digests. That measurement is stored in [oracle_one_shot_binary_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_one_shot_binary_v1.json).

Current end-to-end measurement on a 2000-record JSONL build:

- serial daemon vs. subprocess: about `1.03x`
- `4` oracle workers with batch size `250` vs. serial daemon: about `1.03x`
- `4` oracle workers with batch size `250` vs. subprocess: about `1.06x`

The corresponding artifact-backed measurement is stored in [oracle_e2e_parallel_bench_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_e2e_parallel_bench_v1.json).

After removing a duplicate legal-move generation pass inside the Rust oracle, the same 2000-record build improved again to about `2.03s`, which is:

- about `1.17x` faster than the previous parallel build
- about `1.24x` faster than the original subprocess baseline

That measurement is stored in [oracle_e2e_hotpath_opt_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_e2e_hotpath_opt_v1.json).

After teaching the oracle to skip redundant re-validation when applying a move that is already known legal, the same 2000-record build dropped again to about `1.62s`, which is:

- about `1.25x` faster than the previous hot-path result
- about `1.55x` faster than the original subprocess baseline

That measurement is stored in [oracle_e2e_applyopt_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_e2e_applyopt_v1.json).

To keep the next optimization steps externally checkable, the tooling now also includes a profiling-only oracle binary:

- `cargo run --quiet -p tools --bin dataset-oracle-profile`

It accepts the same newline-delimited request stream as `dataset-oracle` but reports aggregated phase timings instead of labels. The initial hotspot profile is stored in [oracle_profile_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v1.json), the post-serialization profile is stored in [oracle_profile_v2.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v2.json), the post-check-path profile is stored in [oracle_profile_v3.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v3.json), the first fine-grained split is stored in [oracle_profile_v4.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v4.json), the board-snapshot profile is stored in [oracle_profile_v5.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v5.json), the king-square profile is stored in [oracle_profile_v6.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v6.json), the custom-json profile is stored in [oracle_profile_v7.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v7.json), the attack-split profile is stored in [oracle_profile_v8.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v8.json), and the current local-attack profile is stored in [oracle_profile_v9.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v9.json).

The first profile showed:

- `legal_generation`: about `48.7%`
- `json_serialize`: about `32.3%`
- `legal_action_encoding`: about `4.5%`

That directly motivated the next output-path change: the oracle now writes JSON responses straight into the stream instead of first materializing a `String` per record. On the same 2000-record daemon build, wall-clock time dropped from about `1.624s` to about `1.555s`, with byte-identical artifacts. That measurement is stored in [oracle_e2e_jsonopt_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_e2e_jsonopt_v1.json).

For larger reproducible profiling runs, the repository now also includes:

- `python/scripts/profile_dataset_oracle.py`

It runs `dataset-oracle-profile` on a deterministic raw-record slice, records the command and host metadata used for the run, and writes the full JSON report plus a ranked `top_phases` summary. The current large reference run is stored in [oracle_profile_50k_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_50k_v1.json).

That 50k profile keeps the same overall picture, but with a useful large-run nuance:

- `legal_generation`: about `49.4%`
- `self_check_filter`: about `38.6%`
- `json_serialize`: about `23.4%`
- `attack_check_local`: about `18.2%`
- `attack_check_slider`: about `15.2%`

So on larger offline builds, legality remains the primary target, but JSON serialization is still large enough that it cannot be ignored once the obvious rules-kernel wins have been taken.

After that step, the updated profile shifted to:

- `legal_generation`: about `57.0%`
- `json_serialize`: about `20.7%`
- `legal_action_encoding`: about `5.2%`

The next low-risk rules-kernel step trimmed the move-application path used only for the self-check filter in `legal_moves`. That path now updates only the board state needed for `is_in_check`, not the full metadata state used for committed next positions. On the same 2000-record daemon benchmark, two back-to-back runs landed at about `1.547s` and `1.538s`, both below the previous `1.555s` jsonopt baseline. That measurement is stored in [oracle_e2e_checkpath_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_e2e_checkpath_v1.json).

After that step, the profile tightened again to roughly:

- `legal_generation`: about `56.0%`
- `json_serialize`: about `21.2%`
- `legal_action_encoding`: about `5.3%`

That is the current guide for further throughput work. In other words, the builder and daemon transport are no longer the main pressure, JSON serialization is no longer the obvious second target, and the dominant remaining work is still exact legal move generation inside the Rust oracle.

The finer split in [oracle_profile_v4.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v4.json) made that more concrete:

- `pseudo_legal_generation`: about `6.0%`
- `self_check_filter`: about `49.8%`

That led to the next rules-kernel change: the self-check filter now mutates a board snapshot instead of cloning a full `Position` for each pseudo-legal candidate. On the same 2000-record daemon benchmark, wall-clock time dropped from about `1.542s` to about `1.443s` on average across two runs. That measurement is stored in [oracle_e2e_boardcheck_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_e2e_boardcheck_v1.json).

After that change, [oracle_profile_v5.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v5.json) shifted to roughly:

- `legal_generation`: about `47.8%`
- `self_check_filter`: about `40.9%`
- `json_serialize`: about `25.5%`

So the current picture is better balanced: legality remains the largest block, but the dominant part of it has already been materially reduced.

The next refinement reused the already known king square inside the self-check filter instead of re-scanning the board after each candidate move. On the same 2000-record daemon benchmark, wall-clock time dropped again from about `1.443s` to about `1.375s` on average across two runs. That measurement is stored in [oracle_e2e_kingsquare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_e2e_kingsquare_v1.json).

After that change, [oracle_profile_v6.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v6.json) shifted to roughly:

- `legal_generation`: about `42.7%`
- `self_check_filter`: about `34.1%`
- `json_serialize`: about `28.4%`

So the current picture is tighter still: the legality path has been reduced enough that output serialization is again a comparatively important secondary cost.

The next low-risk step addressed exactly that secondary cost: the oracle now uses a specialized schema-identical JSON writer for `DatasetOracleOutput`. A focused unit test checks byte equality against `serde_json` on a representative labeled record. On the same 2000-record daemon benchmark, wall-clock time dropped from about `1.375s` to about `1.358s` on average across two runs. That measurement is stored in [oracle_e2e_customjson_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_e2e_customjson_v1.json).

After that change, [oracle_profile_v7.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v7.json) shifted to roughly:

- `legal_generation`: about `47.9%`
- `self_check_filter`: about `38.4%`
- `json_serialize`: about `19.8%`

So the current picture is now cleaner again: the legality path is once more the clear primary target, and the output path is materially cheaper than it was before the custom writer.

The newest small JSON-path refinement adds a fast ASCII string path to the specialized writer and falls back to `serde_json` only when escaping is actually needed. That keeps byte-for-byte compatibility for quoted, backslash-containing, or control-character strings, but avoids the generic serializer on the dominant FEN/UCI/field-name path. The latest large-build reference is stored in [oracle_pair_50k_json_v2.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_pair_50k_json_v2.json), where the same 50k build lands at about:

- `auto_w4`: `26.052s` vs. `26.577s` in the previous 50k reference
- `w4_b500`: `26.424s` vs. `26.547s` in the previous 50k reference

The matching large-run profile is stored in [oracle_profile_50k_v2.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_50k_v2.json). Its absolute timings still move around enough that the profile should be treated as directional rather than exact for this micro-step, but it keeps the same priority order: `legal_generation` first, `json_serialize` still a real secondary cost.

The newest attack-check refinement is purely profiling-oriented: it does not change legality semantics, but it splits the remaining attack path into:

- `attack_check_pawn`
- `attack_check_knight`
- `attack_check_king`
- `attack_check_bishop_ray`
- `attack_check_rook_ray`

The current 50k reference is [oracle_profile_50k_v3.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_50k_v3.json). On that run, the remaining attack-check cost is led by:

- `attack_check_rook_ray`: about `9.1%`
- `attack_check_king`: about `8.8%`
- `attack_check_knight`: about `8.1%`
- `attack_check_bishop_ray`: about `7.2%`
- `attack_check_pawn`: about `2.5%`

So the remaining self-check pressure is not pawn-dominated. The next likely rules-side wins, if any remain, are king-local checks, knight-local checks, and rook-ray scans.

The next profiling-and-throughput step tightened the move-label path itself. The oracle now:

- reuses the already formatted legal UCI strings for `selected_move_resolution` instead of calling `to_uci()` again for each candidate
- splits `legal_action_encoding` into `legal_action_encode` and `legal_action_sort`

The current large-run profile is [oracle_profile_50k_v4.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_50k_v4.json). On that run:

- `selected_move_resolution` drops to about `0.7%`
- `legal_action_encode` lands at about `3.9%`
- `legal_action_sort` lands at about `4.6%`

The matching end-to-end build result is stored in [oracle_pair_50k_encode_v4.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_pair_50k_encode_v4.json). Compared with the previous 50k reference:

- `auto_w4`: `23.149s` vs. `26.052s`
- `w4_b500`: `23.861s` vs. `26.424s`

That is the clearest recent large-run win after the rules-kernel changes: avoiding duplicate UCI formatting in `selected_move_resolution` and making the encoding path more observable produces a real throughput improvement without changing label semantics.

The same profiled `50k` run was then mirrored on `raspberrypi` and fetched back as [oracle_profile_50k_pi_v4.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_50k_pi_v4.json). That host-side profile keeps the same priority order:

- `legal_generation`: about `51.5%`
- `self_check_filter`: about `43.0%`
- `attack_check_local`: about `21.7%`
- `json_serialize`: about `17.9%`
- `attack_check_slider`: about `16.8%`

So the next optimization choice is now host-stable as well: legality generation remains the first target, but JSON serialization is still large enough on the slower ARM host to justify another serializer pass before deeper rules-kernel work.

To make the serializer itself more directly inspectable, the large local profile now also records JSON subsection byte shares in [oracle_profile_50k_v10.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_50k_v10.json). On that run, the emitted bytes break down as:

- `position_encoding`: about `36.3%`
- `annotations`: about `19.2%`
- `legal_action_encodings`: about `16.9%`
- `top_level`: about `15.3%`
- `legal_moves`: about `12.2%`

That does not replace the time-based profile, but it does answer a previously ambiguous question: the largest JSON payload pressure is not the legal-move list, it is the encoded feature payload.

The next split in [oracle_profile_50k_v12.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_50k_v12.json) breaks `position_encoding` down one level further. Within that payload, `square_tokens` clearly dominates by bytes, with `piece_tokens` and `rule_token` far behind. A first focused writer variant for `square_tokens` was benchmarked in [oracle_pair_50k_squaretokens_v13.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_pair_50k_squaretokens_v13.json), but it regressed the real `50k` build, so it was discarded. The takeaway is narrower now: `square_tokens` is the biggest payload block, but not every attempt to buffer it more aggressively improves end-to-end throughput.

The next local profile [oracle_profile_50k_v15.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_50k_v15.json) splits `annotations` into three payload groups:

- `core_flags`
- `count_fields`
- `selected_move_fields`

On the current `50k` run, `core_flags` and `selected_move_fields` are very close in byte share, while the numeric count fields are much smaller. A first annotations-specific buffered writer was benchmarked in [oracle_pair_50k_annotations_v16.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_pair_50k_annotations_v16.json), but it also regressed the real `50k` build and was discarded. That keeps the current conclusion consistent: the remaining serializer work is measurable, but the obvious "buffer the whole section" variants are not producing real throughput wins.

The next profiling refinement split the remaining self-check attack cost into local attackers and slider attackers. The current result in [oracle_profile_v8.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v8.json) is:

- `attack_check_local`: about `23.0%`
- `attack_check_slider`: about `12.5%`

So the next meaningful target is now narrower still: pawn/knight/king attack detection is more expensive than the ray scans in the remaining attack-check path.

That directly motivated the next rules-kernel change: the local pawn, knight, and king attack checks now use direct board-index arithmetic instead of `Square::offset`-driven coordinate reconstruction. On the same 2000-record daemon benchmark, wall-clock time dropped from about `1.358s` to about `1.343s` on average across two runs. That measurement is stored in [oracle_e2e_localattack_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_e2e_localattack_v1.json).

After that change, [oracle_profile_v9.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v9.json) shifted to roughly:

- `legal_generation`: about `44.8%`
- `self_check_filter`: about `34.7%`
- `attack_check_local`: about `16.3%`
- `attack_check_slider`: about `13.8%`
- `json_serialize`: about `21.3%`

So the remaining attack-check cost is now more balanced than before: the local attacker path is cheaper, but legality generation as a whole is still the dominant block.

The summary reports:

- split counts
- source counts
- available WDL counts
- annotation coverage for checks, mate, stalemate, castling, en passant, promotion, and low-material endgames
- legal-move-count and piece-count statistics
- the effective offline oracle schedule in `oracle_schedule`, including worker count, requested batch size, effective batch size, and batch count

## Current Limits

- PGN ingestion is bounded, offline-only, and intended for dataset generation rather than runtime support
- no selfplay yet
- no policy probabilities yet
- no learned planner targets yet
- `is_low_material_endgame` is intentionally a conservative proxy based on total piece count, used only for Phase-4 reporting
