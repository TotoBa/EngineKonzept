## External Benchmark Curriculum Pressure

### Goal
Bias Phase-10 LAP training toward the positions that matter for the current
external bottleneck: arena games against `stockfish18_skill_*` and `vice_*`.

### Non-goals
- no classical search fallback
- no arena scheduling changes
- no runtime-only hack that bypasses training

### Pressure
Recent generations improve internal validation and verify metrics, but external
arena results remain flat:
- `stockfish18_skill_00`: still only a small draw signal
- `vice_v2`: still no points

The current workflow already carries useful metadata from arena PGNs, but that
metadata is not yet converted into stronger curriculum pressure inside LAP
training.

### Plan
1. Detect external arena feedback rows inside the `dataset -> planner_head`
   bridge using existing raw metadata.
2. Boost `curriculum_priority` and add explicit bucket labels for:
   - external arena feedback
   - stockfish benchmark rows
   - vice benchmark rows
   - non-win and loss recovery slices against those opponents
3. Make LAP training consume `curriculum_priority` as sample weight instead of
   treating it as summary-only metadata.
4. Keep the weighting bounded with `log1p` normalization so the curriculum does
   not collapse around a tiny set of rows.
5. Add focused tests for:
   - metadata-driven external boosts
   - curriculum-weighted LAP losses

### Expected outcome
Future generations should spend more effective optimization budget on the
positions that come from failing or barely-holding external arena games, while
remaining within the latent-planning architecture.
