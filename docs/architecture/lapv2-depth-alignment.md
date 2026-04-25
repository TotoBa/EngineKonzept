# ExecPlan: LAPv2 Depth Alignment

## Status

Started `2026-04-25`.

Step 1 is implemented in the trainer:

- `optimization.deliberation_rank_progress_weight`
- `deliberation_rank_progress_loss`
- `step_rank_improved_rate`
- `step_rank_degraded_rate`
- `mean_step_rank_delta`

Step 2 is implemented in the trainer:

- `optimization.deliberation_step_utility_weight`
- `deliberation_step_utility_loss`
- `step_utility_continue_rate`
- `step_utility_predicted_continue_rate`

The current external-focus runs show a consistent pattern: internal LAPv2
metrics keep improving, but arena strength against `stockfish18_skill_00` and
`vice_v2` does not move reliably. The next work therefore targets the learned
deliberation contract itself, not a classical runtime search layer.

## Goal

Make deeper LAPv2 inner steps useful by construction:

- every additional step should be trained against teacher-rank progress
- the model should learn when more compute is worth spending
- frontier candidates should keep enough diversity to avoid collapsing onto
  internally comfortable but externally weak moves
- external-arena examples should keep higher priority without becoming the only
  signal

The success criterion is not just higher validation top-1. A step is useful only
if it improves teacher rank, avoids degrading already-correct root choices, and
eventually moves the external arena numbers.

## Non-Goals

- no alpha-beta, negamax, PVS, quiescence, TT search, or handcrafted eval
- no MCTS/PUCT tree at runtime
- no hidden symbolic fallback for move choice
- no interruption of the currently running training cluster from this plan

## Step Plan

1. Add a step-rank-progress training loss and metrics. **Implemented.**
   - Compare each active inner step against the best teacher rank seen so far.
   - Penalize root-incorrect examples unless later steps improve the teacher CE
     by a small margin.
   - Penalize root-correct examples when deeper steps degrade the teacher move.
   - Emit train/validation metrics so bad depth behavior is visible immediately.

2. Train halting against realized step value. **Implemented.**
   - Derive a continuation target from whether the next step improved teacher
     rank or avoided degradation.
   - Make sharpness/halting learn "continue because the next step helps", not
     just "continue because the position looks complex".

3. Add depth-conditioned update control.
   - Give the recurrent cell an explicit learned depth embedding.
   - Let early steps explore and later steps consolidate without relying on
     implicit memory state alone.

4. Make frontier diversity a guarded objective.
   - Penalize premature collapse when all frontier slots chase the same move.
   - Keep a small amount of novelty pressure only when teacher rank is not yet
     good.

5. Add an external-hard validation slice.
   - Report external/arena-origin rows separately from the blended validation
     set.
   - Use those metrics for model selection only after they are stable enough not
     to amplify noise.

## Implementation Discipline

Each step must be implemented independently:

- code change
- focused tests
- documentation update
- commit and push
- then reassess before the next step

## Risks

- Strong step-progress pressure may overfit to the teacher top-1 and reduce
  policy calibration.
- Too much anti-degradation pressure can make the loop conservative and remove
  useful exploration.
- External-arena data is sparse and noisy; weighting it too aggressively can
  distort the broader position distribution.
- Better internal depth metrics may still fail externally, which would point to
  missing state representation or opponent modeling rather than loss shaping.
