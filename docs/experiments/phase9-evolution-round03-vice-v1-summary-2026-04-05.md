# Phase 9 Evolution Round03 Vice V1 Summary

This note summarizes the completed continuation run under:

- [/srv/schach/engine_training/phase9/evolution_round03_vice_v1](/srv/schach/engine_training/phase9/evolution_round03_vice_v1)
- config: [phase9_evolution_round03_vice_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_evolution_round03_vice_v1.json)

The run resumes from the latest stable checkpoint frontier of the interrupted prior campaign:

- seed summary: [round_03/summary.json](/srv/schach/engine_training/phase9/evolution_fullmatrix_custom_v1/iterations/round_03/summary.json)

Unlike the previous long run, this continuation:

- skips the original `start` and `after_fulltrain` blocks
- starts directly from `bootstrap`
- injects `vice` as an offline arena opponent every round
- runs `10` selfplay/review/retrain rounds

The final aggregate report is:

- [final_report.json](/srv/schach/engine_training/phase9/evolution_round03_vice_v1/final_report.json)

## Evaluation frame

The run contains two different arm classes:

1. static benchmark arms
   - `planner_active_expanded_v2`
   - `planner_set_v6_replay_expanded_v2`
   - `symbolic_root_v1`
   - `vice_v2`
2. trainable arms
   - `planner_set_v2_expanded_v1`
   - `planner_set_v2_wide_expanded_v1`
   - `planner_set_v5_expanded_v1`
   - `planner_set_v6_expanded_v1`
   - `planner_set_v6_margin_expanded_v1`
   - `planner_set_v6_rank_expanded_v1`
   - `planner_recurrent_expanded_v1`
   - `planner_moe_v1_expanded_v1`
   - `planner_moe_v2_expanded_v1`

This distinction matters for interpretation:

- the static benchmark arms should not drift on held-out verify
- only the trainable arms are expected to react to the `vice`-conditioned selfplay correction loop

## Stage winners

### Verify leaders by stage

- `bootstrap`: `planner_moe_v1_expanded_v1` with `top1=0.817730`, `MRR=0.891667`
- `round_01`: `planner_set_v2_expanded_v1` with `top1=0.814894`
- `round_02`: `planner_set_v6_margin_expanded_v1` with `top1=0.814184`
- `round_03`: `planner_recurrent_expanded_v1` with `top1=0.819858`, `MRR=0.894090`
- `round_04`: `planner_active_expanded_v2` with `top1=0.813475`
- `round_05`: `planner_set_v6_margin_expanded_v1` with `top1=0.814894`
- `round_06`: `planner_active_expanded_v2` with `top1=0.813475`
- `round_07`: `planner_recurrent_expanded_v1` with `top1=0.819858`
- `round_08`: `planner_active_expanded_v2` with `top1=0.813475`
- `round_09`: `planner_active_expanded_v2` with `top1=0.813475`
- `round_10`: `planner_recurrent_expanded_v1` with `top1=0.819858`
- `final`: `planner_recurrent_expanded_v1` with `top1=0.819858`, `MRR=0.891312`

### Arena leaders by stage

- every stage was won by `vice_v2`
- `score_rate` was `1.0` in rounds `01-07` and `09`
- `score_rate` was `0.979167` in `round_08`, `round_10`, and `final`

That means the external rung is still far above the current learned family and should remain a benchmark, not a progression gate to stronger foreign engines yet.

## Final arm assessment

### Best trainable arm overall

The clearest final winner is:

- `planner_recurrent_expanded_v1`

Why:

- final held-out verify leader:
  - `root_top1_accuracy=0.819858`
  - `teacher_root_mean_reciprocal_rank=0.891312`
- final internal arena leader among trainable arms:
  - `15.0 / 24`
  - `score_rate=0.625`

It is the only trainable arm that finishes on top of both:

- the final verify table
- the final internal arena table

### Final trainable verify ranking

1. `planner_recurrent_expanded_v1`: `0.819858 / 0.891312`
2. `planner_set_v6_expanded_v1`: `0.809220 / 0.886407`
3. `planner_set_v6_margin_expanded_v1`: `0.807092 / 0.884870`
4. `planner_set_v6_rank_expanded_v1`: `0.805674 / 0.885225`
5. `planner_set_v2_expanded_v1`: `0.803546 / 0.884811`
6. `planner_set_v2_wide_expanded_v1`: `0.800709 / 0.878251`
7. `planner_moe_v1_expanded_v1`: `0.800709 / 0.880260`
8. `planner_moe_v2_expanded_v1`: `0.795035 / 0.880496`
9. `planner_set_v5_expanded_v1`: `0.786525 / 0.874409`

### Final trainable arena ranking

1. `planner_recurrent_expanded_v1`: `15.0 / 24`, `0.625000`
2. `planner_set_v2_expanded_v1`: `12.5 / 24`, `0.520833`
3. `planner_moe_v2_expanded_v1`: `12.5 / 24`, `0.520833`
4. `planner_set_v6_expanded_v1`: `11.5 / 24`, `0.479167`
5. `planner_set_v6_rank_expanded_v1`: `11.5 / 24`, `0.479167`
6. `planner_set_v5_expanded_v1`: `11.0 / 24`, `0.458333`
7. `planner_set_v2_wide_expanded_v1`: `10.5 / 24`, `0.437500`
8. `planner_set_v6_margin_expanded_v1`: `10.5 / 24`, `0.437500`
9. `planner_moe_v1_expanded_v1`: `9.5 / 24`, `0.395833`

## Architecture conclusions

### Recurrent arm

`planner_recurrent_expanded_v1` is the strongest current direction.

It improved the most on held-out verify among the serious contenders:

- `bootstrap top1=0.807092`
- `final top1=0.819858`
- delta `+0.012766`

MRR also improved materially:

- `0.885638 -> 0.891312`
- delta `+0.005674`

This is the most convincing evidence in the run that Phase-9 recurrence is beginning to pay off under selfplay correction.

### MoE arms

The MoE picture is mixed but directionally useful:

- `planner_moe_v1_expanded_v1` started as the bootstrap verify leader
- but degraded over the 10-round loop:
  - `top1: 0.817730 -> 0.800709`
  - `MRR: 0.891667 -> 0.880260`
- it also finished last among the trainable arms in internal arena score rate

`planner_moe_v2_expanded_v1` was more robust than `moe_v1` in arena:

- final arena tie for second among trainables at `12.5 / 24`

but still weak on final verify:

- `top1=0.795035`

Current conclusion:

- `moe_v1` is not stable under the current selfplay correction loop
- `moe_v2` is more survivable in arena but still not a promotion candidate
- MoE should remain experimental until routing survives repeated selfplay-retrain cycles better

### Set-v6 family

The `set_v6` family split cleanly:

- plain `set_v6` ended as the best non-recurrent trainable verify arm
- `set_v6_margin` briefly won verify rounds, but decayed by the end
- `set_v6_rank` no longer looked like the strongest selfplay arm once `vice` was mixed in

So the useful conclusion is:

- keep `set_v6` as the strongest feed-forward trainable baseline
- do not treat `set_v6_margin` or `set_v6_rank` as clearly superior follow-ups at this point

## Vice benchmark conclusion

`vice_v2` remains clearly stronger than the whole learned family.

Final result:

- `23.5 / 24`
- `23` wins
- `1` draw
- `0` losses

The one final non-loss for the learned side was:

- `vice_v2` as White vs `planner_recurrent_expanded_v1` as Black
- result `1/2-1/2`
- termination `threefold_repetition`

Everything else in the final `vice` ladder was lost, mostly by checkmate.

That means:

- the external benchmark is useful
- but the repo is not ready to progress to stronger external rung engines yet

## Selfplay training signal

All trainable arms received substantial correction volume over the 10 rounds.

Total `planner_head` mistake rows by arm:

- `planner_set_v6_margin_expanded_v1`: `7132`
- `planner_set_v2_expanded_v1`: `7119`
- `planner_set_v6_rank_expanded_v1`: `6965`
- `planner_moe_v2_expanded_v1`: `6959`
- `planner_recurrent_expanded_v1`: `6880`
- `planner_set_v6_expanded_v1`: `6868`
- `planner_set_v2_wide_expanded_v1`: `6833`
- `planner_moe_v1_expanded_v1`: `6801`
- `planner_set_v5_expanded_v1`: `6561`

So this was not a weak data regime. The relative outcomes are informative.

## Recommended next step

The next step should center the recurrent arm.

Recommended order:

1. promote `planner_recurrent_expanded_v1` to the main active internal planner reference
2. keep `planner_set_v6_expanded_v1` as the main feed-forward baseline
3. keep `planner_moe_v2_expanded_v1` as the only MoE carry-forward arm
4. drop `moe_v1` from the next expensive long run unless a router-focused fix is implemented first
5. keep `vice` as the external benchmark opponent for the next cycle

The run clearly argues for:

- more recurrent-planner work
- less broad MoE exploration
- and continued external benchmarking without escalating beyond `vice` yet
