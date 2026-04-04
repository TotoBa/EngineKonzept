# Phase 9 Evolution Custom V1 Stop Summary

This note records the stable state of the interrupted long run under:

- [/srv/schach/engine_training/phase9/evolution_fullmatrix_custom_v1](/srv/schach/engine_training/phase9/evolution_fullmatrix_custom_v1)
- config: [phase9_evolution_fullmatrix_filtered_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_evolution_fullmatrix_filtered_v1.json)

The run was stopped intentionally after `round_04` had started so a smaller follow-up campaign could be prepared with:

- `10` rounds instead of `20`
- injected offline `vice` matches
- wider max-plies adjudication band
- direct resume from the latest stable planner checkpoints

## Stable checkpoint boundary

The latest fully completed selfplay/retrain stage is:

- [round_03/summary.json](/srv/schach/engine_training/phase9/evolution_fullmatrix_custom_v1/iterations/round_03/summary.json)

That stage is the seed boundary for the next evolution run because it contains the newest fully materialized `active_agent_specs` for every planner arm.

`round_04` was only partially through arena play when the run was stopped:

- [round_04/arena/progress.json](/srv/schach/engine_training/phase9/evolution_fullmatrix_custom_v1/iterations/round_04/arena/progress.json)
- progress at stop: `22 / 132` games, `22 / 132` matchups

## Main findings

### After fulltrain

Held-out verify leader:

- `planner_moe_v1_expanded_v1`
- `root_top1_accuracy=0.824113`
- `teacher_root_mean_reciprocal_rank=0.894681`

Arena leader:

- `planner_set_v6_rank_expanded_v1`
- `14.5 / 22`
- `score_rate=0.659091`

This was the first clear split between:

- best verify arm: `moe_v1`
- best arena arm: `set_v6_rank`

### Round 1

Held-out verify leader:

- `planner_recurrent_expanded_v1`
- `root_top1_accuracy=0.816312`
- `teacher_root_mean_reciprocal_rank=0.893203`

Arena leader stayed:

- `planner_set_v6_rank_expanded_v1`
- `14.5 / 22`
- `score_rate=0.659091`

Retrain signal size was already non-trivial:

- all `9` trainable planner arms retrained
- per-arm `planner_head` correction rows roughly `620-744`

### Round 2

Held-out verify leader:

- `planner_active_expanded_v2`
- `root_top1_accuracy=0.813475`
- `teacher_root_mean_reciprocal_rank=0.889894`

Arena leader shifted to:

- `planner_set_v6_replay_expanded_v2`
- `14.5 / 22`
- `score_rate=0.659091`

This is the strongest sign so far that:

- replay-aware arms are robust in direct arena play
- but verify leadership is still moving between arms

### Round 3

Held-out verify leader:

- `planner_moe_v1_expanded_v1`
- `root_top1_accuracy=0.817730`
- `teacher_root_mean_reciprocal_rank=0.891667`

Arena co-lead by score:

- `planner_recurrent_expanded_v1`
- `planner_set_v6_replay_expanded_v2`
- both `14.0 / 22`
- both `score_rate=0.636364`

This is the latest stable picture before the stop:

- verify still likes `moe_v1`
- arena now likes either `recurrent_v1` or `set_v6_replay`
- no single arm dominates both views yet

## Interpretation

Three families remain plausible:

1. `planner_moe_v1_expanded_v1`
   Best verify arm after fulltrain and again after `round_03`, but not the most reliable arena winner.
2. `planner_set_v6_replay_expanded_v2`
   Most convincing replay-aware arena arm so far, especially after `round_02` and still tied at the top in `round_03`.
3. `planner_recurrent_expanded_v1`
   First recurrent arm to win a full verify round and then reach the `round_03` arena co-lead.

The weaker signal from the stopped run is that `planner_set_v6_rank_expanded_v1` can win early arena slices, but that lead did not stay uniquely dominant through the later rounds.

## Follow-up decision

The next run should not restart from scratch.

It should resume from the latest stable active-agent snapshot in:

- [round_03/active_agent_specs](/srv/schach/engine_training/phase9/evolution_fullmatrix_custom_v1/iterations/round_03/active_agent_specs)

And it should change the campaign conditions:

- inject `vice` as an offline arena opponent every round
- keep `stockfish18` only as post-game reviewer and adjudicator
- reduce the campaign to `10` rounds
- start directly with arena instead of repeating fulltrain

The prepared follow-up config for that is:

- [phase9_evolution_round03_vice_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_evolution_round03_vice_v1.json)
