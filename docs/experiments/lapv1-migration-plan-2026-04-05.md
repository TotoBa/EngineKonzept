# LAPv1 Migration Plan

This note records the intended arm-by-arm migration path from the current
Phase-8/9 planner family toward LAPv1.

It is a migration plan, not a promotion declaration. Existing planner arms stay
materialized until LAPv1 beats them empirically.

## Keep / Deprecate / Remove

| Arm | Decision | Reason |
|-----|----------|--------|
| `planner_recurrent_expanded_v1` | `KEEP` | Current strongest learned planner reference and the main regression target for LAPv1 |
| `planner_set_v6_expanded_v1` | `KEEP` | Strongest stable feed-forward baseline |
| `planner_moe_v2_expanded_v1` | `KEEP` | Experimental MoE branch with better arena robustness than `moe_v1` |
| `planner_set_v2_expanded_v1` | `DEPRECATE` | Historical baseline that no longer needs first-class promotion priority |
| `planner_set_v2_wide_expanded_v1` | `REMOVE` | Wider feed-forward variant did not earn a durable quality gain |
| `planner_set_v5_expanded_v1` | `REMOVE` | Underperforms the stronger retained feed-forward arms |
| `planner_set_v6_margin_expanded_v1` | `REMOVE` | Selfplay evolution degraded it versus the retained references |
| `planner_set_v6_rank_expanded_v1` | `REMOVE` | No longer leads either the preferred verify slice or the later vice-conditioned run |
| `planner_moe_v1_expanded_v1` | `REMOVE` | Strong early verify signal but unstable under selfplay correction |
| `planner_active_expanded_v2` | `KEEP` | Current promoted static benchmark inside the existing phase-9 stack |
| `vice_v2` | `KEEP` | External calibration gate, not a promotion target |
| `symbolic_root_v1` | `KEEP` | Minimal legal-move sanity benchmark |

## Migration Gate

LAPv1 becomes the next active-promotion candidate only when all three gates are met:

1. Verify `root_top1_accuracy >= 0.825` on the preferred validation slice.
2. Internal arena `score_rate >= 0.65` against the retained internal reference family.
3. Against `vice_v2`, at least `2.0 / 24`.

## Notes

- These gates are intentionally empirical and conservative.
- They do not authorize removal of checkpoints or configs by themselves.
- `vice_v2` remains a calibration rung only; it does not redefine the repository
  into a classical-engine benchmark program.
