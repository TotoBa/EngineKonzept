"""Evaluation helpers for offline workflow and baseline measurements."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from train.eval.opponent import OpponentBaselineMetrics


def __getattr__(name: str) -> Any:
    """Lazily expose evaluation helpers to avoid dataset/eval import cycles."""
    if name in {
        "OpponentBaselineMetrics",
        "evaluate_symbolic_opponent_baseline",
        "load_opponent_head_checkpoint",
        "score_opponent_candidates",
    }:
        from train.eval.opponent import (
            OpponentBaselineMetrics,
            evaluate_symbolic_opponent_baseline,
            load_opponent_head_checkpoint,
            score_opponent_candidates,
        )

        return {
            "OpponentBaselineMetrics": OpponentBaselineMetrics,
            "evaluate_symbolic_opponent_baseline": evaluate_symbolic_opponent_baseline,
            "load_opponent_head_checkpoint": load_opponent_head_checkpoint,
            "score_opponent_candidates": score_opponent_candidates,
        }[name]
    if name in {"load_symbolic_proposer_checkpoint", "score_symbolic_candidates"}:
        from train.eval.symbolic_proposer import (
            load_symbolic_proposer_checkpoint,
            score_symbolic_candidates,
        )

        return {
            "load_symbolic_proposer_checkpoint": load_symbolic_proposer_checkpoint,
            "score_symbolic_candidates": score_symbolic_candidates,
        }[name]
    if name in {"PlannerBaselineMetrics", "evaluate_two_ply_planner_baseline"}:
        from train.eval.planner import (
            PlannerBaselineMetrics,
            evaluate_two_ply_planner_baseline,
        )

        return {
            "PlannerBaselineMetrics": PlannerBaselineMetrics,
            "evaluate_two_ply_planner_baseline": evaluate_two_ply_planner_baseline,
        }[name]
    if name in {
        "PlannerRootDecision",
        "LoadedPlannerRuntime",
        "build_planner_runtime",
        "load_planner_head_checkpoint",
    }:
        from train.eval.planner_runtime import (
            LoadedPlannerRuntime,
            PlannerRootDecision,
            build_planner_runtime,
            load_planner_head_checkpoint,
        )

        return {
            "PlannerRootDecision": PlannerRootDecision,
            "LoadedPlannerRuntime": LoadedPlannerRuntime,
            "build_planner_runtime": build_planner_runtime,
            "load_planner_head_checkpoint": load_planner_head_checkpoint,
        }[name]
    if name in {"STARTING_FEN", "SelfplayGameRecord", "SelfplaySessionRecord", "play_selfplay_game", "run_selfplay_session"}:
        from train.eval.selfplay import (
            STARTING_FEN,
            SelfplayGameRecord,
            SelfplaySessionRecord,
            play_selfplay_game,
            run_selfplay_session,
        )

        return {
            "STARTING_FEN": STARTING_FEN,
            "SelfplayGameRecord": SelfplayGameRecord,
            "SelfplaySessionRecord": SelfplaySessionRecord,
            "play_selfplay_game": play_selfplay_game,
            "run_selfplay_session": run_selfplay_session,
        }[name]
    if name in {"load_dynamics_checkpoint", "predict_dynamics_latent"}:
        from train.eval.dynamics import (
            load_dynamics_checkpoint,
            predict_dynamics_latent,
        )

        return {
            "load_dynamics_checkpoint": load_dynamics_checkpoint,
            "predict_dynamics_latent": predict_dynamics_latent,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "OpponentBaselineMetrics",
    "LoadedPlannerRuntime",
    "PlannerBaselineMetrics",
    "PlannerRootDecision",
    "STARTING_FEN",
    "SelfplayGameRecord",
    "SelfplaySessionRecord",
    "build_planner_runtime",
    "evaluate_symbolic_opponent_baseline",
    "evaluate_two_ply_planner_baseline",
    "load_dynamics_checkpoint",
    "load_opponent_head_checkpoint",
    "load_planner_head_checkpoint",
    "load_symbolic_proposer_checkpoint",
    "play_selfplay_game",
    "predict_dynamics_latent",
    "run_selfplay_session",
    "score_opponent_candidates",
    "score_symbolic_candidates",
]
