"""Evaluation helpers for offline workflow, selfplay, and baseline measurements."""

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
        "build_planner_runtime_from_spec",
        "load_planner_head_checkpoint",
        "load_planner_runtime_from_spec_path",
    }:
        from train.eval.planner_runtime import (
            LoadedPlannerRuntime,
            PlannerRootDecision,
            build_planner_runtime,
            build_planner_runtime_from_spec,
            load_planner_head_checkpoint,
            load_planner_runtime_from_spec_path,
        )

        return {
            "PlannerRootDecision": PlannerRootDecision,
            "LoadedPlannerRuntime": LoadedPlannerRuntime,
            "build_planner_runtime": build_planner_runtime,
            "build_planner_runtime_from_spec": build_planner_runtime_from_spec,
            "load_planner_head_checkpoint": load_planner_head_checkpoint,
            "load_planner_runtime_from_spec_path": load_planner_runtime_from_spec_path,
        }[name]
    if name in {
        "STARTING_FEN",
        "SelfplayAdjudicationOutcome",
        "SelfplayGameRecord",
        "SelfplayMaxPliesAdjudicationSpec",
        "SelfplaySessionRecord",
        "open_max_plies_adjudicator",
        "play_selfplay_game",
        "run_selfplay_session",
    }:
        from train.eval.selfplay import (
            STARTING_FEN,
            SelfplayAdjudicationOutcome,
            SelfplayGameRecord,
            SelfplayMaxPliesAdjudicationSpec,
            SelfplaySessionRecord,
            open_max_plies_adjudicator,
            play_selfplay_game,
            run_selfplay_session,
        )

        return {
            "STARTING_FEN": STARTING_FEN,
            "SelfplayAdjudicationOutcome": SelfplayAdjudicationOutcome,
            "SelfplayGameRecord": SelfplayGameRecord,
            "SelfplayMaxPliesAdjudicationSpec": SelfplayMaxPliesAdjudicationSpec,
            "SelfplaySessionRecord": SelfplaySessionRecord,
            "open_max_plies_adjudicator": open_max_plies_adjudicator,
            "play_selfplay_game": play_selfplay_game,
            "run_selfplay_session": run_selfplay_session,
        }[name]
    if name in {"SelfplayAgentSpec", "load_selfplay_agent_spec", "write_selfplay_agent_spec"}:
        from train.eval.agent_spec import (
            SelfplayAgentSpec,
            load_selfplay_agent_spec,
            write_selfplay_agent_spec,
        )

        return {
            "SelfplayAgentSpec": SelfplayAgentSpec,
            "load_selfplay_agent_spec": load_selfplay_agent_spec,
            "write_selfplay_agent_spec": write_selfplay_agent_spec,
        }[name]
    if name in {"ExternalUciEngineAgent", "build_external_engine_agent_from_spec"}:
        from train.eval.external_engine import (
            ExternalUciEngineAgent,
            build_external_engine_agent_from_spec,
        )

        return {
            "ExternalUciEngineAgent": ExternalUciEngineAgent,
            "build_external_engine_agent_from_spec": build_external_engine_agent_from_spec,
        }[name]
    if name in {
        "SelfplayArenaMatchupSpec",
        "SelfplayArenaSpec",
        "load_selfplay_arena_spec",
        "run_selfplay_arena",
        "write_selfplay_arena_spec",
    }:
        from train.eval.arena import (
            SelfplayArenaMatchupSpec,
            SelfplayArenaSpec,
            load_selfplay_arena_spec,
            run_selfplay_arena,
            write_selfplay_arena_spec,
        )

        return {
            "SelfplayArenaMatchupSpec": SelfplayArenaMatchupSpec,
            "SelfplayArenaSpec": SelfplayArenaSpec,
            "load_selfplay_arena_spec": load_selfplay_arena_spec,
            "run_selfplay_arena": run_selfplay_arena,
            "write_selfplay_arena_spec": write_selfplay_arena_spec,
        }[name]
    if name in {
        "build_selfplay_arena_matrix",
        "load_selfplay_arena_summary",
        "write_selfplay_arena_matrix",
    }:
        from train.eval.matrix import (
            build_selfplay_arena_matrix,
            load_selfplay_arena_summary,
            write_selfplay_arena_matrix,
        )

        return {
            "build_selfplay_arena_matrix": build_selfplay_arena_matrix,
            "load_selfplay_arena_summary": load_selfplay_arena_summary,
            "write_selfplay_arena_matrix": write_selfplay_arena_matrix,
        }[name]
    if name in {
        "PlannerRunSpec",
        "SelfplayCurriculumPlan",
        "SelfplayCurriculumStage",
        "build_phase9_expanded_curriculum_plan",
        "write_selfplay_curriculum_plan",
    }:
        from train.eval.curriculum import (
            PlannerRunSpec,
            SelfplayCurriculumPlan,
            SelfplayCurriculumStage,
            build_phase9_expanded_curriculum_plan,
            write_selfplay_curriculum_plan,
        )

        return {
            "PlannerRunSpec": PlannerRunSpec,
            "SelfplayCurriculumPlan": SelfplayCurriculumPlan,
            "SelfplayCurriculumStage": SelfplayCurriculumStage,
            "build_phase9_expanded_curriculum_plan": build_phase9_expanded_curriculum_plan,
            "write_selfplay_curriculum_plan": write_selfplay_curriculum_plan,
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
    if name in {
        "PlannerReplayCampaignRunSpec",
        "SelfplayReplayCampaignSpec",
        "build_planner_verify_matrix",
        "load_selfplay_replay_campaign_spec",
        "materialize_replay_campaign_planner_config",
        "run_selfplay_replay_campaign",
        "write_selfplay_replay_campaign_spec",
    }:
        from train.eval.campaign import (
            PlannerReplayCampaignRunSpec,
            SelfplayReplayCampaignSpec,
            build_planner_verify_matrix,
            load_selfplay_replay_campaign_spec,
            materialize_replay_campaign_planner_config,
            run_selfplay_replay_campaign,
            write_selfplay_replay_campaign_spec,
        )

        return {
            "PlannerReplayCampaignRunSpec": PlannerReplayCampaignRunSpec,
            "SelfplayReplayCampaignSpec": SelfplayReplayCampaignSpec,
            "build_planner_verify_matrix": build_planner_verify_matrix,
            "load_selfplay_replay_campaign_spec": load_selfplay_replay_campaign_spec,
            "materialize_replay_campaign_planner_config": materialize_replay_campaign_planner_config,
            "run_selfplay_replay_campaign": run_selfplay_replay_campaign,
            "write_selfplay_replay_campaign_spec": write_selfplay_replay_campaign_spec,
        }[name]
    if name in {
        "PlannerFulltrainRunSpec",
        "PlannerFulltrainArenaCampaignSpec",
        "load_planner_fulltrain_arena_campaign_spec",
        "materialize_fulltrain_planner_config",
        "run_planner_fulltrain_arena_campaign",
        "write_planner_fulltrain_arena_campaign_spec",
    }:
        from train.eval.fulltrain_campaign import (
            PlannerFulltrainArenaCampaignSpec,
            PlannerFulltrainRunSpec,
            load_planner_fulltrain_arena_campaign_spec,
            materialize_fulltrain_planner_config,
            run_planner_fulltrain_arena_campaign,
            write_planner_fulltrain_arena_campaign_spec,
        )

        return {
            "PlannerFulltrainRunSpec": PlannerFulltrainRunSpec,
            "PlannerFulltrainArenaCampaignSpec": PlannerFulltrainArenaCampaignSpec,
            "load_planner_fulltrain_arena_campaign_spec": load_planner_fulltrain_arena_campaign_spec,
            "materialize_fulltrain_planner_config": materialize_fulltrain_planner_config,
            "run_planner_fulltrain_arena_campaign": run_planner_fulltrain_arena_campaign,
            "write_planner_fulltrain_arena_campaign_spec": write_planner_fulltrain_arena_campaign_spec,
        }[name]
    if name in {
        "SelfplayTeacherRetrainAgentSpec",
        "SelfplayTeacherRetrainCycleSpec",
        "load_selfplay_teacher_retrain_cycle_spec",
        "run_selfplay_teacher_retrain_cycle",
        "write_selfplay_teacher_retrain_cycle_spec",
    }:
        from train.eval.selfplay_training_cycle import (
            SelfplayTeacherRetrainAgentSpec,
            SelfplayTeacherRetrainCycleSpec,
            load_selfplay_teacher_retrain_cycle_spec,
            run_selfplay_teacher_retrain_cycle,
            write_selfplay_teacher_retrain_cycle_spec,
        )

        return {
            "SelfplayTeacherRetrainAgentSpec": SelfplayTeacherRetrainAgentSpec,
            "SelfplayTeacherRetrainCycleSpec": SelfplayTeacherRetrainCycleSpec,
            "load_selfplay_teacher_retrain_cycle_spec": load_selfplay_teacher_retrain_cycle_spec,
            "run_selfplay_teacher_retrain_cycle": run_selfplay_teacher_retrain_cycle,
            "write_selfplay_teacher_retrain_cycle_spec": write_selfplay_teacher_retrain_cycle_spec,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "OpponentBaselineMetrics",
    "LoadedPlannerRuntime",
    "PlannerBaselineMetrics",
    "PlannerReplayCampaignRunSpec",
    "PlannerRootDecision",
    "PlannerRunSpec",
    "STARTING_FEN",
    "SelfplayAdjudicationOutcome",
    "SelfplayAgentSpec",
    "SelfplayArenaMatchupSpec",
    "SelfplayArenaSpec",
    "SelfplayCurriculumPlan",
    "SelfplayCurriculumStage",
    "SelfplayReplayCampaignSpec",
    "SelfplayGameRecord",
    "SelfplayMaxPliesAdjudicationSpec",
    "SelfplaySessionRecord",
    "SelfplayTeacherRetrainAgentSpec",
    "SelfplayTeacherRetrainCycleSpec",
    "build_planner_runtime",
    "build_planner_runtime_from_spec",
    "build_planner_verify_matrix",
    "build_phase9_expanded_curriculum_plan",
    "build_selfplay_arena_matrix",
    "evaluate_symbolic_opponent_baseline",
    "evaluate_two_ply_planner_baseline",
    "load_dynamics_checkpoint",
    "load_opponent_head_checkpoint",
    "load_planner_head_checkpoint",
    "load_selfplay_arena_summary",
    "load_selfplay_arena_spec",
    "load_planner_runtime_from_spec_path",
    "load_selfplay_agent_spec",
    "load_symbolic_proposer_checkpoint",
    "load_selfplay_replay_campaign_spec",
    "load_selfplay_teacher_retrain_cycle_spec",
    "materialize_replay_campaign_planner_config",
    "open_max_plies_adjudicator",
    "play_selfplay_game",
    "predict_dynamics_latent",
    "run_selfplay_arena",
    "run_selfplay_replay_campaign",
    "run_selfplay_teacher_retrain_cycle",
    "run_selfplay_session",
    "score_opponent_candidates",
    "score_symbolic_candidates",
    "write_selfplay_arena_matrix",
    "write_selfplay_arena_spec",
    "write_selfplay_curriculum_plan",
    "write_selfplay_agent_spec",
    "write_selfplay_replay_campaign_spec",
    "write_selfplay_teacher_retrain_cycle_spec",
]
