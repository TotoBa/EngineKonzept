"""Evaluation helpers for offline workflow and baseline measurements."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from train.eval.opponent import OpponentBaselineMetrics


def __getattr__(name: str) -> Any:
    """Lazily expose evaluation helpers to avoid dataset/eval import cycles."""
    if name in {"OpponentBaselineMetrics", "evaluate_symbolic_opponent_baseline"}:
        from train.eval.opponent import (
            OpponentBaselineMetrics,
            evaluate_symbolic_opponent_baseline,
        )

        return {
            "OpponentBaselineMetrics": OpponentBaselineMetrics,
            "evaluate_symbolic_opponent_baseline": evaluate_symbolic_opponent_baseline,
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "OpponentBaselineMetrics",
    "evaluate_symbolic_opponent_baseline",
    "load_symbolic_proposer_checkpoint",
    "score_symbolic_candidates",
]
