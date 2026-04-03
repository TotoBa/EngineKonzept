from __future__ import annotations

from train.datasets.planner_replay import (
    build_planner_replay_examples,
    planner_replay_summary,
)
from train.datasets.replay_buffer import ReplayBufferEntry


def _entry(*, sample_id: str, outcome_pov: str, termination_reason: str) -> ReplayBufferEntry:
    return ReplayBufferEntry(
        sample_id=sample_id,
        game_id="game",
        ply_index=0,
        side_to_move="w",
        fen="8/8/8/8/8/8/8/K6k w - - 0 1",
        move_uci="a1a2",
        action_index=0,
        next_fen="8/8/8/8/8/8/K7/7k b - - 1 1",
        selector_name="planner",
        white_agent="planner",
        black_agent="opponent",
        legal_candidate_count=3,
        considered_candidate_count=2,
        proposer_score=0.5,
        planner_score=0.75,
        reply_peak_probability=0.2,
        pressure=0.1,
        uncertainty=0.8,
        game_result="1-0" if outcome_pov == "win" else "0-1",
        outcome_pov=outcome_pov,
        termination_reason=termination_reason,
        game_move_count=12,
    )


def test_build_planner_replay_examples_filters_unfinished() -> None:
    examples = build_planner_replay_examples(
        [
            _entry(sample_id="a", outcome_pov="win", termination_reason="checkmate"),
            _entry(sample_id="b", outcome_pov="unfinished", termination_reason="max_plies"),
        ],
        split="train",
    )
    assert [example.sample_id for example in examples] == ["a"]
    assert examples[0].root_value_cp == 256.0


def test_planner_replay_summary_aggregates_examples() -> None:
    examples = build_planner_replay_examples(
        [
            _entry(sample_id="a", outcome_pov="win", termination_reason="checkmate"),
            _entry(sample_id="b", outcome_pov="draw", termination_reason="threefold_repetition"),
        ],
        split="train",
        include_unfinished=True,
    )
    summary = planner_replay_summary(examples)
    assert summary["example_count"] == 2
    assert summary["outcome_counts"] == {"draw": 1, "win": 1}
