from __future__ import annotations

from train.datasets.replay_buffer import (
    ReplayBufferEntry,
    build_replay_buffer_entries,
    replay_buffer_summary,
)
from train.eval.agent_spec import SelfplayAgentSpec
from train.eval.selfplay import (
    SelfplayGameRecord,
    SelfplayMoveRecord,
    SelfplaySessionRecord,
)


def _sample_session() -> SelfplaySessionRecord:
    return SelfplaySessionRecord(
        games=[
            SelfplayGameRecord(
                game_id="game_0001",
                initial_fen="8/8/8/8/8/8/8/K6k w - - 0 1",
                final_fen="8/8/8/8/8/8/8/K6k b - - 1 1",
                result="1-0",
                termination_reason="checkmate",
                move_count=1,
                white_agent="white_agent",
                black_agent="black_agent",
                moves=[
                    SelfplayMoveRecord(
                        ply_index=0,
                        side_to_move="w",
                        fen="8/8/8/8/8/8/8/K6k w - - 0 1",
                        move_uci="a1a2",
                        action_index=0,
                        selector_name="white_agent",
                        legal_candidate_count=3,
                        considered_candidate_count=2,
                        proposer_score=1.25,
                        planner_score=0.5,
                        reply_peak_probability=0.2,
                        pressure=0.0,
                        uncertainty=0.8,
                        next_fen="8/8/8/8/8/8/K7/7k b - - 1 1",
                    )
                ],
            )
        ]
    )


def test_selfplay_agent_spec_round_trip() -> None:
    spec = SelfplayAgentSpec(
        name="planner_set_v2",
        proposer_checkpoint="models/proposer/stockfish_pgn_symbolic_v1_v1/checkpoint.pt",
        planner_checkpoint="models/planner/corpus_suite_set_v2_10k_122k_expanded_v1/checkpoint.pt",
        opponent_checkpoint="models/opponent/corpus_suite_set_v2_v1/checkpoint.pt",
        opponent_mode="learned",
        root_top_k=4,
        tags=["active"],
    )
    restored = SelfplayAgentSpec.from_dict(spec.to_dict())
    assert restored == spec


def test_build_replay_buffer_entries_flattens_session() -> None:
    entries = build_replay_buffer_entries(_sample_session())
    assert len(entries) == 1
    entry = entries[0]
    assert isinstance(entry, ReplayBufferEntry)
    assert entry.sample_id == "game_0001:0"
    assert entry.outcome_pov == "win"
    assert entry.selector_name == "white_agent"
    assert entry.game_result == "1-0"


def test_replay_buffer_summary_aggregates_entries() -> None:
    entries = build_replay_buffer_entries(_sample_session())
    summary = replay_buffer_summary(entries)
    assert summary["entry_count"] == 1
    assert summary["game_count"] == 1
    assert summary["selector_counts"] == {"white_agent": 1}
    assert summary["outcome_counts"] == {"win": 1}
