from __future__ import annotations

import json
from pathlib import Path

from train.datasets.replay_buffer import (
    ReplayBufferEntry,
    build_replay_buffer_entries,
    build_replay_buffer_entries_from_sessions,
    load_arena_session_paths,
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


def test_uci_engine_agent_spec_round_trip() -> None:
    spec = SelfplayAgentSpec(
        name="vice_v1",
        agent_kind="uci_engine",
        proposer_checkpoint=None,
        opponent_mode="none",
        external_engine_path="/usr/games/vice",
        external_engine_nodes=128,
        external_engine_threads=1,
        external_engine_hash_mb=16,
        external_engine_options={"Book": "false"},
        tags=["external", "offline_benchmark"],
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


def test_build_replay_buffer_entries_from_sessions_flattens_multiple_sessions() -> None:
    entries = build_replay_buffer_entries_from_sessions(
        [_sample_session(), _sample_session()],
        session_labels=["session_a", "session_b"],
    )
    assert len(entries) == 2
    assert entries[0].game_id == "session_a:game_0001"
    assert entries[1].sample_id == "session_b:game_0001:0"


def test_load_arena_session_paths_reads_matchup_paths(tmp_path: Path) -> None:
    session_one = tmp_path / "session_one.json"
    session_two = tmp_path / "session_two.json"
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "matchups": [
                    {"name": "a_vs_b", "session_path": str(session_one)},
                    {"name": "b_vs_c", "session_path": str(session_two)},
                    {"name": "duplicate", "session_path": str(session_one)},
                ]
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    assert load_arena_session_paths(summary_path) == [session_one, session_two]
