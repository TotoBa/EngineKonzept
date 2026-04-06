from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from train.datasets.schema import DatasetExample, PositionEncoding, TacticalAnnotations
from train.eval.arena import (
    SelfplayArenaMatchupSpec,
    SelfplayArenaSpec,
    _selected_initial_fens_for_arena,
    rebuild_selfplay_arena_summary,
    run_selfplay_arena,
)
from train.eval.planner_runtime import PlannerRootDecision
from train.eval.selfplay import SelfplayMaxPliesAdjudicationSpec


def _make_example(
    *,
    sample_id: str,
    fen: str,
    side_to_move: str,
    legal_moves: list[str],
    is_checkmate: bool = False,
) -> DatasetExample:
    return DatasetExample(
        sample_id=sample_id,
        split="test",
        source="arena_test",
        fen=fen,
        side_to_move=side_to_move,
        selected_move_uci=None,
        selected_action_encoding=None,
        next_fen=None,
        legal_moves=legal_moves,
        legal_action_encodings=[[0, 0, index] for index, _move in enumerate(legal_moves)],
        position_encoding=PositionEncoding(
            piece_tokens=[],
            square_tokens=[[index, 0] for index in range(64)],
            rule_token=[0, 0, -1, 0, 1, 0],
        ),
        wdl_target=None,
        annotations=TacticalAnnotations(
            in_check=is_checkmate,
            is_checkmate=is_checkmate,
            is_stalemate=False,
            has_legal_en_passant=False,
            has_legal_castle=False,
            has_legal_promotion=False,
            is_low_material_endgame=False,
            legal_move_count=len(legal_moves),
            piece_count=2,
            selected_move_is_capture=None,
            selected_move_is_promotion=None,
            selected_move_is_castle=None,
            selected_move_is_en_passant=None,
            selected_move_gives_check=None,
        ),
        result=None,
        metadata={},
    )


@dataclass
class _FakeAgent:
    name: str
    move_map: dict[str, PlannerRootDecision]

    def select_move(self, example: DatasetExample) -> PlannerRootDecision:
        return self.move_map[example.fen]


def test_round_robin_expands_without_self_matches() -> None:
    spec = SelfplayArenaSpec(
        name="round_robin",
        agent_specs={"a": "a.json", "b": "b.json", "c": "c.json"},
        schedule_mode="round_robin",
        default_games=2,
        default_max_plies=12,
        default_initial_fens=["startpos"],
    )
    matchups = spec.expanded_matchups()
    assert len(matchups) == 6
    assert {f"{matchup.white_agent}->{matchup.black_agent}" for matchup in matchups} == {
        "a->b",
        "a->c",
        "b->a",
        "b->c",
        "c->a",
        "c->b",
    }


def test_arena_spec_round_trip_with_max_plies_adjudication() -> None:
    spec = SelfplayArenaSpec(
        name="round_robin_adjudicated",
        agent_specs={"a": "a.json", "b": "b.json"},
        schedule_mode="round_robin",
        default_games=2,
        default_max_plies=12,
        default_initial_fens=["startpos"],
        parallel_workers=6,
        opening_selection_seed=17,
        max_plies_adjudication=SelfplayMaxPliesAdjudicationSpec(
            engine_path="/usr/games/stockfish18",
            nodes=64,
            score_threshold_pawns=0.3,
            extension_step_plies=8,
            max_extensions=2,
        ),
    )
    restored = SelfplayArenaSpec.from_dict(spec.to_dict())
    assert restored.parallel_workers == 6
    assert restored.opening_selection_seed == 17
    assert restored.max_plies_adjudication is not None
    assert restored.max_plies_adjudication.engine_path == "/usr/games/stockfish18"
    assert restored.max_plies_adjudication.max_extensions == 2


def test_run_selfplay_arena_writes_sessions_and_standings(tmp_path: Path) -> None:
    start_fen = "8/8/8/8/8/8/8/K6k w - - 0 1"
    mate_fen = "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"
    oracle_examples = {
        start_fen: _make_example(
            sample_id="root",
            fen=start_fen,
            side_to_move="w",
            legal_moves=["e2e4"],
        ),
        mate_fen: _make_example(
            sample_id="mate",
            fen=mate_fen,
            side_to_move="b",
            legal_moves=[],
            is_checkmate=True,
        ),
    }

    def _agent_builder(agent_name: str, _spec_path: Path, _repo_root: Path) -> _FakeAgent:
        return _FakeAgent(
            name=agent_name,
            move_map={
                start_fen: PlannerRootDecision(
                    move_uci="e2e4",
                    action_index=0,
                    next_fen=mate_fen,
                    selector_name=agent_name,
                    legal_candidate_count=1,
                    considered_candidate_count=1,
                    proposer_score=1.0,
                    planner_score=1.0,
                    reply_peak_probability=0.0,
                    pressure=0.0,
                    uncertainty=0.0,
                )
            },
        )

    spec = SelfplayArenaSpec(
        name="explicit_arena",
        agent_specs={"white_arm": "white.json", "black_arm": "black.json"},
        schedule_mode="explicit",
        matchups=[
            SelfplayArenaMatchupSpec(
                white_agent="white_arm",
                black_agent="black_arm",
                games=1,
                max_plies=8,
                initial_fens=[start_fen],
            )
        ],
    )
    output_root = tmp_path / "arena"
    summary = run_selfplay_arena(
        spec=spec,
        repo_root=tmp_path,
        output_root=output_root,
        agent_builder=_agent_builder,
        oracle_loader=lambda fen: oracle_examples[fen],
    )
    assert summary["aggregate"]["game_count"] == 1
    assert summary["standings"]["white_arm"]["wins"] == 1
    assert summary["standings"]["black_arm"]["losses"] == 1
    summary_path = output_root / "summary.json"
    assert summary_path.exists()
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["aggregate"]["game_count"] == 1
    session_path = output_root / "sessions" / "01_white_arm_vs_black_arm.json"
    assert session_path.exists()
    progress_path = output_root / "progress.json"
    assert progress_path.exists()


def test_rebuild_selfplay_arena_summary_recovers_missing_summary(tmp_path: Path) -> None:
    start_fen = "8/8/8/8/8/8/8/K6k w - - 0 1"
    mate_fen = "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"
    oracle_examples = {
        start_fen: _make_example(
            sample_id="root",
            fen=start_fen,
            side_to_move="w",
            legal_moves=["e2e4"],
        ),
        mate_fen: _make_example(
            sample_id="mate",
            fen=mate_fen,
            side_to_move="b",
            legal_moves=[],
            is_checkmate=True,
        ),
    }

    def _agent_builder(agent_name: str, _spec_path: Path, _repo_root: Path) -> _FakeAgent:
        return _FakeAgent(
            name=agent_name,
            move_map={
                start_fen: PlannerRootDecision(
                    move_uci="e2e4",
                    action_index=0,
                    next_fen=mate_fen,
                    selector_name=agent_name,
                    legal_candidate_count=1,
                    considered_candidate_count=1,
                    proposer_score=1.0,
                    planner_score=1.0,
                    reply_peak_probability=0.0,
                    pressure=0.0,
                    uncertainty=0.0,
                )
            },
        )

    output_root = tmp_path / "arena_output"
    spec = SelfplayArenaSpec(
        name="arena_rebuild",
        agent_specs={"white_arm": "white_spec.json", "black_arm": "black_spec.json"},
        schedule_mode="explicit",
        matchups=[
            SelfplayArenaMatchupSpec(
                white_agent="white_arm",
                black_agent="black_arm",
                games=1,
                max_plies=2,
                initial_fens=[start_fen],
            )
        ],
        default_games=1,
        default_max_plies=2,
    )

    run_selfplay_arena(
        spec=spec,
        repo_root=tmp_path,
        output_root=output_root,
        agent_builder=_agent_builder,
        oracle_loader=lambda fen: oracle_examples[fen],
    )
    summary_path = output_root / "summary.json"
    session_path = output_root / "sessions" / "01_white_arm_vs_black_arm.json"
    progress_path = output_root / "progress.json"
    summary_path.unlink()

    rebuilt = rebuild_selfplay_arena_summary(
        spec=spec,
        output_root=output_root,
    )

    assert summary_path.exists()
    assert rebuilt["aggregate"]["game_count"] == 1
    assert rebuilt["standings"]["white_arm"]["wins"] == 1
    progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress_payload["status"] == "completed"
    assert progress_payload["completed_games"] == 1
    payload = json.loads(session_path.read_text(encoding="utf-8"))
    assert payload["aggregate"]["termination_counts"] == {"checkmate": 1}


def test_parallel_selfplay_arena_requires_default_builders(tmp_path: Path) -> None:
    spec = SelfplayArenaSpec(
        name="parallel_arena",
        agent_specs={"white_arm": "white.json", "black_arm": "black.json"},
        schedule_mode="explicit",
        matchups=[
            SelfplayArenaMatchupSpec(
                white_agent="white_arm",
                black_agent="black_arm",
                games=1,
                max_plies=8,
                initial_fens=["8/8/8/8/8/8/8/K6k w - - 0 1"],
            )
        ],
        parallel_workers=2,
    )
    try:
        run_selfplay_arena(
            spec=spec,
            repo_root=tmp_path,
            output_root=tmp_path / "arena",
            agent_builder=lambda *_args: None,
        )
    except ValueError as exc:
        assert "parallel selfplay arena currently requires" in str(exc)
    else:
        raise AssertionError("parallel arena with custom builder should fail")


def test_opening_selection_seed_reuses_same_openings_across_swapped_colors(tmp_path: Path) -> None:
    fen_a = "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"
    fen_b = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
    fen_c = "7k/4Q3/6K1/8/8/8/8/8 b - - 0 1"
    fen_d = "7k/3Q4/6K1/8/8/8/8/8 b - - 0 1"
    oracle_examples = {
        fen_a: _make_example(
            sample_id="fen_a",
            fen=fen_a,
            side_to_move="b",
            legal_moves=[],
            is_checkmate=True,
        ),
        fen_b: _make_example(
            sample_id="fen_b",
            fen=fen_b,
            side_to_move="b",
            legal_moves=[],
            is_checkmate=True,
        ),
        fen_c: _make_example(
            sample_id="fen_c",
            fen=fen_c,
            side_to_move="b",
            legal_moves=[],
            is_checkmate=True,
        ),
        fen_d: _make_example(
            sample_id="fen_d",
            fen=fen_d,
            side_to_move="b",
            legal_moves=[],
            is_checkmate=True,
        ),
    }

    spec = SelfplayArenaSpec(
        name="seeded_openings",
        agent_specs={"white_arm": "white.json", "black_arm": "black.json"},
        schedule_mode="explicit",
        matchups=[
            SelfplayArenaMatchupSpec(
                white_agent="white_arm",
                black_agent="black_arm",
                games=4,
                max_plies=8,
                initial_fens=[fen_a, fen_b, fen_c, fen_d],
            ),
            SelfplayArenaMatchupSpec(
                white_agent="black_arm",
                black_agent="white_arm",
                games=4,
                max_plies=8,
                initial_fens=[fen_a, fen_b, fen_c, fen_d],
            ),
        ],
        opening_selection_seed=23,
    )
    output_root = tmp_path / "arena"
    summary = run_selfplay_arena(
        spec=spec,
        repo_root=tmp_path,
        output_root=output_root,
        agent_builder=lambda *_args: _FakeAgent(name="unused", move_map={}),
        oracle_loader=lambda fen: oracle_examples[fen],
    )
    assert summary["aggregate"]["game_count"] == 8
    first_session = json.loads(
        (output_root / "sessions" / "01_white_arm_vs_black_arm.json").read_text(encoding="utf-8")
    )
    second_session = json.loads(
        (output_root / "sessions" / "02_black_arm_vs_white_arm.json").read_text(encoding="utf-8")
    )
    first_initials = [game["initial_fen"] for game in first_session["games"]]
    second_initials = [game["initial_fen"] for game in second_session["games"]]
    assert first_initials == second_initials
    assert len(set(first_initials)) == 4


def test_round_robin_unique_openings_are_global_per_unordered_pair() -> None:
    openings = [f"opening_{index}" for index in range(6)]
    spec = SelfplayArenaSpec(
        name="unique_round_robin",
        agent_specs={"a": "a.json", "b": "b.json", "c": "c.json"},
        schedule_mode="round_robin",
        default_games=2,
        default_max_plies=12,
        default_initial_fens=openings,
        opening_selection_seed=19,
        round_robin_swap_colors=True,
    )
    matchups = spec.expanded_matchups()
    selected = _selected_initial_fens_for_arena(spec=spec, matchups=matchups)
    selected_by_name = {
        (matchup.white_agent, matchup.black_agent): openings_for_matchup
        for matchup, openings_for_matchup in zip(matchups, selected, strict=True)
    }

    assert selected_by_name[("a", "b")] == selected_by_name[("b", "a")]
    assert selected_by_name[("a", "c")] == selected_by_name[("c", "a")]
    assert selected_by_name[("b", "c")] == selected_by_name[("c", "b")]

    unique_unordered_openings = {
        tuple(sorted((white, black))): tuple(openings_for_matchup)
        for (white, black), openings_for_matchup in selected_by_name.items()
    }
    all_unique_games = [
        opening
        for openings_for_matchup in unique_unordered_openings.values()
        for opening in openings_for_matchup
    ]
    assert len(all_unique_games) == 6
    assert len(set(all_unique_games)) == 6
