from __future__ import annotations

from dataclasses import dataclass

from train.datasets.schema import DatasetExample, PositionEncoding, TacticalAnnotations
from train.eval.planner_runtime import PlannerRootDecision
from train.eval.selfplay import play_selfplay_game, run_selfplay_session


def _make_example(
    *,
    sample_id: str,
    fen: str,
    side_to_move: str,
    legal_moves: list[str],
    is_checkmate: bool = False,
    is_stalemate: bool = False,
) -> DatasetExample:
    return DatasetExample(
        sample_id=sample_id,
        split="test",
        source="selfplay_test",
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
            is_stalemate=is_stalemate,
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


def test_play_selfplay_game_stops_on_checkmate_after_selected_move() -> None:
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
    white = _FakeAgent(
        name="white",
        move_map={
            start_fen: PlannerRootDecision(
                move_uci="e2e4",
                action_index=0,
                next_fen=mate_fen,
                selector_name="white",
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
    black = _FakeAgent(name="black", move_map={})
    game = play_selfplay_game(
        game_id="game_0001",
        white_agent=white,
        black_agent=black,
        repo_root=__import__("pathlib").Path("."),
        initial_fen=start_fen,
        max_plies=8,
        oracle_loader=lambda fen: oracle_examples[fen],
    )
    assert game.termination_reason == "checkmate"
    assert game.result == "1-0"
    assert game.move_count == 1
    assert game.moves[0].move_uci == "e2e4"


def test_play_selfplay_game_stops_on_threefold_repetition() -> None:
    fen_a = "8/8/8/8/8/8/8/K6k w - - 0 1"
    fen_b = "8/8/8/8/8/8/8/K6k b - - 0 1"
    oracle_examples = {
        fen_a: _make_example(sample_id="a", fen=fen_a, side_to_move="w", legal_moves=["a2a3"]),
        fen_b: _make_example(sample_id="b", fen=fen_b, side_to_move="b", legal_moves=["a7a6"]),
    }
    white = _FakeAgent(
        name="white",
        move_map={
            fen_a: PlannerRootDecision(
                move_uci="a2a3",
                action_index=0,
                next_fen=fen_b,
                selector_name="white",
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
    black = _FakeAgent(
        name="black",
        move_map={
            fen_b: PlannerRootDecision(
                move_uci="a7a6",
                action_index=0,
                next_fen=fen_a,
                selector_name="black",
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
    game = play_selfplay_game(
        game_id="game_0002",
        white_agent=white,
        black_agent=black,
        repo_root=__import__("pathlib").Path("."),
        initial_fen=fen_a,
        max_plies=12,
        oracle_loader=lambda fen: oracle_examples[fen],
    )
    assert game.termination_reason == "threefold_repetition"
    assert game.result == "1/2-1/2"
    assert game.move_count == 4


def test_run_selfplay_session_aggregates_results() -> None:
    terminal_fen = "terminal"
    oracle_examples = {
        terminal_fen: _make_example(
            sample_id="terminal",
            fen=terminal_fen,
            side_to_move="b",
            legal_moves=[],
            is_checkmate=True,
        )
    }
    white = _FakeAgent(name="white", move_map={})
    black = _FakeAgent(name="black", move_map={})
    session = run_selfplay_session(
        white_agent=white,
        black_agent=black,
        repo_root=__import__("pathlib").Path("."),
        games=2,
        initial_fens=[terminal_fen],
        oracle_loader=lambda fen: oracle_examples[fen],
    )
    payload = session.to_dict()
    assert payload["aggregate"]["game_count"] == 2
    assert payload["aggregate"]["termination_counts"] == {"checkmate": 2}
    assert payload["aggregate"]["result_counts"] == {"1-0": 2}
