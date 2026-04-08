from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from train.datasets.artifacts import (
    SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE,
    pack_position_features,
)
from train.datasets.lapv1_training import (
    lapv1_training_example_from_planner_head,
    load_lapv1_training_examples,
)
from train.datasets.move_delta import halfka_delta, is_king_move, move_type_hash
from train.datasets.nnue_features import FEATURES_PER_KING, halfka_active_indices, halfka_index
from train.datasets.phase_features import phase_index
from train.datasets.planner_head import PlannerHeadExample
from train.datasets.schema import PositionEncoding


chess = pytest.importorskip("chess")


def test_phase_index_start_position() -> None:
    assert phase_index(chess.Board()) == 0


def test_phase_index_kbk_endgame() -> None:
    board = chess.Board("8/8/8/8/8/8/4k3/4K2B w - - 0 1")
    assert phase_index(board) == 3


def test_halfka_start_position_counts() -> None:
    board = chess.Board()
    white = halfka_active_indices(board, "w")
    black = halfka_active_indices(board, "b")

    assert len(white) == 30
    assert len(black) == 30


def test_halfka_index_uniqueness() -> None:
    values = {
        halfka_index(king_sq, piece_sq, piece_type_color)
        for king_sq in range(64)
        for piece_sq in range(64)
        for piece_type_color in range(12)
    }
    assert len(values) == 64 * FEATURES_PER_KING


def test_move_delta_quiet_pawn() -> None:
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")

    white_leave, white_enter = halfka_delta(board, move, "w")
    black_leave, black_enter = halfka_delta(board, move, "b")

    assert len(white_leave) == 1
    assert len(white_enter) == 1
    assert len(black_leave) == 1
    assert len(black_enter) == 1


def test_move_delta_capture() -> None:
    board = chess.Board("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1")
    move = chess.Move.from_uci("e4d5")

    white_leave, white_enter = halfka_delta(board, move, "w")
    black_leave, black_enter = halfka_delta(board, move, "b")

    assert len(white_leave) == 2
    assert len(white_enter) == 1
    assert len(black_leave) == 2
    assert len(black_enter) == 1


def test_move_delta_promotion() -> None:
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    move = chess.Move.from_uci("a7a8q")

    white_leave, white_enter = halfka_delta(board, move, "w")
    black_leave, black_enter = halfka_delta(board, move, "b")

    assert len(white_leave) == 1
    assert len(white_enter) == 1
    assert len(black_leave) == 1
    assert len(black_enter) == 1


def test_move_delta_castling_marks_king_move() -> None:
    board = chess.Board("4k2r/8/8/8/8/8/8/R3K2R w K - 0 1")
    move = chess.Move.from_uci("e1g1")

    assert is_king_move(board, move, "w") is True
    assert is_king_move(board, move, "b") is False

    black_leave, black_enter = halfka_delta(board, move, "b")
    assert len(black_leave) == 1
    assert len(black_enter) == 1


def test_move_delta_consistency_with_full_rebuild() -> None:
    rng = random.Random(11)
    boards: list[chess.Board] = []
    current = chess.Board()
    for _ in range(25):
        board = current.copy()
        for _step in range(rng.randint(0, 12)):
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            board.push(rng.choice(legal_moves))
        boards.append(board)

    for board in boards:
        legal_moves = list(board.legal_moves)[:10]
        for move in legal_moves:
            for perspective in ("w", "b"):
                if is_king_move(board, move, perspective):
                    continue
                before = set(halfka_active_indices(board, perspective))
                leave, enter = halfka_delta(board, move, perspective)
                after_board = board.copy()
                after_board.push(move)
                after = set(halfka_active_indices(after_board, perspective))
                rebuilt = before.difference(leave).union(enter)
                assert rebuilt == after


def test_old_artifact_still_loads(tmp_path: Path) -> None:
    source = lapv1_training_example_from_planner_head(
        _planner_example("legacy", teacher_index=0, teacher_cp=25.0, teacher_gap=15.0)
    )
    legacy_payload = source.to_dict()
    for field_name in (
        "side_to_move",
        "phase_index",
        "king_sq_white",
        "king_sq_black",
        "nnue_feat_white",
        "nnue_feat_black",
        "candidate_move_types",
        "candidate_delta_white_leave",
        "candidate_delta_white_enter",
        "candidate_delta_black_leave",
        "candidate_delta_black_enter",
        "candidate_is_white_king_move",
        "candidate_is_black_king_move",
    ):
        legacy_payload.pop(field_name)

    artifact_path = tmp_path / "legacy.jsonl"
    artifact_path.write_text(json.dumps(legacy_payload) + "\n", encoding="utf-8")

    with pytest.warns(RuntimeWarning):
        loaded = load_lapv1_training_examples(artifact_path)

    assert loaded[0].side_to_move == 0
    assert loaded[0].phase_index == 0
    assert loaded[0].king_sq_white == -1
    assert loaded[0].candidate_move_types == [0] * len(source.candidate_action_indices)


def test_move_type_hash_in_range() -> None:
    board = chess.Board()
    for move in list(board.legal_moves):
        value = move_type_hash(board, move)
        assert 0 <= value < 128


def test_lapv1_training_example_contains_phase_and_delta_fields() -> None:
    example = lapv1_training_example_from_planner_head(
        _planner_example("roundtrip", teacher_index=0, teacher_cp=25.0, teacher_gap=15.0)
    )

    assert example.phase_index == 3
    assert example.side_to_move == 0
    assert example.king_sq_white == chess.E1
    assert example.king_sq_black == chess.E8
    assert len(example.nnue_feat_white) > 0
    assert len(example.candidate_move_types) == len(example.candidate_action_indices)
    assert len(example.candidate_delta_white_leave) == len(example.candidate_action_indices)
    assert len(example.candidate_is_black_king_move) == len(example.candidate_action_indices)


def _planner_example(
    sample_id: str,
    *,
    teacher_index: int,
    teacher_cp: float,
    teacher_gap: float,
) -> PlannerHeadExample:
    feature_vector = pack_position_features(
        PositionEncoding(
            piece_tokens=[[4, 0, 5], [60, 1, 5], [0, 0, 3], [63, 1, 3]],
            square_tokens=[[square_index, 0] for square_index in range(64)],
            rule_token=[0, 0, -1, 0, 1, 0],
        )
    )
    candidate_features = [
        [1.0] + [0.0] * 34,
        [0.0, 1.0] + [0.0] * 33,
    ]
    teacher_policy = [0.0, 0.0]
    teacher_policy[teacher_index] = 1.0
    return PlannerHeadExample(
        sample_id=sample_id,
        split="train",
        fen="4k2r/8/8/8/8/8/8/R3K3 w - - 0 1",
        feature_vector=feature_vector,
        candidate_context_version=2,
        global_context_version=1,
        global_features=[0.0] * SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE,
        candidate_action_indices=[1, 2],
        candidate_features=candidate_features,
        proposer_scores=[0.1, 0.2],
        transition_context_version=1,
        transition_features=[[0.0] * 8, [0.0] * 8],
        reply_peak_probabilities=[0.2, 0.1],
        pressures=[0.1, 0.2],
        uncertainties=[0.2, 0.1],
        curriculum_bucket_labels=["unit_test"],
        curriculum_priority=1.0,
        teacher_top1_action_index=teacher_index + 1,
        teacher_top1_candidate_index=teacher_index,
        teacher_policy=teacher_policy,
        teacher_root_value_cp=teacher_cp,
        teacher_top1_minus_top2_cp=teacher_gap,
        teacher_candidate_rank_bucket_targets=[0, 1],
    )
