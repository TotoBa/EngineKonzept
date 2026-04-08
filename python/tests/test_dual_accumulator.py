from __future__ import annotations

import random

import pytest

from train.datasets.artifacts import (
    SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE,
    pack_position_features,
)
from train.datasets.lapv1_training import lapv1_training_example_from_planner_head
from train.datasets.move_delta import halfka_delta, is_king_move
from train.datasets.nnue_features import TOTAL_FEATURES, halfka_active_indices
from train.datasets.planner_head import PlannerHeadExample
from train.datasets.schema import PositionEncoding
from train.models.dual_accumulator import (
    DualAccumulatorBuilder,
    IncrementalAccumulator,
    pack_sparse_feature_lists,
)
from train.models.feature_transformer import FeatureTransformer
from train.trainers.lapv1 import _collate_examples


torch = pytest.importorskip("torch")
chess = pytest.importorskip("chess")


def test_ft_build_shape() -> None:
    ft = FeatureTransformer(num_features=TOTAL_FEATURES, accumulator_dim=8)
    indices, offsets = pack_sparse_feature_lists([[1, 2, 3], [4, 5]])

    built = ft.build(indices, offsets)

    assert tuple(built.shape) == (2, 8)


def test_ft_build_matches_manual_sum() -> None:
    ft = FeatureTransformer(num_features=TOTAL_FEATURES, accumulator_dim=4)
    with torch.no_grad():
        ft.ft.weight.copy_(
            torch.arange(TOTAL_FEATURES * 4, dtype=torch.float32).reshape(TOTAL_FEATURES, 4)
        )
    indices, offsets = pack_sparse_feature_lists([[1, 7], [2, 3, 9]])

    built = ft.build(indices, offsets)
    expected_0 = ft.ft.weight[torch.tensor([1, 7], dtype=torch.long)].sum(dim=0)
    expected_1 = ft.ft.weight[torch.tensor([2, 3, 9], dtype=torch.long)].sum(dim=0)

    assert torch.allclose(built[0], expected_0)
    assert torch.allclose(built[1], expected_1)


def test_dual_accumulator_independent_white_black() -> None:
    ft = FeatureTransformer(num_features=TOTAL_FEATURES, accumulator_dim=4)
    builder = DualAccumulatorBuilder()
    batch = {
        "nnue_feat_white_indices": torch.tensor([1, 2, 3], dtype=torch.long),
        "nnue_feat_white_offsets": torch.tensor([0, 2], dtype=torch.long),
        "nnue_feat_black_indices": torch.tensor([9, 10, 11], dtype=torch.long),
        "nnue_feat_black_offsets": torch.tensor([0, 1], dtype=torch.long),
    }

    a_white, a_black = builder(ft, batch)

    assert tuple(a_white.shape) == (2, 4)
    assert tuple(a_black.shape) == (2, 4)
    assert not torch.allclose(a_white, a_black)


def test_incremental_apply_move_matches_full_rebuild() -> None:
    rng = random.Random(19)
    ft = FeatureTransformer(num_features=TOTAL_FEATURES, accumulator_dim=8)
    with torch.no_grad():
        ft.ft.weight.normal_(mean=0.0, std=0.1)

    boards: list[chess.Board] = []
    for _ in range(12):
        board = chess.Board()
        for _step in range(rng.randint(0, 10)):
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            board.push(rng.choice(legal_moves))
        boards.append(board)

    for board in boards:
        for move in list(board.legal_moves)[:10]:
            inc = IncrementalAccumulator(ft)
            inc.init_from_position(
                {
                    "nnue_feat_white": halfka_active_indices(board, "w"),
                    "nnue_feat_black": halfka_active_indices(board, "b"),
                }
            )
            after_board = board.copy()
            after_board.push(move)

            def full_rebuild() -> tuple[torch.Tensor, torch.Tensor]:
                white_indices, white_offsets = pack_sparse_feature_lists(
                    [halfka_active_indices(after_board, "w")]
                )
                black_indices, black_offsets = pack_sparse_feature_lists(
                    [halfka_active_indices(after_board, "b")]
                )
                return (
                    ft.build(white_indices, white_offsets),
                    ft.build(black_indices, black_offsets),
                )

            leave_w, enter_w = halfka_delta(board, move, "w")
            leave_b, enter_b = halfka_delta(board, move, "b")
            inc.apply_move(
                leave_w,
                enter_w,
                leave_b,
                enter_b,
                is_king_w=is_king_move(board, move, "w"),
                is_king_b=is_king_move(board, move, "b"),
                full_rebuild_fn=full_rebuild,
            )

            rebuilt_w, rebuilt_b = full_rebuild()
            assert torch.allclose(inc.get("w"), rebuilt_w, atol=1e-6)
            assert torch.allclose(inc.get("b"), rebuilt_b, atol=1e-6)


def test_incremental_king_move_triggers_rebuild() -> None:
    ft = FeatureTransformer(num_features=TOTAL_FEATURES, accumulator_dim=4)
    board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    move = chess.Move.from_uci("e1e2")
    inc = IncrementalAccumulator(ft)
    inc.init_from_position(
        {
            "nnue_feat_white": halfka_active_indices(board, "w"),
            "nnue_feat_black": halfka_active_indices(board, "b"),
        }
    )

    after_board = board.copy()
    after_board.push(move)

    def full_rebuild() -> tuple[torch.Tensor, torch.Tensor]:
        white_indices, white_offsets = pack_sparse_feature_lists(
            [halfka_active_indices(after_board, "w")]
        )
        black_indices, black_offsets = pack_sparse_feature_lists(
            [halfka_active_indices(after_board, "b")]
        )
        return (
            ft.build(white_indices, white_offsets),
            ft.build(black_indices, black_offsets),
        )

    inc.apply_move(
        *halfka_delta(board, move, "w"),
        *halfka_delta(board, move, "b"),
        is_king_w=True,
        is_king_b=False,
        full_rebuild_fn=full_rebuild,
    )

    assert inc.dirty_white is True
    assert torch.allclose(inc.get("w"), full_rebuild()[0], atol=1e-6)
    assert inc.dirty_white is False


def test_dirty_flag_lazy_rebuild() -> None:
    ft = FeatureTransformer(num_features=TOTAL_FEATURES, accumulator_dim=4)
    board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    move = chess.Move.from_uci("e1e2")
    inc = IncrementalAccumulator(ft)
    inc.init_from_position(
        {
            "nnue_feat_white": halfka_active_indices(board, "w"),
            "nnue_feat_black": halfka_active_indices(board, "b"),
        }
    )
    after_board = board.copy()
    after_board.push(move)
    rebuild_calls = 0

    def full_rebuild() -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal rebuild_calls
        rebuild_calls += 1
        white_indices, white_offsets = pack_sparse_feature_lists(
            [halfka_active_indices(after_board, "w")]
        )
        black_indices, black_offsets = pack_sparse_feature_lists(
            [halfka_active_indices(after_board, "b")]
        )
        return (
            ft.build(white_indices, white_offsets),
            ft.build(black_indices, black_offsets),
        )

    inc.apply_move(
        *halfka_delta(board, move, "w"),
        *halfka_delta(board, move, "b"),
        is_king_w=True,
        is_king_b=False,
        full_rebuild_fn=full_rebuild,
    )

    assert rebuild_calls == 0
    inc.get("w")
    assert rebuild_calls == 1


def test_collate_examples_emits_sparse_embedding_bag_inputs() -> None:
    examples = [
        lapv1_training_example_from_planner_head(
            _planner_example("one", teacher_index=0, teacher_cp=20.0, teacher_gap=10.0)
        ),
        lapv1_training_example_from_planner_head(
            _planner_example("two", teacher_index=1, teacher_cp=15.0, teacher_gap=5.0)
        ),
    ]

    batch = _collate_examples(examples)

    assert "nnue_feat_white_indices" in batch
    assert "nnue_feat_white_offsets" in batch
    assert "nnue_feat_black_indices" in batch
    assert "nnue_feat_black_offsets" in batch
    assert batch["nnue_feat_white_offsets"].shape[0] == 2
    assert batch["nnue_feat_black_offsets"].shape[0] == 2


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
