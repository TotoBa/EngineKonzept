from __future__ import annotations

import json
from pathlib import Path

import pytest
import chess

from train.config import load_planner_train_config
from train.datasets.contracts import candidate_context_feature_dim
from train.datasets.move_delta import halfka_delta, is_king_move, move_type_hash
from train.datasets.nnue_features import halfka_active_indices
from train.models.intention_encoder import STATE_CONTEXT_V1_GLOBAL_FEATURE_DIM, torch
from train.models.dual_accumulator import build_sparse_rows, pack_sparse_feature_lists
from train.models.lapv1 import LAPv1Config, LAPv1Model


pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch not installed")


def _sample_inputs(
    *,
    batch_size: int = 2,
    candidate_count: int = 5,
) -> dict[str, torch.Tensor]:
    piece_tokens = torch.full((batch_size, 32, 3), -1, dtype=torch.long)
    piece_tokens[:, 0] = torch.tensor([4, 0, 5], dtype=torch.long)
    piece_tokens[:, 1] = torch.tensor([0, 0, 3], dtype=torch.long)
    piece_tokens[:, 2] = torch.tensor([7, 0, 1], dtype=torch.long)
    piece_tokens[:, 3] = torch.tensor([60, 1, 5], dtype=torch.long)
    piece_tokens[:, 4] = torch.tensor([63, 1, 3], dtype=torch.long)

    square_indices = torch.arange(64, dtype=torch.float32).reshape(1, 64, 1) / 63.0
    square_tokens = torch.cat(
        [square_indices, torch.zeros((1, 64, 1), dtype=torch.float32)],
        dim=2,
    ).repeat(batch_size, 1, 1)
    global_features = torch.linspace(
        0.0,
        1.0,
        STATE_CONTEXT_V1_GLOBAL_FEATURE_DIM,
        dtype=torch.float32,
    ).unsqueeze(0).repeat(batch_size, 1)
    reachability_edges = torch.full((batch_size, 16, 3), -1, dtype=torch.long)
    reachability_edges[:, :7, :] = torch.tensor(
        [
            [4, 12, 5],
            [0, 8, 3],
            [7, 15, 1],
            [60, 52, 5],
            [63, 55, 3],
            [12, 20, 5],
            [52, 44, 5],
        ],
        dtype=torch.long,
    )
    feature_dim = candidate_context_feature_dim(2)
    candidate_context = torch.randn(
        (batch_size, candidate_count, feature_dim),
        dtype=torch.float32,
    )
    candidate_action_indices = torch.tensor(
        [[1, 17, 42, 103, 511], [5, 9, 18, 0, 0]],
        dtype=torch.long,
    )
    candidate_mask = torch.tensor(
        [[True, True, True, True, False], [True, True, True, False, False]],
        dtype=torch.bool,
    )
    if batch_size != 2:
        candidate_action_indices = torch.arange(
            1,
            batch_size * candidate_count + 1,
            dtype=torch.long,
        ).reshape(batch_size, candidate_count)
        candidate_mask = torch.ones((batch_size, candidate_count), dtype=torch.bool)
    nnue_feat_white_indices, nnue_feat_white_offsets = pack_sparse_feature_lists(
        [
            [1 + index, 11 + index, 21 + index]
            for index in range(batch_size)
        ]
    )
    nnue_feat_black_indices, nnue_feat_black_offsets = pack_sparse_feature_lists(
        [
            [31 + index, 41 + index, 51 + index]
            for index in range(batch_size)
        ]
    )
    candidate_delta_white_leave_indices, candidate_delta_white_leave_offsets = (
        pack_sparse_feature_lists(
            [
                [1 + row_index]
                for row_index in range(batch_size * candidate_count)
            ]
        )
    )
    candidate_delta_white_enter_indices, candidate_delta_white_enter_offsets = (
        pack_sparse_feature_lists(
            [
                [101 + row_index]
                for row_index in range(batch_size * candidate_count)
            ]
        )
    )
    candidate_delta_black_leave_indices, candidate_delta_black_leave_offsets = (
        pack_sparse_feature_lists(
            [
                [201 + row_index]
                for row_index in range(batch_size * candidate_count)
            ]
        )
    )
    candidate_delta_black_enter_indices, candidate_delta_black_enter_offsets = (
        pack_sparse_feature_lists(
            [
                [301 + row_index]
                for row_index in range(batch_size * candidate_count)
            ]
        )
    )
    candidate_nnue_feat_white_after_move_indices, candidate_nnue_feat_white_after_move_offsets = (
        pack_sparse_feature_lists(
            [
                [401 + row_index, 501 + row_index]
                for row_index in range(batch_size * candidate_count)
            ]
        )
    )
    candidate_nnue_feat_black_after_move_indices, candidate_nnue_feat_black_after_move_offsets = (
        pack_sparse_feature_lists(
            [
                [601 + row_index, 701 + row_index]
                for row_index in range(batch_size * candidate_count)
            ]
        )
    )
    return {
        "piece_tokens": piece_tokens,
        "square_tokens": square_tokens,
        "state_context_v1_global": global_features,
        "reachability_edges": reachability_edges,
        "candidate_context_v2": candidate_context,
        "candidate_action_indices": candidate_action_indices,
        "candidate_mask": candidate_mask,
        "side_to_move": (torch.arange(batch_size, dtype=torch.long) % 2),
        "nnue_feat_white_indices": nnue_feat_white_indices,
        "nnue_feat_white_offsets": nnue_feat_white_offsets,
        "nnue_feat_black_indices": nnue_feat_black_indices,
        "nnue_feat_black_offsets": nnue_feat_black_offsets,
        "candidate_move_types": (
            torch.arange(batch_size * candidate_count, dtype=torch.long).reshape(
                batch_size,
                candidate_count,
            )
            % 128
        ),
        "candidate_delta_white_leave_indices": candidate_delta_white_leave_indices,
        "candidate_delta_white_leave_offsets": candidate_delta_white_leave_offsets,
        "candidate_delta_white_enter_indices": candidate_delta_white_enter_indices,
        "candidate_delta_white_enter_offsets": candidate_delta_white_enter_offsets,
        "candidate_delta_black_leave_indices": candidate_delta_black_leave_indices,
        "candidate_delta_black_leave_offsets": candidate_delta_black_leave_offsets,
        "candidate_delta_black_enter_indices": candidate_delta_black_enter_indices,
        "candidate_delta_black_enter_offsets": candidate_delta_black_enter_offsets,
        "candidate_nnue_feat_white_after_move_indices": candidate_nnue_feat_white_after_move_indices,
        "candidate_nnue_feat_white_after_move_offsets": candidate_nnue_feat_white_after_move_offsets,
        "candidate_nnue_feat_black_after_move_indices": candidate_nnue_feat_black_after_move_indices,
        "candidate_nnue_feat_black_after_move_offsets": candidate_nnue_feat_black_after_move_offsets,
        "candidate_has_king_move": torch.zeros((batch_size, candidate_count), dtype=torch.bool),
        "phase_index": (torch.arange(batch_size, dtype=torch.long) % 4),
    }


def test_lapv1_forward_pass_produces_expected_shapes() -> None:
    model = LAPv1Model(LAPv1Config())
    outputs = model(**_sample_inputs())

    assert tuple(outputs["final_policy_logits"].shape) == (2, 5)
    assert tuple(outputs["final_value"]["wdl_logits"].shape) == (2, 3)
    assert tuple(outputs["final_value"]["cp_score"].shape) == (2, 1)
    assert tuple(outputs["final_value"]["sigma_value"].shape) == (2, 1)
    assert tuple(outputs["refined_top1_action_index"].shape) == (2,)
    assert tuple(outputs["initial_policy_logits"].shape) == (2, 5)
    assert tuple(outputs["final_policy_deltas"].shape) == (2, 5)
    assert "step_candidate_score_tensors" in outputs
    assert "step_active_masks" in outputs
    assert "step_rollback_masks" in outputs
    assert "root_candidate_scores" in outputs


def test_lapv1_trace_length_respects_max_inner_steps() -> None:
    config = LAPv1Config.from_mapping({"deliberation": {"max_inner_steps": 2, "min_inner_steps": 1}})
    model = LAPv1Model(config)
    outputs = model(**_sample_inputs())

    assert len(outputs["deliberation_trace"].steps) <= 2


def test_lapv1_output_is_differentiable() -> None:
    model = LAPv1Model(LAPv1Config.from_mapping({"deliberation": {"max_inner_steps": 2}}))
    outputs = model(**_sample_inputs())
    finite_policy = torch.where(
        torch.isfinite(outputs["final_policy_logits"]),
        outputs["final_policy_logits"],
        torch.zeros_like(outputs["final_policy_logits"]),
    )
    loss = (
        finite_policy.sum()
        + outputs["final_value"]["wdl_logits"].sum()
        + outputs["final_value"]["cp_score"].sum()
        + outputs["final_value"]["sigma_value"].sum()
    )
    loss.backward()

    assert model.intention_encoder.square_embedding.weight.grad is not None
    assert model.state_embedder.state_projection[0].weight.grad is not None
    assert model.policy_head.scorer[0].weight.grad is not None
    assert model.value_head.wdl_head.weight.grad is not None
    assert model.opponent_head.context_projection[0].weight.grad is not None


def test_phase_moe_off_is_bit_identical_to_v1() -> None:
    baseline = LAPv1Model(LAPv1Config())
    flagged = LAPv1Model(
        LAPv1Config.from_mapping(
            {
                "lapv2": {
                    "enabled": False,
                    "phase_moe": False,
                }
            }
        )
    )
    flagged.load_state_dict(baseline.state_dict())
    inputs = _sample_inputs()

    baseline_outputs = baseline(**inputs)
    flagged_outputs = flagged(**inputs)

    assert torch.equal(
        baseline_outputs["initial_policy_logits"],
        flagged_outputs["initial_policy_logits"],
    )
    assert torch.equal(
        baseline_outputs["final_policy_logits"],
        flagged_outputs["final_policy_logits"],
    )
    assert torch.equal(baseline_outputs["z_root"], flagged_outputs["z_root"])


def test_flag_off_uses_legacy_value_head() -> None:
    baseline = LAPv1Model(LAPv1Config())
    flagged = LAPv1Model(
        LAPv1Config.from_mapping(
            {
                "lapv2": {
                    "enabled": True,
                    "nnue_value": False,
                }
            }
        )
    )
    flagged.load_state_dict(baseline.state_dict())
    inputs = _sample_inputs()
    outputs_baseline = baseline(**inputs)
    outputs_flagged = flagged(**inputs)

    assert torch.equal(
        outputs_baseline["final_value"]["wdl_logits"],
        outputs_flagged["final_value"]["wdl_logits"],
    )
    assert torch.equal(
        outputs_baseline["final_value"]["cp_score"],
        outputs_flagged["final_value"]["cp_score"],
    )


def test_ft_attribute_exists_for_future_policy() -> None:
    model = LAPv1Model(
        LAPv1Config.from_mapping(
            {
                "lapv2": {
                    "enabled": True,
                    "nnue_value": True,
                    "N_accumulator": 8,
                }
            }
        )
    )

    assert model.ft is not None
    assert model.value_head_nnue is not None


def test_flag_off_uses_legacy_policy() -> None:
    baseline = LAPv1Model(LAPv1Config())
    flagged = LAPv1Model(
        LAPv1Config.from_mapping(
            {
                "lapv2": {
                    "enabled": True,
                    "nnue_value": True,
                    "nnue_policy": False,
                }
            }
        )
    )
    flagged.load_state_dict(baseline.state_dict(), strict=False)
    inputs = _sample_inputs()
    outputs_baseline = baseline(**inputs)
    outputs_flagged = flagged(**inputs)

    assert torch.equal(
        outputs_baseline["initial_policy_logits"],
        outputs_flagged["initial_policy_logits"],
    )


def test_value_and_policy_share_ft_object() -> None:
    model = LAPv1Model(
        LAPv1Config.from_mapping(
            {
                "lapv2": {
                    "enabled": True,
                    "nnue_value": True,
                    "nnue_policy": True,
                    "N_accumulator": 8,
                }
            }
        )
    )

    assert model.ft is not None
    assert model.value_head_nnue is not None
    assert model.policy_head_nnue is not None


def test_value_and_policy_gradients_both_flow_to_ft() -> None:
    model = LAPv1Model(
        LAPv1Config.from_mapping(
            {
                "lapv2": {
                    "enabled": True,
                    "nnue_value": True,
                    "nnue_policy": True,
                    "N_accumulator": 8,
                },
                "deliberation": {"max_inner_steps": 0, "min_inner_steps": 0},
            }
        )
    )
    inputs = _sample_inputs()
    ft_module = model.ft
    assert ft_module is not None
    ft_weight = ft_module.ft.weight

    outputs = model(**inputs)
    model.zero_grad(set_to_none=True)
    outputs["initial_policy_logits"].sum().backward(retain_graph=True)
    policy_grad = ft_weight.grad.detach().clone()
    model.zero_grad(set_to_none=True)
    outputs["final_value"]["cp_score"].sum().backward()
    value_grad = ft_weight.grad.detach().clone()

    assert float(policy_grad.abs().sum().item()) > 0.0
    assert float(value_grad.abs().sum().item()) > 0.0


def test_policy_nnue_score_invariant_under_kings_move() -> None:
    model = LAPv1Model(
        LAPv1Config.from_mapping(
            {
                "lapv2": {
                    "enabled": True,
                    "nnue_value": True,
                    "nnue_policy": True,
                    "N_accumulator": 8,
                },
                "deliberation": {"max_inner_steps": 0, "min_inner_steps": 0},
            }
        )
    )
    board = chess.Board("4k2r/8/8/8/8/8/8/R3K2R w K - 0 1")
    move = chess.Move.from_uci("e1g1")
    phase_idx = torch.tensor([0], dtype=torch.long)
    side_to_move = torch.tensor([0], dtype=torch.long)
    nnue_feat_white_indices, nnue_feat_white_offsets = pack_sparse_feature_lists(
        [halfka_active_indices(board, "w")]
    )
    nnue_feat_black_indices, nnue_feat_black_offsets = pack_sparse_feature_lists(
        [halfka_active_indices(board, "b")]
    )
    a_white, a_black = model._root_dual_accumulators(
        phase_idx=phase_idx,
        nnue_feat_white_indices=nnue_feat_white_indices,
        nnue_feat_white_offsets=nnue_feat_white_offsets,
        nnue_feat_black_indices=nnue_feat_black_indices,
        nnue_feat_black_offsets=nnue_feat_black_offsets,
    )
    after_board = board.copy(stack=False)
    after_board.push(move)
    candidate_nnue_feat_white_after_move_indices, candidate_nnue_feat_white_after_move_offsets = (
        pack_sparse_feature_lists([halfka_active_indices(after_board, "w")])
    )
    candidate_nnue_feat_black_after_move_indices, candidate_nnue_feat_black_after_move_offsets = (
        pack_sparse_feature_lists([halfka_active_indices(after_board, "b")])
    )
    candidate_delta_white_leave_indices, candidate_delta_white_leave_offsets = (
        pack_sparse_feature_lists([halfka_delta(board, move, "w")[0]])
    )
    candidate_delta_white_enter_indices, candidate_delta_white_enter_offsets = (
        pack_sparse_feature_lists([halfka_delta(board, move, "w")[1]])
    )
    candidate_delta_black_leave_indices, candidate_delta_black_leave_offsets = (
        pack_sparse_feature_lists([halfka_delta(board, move, "b")[0]])
    )
    candidate_delta_black_enter_indices, candidate_delta_black_enter_offsets = (
        pack_sparse_feature_lists([halfka_delta(board, move, "b")[1]])
    )
    logits, _caches, _cache_stats = model._nnue_policy_logits(
        a_white=a_white,
        a_black=a_black,
        phase_idx=phase_idx,
        side_to_move=side_to_move,
        candidate_mask=torch.tensor([[True]], dtype=torch.bool),
        candidate_move_types=torch.tensor([[move_type_hash(board, move)]], dtype=torch.long),
        candidate_delta_white_leave_indices=candidate_delta_white_leave_indices,
        candidate_delta_white_leave_offsets=candidate_delta_white_leave_offsets,
        candidate_delta_white_enter_indices=candidate_delta_white_enter_indices,
        candidate_delta_white_enter_offsets=candidate_delta_white_enter_offsets,
        candidate_delta_black_leave_indices=candidate_delta_black_leave_indices,
        candidate_delta_black_leave_offsets=candidate_delta_black_leave_offsets,
        candidate_delta_black_enter_indices=candidate_delta_black_enter_indices,
        candidate_delta_black_enter_offsets=candidate_delta_black_enter_offsets,
        candidate_nnue_feat_white_after_move_indices=candidate_nnue_feat_white_after_move_indices,
        candidate_nnue_feat_white_after_move_offsets=candidate_nnue_feat_white_after_move_offsets,
        candidate_nnue_feat_black_after_move_indices=candidate_nnue_feat_black_after_move_indices,
        candidate_nnue_feat_black_after_move_offsets=candidate_nnue_feat_black_after_move_offsets,
        candidate_has_king_move=torch.tensor(
            [[is_king_move(board, move, "w") or is_king_move(board, move, "b")]],
            dtype=torch.bool,
        ),
    )
    assert model.ft is not None
    after_black = build_sparse_rows(
        model.ft,
        candidate_nnue_feat_black_after_move_indices,
        candidate_nnue_feat_black_after_move_offsets,
        phase_idx=phase_idx,
    )
    assert model.policy_head_nnue is not None
    manual_logits = model.policy_head_nnue(
        a_white,
        after_black.unsqueeze(1),
        torch.tensor([[move_type_hash(board, move)]], dtype=torch.long),
    )

    assert torch.allclose(logits, manual_logits, atol=1e-6)


def test_phase_nnue_value_runs_all_phases() -> None:
    model = LAPv1Model(
        LAPv1Config.from_mapping(
            {
                "lapv2": {
                    "enabled": True,
                    "nnue_value": True,
                    "nnue_value_phase_moe": True,
                    "N_accumulator": 8,
                }
            }
        )
    )

    outputs = model(**_sample_inputs(batch_size=4))

    assert tuple(outputs["final_value"]["wdl_logits"].shape) == (4, 3)
    assert torch.isfinite(outputs["final_value"]["cp_score"]).all()


def test_sharpness_phase_moe_runs() -> None:
    model = LAPv1Model(
        LAPv1Config.from_mapping(
            {
                "lapv2": {
                    "enabled": True,
                    "sharpness_phase_moe": True,
                }
            }
        )
    )

    outputs = model(**_sample_inputs(batch_size=4))

    assert tuple(outputs["root_sharpness"].shape) == (4,)
    assert torch.isfinite(outputs["root_sharpness"]).all()


def test_sharpness_phase_flag_off_is_bit_identical_to_v1() -> None:
    baseline = LAPv1Model(LAPv1Config())
    flagged = LAPv1Model(
        LAPv1Config.from_mapping(
            {
                "lapv2": {
                    "enabled": True,
                    "sharpness_phase_moe": False,
                }
            }
        )
    )
    flagged.load_state_dict(baseline.state_dict(), strict=False)
    inputs = _sample_inputs()

    baseline_outputs = baseline(**inputs)
    flagged_outputs = flagged(**inputs)

    assert torch.equal(
        baseline_outputs["root_sharpness"],
        flagged_outputs["root_sharpness"],
    )


def test_shared_opponent_readout_runs() -> None:
    model = LAPv1Model(
        LAPv1Config.from_mapping(
            {
                "lapv2": {
                    "enabled": True,
                    "shared_opponent_readout": True,
                },
                "deliberation": {"max_inner_steps": 2, "min_inner_steps": 1},
            }
        )
    )

    outputs = model(**_sample_inputs(batch_size=4))

    assert tuple(outputs["final_policy_logits"].shape) == (4, 5)
    assert torch.isfinite(outputs["final_policy_logits"]).all()


def test_flag_off_uses_legacy_opponent() -> None:
    baseline = LAPv1Model(
        LAPv1Config.from_mapping(
            {
                "deliberation": {"max_inner_steps": 2, "min_inner_steps": 1},
            }
        )
    )
    flagged = LAPv1Model(
        LAPv1Config.from_mapping(
            {
                "lapv2": {
                    "enabled": True,
                    "shared_opponent_readout": False,
                },
                "deliberation": {"max_inner_steps": 2, "min_inner_steps": 1},
            }
        )
    )
    flagged.load_state_dict(baseline.state_dict(), strict=False)
    inputs = _sample_inputs()

    baseline_outputs = baseline(**inputs)
    flagged_outputs = flagged(**inputs)

    assert torch.equal(
        baseline_outputs["initial_policy_logits"],
        flagged_outputs["initial_policy_logits"],
    )
    assert torch.equal(
        baseline_outputs["final_policy_logits"],
        flagged_outputs["final_policy_logits"],
    )


def test_runtime_unchanged_with_distill_flag_on() -> None:
    baseline = LAPv1Model(
        LAPv1Config.from_mapping(
            {
                "lapv2": {
                    "enabled": True,
                    "shared_opponent_readout": True,
                },
                "deliberation": {"max_inner_steps": 2, "min_inner_steps": 1},
            }
        )
    )
    flagged = LAPv1Model(
        LAPv1Config.from_mapping(
            {
                "lapv2": {
                    "enabled": True,
                    "shared_opponent_readout": True,
                    "distill_opponent": True,
                    "distill_fraction": 1.0,
                },
                "deliberation": {"max_inner_steps": 2, "min_inner_steps": 1},
            }
        )
    )
    flagged.load_state_dict(baseline.state_dict(), strict=False)
    inputs = _sample_inputs()

    baseline_outputs = baseline(**inputs)
    flagged_outputs = flagged(**inputs)

    assert torch.equal(
        baseline_outputs["initial_policy_logits"],
        flagged_outputs["initial_policy_logits"],
    )
    assert torch.equal(
        baseline_outputs["final_policy_logits"],
        flagged_outputs["final_policy_logits"],
    )


def test_phase_constant_over_loop() -> None:
    model = LAPv1Model(
        LAPv1Config.from_mapping(
            {
                "deliberation": {
                    "max_inner_steps": 2,
                    "min_inner_steps": 2,
                    "top_k_refine": 2,
                },
                "lapv2": {
                    "enabled": True,
                    "nnue_value": True,
                    "nnue_value_phase_moe": True,
                    "nnue_policy": True,
                    "accumulator_cache": True,
                    "N_accumulator": 8,
                },
            }
        )
    )
    model.eval()
    inputs = _sample_inputs(batch_size=1, candidate_count=5)
    inputs["phase_index"] = torch.tensor([3], dtype=torch.long)

    with torch.inference_mode():
        outputs = model(**inputs)

    assert len(outputs["step_phase_indices"]) == 2
    for step_phase in outputs["step_phase_indices"]:
        assert torch.equal(step_phase, inputs["phase_index"])
    assert outputs["accumulator_cache_stats"]["phase_fixed"] is True
    assert outputs["accumulator_cache_stats"]["phase_indices"] == (3,)


def test_accumulator_cache_eval_matches_no_cache() -> None:
    baseline = LAPv1Model(
        LAPv1Config.from_mapping(
            {
                "deliberation": {
                    "max_inner_steps": 2,
                    "min_inner_steps": 2,
                    "top_k_refine": 2,
                },
                "lapv2": {
                    "enabled": True,
                    "nnue_value": True,
                    "nnue_value_phase_moe": True,
                    "nnue_policy": True,
                    "accumulator_cache": False,
                    "N_accumulator": 8,
                },
            }
        )
    )
    cached = LAPv1Model(
        LAPv1Config.from_mapping(
            {
                "deliberation": {
                    "max_inner_steps": 2,
                    "min_inner_steps": 2,
                    "top_k_refine": 2,
                },
                "lapv2": {
                    "enabled": True,
                    "nnue_value": True,
                    "nnue_value_phase_moe": True,
                    "nnue_policy": True,
                    "accumulator_cache": True,
                    "N_accumulator": 8,
                },
            }
        )
    )
    cached.load_state_dict(baseline.state_dict())
    baseline.eval()
    cached.eval()
    inputs = _sample_inputs(batch_size=1, candidate_count=5)
    inputs["phase_index"] = torch.tensor([2], dtype=torch.long)
    inputs["candidate_has_king_move"][0, 1] = True

    with torch.inference_mode():
        baseline_outputs = baseline(**inputs)
        cached_outputs = cached(**inputs)

    assert torch.equal(
        baseline_outputs["initial_policy_logits"],
        cached_outputs["initial_policy_logits"],
    )
    assert torch.equal(
        baseline_outputs["final_policy_logits"],
        cached_outputs["final_policy_logits"],
    )
    assert torch.equal(
        baseline_outputs["final_value"]["wdl_logits"],
        cached_outputs["final_value"]["wdl_logits"],
    )
    assert torch.equal(
        baseline_outputs["final_value"]["cp_score"],
        cached_outputs["final_value"]["cp_score"],
    )
    assert torch.equal(
        baseline_outputs["final_value"]["sigma_value"],
        cached_outputs["final_value"]["sigma_value"],
    )
    assert torch.equal(
        baseline_outputs["refined_top1_action_index"],
        cached_outputs["refined_top1_action_index"],
    )


def test_cache_hit_for_top_k_candidates() -> None:
    model = LAPv1Model(
        LAPv1Config.from_mapping(
            {
                "deliberation": {
                    "max_inner_steps": 2,
                    "min_inner_steps": 2,
                    "top_k_refine": 2,
                },
                "lapv2": {
                    "enabled": True,
                    "nnue_value": True,
                    "nnue_value_phase_moe": True,
                    "nnue_policy": True,
                    "accumulator_cache": True,
                    "N_accumulator": 8,
                },
            }
        )
    )
    model.eval()
    inputs = _sample_inputs(batch_size=1, candidate_count=5)
    inputs["phase_index"] = torch.tensor([1], dtype=torch.long)

    with torch.inference_mode():
        outputs = model(**inputs)

    assert len(outputs["step_selected_candidate_tensors"]) == 2
    assert outputs["accumulator_cache_stats"]["enabled"] is True
    assert outputs["accumulator_cache_stats"]["lookup_hits"] == 0
    assert outputs["accumulator_cache_stats"]["lookup_misses"] == 5
    assert outputs["accumulator_cache_stats"]["touch_hits"] == 4
    assert outputs["accumulator_cache_stats"]["touch_misses"] == 0
    assert outputs["accumulator_cache_stats"]["cached_candidate_count"] == 5


def test_lapv1_total_parameter_budget_is_within_target_band() -> None:
    model = LAPv1Model(LAPv1Config())
    parameter_bytes = sum(parameter.numel() for parameter in model.parameters()) * 4
    lower_bound = int(200 * 1024 * 1024 * 0.7)
    upper_bound = int(300 * 1024 * 1024 * 1.3)

    assert lower_bound <= parameter_bytes <= upper_bound


def test_load_planner_train_config_accepts_lapv1_wrapper(tmp_path: Path) -> None:
    config_path = tmp_path / "planner_lapv1_config.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 7,
                "output_dir": "planner_out",
                "lapv1": {
                    "deliberation": {
                        "max_inner_steps": 2,
                        "min_inner_steps": 1,
                    }
                },
                "data": {
                    "train_path": "planner_head_train.jsonl",
                    "validation_path": "planner_head_validation.jsonl",
                },
                "model": {
                    "architecture": "lapv1",
                    "hidden_dim": 64,
                    "hidden_layers": 1,
                    "action_embedding_dim": 16,
                    "latent_feature_dim": 0,
                    "dropout": 0.0,
                },
                "optimization": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "teacher_policy_loss_weight": 1.0,
                    "teacher_kl_loss_weight": 0.25,
                },
                "evaluation": {"top_k": 3},
                "runtime": {"torch_threads": 1, "dataloader_workers": 0},
                "export": {"bundle_dir": "planner_bundle"},
            }
        ),
        encoding="utf-8",
    )

    loaded = load_planner_train_config(config_path)

    assert loaded.model.architecture == "lapv1"
    assert loaded.lapv1 is not None
    assert loaded.lapv1["deliberation"]["max_inner_steps"] == 2
