from __future__ import annotations

import json
from pathlib import Path

import pytest

from train.config import load_planner_train_config
from train.datasets.contracts import candidate_context_feature_dim
from train.models.intention_encoder import STATE_CONTEXT_V1_GLOBAL_FEATURE_DIM, torch
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
    return {
        "piece_tokens": piece_tokens,
        "square_tokens": square_tokens,
        "state_context_v1_global": global_features,
        "reachability_edges": reachability_edges,
        "candidate_context_v2": candidate_context,
        "candidate_action_indices": candidate_action_indices,
        "candidate_mask": candidate_mask,
        "phase_index": torch.tensor([0, 1], dtype=torch.long)[:batch_size],
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
