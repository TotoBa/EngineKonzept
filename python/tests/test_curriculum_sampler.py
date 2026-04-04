from __future__ import annotations

from train.config import PlannerTrainConfig
from train.datasets.curriculum import CurriculumSampler
from train.datasets.planner_head import PlannerHeadExample, compute_curriculum_weights
from train.models.planner import torch


def _planner_example(
    *,
    sample_id: str,
    candidate_count: int,
    teacher_root_value_cp: float,
    teacher_top1_minus_top2_cp: float | None,
    teacher_policy: list[float],
) -> PlannerHeadExample:
    candidate_action_indices = list(range(candidate_count))
    return PlannerHeadExample(
        sample_id=sample_id,
        split="train",
        fen="startpos",
        feature_vector=[0.0, 1.0],
        candidate_context_version=2,
        global_context_version=1,
        global_features=[0.0],
        candidate_action_indices=candidate_action_indices,
        candidate_features=[[0.0] for _ in candidate_action_indices],
        proposer_scores=[0.0 for _ in candidate_action_indices],
        transition_context_version=1,
        transition_features=[[0.0] for _ in candidate_action_indices],
        reply_peak_probabilities=[0.0 for _ in candidate_action_indices],
        pressures=[0.0 for _ in candidate_action_indices],
        uncertainties=[0.0 for _ in candidate_action_indices],
        curriculum_bucket_labels=["test"],
        curriculum_priority=0.0,
        teacher_top1_action_index=0,
        teacher_top1_candidate_index=0,
        teacher_policy=teacher_policy,
        teacher_root_value_cp=teacher_root_value_cp,
        teacher_top1_minus_top2_cp=teacher_top1_minus_top2_cp,
    )


def test_compute_curriculum_weights_sum_matches_example_count() -> None:
    examples = [
        _planner_example(
            sample_id="easy",
            candidate_count=2,
            teacher_root_value_cp=20.0,
            teacher_top1_minus_top2_cp=160.0,
            teacher_policy=[0.95, 0.05],
        ),
        _planner_example(
            sample_id="hard",
            candidate_count=5,
            teacher_root_value_cp=320.0,
            teacher_top1_minus_top2_cp=5.0,
            teacher_policy=[0.25, 0.22, 0.20, 0.18, 0.15],
        ),
    ]
    weights = compute_curriculum_weights(
        examples,
        strategy="linear_ramp",
        epoch=2,
        total_epochs=5,
    )
    assert len(weights) == 2
    assert all(weight > 0.0 for weight in weights)
    assert abs(sum(weights) - len(examples)) < 1e-6


def test_linear_ramp_increases_hard_example_weight_over_epochs() -> None:
    examples = [
        _planner_example(
            sample_id="easy",
            candidate_count=2,
            teacher_root_value_cp=10.0,
            teacher_top1_minus_top2_cp=200.0,
            teacher_policy=[0.98, 0.02],
        ),
        _planner_example(
            sample_id="hard",
            candidate_count=6,
            teacher_root_value_cp=400.0,
            teacher_top1_minus_top2_cp=4.0,
            teacher_policy=[0.21, 0.20, 0.18, 0.16, 0.14, 0.11],
        ),
    ]
    early_weights = compute_curriculum_weights(
        examples,
        strategy="linear_ramp",
        epoch=0,
        total_epochs=12,
    )
    late_weights = compute_curriculum_weights(
        examples,
        strategy="linear_ramp",
        epoch=11,
        total_epochs=12,
    )
    assert early_weights[0] > early_weights[1]
    assert late_weights[1] > late_weights[0]
    assert late_weights[1] > early_weights[1]


def test_uniform_curriculum_sampler_matches_current_shuffle_order() -> None:
    assert torch is not None
    examples = [
        _planner_example(
            sample_id=f"example_{index}",
            candidate_count=2,
            teacher_root_value_cp=float(index),
            teacher_top1_minus_top2_cp=40.0,
            teacher_policy=[0.8, 0.2],
        )
        for index in range(8)
    ]
    sampler = CurriculumSampler(examples, strategy="uniform", seed=17, total_epochs=4)
    sampler.set_epoch(0)
    sampled_indices = list(iter(sampler))

    generator = torch.Generator()
    generator.manual_seed(17)
    expected_indices = [int(index) for index in torch.randperm(len(examples), generator=generator).tolist()]
    assert sampled_indices == expected_indices


def test_planner_config_accepts_optional_curriculum_section() -> None:
    config = PlannerTrainConfig.from_dict(
        {
            "seed": 7,
            "output_dir": "/tmp/planner",
            "data": {
                "train_path": "artifacts/train.jsonl",
                "validation_path": "artifacts/validation.jsonl",
            },
            "curriculum": {
                "strategy": "sqrt_ramp",
                "value_spread_weight": 0.5,
                "candidate_count_weight": 1.0,
                "agreement_weight": 2.0,
            },
            "model": {
                "architecture": "set_v2",
                "hidden_dim": 64,
                "hidden_layers": 1,
                "action_embedding_dim": 32,
                "dropout": 0.0,
            },
            "optimization": {
                "epochs": 1,
                "batch_size": 8,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "teacher_policy_loss_weight": 1.0,
            },
            "evaluation": {"top_k": 3},
            "runtime": {"torch_threads": 0, "dataloader_workers": 0},
            "export": {"bundle_dir": "/tmp/model", "checkpoint_name": "checkpoint.pt"},
        }
    )
    assert config.curriculum is not None
    assert config.curriculum.strategy == "sqrt_ramp"
    assert config.curriculum.agreement_weight == 2.0
