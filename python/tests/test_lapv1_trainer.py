from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from train.config import (
    PlannerDataConfig,
    PlannerEvaluationConfig,
    PlannerExportConfig,
    PlannerRuntimeConfig,
)
from train.datasets.artifacts import (
    SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE,
    pack_position_features,
)
from train.datasets.lapv1_training import (
    lapv1_training_example_from_planner_head,
    load_lapv1_training_examples,
    write_lapv1_training_artifact,
)
from train.datasets.planner_head import PlannerHeadExample
from train.datasets.schema import PositionEncoding
from train.models.lapv1 import LAPV1_MODEL_NAME, LAPv1Config, LAPv1Model
from train.models.intention_encoder import torch
from train.trainers import evaluate_lapv1_checkpoint, train_lapv1
from train.trainers.lapv1 import (
    LAPv1OptimizationConfig,
    LAPv1Stage2Config,
    LAPv1Stage2PhaseConfig,
    LAPv1TrainConfig,
    build_lapv2_warm_start_checkpoint,
    _apply_lapv2_phase_gate_mean_pull,
    _load_lapv1_model_state,
    _build_lazy_dataset,
    _collate_examples,
    _improvement_over_root_loss,
    _normalize_lapv2_shared_loss,
    _phase_load_balance_weights,
    _opponent_distill_loss,
    _policy_margin_loss,
    _prepare_example,
    _trace_policy_ce_loss,
)


pytest.importorskip("torch")


def test_train_and_evaluate_lapv1_stage1_on_tiny_cpu_dataset(tmp_path: Path) -> None:
    train_path = tmp_path / "lapv1_train.jsonl"
    validation_path = tmp_path / "lapv1_validation.jsonl"
    _write_examples(
        train_path,
        [
            _planner_example("train-1", teacher_index=0, teacher_cp=60.0, teacher_gap=40.0),
            _planner_example("train-2", teacher_index=1, teacher_cp=10.0, teacher_gap=10.0),
            _planner_example("train-3", teacher_index=0, teacher_cp=-40.0, teacher_gap=25.0),
            _planner_example("train-4", teacher_index=1, teacher_cp=35.0, teacher_gap=15.0),
        ],
    )
    _write_examples(
        validation_path,
        [
            _planner_example("validation-1", teacher_index=0, teacher_cp=50.0, teacher_gap=30.0),
            _planner_example("validation-2", teacher_index=1, teacher_cp=0.0, teacher_gap=5.0),
        ],
    )

    config = LAPv1TrainConfig(
        seed=5,
        output_dir=str(tmp_path / "lapv1_out"),
        stage="T1",
        data=PlannerDataConfig(
            train_path=str(train_path),
            validation_path=str(validation_path),
        ),
        model=LAPv1Config.from_mapping(
            {
                "deliberation": {"max_inner_steps": 0, "min_inner_steps": 0},
                "opponent_head": {
                    "architecture": "set_v2",
                    "hidden_dim": 64,
                    "hidden_layers": 1,
                    "action_embedding_dim": 16,
                    "dropout": 0.0,
                },
                "value_head": {"hidden_dim": 1024},
                "policy_head": {
                    "hidden_dim": 512,
                    "action_embedding_dim": 32,
                    "feedforward_dim": 1024,
                },
                "state_embedder": {"feedforward_dim": 1024},
                "intention_encoder": {"feedforward_dim": 1024},
            }
        ),
        optimization=LAPv1OptimizationConfig(
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            max_grad_norm=1.0,
            value_wdl_weight=1.0,
            value_cp_weight=0.25,
            sharpness_weight=0.1,
            policy_ce_weight=1.0,
            policy_kl_weight=0.25,
            policy_margin_weight=0.1,
            policy_rank_weight=0.1,
            intention_aux_weight=0.05,
        ),
        evaluation=PlannerEvaluationConfig(top_k=3),
        runtime=PlannerRuntimeConfig(torch_threads=1, dataloader_workers=0),
        export=PlannerExportConfig(bundle_dir=str(tmp_path / "bundle")),
    )

    run = train_lapv1(config, repo_root=tmp_path)

    checkpoint_path = Path(run.export_paths["checkpoint"])
    assert checkpoint_path.exists()
    assert Path(run.summary_path).exists()
    assert run.best_epoch == 1

    metrics = evaluate_lapv1_checkpoint(checkpoint_path, dataset_path=validation_path)

    assert metrics.total_examples == 2
    assert 0.0 <= metrics.root_top1_accuracy <= 1.0
    assert 0.0 <= metrics.root_top3_accuracy <= 1.0
    assert metrics.total_loss >= 0.0


def test_train_lapv1_accepts_matching_initial_checkpoint(tmp_path: Path) -> None:
    train_path = tmp_path / "lapv1_train.jsonl"
    validation_path = tmp_path / "lapv1_validation.jsonl"
    _write_examples(
        train_path,
        [_planner_example("train-1", teacher_index=0, teacher_cp=60.0, teacher_gap=40.0)],
    )
    _write_examples(
        validation_path,
        [_planner_example("validation-1", teacher_index=0, teacher_cp=50.0, teacher_gap=30.0)],
    )

    model_config = LAPv1Config.from_mapping(
        {
            "deliberation": {"max_inner_steps": 0, "min_inner_steps": 0},
            "opponent_head": {
                "architecture": "set_v2",
                "hidden_dim": 64,
                "hidden_layers": 1,
                "action_embedding_dim": 16,
                "dropout": 0.0,
            },
            "value_head": {"hidden_dim": 1024},
            "policy_head": {
                "hidden_dim": 512,
                "action_embedding_dim": 32,
                "feedforward_dim": 1024,
            },
            "state_embedder": {"feedforward_dim": 1024},
            "intention_encoder": {"feedforward_dim": 1024},
        }
    )
    checkpoint_path = tmp_path / "initial_checkpoint.pt"
    model = LAPv1Model(model_config)
    aux_probe_state = {
        "network.weight": torch.randn(
            (7, model_config.intention_encoder.intention_dim),
            dtype=torch.float32,
        ),
        "network.bias": torch.randn((7,), dtype=torch.float32),
    }
    torch.save(
        {
            "model_name": LAPV1_MODEL_NAME,
            "model_state_dict": model.state_dict(),
            "aux_state_dict": aux_probe_state,
        },
        checkpoint_path,
    )

    config = LAPv1TrainConfig(
        seed=23,
        output_dir=str(tmp_path / "lapv1_out"),
        stage="T1",
        data=PlannerDataConfig(
            train_path=str(train_path),
            validation_path=str(validation_path),
        ),
        model=model_config,
        optimization=LAPv1OptimizationConfig(
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            weight_decay=0.0,
        ),
        evaluation=PlannerEvaluationConfig(top_k=3),
        runtime=PlannerRuntimeConfig(torch_threads=1, dataloader_workers=0),
        export=PlannerExportConfig(bundle_dir=str(tmp_path / "bundle")),
        initial_checkpoint=str(checkpoint_path),
    )

    run = train_lapv1(config, repo_root=tmp_path)

    assert Path(run.export_paths["checkpoint"]).exists()


def test_nnue_policy_on_runs_training_step(tmp_path: Path) -> None:
    train_path = tmp_path / "lapv1_train.jsonl"
    validation_path = tmp_path / "lapv1_validation.jsonl"
    _write_examples(
        train_path,
        [
            _planner_example("train-1", teacher_index=0, teacher_cp=60.0, teacher_gap=40.0),
            _planner_example("train-2", teacher_index=1, teacher_cp=10.0, teacher_gap=10.0),
        ],
    )
    _write_examples(
        validation_path,
        [_planner_example("validation-1", teacher_index=0, teacher_cp=50.0, teacher_gap=30.0)],
    )

    config = LAPv1TrainConfig(
        seed=11,
        output_dir=str(tmp_path / "lapv1_out"),
        stage="T1",
        data=PlannerDataConfig(
            train_path=str(train_path),
            validation_path=str(validation_path),
        ),
        model=LAPv1Config.from_mapping(
            {
                "deliberation": {"max_inner_steps": 0, "min_inner_steps": 0},
                "lapv2": {
                    "enabled": True,
                    "nnue_value": True,
                    "nnue_policy": True,
                    "N_accumulator": 8,
                    "loss_balance": {
                        "value_loss_norm": "ema",
                        "policy_loss_norm": "ema",
                    },
                },
                "opponent_head": {
                    "architecture": "set_v2",
                    "hidden_dim": 64,
                    "hidden_layers": 1,
                    "action_embedding_dim": 16,
                    "dropout": 0.0,
                },
                "value_head": {"hidden_dim": 256},
                "policy_head": {
                    "hidden_dim": 256,
                    "action_embedding_dim": 32,
                    "feedforward_dim": 512,
                },
                "state_embedder": {"feedforward_dim": 512},
                "intention_encoder": {"feedforward_dim": 512},
            }
        ),
        optimization=LAPv1OptimizationConfig(
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            max_grad_norm=1.0,
            log_interval_batches=1,
        ),
        evaluation=PlannerEvaluationConfig(top_k=3),
        runtime=PlannerRuntimeConfig(torch_threads=1, dataloader_workers=0),
        export=PlannerExportConfig(bundle_dir=str(tmp_path / "bundle")),
    )

    run = train_lapv1(config, repo_root=tmp_path)

    assert Path(run.export_paths["checkpoint"]).exists()


def test_loss_balance_keeps_ema_in_range() -> None:
    state: dict[str, float] = {}
    losses = [
        _normalize_lapv2_shared_loss(
            raw_loss=torch.tensor(value, dtype=torch.float32),
            state=state,
            key="policy_shared",
            mode="ema",
            training=True,
        )
        for value in (10.0, 5.0, 2.5, 1.25)
    ]

    assert "policy_shared" in state
    assert 1e-6 <= state["policy_shared"] <= 10.0
    assert all(torch.isfinite(loss) for loss in losses)


def test_distill_loss_positive_on_random_init() -> None:
    loss = _opponent_distill_loss(
        step_student_reply_logits_tensors=(torch.randn(2, 3, 4),),
        step_student_pressure_tensors=(torch.sigmoid(torch.randn(2, 3)),),
        step_student_uncertainty_tensors=(torch.sigmoid(torch.randn(2, 3)),),
        step_teacher_reply_logits_tensors=(torch.randn(2, 3, 4),),
        step_teacher_pressure_tensors=(torch.sigmoid(torch.randn(2, 3)),),
        step_teacher_uncertainty_tensors=(torch.sigmoid(torch.randn(2, 3)),),
        step_active_masks=(torch.tensor([True, True], dtype=torch.bool),),
        reply_weight=1.0,
        pressure_weight=0.5,
        uncertainty_weight=0.5,
    )

    assert float(loss.item()) > 0.0


def test_distill_loss_zero_when_match_teacher() -> None:
    teacher_reply = torch.randn(2, 3, 4)
    teacher_pressure = torch.sigmoid(torch.randn(2, 3))
    teacher_uncertainty = torch.sigmoid(torch.randn(2, 3))
    loss = _opponent_distill_loss(
        step_student_reply_logits_tensors=(teacher_reply.clone(),),
        step_student_pressure_tensors=(teacher_pressure.clone(),),
        step_student_uncertainty_tensors=(teacher_uncertainty.clone(),),
        step_teacher_reply_logits_tensors=(teacher_reply,),
        step_teacher_pressure_tensors=(teacher_pressure,),
        step_teacher_uncertainty_tensors=(teacher_uncertainty,),
        step_active_masks=(torch.tensor([True, True], dtype=torch.bool),),
        reply_weight=1.0,
        pressure_weight=0.5,
        uncertainty_weight=0.5,
    )

    assert float(loss.item()) == pytest.approx(0.0, abs=1e-7)


def test_sharpness_phase_moe_on_runs_training_step(tmp_path: Path) -> None:
    train_path = tmp_path / "lapv1_train.jsonl"
    validation_path = tmp_path / "lapv1_validation.jsonl"
    _write_examples(
        train_path,
        [
            _planner_example("train-1", teacher_index=0, teacher_cp=60.0, teacher_gap=40.0),
            _planner_example("train-2", teacher_index=1, teacher_cp=10.0, teacher_gap=10.0),
        ],
    )
    _write_examples(
        validation_path,
        [_planner_example("validation-1", teacher_index=0, teacher_cp=50.0, teacher_gap=30.0)],
    )

    config = LAPv1TrainConfig(
        seed=12,
        output_dir=str(tmp_path / "lapv1_out"),
        stage="T1",
        data=PlannerDataConfig(
            train_path=str(train_path),
            validation_path=str(validation_path),
        ),
        model=LAPv1Config.from_mapping(
            {
                "deliberation": {"max_inner_steps": 0, "min_inner_steps": 0},
                "lapv2": {
                    "enabled": True,
                    "sharpness_phase_moe": True,
                },
                "opponent_head": {
                    "architecture": "set_v2",
                    "hidden_dim": 64,
                    "hidden_layers": 1,
                    "action_embedding_dim": 16,
                    "dropout": 0.0,
                },
                "value_head": {"hidden_dim": 256},
                "policy_head": {
                    "hidden_dim": 256,
                    "action_embedding_dim": 32,
                    "feedforward_dim": 512,
                },
                "state_embedder": {"feedforward_dim": 512},
                "intention_encoder": {"feedforward_dim": 512},
            }
        ),
        optimization=LAPv1OptimizationConfig(
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            weight_decay=0.0,
        ),
        evaluation=PlannerEvaluationConfig(top_k=3),
        runtime=PlannerRuntimeConfig(torch_threads=1, dataloader_workers=0),
        export=PlannerExportConfig(bundle_dir=str(tmp_path / "bundle")),
    )

    run = train_lapv1(config, repo_root=tmp_path)

    assert Path(run.export_paths["checkpoint"]).exists()


def test_shared_opponent_readout_on_runs_training_step(tmp_path: Path) -> None:
    train_path = tmp_path / "lapv1_train.jsonl"
    validation_path = tmp_path / "lapv1_validation.jsonl"
    _write_examples(
        train_path,
        [
            _planner_example("train-1", teacher_index=0, teacher_cp=60.0, teacher_gap=40.0),
            _planner_example("train-2", teacher_index=1, teacher_cp=10.0, teacher_gap=10.0),
        ],
    )
    _write_examples(
        validation_path,
        [_planner_example("validation-1", teacher_index=0, teacher_cp=50.0, teacher_gap=30.0)],
    )

    config = LAPv1TrainConfig(
        seed=13,
        output_dir=str(tmp_path / "lapv1_out"),
        stage="T2",
        data=PlannerDataConfig(
            train_path=str(train_path),
            validation_path=str(validation_path),
        ),
        model=LAPv1Config.from_mapping(
            {
                "lapv2": {
                    "enabled": True,
                    "shared_opponent_readout": True,
                },
                "deliberation": {"max_inner_steps": 2, "min_inner_steps": 1},
                "opponent_head": {
                    "architecture": "set_v2",
                    "hidden_dim": 64,
                    "hidden_layers": 1,
                    "action_embedding_dim": 16,
                    "dropout": 0.0,
                },
                "value_head": {"hidden_dim": 256},
                "policy_head": {
                    "hidden_dim": 256,
                    "action_embedding_dim": 32,
                    "feedforward_dim": 512,
                },
                "state_embedder": {"feedforward_dim": 512},
                "intention_encoder": {"feedforward_dim": 512},
            }
        ),
        optimization=LAPv1OptimizationConfig(
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            weight_decay=0.0,
        ),
        evaluation=PlannerEvaluationConfig(top_k=3),
        runtime=PlannerRuntimeConfig(torch_threads=1, dataloader_workers=0),
        export=PlannerExportConfig(bundle_dir=str(tmp_path / "bundle")),
        stage2=LAPv1Stage2Config(max_inner_steps_schedule=(1,)),
    )

    run = train_lapv1(config, repo_root=tmp_path)

    assert Path(run.export_paths["checkpoint"]).exists()


def test_distill_opponent_on_runs_training_step(tmp_path: Path) -> None:
    train_path = tmp_path / "lapv1_train.jsonl"
    validation_path = tmp_path / "lapv1_validation.jsonl"
    _write_examples(
        train_path,
        [
            _planner_example("train-1", teacher_index=0, teacher_cp=60.0, teacher_gap=40.0),
            _planner_example("train-2", teacher_index=1, teacher_cp=10.0, teacher_gap=10.0),
        ],
    )
    _write_examples(
        validation_path,
        [_planner_example("validation-1", teacher_index=0, teacher_cp=50.0, teacher_gap=30.0)],
    )

    config = LAPv1TrainConfig(
        seed=14,
        output_dir=str(tmp_path / "lapv1_out"),
        stage="T2",
        data=PlannerDataConfig(
            train_path=str(train_path),
            validation_path=str(validation_path),
        ),
        model=LAPv1Config.from_mapping(
            {
                "lapv2": {
                    "enabled": True,
                    "shared_opponent_readout": True,
                    "distill_opponent": True,
                    "distill_fraction": 1.0,
                },
                "deliberation": {"max_inner_steps": 2, "min_inner_steps": 1},
                "opponent_head": {
                    "architecture": "set_v2",
                    "hidden_dim": 64,
                    "hidden_layers": 1,
                    "action_embedding_dim": 16,
                    "dropout": 0.0,
                },
                "value_head": {"hidden_dim": 256},
                "policy_head": {
                    "hidden_dim": 256,
                    "action_embedding_dim": 32,
                    "feedforward_dim": 512,
                },
                "state_embedder": {"feedforward_dim": 512},
                "intention_encoder": {"feedforward_dim": 512},
            }
        ),
        optimization=LAPv1OptimizationConfig(
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            weight_decay=0.0,
        ),
        evaluation=PlannerEvaluationConfig(top_k=3),
        runtime=PlannerRuntimeConfig(torch_threads=1, dataloader_workers=0),
        export=PlannerExportConfig(bundle_dir=str(tmp_path / "bundle")),
        stage2=LAPv1Stage2Config(max_inner_steps_schedule=(1,)),
    )

    run = train_lapv1(config, repo_root=tmp_path)

    assert Path(run.export_paths["checkpoint"]).exists()


def test_train_lapv1_accepts_older_checkpoint_missing_residual_delta_net(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "lapv1_train.jsonl"
    validation_path = tmp_path / "lapv1_validation.jsonl"
    _write_examples(
        train_path,
        [_planner_example("train-1", teacher_index=0, teacher_cp=60.0, teacher_gap=40.0)],
    )
    _write_examples(
        validation_path,
        [_planner_example("validation-1", teacher_index=0, teacher_cp=50.0, teacher_gap=30.0)],
    )

    model_config = LAPv1Config.from_mapping(
        {
            "deliberation": {"max_inner_steps": 0, "min_inner_steps": 0},
            "opponent_head": {
                "architecture": "set_v2",
                "hidden_dim": 64,
                "hidden_layers": 1,
                "action_embedding_dim": 16,
                "dropout": 0.0,
            },
            "value_head": {"hidden_dim": 1024},
            "policy_head": {
                "hidden_dim": 512,
                "action_embedding_dim": 32,
                "feedforward_dim": 1024,
            },
            "state_embedder": {"feedforward_dim": 1024},
            "intention_encoder": {"feedforward_dim": 1024},
        }
    )
    checkpoint_path = tmp_path / "initial_checkpoint_legacy.pt"
    model = LAPv1Model(model_config)
    legacy_state_dict = dict(model.state_dict())
    for key in list(legacy_state_dict):
        if key.startswith("deliberation_loop.cell.candidate_delta_network."):
            del legacy_state_dict[key]
    torch.save(
        {
            "model_name": LAPV1_MODEL_NAME,
            "model_state_dict": legacy_state_dict,
        },
        checkpoint_path,
    )

    config = LAPv1TrainConfig(
        seed=31,
        output_dir=str(tmp_path / "lapv1_out"),
        stage="T1",
        data=PlannerDataConfig(
            train_path=str(train_path),
            validation_path=str(validation_path),
        ),
        model=model_config,
        optimization=LAPv1OptimizationConfig(
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            weight_decay=0.0,
        ),
        evaluation=PlannerEvaluationConfig(top_k=3),
        runtime=PlannerRuntimeConfig(torch_threads=1, dataloader_workers=0),
        export=PlannerExportConfig(bundle_dir=str(tmp_path / "bundle")),
        initial_checkpoint=str(checkpoint_path),
    )

    run = train_lapv1(config, repo_root=tmp_path)

    assert Path(run.export_paths["checkpoint"]).exists()


def test_train_lapv1_accepts_older_checkpoint_missing_shared_opponent_readout(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "lapv1_train.jsonl"
    validation_path = tmp_path / "lapv1_validation.jsonl"
    _write_examples(
        train_path,
        [_planner_example("train-1", teacher_index=0, teacher_cp=60.0, teacher_gap=40.0)],
    )
    _write_examples(
        validation_path,
        [_planner_example("validation-1", teacher_index=0, teacher_cp=50.0, teacher_gap=30.0)],
    )

    legacy_model_config = LAPv1Config.from_mapping(
        {
            "deliberation": {"max_inner_steps": 2, "min_inner_steps": 1},
            "opponent_head": {
                "architecture": "set_v2",
                "hidden_dim": 64,
                "hidden_layers": 1,
                "action_embedding_dim": 16,
                "dropout": 0.0,
            },
            "value_head": {"hidden_dim": 256},
            "policy_head": {
                "hidden_dim": 256,
                "action_embedding_dim": 32,
                "feedforward_dim": 512,
            },
            "state_embedder": {"feedforward_dim": 512},
            "intention_encoder": {"feedforward_dim": 512},
        }
    )
    checkpoint_path = tmp_path / "initial_checkpoint_legacy_shared_opponent.pt"
    model = LAPv1Model(legacy_model_config)
    torch.save(
        {
            "model_name": LAPV1_MODEL_NAME,
            "model_state_dict": dict(model.state_dict()),
        },
        checkpoint_path,
    )

    config = LAPv1TrainConfig(
        seed=32,
        output_dir=str(tmp_path / "lapv1_out"),
        stage="T2",
        data=PlannerDataConfig(
            train_path=str(train_path),
            validation_path=str(validation_path),
        ),
        model=LAPv1Config.from_mapping(
            {
                "lapv2": {
                    "enabled": True,
                    "shared_opponent_readout": True,
                },
                "deliberation": {"max_inner_steps": 2, "min_inner_steps": 1},
                "opponent_head": {
                    "architecture": "set_v2",
                    "hidden_dim": 64,
                    "hidden_layers": 1,
                    "action_embedding_dim": 16,
                    "dropout": 0.0,
                },
                "value_head": {"hidden_dim": 256},
                "policy_head": {
                    "hidden_dim": 256,
                    "action_embedding_dim": 32,
                    "feedforward_dim": 512,
                },
                "state_embedder": {"feedforward_dim": 512},
                "intention_encoder": {"feedforward_dim": 512},
            }
        ),
        optimization=LAPv1OptimizationConfig(
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            weight_decay=0.0,
        ),
        evaluation=PlannerEvaluationConfig(top_k=3),
        runtime=PlannerRuntimeConfig(torch_threads=1, dataloader_workers=0),
        export=PlannerExportConfig(bundle_dir=str(tmp_path / "bundle")),
        initial_checkpoint=str(checkpoint_path),
        stage2=LAPv1Stage2Config(max_inner_steps_schedule=(1,)),
    )

    run = train_lapv1(config, repo_root=tmp_path)

    assert Path(run.export_paths["checkpoint"]).exists()


def test_evaluate_lapv1_checkpoint_accepts_stage2_phase_config_payload(
    tmp_path: Path,
) -> None:
    validation_path = tmp_path / "lapv1_validation.jsonl"
    _write_examples(
        validation_path,
        [_planner_example("validation-1", teacher_index=0, teacher_cp=50.0, teacher_gap=30.0)],
    )

    model_config = LAPv1Config.from_mapping(
        {
            "deliberation": {"max_inner_steps": 4, "min_inner_steps": 1},
            "opponent_head": {
                "architecture": "set_v2",
                "hidden_dim": 64,
                "hidden_layers": 1,
                "action_embedding_dim": 16,
                "dropout": 0.0,
            },
            "value_head": {"hidden_dim": 1024},
            "policy_head": {
                "hidden_dim": 512,
                "action_embedding_dim": 32,
                "feedforward_dim": 1024,
            },
            "state_embedder": {"feedforward_dim": 1024},
            "intention_encoder": {"feedforward_dim": 1024},
        }
    )
    checkpoint_path = tmp_path / "stage2_phased_checkpoint.pt"
    model = LAPv1Model(model_config)
    torch.save(
        {
            "model_name": LAPV1_MODEL_NAME,
            "model_state_dict": model.state_dict(),
            "training_config": LAPv1TrainConfig(
                seed=41,
                output_dir=str(tmp_path / "lapv1_out"),
                stage="T2",
                data=PlannerDataConfig(
                    train_path="train.jsonl",
                    validation_path=str(validation_path),
                ),
                model=model_config,
                optimization=LAPv1OptimizationConfig(
                    epochs=4,
                    batch_size=1,
                    learning_rate=1e-3,
                    weight_decay=0.0,
                ),
                evaluation=PlannerEvaluationConfig(top_k=3),
                runtime=PlannerRuntimeConfig(torch_threads=1, dataloader_workers=0),
                export=PlannerExportConfig(bundle_dir=str(tmp_path / "bundle")),
                stage2=LAPv1Stage2Config(
                    phases=(
                        LAPv1Stage2PhaseConfig(
                            name="freeze_inner",
                            epochs=2,
                            trainable_parameter_groups=("inner_loop",),
                            max_inner_steps_schedule=(1, 2),
                        ),
                        LAPv1Stage2PhaseConfig(
                            name="joint_finetune",
                            epochs=2,
                            trainable_parameter_groups=("all",),
                            max_inner_steps_schedule=(2, 4),
                        ),
                    ),
                ),
            ).to_dict(),
        },
        checkpoint_path,
    )

    metrics = evaluate_lapv1_checkpoint(checkpoint_path)

    assert metrics.total_examples == 1
    assert 0.0 <= metrics.root_top1_accuracy <= 1.0


def test_lapv1_optimization_config_validates_max_grad_norm() -> None:
    config = LAPv1OptimizationConfig(max_grad_norm=0.5)
    assert config.max_grad_norm == 0.5

    with pytest.raises(ValueError, match="max_grad_norm"):
        LAPv1OptimizationConfig(max_grad_norm=0.0)


def test_lapv1_optimization_config_validates_log_interval_batches() -> None:
    config = LAPv1OptimizationConfig(log_interval_batches=4)
    assert config.log_interval_batches == 4

    with pytest.raises(ValueError, match="log_interval_batches"):
        LAPv1OptimizationConfig(log_interval_batches=0)


def test_lapv1_collate_clips_extreme_root_value_targets() -> None:
    example = _planner_example(
        "mate-like",
        teacher_index=0,
        teacher_cp=99999.0,
        teacher_gap=20.0,
    )
    prepared = _prepare_example(example)
    batch = _collate_examples([prepared])
    assert float(batch["teacher_root_value_cp"][0].item()) == 1024.0


def test_lapv1_collate_clips_extreme_root_gap_targets() -> None:
    example = _planner_example(
        "gap-like",
        teacher_index=0,
        teacher_cp=25.0,
        teacher_gap=100063.0,
    )
    prepared = _prepare_example(example)
    batch = _collate_examples([prepared])
    assert float(batch["teacher_top1_minus_top2_cp"][0].item()) == 512.0


def test_policy_margin_loss_ignores_single_candidate_rows() -> None:
    assert torch is not None

    logits = torch.tensor([[5.0, -1.0e9], [2.0, 1.0]], dtype=torch.float32)
    candidate_mask = torch.tensor([[True, False], [True, True]], dtype=torch.bool)
    teacher_top1 = torch.tensor([0, 0], dtype=torch.long)
    gap_targets = torch.tensor([40.0, 20.0], dtype=torch.float32)

    loss = _policy_margin_loss(logits, candidate_mask, teacher_top1, gap_targets)

    assert torch.isfinite(loss)
    assert float(loss.item()) >= 0.0


def test_policy_margin_loss_uses_smooth_l1_on_valid_rows() -> None:
    assert torch is not None

    logits = torch.tensor([[3.0, 1.0]], dtype=torch.float32)
    candidate_mask = torch.tensor([[True, True]], dtype=torch.bool)
    teacher_top1 = torch.tensor([0], dtype=torch.long)
    gap_targets = torch.tensor([128.0], dtype=torch.float32)

    loss = _policy_margin_loss(logits, candidate_mask, teacher_top1, gap_targets)

    expected = torch.nn.functional.smooth_l1_loss(
        torch.tensor([2.0], dtype=torch.float32),
        torch.tensor([1.0], dtype=torch.float32),
        reduction="none",
    ).mean()
    assert torch.allclose(loss, expected)


def test_policy_margin_loss_caps_raw_gap_targets_locally() -> None:
    assert torch is not None

    logits = torch.tensor([[3.0, 1.0]], dtype=torch.float32)
    candidate_mask = torch.tensor([[True, True]], dtype=torch.bool)
    teacher_top1 = torch.tensor([0], dtype=torch.long)
    gap_targets = torch.tensor([100063.0], dtype=torch.float32)

    loss = _policy_margin_loss(logits, candidate_mask, teacher_top1, gap_targets)

    expected = torch.nn.functional.smooth_l1_loss(
        torch.tensor([2.0], dtype=torch.float32),
        torch.tensor([4.0], dtype=torch.float32),
        reduction="none",
    ).mean()
    assert torch.allclose(loss, expected)


def test_improvement_over_root_loss_only_penalizes_root_incorrect_rows() -> None:
    assert torch is not None

    initial_logits = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    final_logits = initial_logits.clone()
    teacher_top1 = torch.tensor([0, 0], dtype=torch.long)
    candidate_mask = torch.tensor([[True, True], [True, True]], dtype=torch.bool)

    loss = _improvement_over_root_loss(
        initial_logits=initial_logits,
        final_logits=final_logits,
        teacher_top1_candidate_index=teacher_top1,
        candidate_mask=candidate_mask,
        step_candidate_score_tensors=(),
        step_active_masks=(),
    )

    assert torch.allclose(loss, torch.tensor(0.05, dtype=torch.float32), atol=1e-5)


def test_trace_policy_ce_loss_averages_over_step_logits() -> None:
    assert torch is not None

    step_logits = (
        torch.tensor([[2.0, 1.0]], dtype=torch.float32),
        torch.tensor([[3.0, 0.5]], dtype=torch.float32),
    )
    step_active_masks = (
        torch.tensor([True], dtype=torch.bool),
        torch.tensor([True], dtype=torch.bool),
    )
    teacher_top1 = torch.tensor([0], dtype=torch.long)

    loss = _trace_policy_ce_loss(step_logits, teacher_top1, step_active_masks)

    expected = torch.stack(
        [
            torch.nn.functional.cross_entropy(step_logits[0], teacher_top1),
            torch.nn.functional.cross_entropy(step_logits[1], teacher_top1),
        ]
    ).mean()
    assert torch.allclose(loss, expected)


def test_lapv1_lazy_dataset_indexes_multiple_jsonl_files(tmp_path: Path) -> None:
    first_path = tmp_path / "part1.jsonl"
    second_path = tmp_path / "part2.jsonl"
    _write_examples(
        first_path,
        [
            _planner_example("train-1", teacher_index=0, teacher_cp=60.0, teacher_gap=40.0),
            _planner_example("train-2", teacher_index=1, teacher_cp=10.0, teacher_gap=10.0),
        ],
    )
    _write_examples(
        second_path,
        [
            _planner_example("train-3", teacher_index=0, teacher_cp=-40.0, teacher_gap=25.0),
        ],
    )

    dataset = _build_lazy_dataset([first_path, second_path])
    try:
        assert len(dataset) == 3
        assert dataset[0].sample_id == "train-1"
        assert dataset[2].sample_id == "train-3"
    finally:
        dataset.close()


def test_lapv1_lazy_dataset_emits_index_progress_logs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "part1.jsonl"
    _write_examples(
        path,
        [
            _planner_example("train-1", teacher_index=0, teacher_cp=60.0, teacher_gap=40.0),
            _planner_example("train-2", teacher_index=1, teacher_cp=10.0, teacher_gap=10.0),
        ],
    )

    dataset = _build_lazy_dataset(
        [path],
        log_label="train",
        log_every_examples=1,
    )
    try:
        assert len(dataset) == 2
    finally:
        dataset.close()

    captured = capsys.readouterr()
    assert "dataset_index_start label=train" in captured.out
    assert "dataset_index_progress label=train" in captured.out
    assert "dataset_index_done label=train total_examples=2" in captured.out


def test_train_lapv1_stage1_on_precomputed_lapv1_artifact(tmp_path: Path) -> None:
    train_path = tmp_path / "lapv1_train.jsonl"
    validation_path = tmp_path / "lapv1_validation.jsonl"
    train_examples = [
        lapv1_training_example_from_planner_head(
            _planner_example("train-1", teacher_index=0, teacher_cp=60.0, teacher_gap=40.0)
        ),
        lapv1_training_example_from_planner_head(
            _planner_example("train-2", teacher_index=1, teacher_cp=10.0, teacher_gap=10.0)
        ),
    ]
    validation_examples = [
        lapv1_training_example_from_planner_head(
            _planner_example("validation-1", teacher_index=0, teacher_cp=50.0, teacher_gap=30.0)
        ),
    ]
    write_lapv1_training_artifact(train_path, train_examples)
    write_lapv1_training_artifact(validation_path, validation_examples)

    config = LAPv1TrainConfig(
        seed=19,
        output_dir=str(tmp_path / "lapv1_out"),
        stage="T1",
        data=PlannerDataConfig(
            train_path=str(train_path),
            validation_path=str(validation_path),
        ),
        model=LAPv1Config.from_mapping(
            {
                "deliberation": {"max_inner_steps": 0, "min_inner_steps": 0},
                "opponent_head": {
                    "architecture": "set_v2",
                    "hidden_dim": 64,
                    "hidden_layers": 1,
                    "action_embedding_dim": 16,
                    "dropout": 0.0,
                },
                "value_head": {"hidden_dim": 1024},
                "policy_head": {
                    "hidden_dim": 512,
                    "action_embedding_dim": 32,
                    "feedforward_dim": 1024,
                },
                "state_embedder": {"feedforward_dim": 1024},
                "intention_encoder": {"feedforward_dim": 1024},
            }
        ),
        optimization=LAPv1OptimizationConfig(
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            max_grad_norm=1.0,
            log_interval_batches=1,
            value_wdl_weight=1.0,
            value_cp_weight=0.25,
            sharpness_weight=0.1,
            policy_ce_weight=1.0,
            policy_kl_weight=0.25,
            policy_margin_weight=0.1,
            policy_rank_weight=0.1,
            intention_aux_weight=0.05,
        ),
        evaluation=PlannerEvaluationConfig(top_k=3),
        runtime=PlannerRuntimeConfig(torch_threads=1, dataloader_workers=0),
        export=PlannerExportConfig(bundle_dir=str(tmp_path / "bundle")),
    )

    run = train_lapv1(config, repo_root=tmp_path)

    assert Path(run.export_paths["checkpoint"]).exists()
    assert run.best_epoch == 1


def test_lapv1_training_artifact_roundtrip(tmp_path: Path) -> None:
    source_example = lapv1_training_example_from_planner_head(
        _planner_example("roundtrip", teacher_index=0, teacher_cp=25.0, teacher_gap=15.0)
    )
    artifact_path = tmp_path / "lapv1_train.jsonl"
    write_lapv1_training_artifact(artifact_path, [source_example])

    loaded = load_lapv1_training_examples(artifact_path)

    assert len(loaded) == 1
    assert loaded[0].sample_id == "roundtrip"
    assert loaded[0].candidate_action_indices == source_example.candidate_action_indices


def test_train_lapv1_stage1_emits_batch_progress_logs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    train_path = tmp_path / "lapv1_train.jsonl"
    validation_path = tmp_path / "lapv1_validation.jsonl"
    _write_examples(
        train_path,
        [
            _planner_example("train-1", teacher_index=0, teacher_cp=60.0, teacher_gap=40.0),
            _planner_example("train-2", teacher_index=1, teacher_cp=10.0, teacher_gap=10.0),
            _planner_example("train-3", teacher_index=0, teacher_cp=-40.0, teacher_gap=25.0),
            _planner_example("train-4", teacher_index=1, teacher_cp=35.0, teacher_gap=15.0),
        ],
    )
    _write_examples(
        validation_path,
        [
            _planner_example("validation-1", teacher_index=0, teacher_cp=50.0, teacher_gap=30.0),
            _planner_example("validation-2", teacher_index=1, teacher_cp=0.0, teacher_gap=5.0),
        ],
    )

    config = LAPv1TrainConfig(
        seed=17,
        output_dir=str(tmp_path / "lapv1_out"),
        stage="T1",
        data=PlannerDataConfig(
            train_path=str(train_path),
            validation_path=str(validation_path),
        ),
        model=LAPv1Config.from_mapping(
            {
                "deliberation": {"max_inner_steps": 0, "min_inner_steps": 0},
                "opponent_head": {
                    "architecture": "set_v2",
                    "hidden_dim": 64,
                    "hidden_layers": 1,
                    "action_embedding_dim": 16,
                    "dropout": 0.0,
                },
                "value_head": {"hidden_dim": 1024},
                "policy_head": {
                    "hidden_dim": 512,
                    "action_embedding_dim": 32,
                    "feedforward_dim": 1024,
                },
                "state_embedder": {"feedforward_dim": 1024},
                "intention_encoder": {"feedforward_dim": 1024},
            }
        ),
        optimization=LAPv1OptimizationConfig(
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            max_grad_norm=1.0,
            log_interval_batches=1,
            value_wdl_weight=1.0,
            value_cp_weight=0.25,
            sharpness_weight=0.1,
            policy_ce_weight=1.0,
            policy_kl_weight=0.25,
            policy_margin_weight=0.1,
            policy_rank_weight=0.1,
            intention_aux_weight=0.05,
        ),
        evaluation=PlannerEvaluationConfig(top_k=3),
        runtime=PlannerRuntimeConfig(torch_threads=1, dataloader_workers=0),
        export=PlannerExportConfig(bundle_dir=str(tmp_path / "bundle")),
    )

    train_lapv1(config, repo_root=tmp_path)

    captured = capsys.readouterr()
    assert "phase=train" in captured.out
    assert "phase=validation" in captured.out
    assert "batch=1/2" in captured.out
    assert "batch=2/2" in captured.out


def test_lapv1_checkpoint_loads_into_phase_moe(tmp_path: Path) -> None:
    legacy_config = LAPv1Config.from_mapping(
        {
            "deliberation": {"max_inner_steps": 0, "min_inner_steps": 0},
            "opponent_head": {
                "architecture": "set_v2",
                "hidden_dim": 64,
                "hidden_layers": 1,
                "action_embedding_dim": 16,
                "dropout": 0.0,
            },
            "value_head": {"hidden_dim": 1024},
            "policy_head": {
                "hidden_dim": 512,
                "action_embedding_dim": 32,
                "feedforward_dim": 1024,
            },
            "state_embedder": {"feedforward_dim": 1024},
            "intention_encoder": {"feedforward_dim": 1024},
        }
    )
    legacy_model = LAPv1Model(legacy_config)
    checkpoint_path = tmp_path / "lapv1_legacy.pt"
    torch.save(
        {
            "model_name": LAPV1_MODEL_NAME,
            "model_state_dict": legacy_model.state_dict(),
        },
        checkpoint_path,
    )

    phase_moe_config = LAPv1Config.from_mapping(
        {
            "deliberation": {"max_inner_steps": 0, "min_inner_steps": 0},
            "opponent_head": {
                "architecture": "set_v2",
                "hidden_dim": 64,
                "hidden_layers": 1,
                "action_embedding_dim": 16,
                "dropout": 0.0,
            },
            "value_head": {"hidden_dim": 1024},
            "policy_head": {
                "hidden_dim": 512,
                "action_embedding_dim": 32,
                "feedforward_dim": 1024,
            },
            "state_embedder": {"feedforward_dim": 1024},
            "intention_encoder": {"feedforward_dim": 1024},
            "lapv2": {
                "enabled": True,
                "phase_moe": True,
            },
        }
    )
    phase_moe_model = LAPv1Model(phase_moe_config)
    _load_lapv1_model_state(
        phase_moe_model,
        legacy_model.state_dict(),
        checkpoint_path=checkpoint_path,
    )

    batch = _collate_examples(
        [
            lapv1_training_example_from_planner_head(
                _planner_example("pmoe-0", teacher_index=0, teacher_cp=30.0, teacher_gap=20.0)
            ),
            lapv1_training_example_from_planner_head(
                _planner_example("pmoe-1", teacher_index=1, teacher_cp=15.0, teacher_gap=10.0)
            ),
        ]
    )

    with torch.inference_mode():
        legacy_outputs = legacy_model(
            batch["piece_tokens"],
            batch["square_tokens"],
            batch["state_context_global"],
            batch["reachability_edges"],
            batch["candidate_features"],
            batch["candidate_action_indices"],
            batch["candidate_mask"],
            phase_index=batch["phase_index"],
        )
        phase_moe_outputs = phase_moe_model(
            batch["piece_tokens"],
            batch["square_tokens"],
            batch["state_context_global"],
            batch["reachability_edges"],
            batch["candidate_features"],
            batch["candidate_action_indices"],
            batch["candidate_mask"],
            phase_index=batch["phase_index"],
        )

    assert torch.allclose(
        legacy_outputs["initial_policy_logits"],
        phase_moe_outputs["initial_policy_logits"],
        atol=1e-6,
    )
    assert torch.allclose(
        legacy_outputs["final_policy_logits"],
        phase_moe_outputs["final_policy_logits"],
        atol=1e-6,
    )
    assert torch.allclose(legacy_outputs["z_root"], phase_moe_outputs["z_root"], atol=1e-6)


def test_phase_moe_on_runs_training_step(tmp_path: Path) -> None:
    train_path = tmp_path / "lapv1_train.jsonl"
    validation_path = tmp_path / "lapv1_validation.jsonl"
    _write_examples(
        train_path,
        [
            _planner_example("train-1", teacher_index=0, teacher_cp=60.0, teacher_gap=40.0),
            _planner_example("train-2", teacher_index=1, teacher_cp=10.0, teacher_gap=10.0),
        ],
    )
    _write_examples(
        validation_path,
        [
            _planner_example("validation-1", teacher_index=0, teacher_cp=50.0, teacher_gap=30.0),
            _planner_example("validation-2", teacher_index=1, teacher_cp=0.0, teacher_gap=5.0),
        ],
    )

    config = LAPv1TrainConfig(
        seed=29,
        output_dir=str(tmp_path / "lapv1_out"),
        stage="T1",
        data=PlannerDataConfig(
            train_path=str(train_path),
            validation_path=str(validation_path),
        ),
        model=LAPv1Config.from_mapping(
            {
                "deliberation": {"max_inner_steps": 0, "min_inner_steps": 0},
                "opponent_head": {
                    "architecture": "set_v2",
                    "hidden_dim": 64,
                    "hidden_layers": 1,
                    "action_embedding_dim": 16,
                    "dropout": 0.0,
                },
                "value_head": {"hidden_dim": 1024},
                "policy_head": {
                    "hidden_dim": 512,
                    "action_embedding_dim": 32,
                    "feedforward_dim": 1024,
                },
                "state_embedder": {"feedforward_dim": 1024},
                "intention_encoder": {"feedforward_dim": 1024},
                "lapv2": {
                    "enabled": True,
                    "phase_moe": True,
                },
            }
        ),
        optimization=LAPv1OptimizationConfig(
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            max_grad_norm=1.0,
            value_wdl_weight=1.0,
            value_cp_weight=0.25,
            sharpness_weight=0.1,
            policy_ce_weight=1.0,
            policy_kl_weight=0.25,
            policy_margin_weight=0.1,
            policy_rank_weight=0.1,
            intention_aux_weight=0.05,
        ),
        evaluation=PlannerEvaluationConfig(top_k=3),
        runtime=PlannerRuntimeConfig(torch_threads=1, dataloader_workers=0),
        export=PlannerExportConfig(bundle_dir=str(tmp_path / "bundle")),
    )

    run = train_lapv1(config, repo_root=tmp_path)

    assert Path(run.export_paths["checkpoint"]).exists()


def test_nnue_value_on_runs_training_step(tmp_path: Path) -> None:
    train_path = tmp_path / "lapv1_train.jsonl"
    validation_path = tmp_path / "lapv1_validation.jsonl"
    _write_examples(
        train_path,
        [
            _planner_example("train-1", teacher_index=0, teacher_cp=60.0, teacher_gap=40.0),
            _planner_example("train-2", teacher_index=1, teacher_cp=10.0, teacher_gap=10.0),
        ],
    )
    _write_examples(
        validation_path,
        [
            _planner_example("validation-1", teacher_index=0, teacher_cp=50.0, teacher_gap=30.0),
            _planner_example("validation-2", teacher_index=1, teacher_cp=0.0, teacher_gap=5.0),
        ],
    )

    config = LAPv1TrainConfig(
        seed=41,
        output_dir=str(tmp_path / "lapv1_out"),
        stage="T1",
        data=PlannerDataConfig(
            train_path=str(train_path),
            validation_path=str(validation_path),
        ),
        model=LAPv1Config.from_mapping(
            {
                "deliberation": {"max_inner_steps": 0, "min_inner_steps": 0},
                "opponent_head": {
                    "architecture": "set_v2",
                    "hidden_dim": 64,
                    "hidden_layers": 1,
                    "action_embedding_dim": 16,
                    "dropout": 0.0,
                },
                "value_head": {"hidden_dim": 1024},
                "policy_head": {
                    "hidden_dim": 512,
                    "action_embedding_dim": 32,
                    "feedforward_dim": 1024,
                },
                "state_embedder": {"feedforward_dim": 1024},
                "intention_encoder": {"feedforward_dim": 1024},
                "lapv2": {
                    "enabled": True,
                    "nnue_value": True,
                    "N_accumulator": 8,
                },
            }
        ),
        optimization=LAPv1OptimizationConfig(
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            max_grad_norm=1.0,
            value_wdl_weight=1.0,
            value_cp_weight=0.25,
            sharpness_weight=0.1,
            policy_ce_weight=1.0,
            policy_kl_weight=0.25,
            policy_margin_weight=0.1,
            policy_rank_weight=0.1,
            intention_aux_weight=0.05,
        ),
        evaluation=PlannerEvaluationConfig(top_k=3),
        runtime=PlannerRuntimeConfig(torch_threads=1, dataloader_workers=0),
        export=PlannerExportConfig(bundle_dir=str(tmp_path / "bundle")),
    )

    run = train_lapv1(config, repo_root=tmp_path)

    checkpoint_path = Path(run.export_paths["checkpoint"])
    payload = torch.load(checkpoint_path, map_location="cpu")

    assert checkpoint_path.exists()
    assert payload["lapv2_version"] == 1


def test_phase_nnue_value_warm_start_matches_single(tmp_path: Path) -> None:
    single_config = LAPv1Config.from_mapping(
        {
            "deliberation": {"max_inner_steps": 0, "min_inner_steps": 0},
            "opponent_head": {
                "architecture": "set_v2",
                "hidden_dim": 64,
                "hidden_layers": 1,
                "action_embedding_dim": 16,
                "dropout": 0.0,
            },
            "value_head": {"hidden_dim": 1024},
            "policy_head": {
                "hidden_dim": 512,
                "action_embedding_dim": 32,
                "feedforward_dim": 1024,
            },
            "state_embedder": {"feedforward_dim": 1024},
            "intention_encoder": {"feedforward_dim": 1024},
            "lapv2": {
                "enabled": True,
                "nnue_value": True,
                "N_accumulator": 8,
            },
        }
    )
    phase_config = LAPv1Config.from_mapping(
        {
            "deliberation": {"max_inner_steps": 0, "min_inner_steps": 0},
            "opponent_head": {
                "architecture": "set_v2",
                "hidden_dim": 64,
                "hidden_layers": 1,
                "action_embedding_dim": 16,
                "dropout": 0.0,
            },
            "value_head": {"hidden_dim": 1024},
            "policy_head": {
                "hidden_dim": 512,
                "action_embedding_dim": 32,
                "feedforward_dim": 1024,
            },
            "state_embedder": {"feedforward_dim": 1024},
            "intention_encoder": {"feedforward_dim": 1024},
            "lapv2": {
                "enabled": True,
                "nnue_value": True,
                "nnue_value_phase_moe": True,
                "N_accumulator": 8,
            },
        }
    )
    single_model = LAPv1Model(single_config)
    phase_model = LAPv1Model(phase_config)
    checkpoint_path = tmp_path / "lapv2_step6_single.pt"
    torch.save(
        {
            "model_name": LAPV1_MODEL_NAME,
            "lapv2_version": 1,
            "model_state_dict": single_model.state_dict(),
        },
        checkpoint_path,
    )
    _load_lapv1_model_state(
        phase_model,
        single_model.state_dict(),
        checkpoint_path=checkpoint_path,
    )

    batch = _collate_examples(
        [
            lapv1_training_example_from_planner_head(
                _planner_example("phase-0", teacher_index=0, teacher_cp=30.0, teacher_gap=20.0)
            ),
            lapv1_training_example_from_planner_head(
                _planner_example("phase-1", teacher_index=1, teacher_cp=15.0, teacher_gap=10.0)
            ),
        ]
    )

    with torch.inference_mode():
        single_outputs = single_model(
            batch["piece_tokens"],
            batch["square_tokens"],
            batch["state_context_global"],
            batch["reachability_edges"],
            batch["candidate_features"],
            batch["candidate_action_indices"],
            batch["candidate_mask"],
            phase_index=batch["phase_index"],
            side_to_move=batch["side_to_move"],
            nnue_feat_white_indices=batch["nnue_feat_white_indices"],
            nnue_feat_white_offsets=batch["nnue_feat_white_offsets"],
            nnue_feat_black_indices=batch["nnue_feat_black_indices"],
            nnue_feat_black_offsets=batch["nnue_feat_black_offsets"],
        )
        phase_outputs = phase_model(
            batch["piece_tokens"],
            batch["square_tokens"],
            batch["state_context_global"],
            batch["reachability_edges"],
            batch["candidate_features"],
            batch["candidate_action_indices"],
            batch["candidate_mask"],
            phase_index=batch["phase_index"],
            side_to_move=batch["side_to_move"],
            nnue_feat_white_indices=batch["nnue_feat_white_indices"],
            nnue_feat_white_offsets=batch["nnue_feat_white_offsets"],
            nnue_feat_black_indices=batch["nnue_feat_black_indices"],
            nnue_feat_black_offsets=batch["nnue_feat_black_offsets"],
        )

    assert torch.allclose(
        single_outputs["final_value"]["wdl_logits"],
        phase_outputs["final_value"]["wdl_logits"],
        atol=1e-6,
    )
    assert torch.allclose(
        single_outputs["final_value"]["cp_score"],
        phase_outputs["final_value"]["cp_score"],
        atol=1e-6,
    )


def test_warm_start_checkpoint_forward_matches_lapv1(tmp_path: Path) -> None:
    source_config = LAPv1TrainConfig(
        seed=11,
        output_dir=str(tmp_path / "src_out"),
        stage="T2",
        data=PlannerDataConfig(
            train_path=str(tmp_path / "src_train.jsonl"),
            validation_path=str(tmp_path / "src_validation.jsonl"),
        ),
        model=LAPv1Config.from_mapping(
            {
                "deliberation": {"max_inner_steps": 2, "min_inner_steps": 1},
                "opponent_head": {
                    "architecture": "set_v2",
                    "hidden_dim": 64,
                    "hidden_layers": 1,
                    "action_embedding_dim": 16,
                    "dropout": 0.0,
                },
                "value_head": {"hidden_dim": 1024},
                "policy_head": {
                    "hidden_dim": 512,
                    "action_embedding_dim": 32,
                    "feedforward_dim": 1024,
                },
                "state_embedder": {"feedforward_dim": 1024},
                "intention_encoder": {"feedforward_dim": 1024},
            }
        ),
        optimization=LAPv1OptimizationConfig(epochs=1, batch_size=2, learning_rate=1e-3),
        evaluation=PlannerEvaluationConfig(top_k=3),
        runtime=PlannerRuntimeConfig(torch_threads=1, dataloader_workers=0),
        export=PlannerExportConfig(bundle_dir=str(tmp_path / "src_bundle")),
        stage2=LAPv1Stage2Config(
            phases=(
                LAPv1Stage2PhaseConfig(
                    name="joint",
                    epochs=1,
                    trainable_parameter_groups=("all",),
                    max_inner_steps_schedule=(2,),
                ),
            )
        ),
    )
    target_config = LAPv1TrainConfig(
        seed=12,
        output_dir=str(tmp_path / "target_out"),
        stage="T2",
        data=PlannerDataConfig(
            train_path=str(tmp_path / "target_train.jsonl"),
            validation_path=str(tmp_path / "target_validation.jsonl"),
        ),
        model=LAPv1Config.from_mapping(
            {
                "deliberation": {"max_inner_steps": 2, "min_inner_steps": 1},
                "opponent_head": {
                    "architecture": "set_v2",
                    "hidden_dim": 64,
                    "hidden_layers": 1,
                    "action_embedding_dim": 16,
                    "dropout": 0.0,
                },
                "value_head": {"hidden_dim": 1024},
                "policy_head": {
                    "hidden_dim": 512,
                    "action_embedding_dim": 32,
                    "feedforward_dim": 1024,
                },
                "state_embedder": {"feedforward_dim": 1024},
                "intention_encoder": {"feedforward_dim": 1024},
                "lapv2": {
                    "enabled": True,
                    "phase_moe": True,
                    "nnue_value": True,
                    "nnue_value_phase_moe": True,
                    "nnue_policy": True,
                    "shared_opponent_readout": True,
                    "N_accumulator": 8,
                },
            }
        ),
        optimization=LAPv1OptimizationConfig(epochs=1, batch_size=2, learning_rate=1e-3),
        evaluation=PlannerEvaluationConfig(top_k=3),
        runtime=PlannerRuntimeConfig(torch_threads=1, dataloader_workers=0),
        export=PlannerExportConfig(bundle_dir=str(tmp_path / "target_bundle")),
        stage2=LAPv1Stage2Config(
            phases=(
                LAPv1Stage2PhaseConfig(
                    name="joint",
                    epochs=1,
                    trainable_parameter_groups=("all",),
                    max_inner_steps_schedule=(2,),
                ),
            )
        ),
    )
    source_model = LAPv1Model(source_config.model)
    source_checkpoint = tmp_path / "lapv1_t2_source.pt"
    torch.save(
        {
            "model_name": LAPV1_MODEL_NAME,
            "lapv2_version": 0,
            "model_state_dict": source_model.state_dict(),
            "training_config": source_config.to_dict(),
        },
        source_checkpoint,
    )
    output_checkpoint = tmp_path / "lapv2_warm_start.pt"

    result = build_lapv2_warm_start_checkpoint(
        source_checkpoint,
        target_config=target_config,
        output_checkpoint=output_checkpoint,
    )

    payload = torch.load(output_checkpoint, map_location="cpu")
    target_model = LAPv1Model(target_config.model)
    target_model.load_state_dict(payload["model_state_dict"])
    batch = _collate_examples(
        [
            lapv1_training_example_from_planner_head(
                _planner_example("warm-1", teacher_index=0, teacher_cp=30.0, teacher_gap=20.0)
            ),
            lapv1_training_example_from_planner_head(
                _planner_example("warm-2", teacher_index=1, teacher_cp=-10.0, teacher_gap=15.0)
            ),
        ]
    )
    source_model.eval()
    target_model.eval()
    with torch.inference_mode():
        source_outputs = source_model(
            batch["piece_tokens"],
            batch["square_tokens"],
            batch["state_context_global"],
            batch["reachability_edges"],
            batch["candidate_features"],
            batch["candidate_action_indices"],
            batch["candidate_mask"],
            phase_index=batch["phase_index"],
            side_to_move=batch["side_to_move"],
            nnue_feat_white_indices=batch["nnue_feat_white_indices"],
            nnue_feat_white_offsets=batch["nnue_feat_white_offsets"],
            nnue_feat_black_indices=batch["nnue_feat_black_indices"],
            nnue_feat_black_offsets=batch["nnue_feat_black_offsets"],
            candidate_move_types=batch["candidate_move_types"],
            candidate_delta_white_leave_indices=batch["candidate_delta_white_leave_indices"],
            candidate_delta_white_leave_offsets=batch["candidate_delta_white_leave_offsets"],
            candidate_delta_white_enter_indices=batch["candidate_delta_white_enter_indices"],
            candidate_delta_white_enter_offsets=batch["candidate_delta_white_enter_offsets"],
            candidate_delta_black_leave_indices=batch["candidate_delta_black_leave_indices"],
            candidate_delta_black_leave_offsets=batch["candidate_delta_black_leave_offsets"],
            candidate_delta_black_enter_indices=batch["candidate_delta_black_enter_indices"],
            candidate_delta_black_enter_offsets=batch["candidate_delta_black_enter_offsets"],
            candidate_nnue_feat_white_after_move_indices=batch[
                "candidate_nnue_feat_white_after_move_indices"
            ],
            candidate_nnue_feat_white_after_move_offsets=batch[
                "candidate_nnue_feat_white_after_move_offsets"
            ],
            candidate_nnue_feat_black_after_move_indices=batch[
                "candidate_nnue_feat_black_after_move_indices"
            ],
            candidate_nnue_feat_black_after_move_offsets=batch[
                "candidate_nnue_feat_black_after_move_offsets"
            ],
            candidate_has_king_move=batch["candidate_has_king_move"],
        )
        target_outputs = target_model(
            batch["piece_tokens"],
            batch["square_tokens"],
            batch["state_context_global"],
            batch["reachability_edges"],
            batch["candidate_features"],
            batch["candidate_action_indices"],
            batch["candidate_mask"],
            phase_index=batch["phase_index"],
            side_to_move=batch["side_to_move"],
            nnue_feat_white_indices=batch["nnue_feat_white_indices"],
            nnue_feat_white_offsets=batch["nnue_feat_white_offsets"],
            nnue_feat_black_indices=batch["nnue_feat_black_indices"],
            nnue_feat_black_offsets=batch["nnue_feat_black_offsets"],
            candidate_move_types=batch["candidate_move_types"],
            candidate_delta_white_leave_indices=batch["candidate_delta_white_leave_indices"],
            candidate_delta_white_leave_offsets=batch["candidate_delta_white_leave_offsets"],
            candidate_delta_white_enter_indices=batch["candidate_delta_white_enter_indices"],
            candidate_delta_white_enter_offsets=batch["candidate_delta_white_enter_offsets"],
            candidate_delta_black_leave_indices=batch["candidate_delta_black_leave_indices"],
            candidate_delta_black_leave_offsets=batch["candidate_delta_black_leave_offsets"],
            candidate_delta_black_enter_indices=batch["candidate_delta_black_enter_indices"],
            candidate_delta_black_enter_offsets=batch["candidate_delta_black_enter_offsets"],
            candidate_nnue_feat_white_after_move_indices=batch[
                "candidate_nnue_feat_white_after_move_indices"
            ],
            candidate_nnue_feat_white_after_move_offsets=batch[
                "candidate_nnue_feat_white_after_move_offsets"
            ],
            candidate_nnue_feat_black_after_move_indices=batch[
                "candidate_nnue_feat_black_after_move_indices"
            ],
            candidate_nnue_feat_black_after_move_offsets=batch[
                "candidate_nnue_feat_black_after_move_offsets"
            ],
            candidate_has_king_move=batch["candidate_has_king_move"],
        )

    assert result.lapv2_version == 1
    assert output_checkpoint.exists()
    assert payload["warm_start_source_checkpoint"] == str(source_checkpoint)
    assert payload["lapv2_version"] == 1
    assert tuple(payload["warm_start_fresh_init_prefixes"]) == (
        "ft.",
        "value_head_nnue.",
        "policy_head_nnue.",
        "deliberation_loop.reply_signal_projector.opponent_readout.",
    )
    assert torch.allclose(source_outputs["piece_intentions"], target_outputs["piece_intentions"], atol=1e-6)
    assert torch.allclose(source_outputs["z_root"], target_outputs["z_root"], atol=1e-6)
    assert torch.allclose(source_outputs["root_sharpness"], target_outputs["root_sharpness"], atol=1e-6)
    assert target_model.opponent_readout is not None
    assert target_model.ft is not None
    assert math.isclose(
        float(target_model.ft.experts[0].ft.weight.std().item()),
        1.0 / math.sqrt(8.0),
        rel_tol=0.35,
    )
    assert torch.allclose(
        target_model.intention_encoder.experts[0].square_embedding.weight,
        target_model.intention_encoder.experts[3].square_embedding.weight,
        atol=1e-6,
    )


def test_gate_mean_pull_converges_experts() -> None:
    model = LAPv1Model(
        LAPv1Config.from_mapping(
            {
                "lapv2": {
                    "enabled": True,
                    "nnue_value": True,
                    "nnue_value_phase_moe": True,
                    "nnue_phase_gate_steps": 3,
                    "N_accumulator": 8,
                }
            }
        )
    )
    assert model.ft is not None
    experts = model.ft.experts
    with torch.no_grad():
        experts[0].ft.weight.fill_(0.0)
        experts[1].ft.weight.fill_(1.0)
        experts[2].ft.weight.fill_(2.0)
        experts[3].ft.weight.fill_(3.0)

    _apply_lapv2_phase_gate_mean_pull(model)

    assert torch.allclose(experts[0].ft.weight, experts[1].ft.weight)
    assert torch.allclose(experts[1].ft.weight, experts[2].ft.weight)
    assert torch.allclose(experts[2].ft.weight, experts[3].ft.weight)


def test_load_balancing_weights_in_range() -> None:
    weights = _phase_load_balance_weights(
        torch.tensor([0, 0, 0, 1, 2, 3], dtype=torch.long),
        enabled=True,
    )

    assert tuple(weights.shape) == (6,)
    assert torch.all(weights >= 0.5)
    assert pytest.approx(float(weights.mean().item()), rel=1e-6) == 1.0
    assert float(weights[0].item()) < float(weights[-1].item())


def test_phase_nnue_value_on_runs_training_step(tmp_path: Path) -> None:
    train_path = tmp_path / "lapv1_train.jsonl"
    validation_path = tmp_path / "lapv1_validation.jsonl"
    _write_examples(
        train_path,
        [
            _planner_example("train-1", teacher_index=0, teacher_cp=60.0, teacher_gap=40.0),
            _planner_example("train-2", teacher_index=1, teacher_cp=10.0, teacher_gap=10.0),
        ],
    )
    _write_examples(
        validation_path,
        [
            _planner_example("validation-1", teacher_index=0, teacher_cp=50.0, teacher_gap=30.0),
            _planner_example("validation-2", teacher_index=1, teacher_cp=0.0, teacher_gap=5.0),
        ],
    )

    config = LAPv1TrainConfig(
        seed=43,
        output_dir=str(tmp_path / "lapv1_out"),
        stage="T1",
        data=PlannerDataConfig(
            train_path=str(train_path),
            validation_path=str(validation_path),
        ),
        model=LAPv1Config.from_mapping(
            {
                "deliberation": {"max_inner_steps": 0, "min_inner_steps": 0},
                "opponent_head": {
                    "architecture": "set_v2",
                    "hidden_dim": 64,
                    "hidden_layers": 1,
                    "action_embedding_dim": 16,
                    "dropout": 0.0,
                },
                "value_head": {"hidden_dim": 1024},
                "policy_head": {
                    "hidden_dim": 512,
                    "action_embedding_dim": 32,
                    "feedforward_dim": 1024,
                },
                "state_embedder": {"feedforward_dim": 1024},
                "intention_encoder": {"feedforward_dim": 1024},
                "lapv2": {
                    "enabled": True,
                    "nnue_value": True,
                    "nnue_value_phase_moe": True,
                    "nnue_phase_gate_steps": 3,
                    "N_accumulator": 8,
                },
            }
        ),
        optimization=LAPv1OptimizationConfig(
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            max_grad_norm=1.0,
            value_wdl_weight=1.0,
            value_cp_weight=0.25,
            sharpness_weight=0.1,
            policy_ce_weight=1.0,
            policy_kl_weight=0.25,
            policy_margin_weight=0.1,
            policy_rank_weight=0.1,
            intention_aux_weight=0.05,
        ),
        evaluation=PlannerEvaluationConfig(top_k=3),
        runtime=PlannerRuntimeConfig(torch_threads=1, dataloader_workers=0),
        export=PlannerExportConfig(bundle_dir=str(tmp_path / "bundle")),
    )

    run = train_lapv1(config, repo_root=tmp_path)

    assert Path(run.export_paths["checkpoint"]).exists()


def _write_examples(path: Path, examples: list[PlannerHeadExample]) -> None:
    path.write_text(
        "".join(json.dumps(example.to_dict()) + "\n" for example in examples),
        encoding="utf-8",
    )


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
        transition_features=[[0.0] * 45, [0.0] * 45],
        reply_peak_probabilities=[0.5, 0.4],
        pressures=[0.2, 0.3],
        uncertainties=[0.3, 0.4],
        curriculum_bucket_labels=["lapv1_test"],
        curriculum_priority=1.0,
        teacher_top1_action_index=teacher_index + 1,
        teacher_top1_candidate_index=teacher_index,
        teacher_policy=teacher_policy,
        teacher_root_value_cp=teacher_cp,
        teacher_top1_minus_top2_cp=teacher_gap,
        teacher_candidate_scores_cp=[teacher_cp, teacher_cp - teacher_gap],
        teacher_candidate_score_delta_targets_cp=[0.0, -teacher_gap],
        teacher_rank_bucket_version=1,
        teacher_candidate_rank_bucket_targets=[0, 1],
        latent_state_version=None,
        latent_features=None,
    )
