from __future__ import annotations

import json
from pathlib import Path

import pytest

from train.trainers import count_lapv1_model_parameters, load_lapv1_train_config
from train.trainers.lapv1 import LAPv1TrainConfig


pytest.importorskip("torch")


def test_lapv1_train_config_from_dict_accepts_lapv1_architecture() -> None:
    config = LAPv1TrainConfig.from_dict(
        {
            "seed": 7,
            "output_dir": "models/lapv1/test",
            "stage": "T1",
            "data": {
                "train_path": "train.jsonl",
                "validation_path": "validation.jsonl",
            },
            "model": {
                "architecture": "lapv1",
                "intention_encoder": {"feedforward_dim": 1024},
                "state_embedder": {"feedforward_dim": 1024},
                "value_head": {"hidden_dim": 1024},
                "policy_head": {
                    "hidden_dim": 512,
                    "action_embedding_dim": 32,
                    "feedforward_dim": 1024,
                },
                "deliberation": {"max_inner_steps": 0, "min_inner_steps": 0},
                "opponent_head": {
                    "architecture": "set_v2",
                    "hidden_dim": 64,
                    "hidden_layers": 1,
                    "action_embedding_dim": 16,
                    "dropout": 0.0,
                },
            },
            "optimization": {
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "value_wdl_weight": 1.0,
                "value_cp_weight": 0.25,
                "sharpness_weight": 0.1,
                "policy_ce_weight": 1.0,
                "policy_kl_weight": 0.25,
                "policy_margin_weight": 0.1,
                "policy_rank_weight": 0.1,
                "intention_aux_weight": 0.05,
            },
            "evaluation": {"top_k": 3},
            "runtime": {"torch_threads": 1, "dataloader_workers": 0},
            "export": {"bundle_dir": "models/lapv1/test/bundle"},
        }
    )

    assert config.stage == "T1"
    assert config.model.deliberation.max_inner_steps == 0
    assert count_lapv1_model_parameters(config) > 0


def test_lapv1_train_config_from_dict_rejects_non_lapv1_architecture() -> None:
    with pytest.raises(ValueError, match="model.architecture must be 'lapv1'"):
        LAPv1TrainConfig.from_dict(
            {
                "seed": 7,
                "output_dir": "models/lapv1/test",
                "stage": "T1",
                "data": {
                    "train_path": "train.jsonl",
                    "validation_path": "validation.jsonl",
                },
                "model": {
                    "architecture": "set_v2",
                    "deliberation": {"max_inner_steps": 0, "min_inner_steps": 0},
                },
                "optimization": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 1e-3,
                    "weight_decay": 0.0,
                },
                "evaluation": {"top_k": 3},
                "export": {"bundle_dir": "models/lapv1/test/bundle"},
            }
        )


def test_reference_stage1_config_loads_and_targets_filtered_two_tier_slice() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "python" / "configs" / "phase10_lapv1_stage1_10k_122k_v1.json"

    config = load_lapv1_train_config(config_path)

    train_paths = config.data.resolved_train_paths()
    validation_paths = config.data.resolved_validation_paths()

    assert config.stage == "T1"
    assert config.optimization.epochs == 20
    assert config.model.deliberation.max_inner_steps == 0
    assert config.output_dir == "models/lapv1/stage1_10k_122k_v1"
    assert config.export.bundle_dir == "models/lapv1/stage1_10k_122k_v1/bundle"
    assert len(train_paths) == 2
    assert len(validation_paths) == 2
    assert "pgn_10k_train_v1" in train_paths[0]
    assert "merged_unique_122k_train_v1" in train_paths[1]

    raw_payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert raw_payload["model"]["architecture"] == "lapv1"
