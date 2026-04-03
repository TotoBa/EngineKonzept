"""Tests for the exact symbolic reply-scorer Phase-7 baseline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from train.eval import evaluate_symbolic_opponent_baseline
from train.models.proposer import LegalityPolicyProposer


def test_evaluate_symbolic_opponent_baseline_on_artifact_file(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    artifact_path = tmp_path / "opponent_head_test.jsonl"
    artifact_path.write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "split": "test",
                "root_fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
                "root_feature_vector": [0.0] * 230,
                "curriculum_bucket_labels": ["forced_teacher"],
                "curriculum_priority": 1.0,
                "chosen_move_uci": "e2e4",
                "chosen_action_index": 1,
                "transition_context_version": 1,
                "transition_features": [0.0] * 45,
                "next_fen": "4k3/8/8/8/4P3/8/8/4K3 b - - 0 1",
                "next_feature_vector": [0.0] * 230,
                "reply_candidate_context_version": 2,
                "reply_global_context_version": 1,
                "reply_global_features": [0.0] * 9,
                "reply_candidate_action_indices": [2, 3],
                "reply_candidate_features": [
                    [1.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 30,
                    [0.0] * 35,
                ],
                "teacher_reply_uci": "e7e5",
                "teacher_reply_action_index": 2,
                "teacher_reply_policy": [1.0, 0.0],
                "teacher_root_value_cp": 25.0,
                "teacher_root_value_mate": None,
                "teacher_top1_minus_top2_cp": 40.0,
                "pressure_target": 0.5,
                "uncertainty_target": 0.5,
                "reply_is_capture": False,
                "reply_is_promotion": False,
                "reply_is_castle": False,
                "reply_is_en_passant": False,
                "reply_gives_check": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    checkpoint_path = tmp_path / "symbolic_checkpoint.pt"
    torch.manual_seed(0)
    model = LegalityPolicyProposer(
        architecture="symbolic_v1",
        hidden_dim=32,
        hidden_layers=1,
        dropout=0.0,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "training_config": {
                "seed": 0,
                "output_dir": "artifacts/tests",
                "data": {
                    "dataset_path": "artifacts/datasets/phase5_stockfish_pgn_train_pi_10k_v1",
                    "train_split": "train",
                    "validation_split": "validation",
                },
                "model": {
                    "architecture": "symbolic_v1",
                    "hidden_dim": 32,
                    "hidden_layers": 1,
                    "dropout": 0.0,
                },
                "optimization": {
                    "epochs": 1,
                    "batch_size": 4,
                    "learning_rate": 1e-3,
                    "weight_decay": 0.0,
                    "legality_loss_weight": 1.0,
                    "policy_loss_weight": 1.0,
                },
                "evaluation": {
                    "legality_threshold": 0.5,
                    "checkpoint_selection": "legality_first",
                    "selection_policy_weight": 1.0,
                },
                "runtime": {
                    "torch_threads": 0,
                    "dataloader_workers": 0,
                },
                "export": {
                    "bundle_dir": "models/proposer/test",
                    "enabled": False,
                    "checkpoint_name": "checkpoint.pt",
                    "exported_program_name": "proposer.pt2",
                    "runtime_weights_name": "symbolic_runtime.bin",
                    "metadata_name": "metadata.json",
                },
            },
        },
        checkpoint_path,
    )

    metrics = evaluate_symbolic_opponent_baseline(
        checkpoint_path,
        dataset_path=artifact_path,
        split="test",
    )

    assert metrics.total_examples == 1
    assert metrics.supervised_examples == 1
    assert metrics.reply_top1_accuracy == 1.0
    assert metrics.reply_top3_accuracy == 1.0
    assert metrics.teacher_reply_mean_reciprocal_rank == 1.0
    assert abs(metrics.teacher_reply_mean_probability - 0.5) < 1e-6
    assert metrics.pressure_mae == 0.0
    assert metrics.uncertainty_mae == 0.0
