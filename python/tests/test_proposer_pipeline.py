"""Tests for the Phase-5 legality/policy proposer pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from train.action_space import ACTION_SPACE_SIZE, flatten_action, unflatten_action
from train.config import (
    DEFAULT_PROPOSER_METADATA_NAME,
    load_proposer_train_config,
    resolve_repo_path,
)
from train.datasets import (
    POSITION_FEATURE_SIZE,
    build_symbolic_proposer_example,
    candidate_context_feature_dim,
    load_proposer_examples,
    materialize_symbolic_proposer_artifacts,
    proposer_artifact_name,
    symbolic_candidate_context_v2_feature_spec,
    symbolic_proposer_artifact_name,
    write_dataset_artifacts,
)
from train.datasets.artifacts import (
    SYMBOLIC_PROPOSER_CANDIDATE_FEATURE_SIZE,
    SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE,
    to_proposer_example,
)
from train.datasets.schema import DatasetExample
from train.export.proposer import build_export_metadata
from train.models.proposer import LegalityPolicyProposer
from train.trainers.proposer import ProposerMetrics, _is_better_validation
from train.trainers import train_proposer


def test_action_flatten_roundtrip_is_stable() -> None:
    action = [12, 28, 0]
    flat_index = flatten_action(action)

    assert 0 <= flat_index < ACTION_SPACE_SIZE
    assert unflatten_action(flat_index) == action


def test_proposer_examples_load_and_pack_fixed_width_features(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "train.jsonl").write_text(
        json.dumps(_dataset_example_dict(split="train")) + "\n",
        encoding="utf-8",
    )

    examples = load_proposer_examples(dataset_dir, "train")

    assert len(examples) == 1
    assert len(examples[0].feature_vector) == POSITION_FEATURE_SIZE
    assert examples[0].selected_action_index == flatten_action([12, 28, 0])
    assert examples[0].legal_action_indices == sorted(examples[0].legal_action_indices)


def test_proposer_examples_prefer_lean_split_artifacts_when_available(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "train.jsonl").write_text(
        json.dumps(_dataset_example_dict(split="train")) + "\n",
        encoding="utf-8",
    )
    lean_example = to_proposer_example(
        DatasetExample.from_dict(_dataset_example_dict(split="train"))
    )
    lean_payload = lean_example.to_dict()
    lean_payload["sample_id"] = "lean:train"
    (dataset_dir / proposer_artifact_name("train")).write_text(
        json.dumps(lean_payload) + "\n",
        encoding="utf-8",
    )

    examples = load_proposer_examples(dataset_dir, "train")

    assert len(examples) == 1
    assert examples[0].sample_id == "lean:train"
    assert len(examples[0].feature_vector) == POSITION_FEATURE_SIZE


def test_train_proposer_can_load_optional_lean_artifacts(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    dataset_examples = [
        DatasetExample.from_dict(_dataset_example_dict(sample_id="train:1", split="train")),
        DatasetExample.from_dict(_dataset_example_dict(sample_id="validation:1", split="validation")),
    ]
    dataset = SimpleNamespace(examples=dataset_examples, summary={"ok": True})

    write_dataset_artifacts(dataset_dir, dataset, write_proposer_artifacts=True)

    train_examples = load_proposer_examples(dataset_dir, "train")
    validation_examples = load_proposer_examples(dataset_dir, "validation")

    assert len(train_examples) == 1
    assert len(validation_examples) == 1
    assert train_examples[0].sample_id == "train:1"
    assert validation_examples[0].sample_id == "validation:1"


def test_symbolic_proposer_examples_can_be_materialized_and_loaded(tmp_path: Path) -> None:
    pytest.importorskip("chess")

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    dataset_examples = [
        DatasetExample.from_dict(_dataset_example_dict(sample_id="train:1", split="train")),
    ]
    dataset = SimpleNamespace(examples=dataset_examples, summary={"ok": True})

    write_dataset_artifacts(dataset_dir, dataset, write_proposer_artifacts=True)
    counts = materialize_symbolic_proposer_artifacts(dataset_dir)
    examples = load_proposer_examples(dataset_dir, "train", variant="symbolic")

    assert counts["train"] == 1
    assert (dataset_dir / symbolic_proposer_artifact_name("train")).exists()
    assert len(examples) == 1
    assert len(examples[0].global_features) == SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE
    assert len(examples[0].candidate_features[0]) == SYMBOLIC_PROPOSER_CANDIDATE_FEATURE_SIZE
    assert examples[0].candidate_context_version == 1
    assert examples[0].global_context_version == 1


def test_symbolic_candidate_context_v2_spec_is_exposed() -> None:
    spec = symbolic_candidate_context_v2_feature_spec()

    assert spec["candidate_context_version"] == 2
    assert spec["candidate_feature_dim"] == candidate_context_feature_dim(2)
    assert "promotion_to_queen" in spec["candidate_feature_order"]
    assert "castle_queenside" in spec["candidate_feature_order"]
    assert "delta_file_normalized" in spec["candidate_feature_order"]


def test_symbolic_proposer_example_supports_candidate_context_v2() -> None:
    pytest.importorskip("chess")

    example = build_symbolic_proposer_example(
        DatasetExample.from_dict(_dataset_example_dict(split="train")),
        candidate_context_version=2,
        global_context_version=1,
    )

    assert example.candidate_context_version == 2
    assert example.global_context_version == 1
    assert len(example.candidate_features[0]) == candidate_context_feature_dim(2)
    assert len(example.global_features) == SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE


def test_config_accepts_symbolic_v1_architecture(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 7,
                "output_dir": "artifacts/phase5/test-run",
                "data": {
                    "dataset_path": "artifacts/datasets/phase4",
                    "train_split": "train",
                    "validation_split": "validation",
                },
                "model": {
                    "architecture": "symbolic_v1",
                    "hidden_dim": 64,
                    "hidden_layers": 2,
                    "dropout": 0.0,
                },
                "optimization": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "legality_loss_weight": 1.0,
                    "policy_loss_weight": 1.0,
                },
                "evaluation": {"legality_threshold": 0.4},
                "export": {
                    "enabled": False,
                    "bundle_dir": "models/proposer/test-symbolic-v1",
                    "checkpoint_name": "checkpoint.pt",
                    "exported_program_name": "proposer.pt2",
                    "metadata_name": "metadata.json",
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_proposer_train_config(config_path)

    assert config.model.architecture == "symbolic_v1"
    assert config.export.enabled is False


def test_export_metadata_marks_symbolic_legality_source(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 7,
                "output_dir": "artifacts/phase5/test-run",
                "data": {
                    "dataset_path": "artifacts/datasets/phase4",
                    "train_split": "train",
                    "validation_split": "validation",
                },
                "model": {
                    "architecture": "symbolic_v1",
                    "hidden_dim": 64,
                    "hidden_layers": 2,
                    "dropout": 0.0,
                },
                "optimization": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "legality_loss_weight": 1.0,
                    "policy_loss_weight": 1.0,
                },
                "evaluation": {"legality_threshold": 0.4},
                "export": {
                    "enabled": True,
                    "bundle_dir": "models/proposer/test-symbolic",
                    "checkpoint_name": "checkpoint.pt",
                    "exported_program_name": "proposer.pt2",
                    "metadata_name": "metadata.json",
                },
            }
        ),
        encoding="utf-8",
    )

    metadata = build_export_metadata(
        load_proposer_train_config(config_path),
        validation_metrics={},
    )

    assert metadata["outputs"]["legality_source"] == "symbolic_generator"
    assert metadata["input"]["symbolic"]["max_legal_candidates"] > 0


def test_config_loading_and_export_metadata_are_repo_relative(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 7,
                "output_dir": "artifacts/phase5/test-run",
                "data": {
                    "dataset_path": "artifacts/datasets/phase4",
                    "train_split": "train",
                    "validation_split": "validation",
                },
                "model": {"hidden_dim": 64, "hidden_layers": 2, "dropout": 0.0},
                "optimization": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "legality_loss_weight": 1.0,
                    "policy_loss_weight": 1.0,
                },
                "evaluation": {"legality_threshold": 0.4},
                "export": {
                    "bundle_dir": "models/proposer/test",
                    "checkpoint_name": "checkpoint.pt",
                    "exported_program_name": "proposer.pt2",
                    "metadata_name": "metadata.json",
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_proposer_train_config(config_path)
    metadata = build_export_metadata(
        config,
        validation_metrics={"legal_set_precision": 0.5, "legal_set_recall": 0.25},
    )

    assert resolve_repo_path(tmp_path, config.output_dir) == tmp_path / "artifacts/phase5/test-run"
    assert metadata["schema_version"] == 4
    assert metadata["training"]["architecture"] == "mlp_v1"
    assert metadata["action_space"]["flat_size"] == ACTION_SPACE_SIZE
    assert metadata["input"]["feature_dim"] == POSITION_FEATURE_SIZE
    assert metadata["outputs"]["legality_threshold"] == 0.4
    assert metadata["training"]["checkpoint_selection"] == "legality_first"
    assert metadata["artifacts"]["exported_program_file"] == "proposer.pt2"
    assert metadata["outputs"]["legality_source"] == "learned_head"
    assert config.runtime.torch_threads == 0
    assert config.runtime.dataloader_workers == 0


def test_config_accepts_multistream_v2_architecture(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 7,
                "output_dir": "artifacts/phase5/test-run",
                "data": {
                    "dataset_path": "artifacts/datasets/phase4",
                    "train_split": "train",
                    "validation_split": "validation",
                },
                "model": {
                    "architecture": "multistream_v2",
                    "hidden_dim": 64,
                    "hidden_layers": 2,
                    "dropout": 0.0,
                },
                "optimization": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "legality_loss_weight": 1.0,
                    "policy_loss_weight": 1.0,
                },
                "evaluation": {"legality_threshold": 0.4},
                "export": {
                    "bundle_dir": "models/proposer/test",
                    "checkpoint_name": "checkpoint.pt",
                    "exported_program_name": "proposer.pt2",
                    "metadata_name": "metadata.json",
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_proposer_train_config(config_path)

    assert config.model.architecture == "multistream_v2"


def test_config_accepts_balanced_checkpoint_selection(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 7,
                "output_dir": "artifacts/phase5/test-run",
                "data": {
                    "dataset_path": "artifacts/datasets/phase4",
                    "train_split": "train",
                    "validation_split": "validation",
                },
                "model": {
                    "architecture": "factorized_v5",
                    "hidden_dim": 64,
                    "hidden_layers": 2,
                    "dropout": 0.0,
                },
                "optimization": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "legality_loss_weight": 1.0,
                    "policy_loss_weight": 1.0,
                },
                "evaluation": {
                    "legality_threshold": 0.4,
                    "checkpoint_selection": "balanced",
                    "selection_policy_weight": 5.0,
                },
                "export": {
                    "bundle_dir": "models/proposer/test",
                    "checkpoint_name": "checkpoint.pt",
                    "exported_program_name": "proposer.pt2",
                    "metadata_name": "metadata.json",
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_proposer_train_config(config_path)

    assert config.evaluation.checkpoint_selection == "balanced"
    assert config.evaluation.selection_policy_weight == 5.0


def test_config_accepts_factorized_v3_architecture(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 7,
                "output_dir": "artifacts/phase5/test-run",
                "data": {
                    "dataset_path": "artifacts/datasets/phase4",
                    "train_split": "train",
                    "validation_split": "validation",
                },
                "model": {
                    "architecture": "factorized_v3",
                    "hidden_dim": 64,
                    "hidden_layers": 2,
                    "dropout": 0.0,
                },
                "optimization": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "legality_loss_weight": 1.0,
                    "policy_loss_weight": 1.0,
                },
                "evaluation": {"legality_threshold": 0.4},
                "export": {
                    "bundle_dir": "models/proposer/test",
                    "checkpoint_name": "checkpoint.pt",
                    "exported_program_name": "proposer.pt2",
                    "metadata_name": "metadata.json",
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_proposer_train_config(config_path)

    assert config.model.architecture == "factorized_v3"


def test_config_accepts_factorized_v4_architecture(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 7,
                "output_dir": "artifacts/phase5/test-run",
                "data": {
                    "dataset_path": "artifacts/datasets/phase4",
                    "train_split": "train",
                    "validation_split": "validation",
                },
                "model": {
                    "architecture": "factorized_v4",
                    "hidden_dim": 64,
                    "hidden_layers": 2,
                    "dropout": 0.0,
                },
                "optimization": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "legality_loss_weight": 1.0,
                    "policy_loss_weight": 1.0,
                },
                "evaluation": {"legality_threshold": 0.4},
                "export": {
                    "bundle_dir": "models/proposer/test",
                    "checkpoint_name": "checkpoint.pt",
                    "exported_program_name": "proposer.pt2",
                    "metadata_name": "metadata.json",
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_proposer_train_config(config_path)

    assert config.model.architecture == "factorized_v4"


def test_config_accepts_factorized_v5_architecture(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 7,
                "output_dir": "artifacts/phase5/test-run",
                "data": {
                    "dataset_path": "artifacts/datasets/phase4",
                    "train_split": "train",
                    "validation_split": "validation",
                },
                "model": {
                    "architecture": "factorized_v5",
                    "hidden_dim": 64,
                    "hidden_layers": 2,
                    "dropout": 0.0,
                },
                "optimization": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "legality_loss_weight": 1.0,
                    "policy_loss_weight": 1.0,
                },
                "evaluation": {"legality_threshold": 0.4},
                "export": {
                    "bundle_dir": "models/proposer/test",
                    "checkpoint_name": "checkpoint.pt",
                    "exported_program_name": "proposer.pt2",
                    "metadata_name": "metadata.json",
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_proposer_train_config(config_path)

    assert config.model.architecture == "factorized_v5"


def test_config_accepts_factorized_v6_architecture(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 7,
                "output_dir": "artifacts/phase5/test-run",
                "data": {
                    "dataset_path": "artifacts/datasets/phase4",
                    "train_split": "train",
                    "validation_split": "validation",
                },
                "model": {
                    "architecture": "factorized_v6",
                    "hidden_dim": 64,
                    "hidden_layers": 2,
                    "dropout": 0.0,
                },
                "optimization": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "legality_loss_weight": 1.0,
                    "policy_loss_weight": 1.0,
                },
                "evaluation": {"legality_threshold": 0.4},
                "export": {
                    "bundle_dir": "models/proposer/test",
                    "checkpoint_name": "checkpoint.pt",
                    "exported_program_name": "proposer.pt2",
                    "metadata_name": "metadata.json",
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_proposer_train_config(config_path)

    assert config.model.architecture == "factorized_v6"


def test_config_accepts_relational_v1_architecture(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 7,
                "output_dir": "artifacts/phase5/test-run",
                "data": {
                    "dataset_path": "artifacts/datasets/phase4",
                    "train_split": "train",
                    "validation_split": "validation",
                },
                "model": {
                    "architecture": "relational_v1",
                    "hidden_dim": 64,
                    "hidden_layers": 2,
                    "dropout": 0.0,
                },
                "optimization": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "legality_loss_weight": 1.0,
                    "policy_loss_weight": 1.0,
                },
                "evaluation": {"legality_threshold": 0.4},
                "export": {
                    "bundle_dir": "models/proposer/test",
                    "checkpoint_name": "checkpoint.pt",
                    "exported_program_name": "proposer.pt2",
                    "metadata_name": "metadata.json",
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_proposer_train_config(config_path)

    assert config.model.architecture == "relational_v1"


def test_multistream_v2_forward_matches_action_space_shape() -> None:
    torch = pytest.importorskip("torch")
    model = LegalityPolicyProposer(
        architecture="multistream_v2",
        hidden_dim=32,
        hidden_layers=2,
        dropout=0.0,
    )
    features = torch.zeros((3, POSITION_FEATURE_SIZE), dtype=torch.float32)

    legality_logits, policy_logits = model(features)

    assert tuple(legality_logits.shape) == (3, ACTION_SPACE_SIZE)
    assert tuple(policy_logits.shape) == (3, ACTION_SPACE_SIZE)


def test_factorized_v3_forward_matches_action_space_shape() -> None:
    torch = pytest.importorskip("torch")
    model = LegalityPolicyProposer(
        architecture="factorized_v3",
        hidden_dim=32,
        hidden_layers=2,
        dropout=0.0,
    )
    features = torch.zeros((3, POSITION_FEATURE_SIZE), dtype=torch.float32)

    legality_logits, policy_logits = model(features)

    assert tuple(legality_logits.shape) == (3, ACTION_SPACE_SIZE)
    assert tuple(policy_logits.shape) == (3, ACTION_SPACE_SIZE)


def test_factorized_v4_forward_matches_action_space_shape() -> None:
    torch = pytest.importorskip("torch")
    model = LegalityPolicyProposer(
        architecture="factorized_v4",
        hidden_dim=32,
        hidden_layers=2,
        dropout=0.0,
    )
    features = torch.zeros((3, POSITION_FEATURE_SIZE), dtype=torch.float32)

    legality_logits, policy_logits = model(features)

    assert tuple(legality_logits.shape) == (3, ACTION_SPACE_SIZE)
    assert tuple(policy_logits.shape) == (3, ACTION_SPACE_SIZE)


def test_factorized_v5_forward_matches_action_space_shape() -> None:
    torch = pytest.importorskip("torch")
    model = LegalityPolicyProposer(
        architecture="factorized_v5",
        hidden_dim=32,
        hidden_layers=2,
        dropout=0.0,
    )
    features = torch.zeros((3, POSITION_FEATURE_SIZE), dtype=torch.float32)

    legality_logits, policy_logits = model(features)

    assert tuple(legality_logits.shape) == (3, ACTION_SPACE_SIZE)
    assert tuple(policy_logits.shape) == (3, ACTION_SPACE_SIZE)


def test_factorized_v6_forward_matches_action_space_shape() -> None:
    torch = pytest.importorskip("torch")
    model = LegalityPolicyProposer(
        architecture="factorized_v6",
        hidden_dim=32,
        hidden_layers=2,
        dropout=0.0,
    )
    features = torch.zeros((3, POSITION_FEATURE_SIZE), dtype=torch.float32)

    legality_logits, policy_logits = model(features)

    assert tuple(legality_logits.shape) == (3, ACTION_SPACE_SIZE)
    assert tuple(policy_logits.shape) == (3, ACTION_SPACE_SIZE)


def test_relational_v1_forward_matches_action_space_shape() -> None:
    torch = pytest.importorskip("torch")
    model = LegalityPolicyProposer(
        architecture="relational_v1",
        hidden_dim=32,
        hidden_layers=2,
        dropout=0.0,
    )
    features = torch.zeros((3, POSITION_FEATURE_SIZE), dtype=torch.float32)

    legality_logits, policy_logits = model(features)

    assert tuple(legality_logits.shape) == (3, ACTION_SPACE_SIZE)
    assert tuple(policy_logits.shape) == (3, ACTION_SPACE_SIZE)


def test_factorized_v3_uses_fewer_parameters_than_flat_mlp() -> None:
    pytest.importorskip("torch")
    flat_model = LegalityPolicyProposer(
        architecture="mlp_v1",
        hidden_dim=128,
        hidden_layers=2,
        dropout=0.0,
    )
    factorized_model = LegalityPolicyProposer(
        architecture="factorized_v3",
        hidden_dim=128,
        hidden_layers=2,
        dropout=0.0,
    )

    flat_param_count = sum(parameter.numel() for parameter in flat_model.parameters())
    factorized_param_count = sum(
        parameter.numel() for parameter in factorized_model.parameters()
    )

    assert factorized_param_count < flat_param_count


def test_factorized_v4_has_more_capacity_than_factorized_v3() -> None:
    pytest.importorskip("torch")
    factorized_v3_model = LegalityPolicyProposer(
        architecture="factorized_v3",
        hidden_dim=128,
        hidden_layers=2,
        dropout=0.0,
    )
    factorized_v4_model = LegalityPolicyProposer(
        architecture="factorized_v4",
        hidden_dim=128,
        hidden_layers=2,
        dropout=0.0,
    )

    factorized_v3_param_count = sum(
        parameter.numel() for parameter in factorized_v3_model.parameters()
    )
    factorized_v4_param_count = sum(
        parameter.numel() for parameter in factorized_v4_model.parameters()
    )

    assert factorized_v4_param_count > factorized_v3_param_count


def test_factorized_v5_has_more_capacity_than_factorized_v4() -> None:
    pytest.importorskip("torch")
    factorized_v4_model = LegalityPolicyProposer(
        architecture="factorized_v4",
        hidden_dim=128,
        hidden_layers=2,
        dropout=0.0,
    )
    factorized_v5_model = LegalityPolicyProposer(
        architecture="factorized_v5",
        hidden_dim=128,
        hidden_layers=2,
        dropout=0.0,
    )

    factorized_v4_param_count = sum(
        parameter.numel() for parameter in factorized_v4_model.parameters()
    )
    factorized_v5_param_count = sum(
        parameter.numel() for parameter in factorized_v5_model.parameters()
    )

    assert factorized_v5_param_count > factorized_v4_param_count


def test_factorized_v6_has_more_capacity_than_factorized_v5() -> None:
    pytest.importorskip("torch")
    factorized_v5_model = LegalityPolicyProposer(
        architecture="factorized_v5",
        hidden_dim=128,
        hidden_layers=2,
        dropout=0.0,
    )
    factorized_v6_model = LegalityPolicyProposer(
        architecture="factorized_v6",
        hidden_dim=128,
        hidden_layers=2,
        dropout=0.0,
    )

    factorized_v5_param_count = sum(
        parameter.numel() for parameter in factorized_v5_model.parameters()
    )
    factorized_v6_param_count = sum(
        parameter.numel() for parameter in factorized_v6_model.parameters()
    )

    assert factorized_v6_param_count > factorized_v5_param_count


def test_relational_v1_uses_less_capacity_than_multistream_v2() -> None:
    pytest.importorskip("torch")
    multistream_model = LegalityPolicyProposer(
        architecture="multistream_v2",
        hidden_dim=128,
        hidden_layers=2,
        dropout=0.0,
    )
    relational_model = LegalityPolicyProposer(
        architecture="relational_v1",
        hidden_dim=128,
        hidden_layers=2,
        dropout=0.0,
    )

    multistream_param_count = sum(parameter.numel() for parameter in multistream_model.parameters())
    relational_param_count = sum(parameter.numel() for parameter in relational_model.parameters())

    assert relational_param_count < multistream_param_count


def test_balanced_checkpoint_selection_can_prefer_policy_better_epoch() -> None:
    lower_policy_higher_f1 = ProposerMetrics(
        total_examples=1,
        labeled_policy_examples=1,
        total_loss=0.0,
        legality_loss=0.0,
        policy_loss=0.0,
        legal_set_precision=0.0,
        legal_set_recall=0.03,
        legal_set_f1=0.055,
        policy_top1_accuracy=0.014,
        examples_per_second=1.0,
    )
    higher_policy_lower_f1 = ProposerMetrics(
        total_examples=1,
        labeled_policy_examples=1,
        total_loss=0.0,
        legality_loss=0.0,
        policy_loss=0.0,
        legal_set_precision=0.0,
        legal_set_recall=0.02,
        legal_set_f1=0.023,
        policy_top1_accuracy=0.022,
        examples_per_second=1.0,
    )

    assert _is_better_validation(
        higher_policy_lower_f1,
        lower_policy_higher_f1,
        selection_mode="balanced",
        selection_policy_weight=5.0,
    )


def test_config_rejects_non_default_metadata_filename(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 7,
                "output_dir": "artifacts/phase5/test-run",
                "data": {
                    "dataset_path": "artifacts/datasets/phase4",
                    "train_split": "train",
                    "validation_split": "validation",
                },
                "model": {"hidden_dim": 64, "hidden_layers": 2, "dropout": 0.0},
                "optimization": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "legality_loss_weight": 1.0,
                    "policy_loss_weight": 1.0,
                },
                "evaluation": {"legality_threshold": 0.4},
                "export": {
                    "bundle_dir": "models/proposer/test",
                    "checkpoint_name": "checkpoint.pt",
                    "exported_program_name": "proposer.pt2",
                    "metadata_name": "bundle-metadata.json",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=f"export.metadata_name must be '{DEFAULT_PROPOSER_METADATA_NAME}'",
    ):
        load_proposer_train_config(config_path)


def test_train_proposer_exports_bundle_when_torch_is_available(tmp_path: Path) -> None:
    pytest.importorskip("torch")

    dataset_dir = tmp_path / "artifacts" / "datasets" / "phase4"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "train.jsonl").write_text(
        "\n".join(
            [
                json.dumps(_dataset_example_dict(sample_id="train:1", split="train")),
                json.dumps(_dataset_example_dict(sample_id="train:2", split="train")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "validation.jsonl").write_text(
        json.dumps(_dataset_example_dict(sample_id="validation:1", split="validation")) + "\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "python" / "configs" / "phase5_proposer_v1.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "seed": 3,
                "output_dir": "artifacts/phase5/test-run",
                "data": {
                    "dataset_path": "artifacts/datasets/phase4",
                    "train_split": "train",
                    "validation_split": "validation",
                },
                "model": {"hidden_dim": 32, "hidden_layers": 1, "dropout": 0.0},
                "optimization": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "legality_loss_weight": 1.0,
                    "policy_loss_weight": 1.0,
                },
                "evaluation": {"legality_threshold": 0.5},
                "export": {
                    "bundle_dir": "models/proposer/test",
                    "checkpoint_name": "checkpoint.pt",
                    "exported_program_name": "proposer.pt2",
                    "metadata_name": "metadata.json",
                },
            }
        ),
        encoding="utf-8",
    )

    run = train_proposer(load_proposer_train_config(config_path), repo_root=tmp_path)

    assert run.best_epoch == 1
    assert run.model_parameter_count > 0
    assert run.history[0]["examples_per_second"] > 0.0
    assert (tmp_path / "models/proposer/test/checkpoint.pt").exists()
    assert (tmp_path / "models/proposer/test/proposer.pt2").exists()
    assert (tmp_path / "models/proposer/test/metadata.json").exists()
    assert (tmp_path / "artifacts/phase5/test-run/summary.json").exists()


def test_train_proposer_supports_multistream_v2(tmp_path: Path) -> None:
    pytest.importorskip("torch")

    dataset_dir = tmp_path / "artifacts" / "datasets" / "phase4"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "train.jsonl").write_text(
        "\n".join(
            [
                json.dumps(_dataset_example_dict(sample_id="train:1", split="train")),
                json.dumps(_dataset_example_dict(sample_id="train:2", split="train")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "validation.jsonl").write_text(
        json.dumps(_dataset_example_dict(sample_id="validation:1", split="validation")) + "\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "python" / "configs" / "phase5_proposer_v2.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "seed": 3,
                "output_dir": "artifacts/phase5/test-run-v2",
                "data": {
                    "dataset_path": "artifacts/datasets/phase4",
                    "train_split": "train",
                    "validation_split": "validation",
                },
                "model": {
                    "architecture": "multistream_v2",
                    "hidden_dim": 32,
                    "hidden_layers": 1,
                    "dropout": 0.0,
                },
                "optimization": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "legality_loss_weight": 1.0,
                    "policy_loss_weight": 1.0,
                },
                "evaluation": {"legality_threshold": 0.5},
                "export": {
                    "bundle_dir": "models/proposer/test-v2",
                    "checkpoint_name": "checkpoint.pt",
                    "exported_program_name": "proposer.pt2",
                    "metadata_name": "metadata.json",
                },
            }
        ),
        encoding="utf-8",
    )

    run = train_proposer(load_proposer_train_config(config_path), repo_root=tmp_path)

    assert run.best_epoch == 1
    assert run.model_parameter_count > 0
    assert (tmp_path / "models/proposer/test-v2/checkpoint.pt").exists()
    assert (tmp_path / "models/proposer/test-v2/proposer.pt2").exists()


def test_train_proposer_supports_factorized_v3(tmp_path: Path) -> None:
    pytest.importorskip("torch")

    dataset_dir = tmp_path / "artifacts" / "datasets" / "phase4"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "train.jsonl").write_text(
        "\n".join(
            [
                json.dumps(_dataset_example_dict(sample_id="train:1", split="train")),
                json.dumps(_dataset_example_dict(sample_id="train:2", split="train")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "validation.jsonl").write_text(
        json.dumps(_dataset_example_dict(sample_id="validation:1", split="validation")) + "\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "python" / "configs" / "phase5_proposer_v3.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "seed": 3,
                "output_dir": "artifacts/phase5/test-run-v3",
                "data": {
                    "dataset_path": "artifacts/datasets/phase4",
                    "train_split": "train",
                    "validation_split": "validation",
                },
                "model": {
                    "architecture": "factorized_v3",
                    "hidden_dim": 32,
                    "hidden_layers": 1,
                    "dropout": 0.0,
                },
                "optimization": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "legality_loss_weight": 1.0,
                    "policy_loss_weight": 1.0,
                },
                "evaluation": {"legality_threshold": 0.5},
                "export": {
                    "bundle_dir": "models/proposer/test-v3",
                    "checkpoint_name": "checkpoint.pt",
                    "exported_program_name": "proposer.pt2",
                    "metadata_name": "metadata.json",
                },
            }
        ),
        encoding="utf-8",
    )

    run = train_proposer(load_proposer_train_config(config_path), repo_root=tmp_path)

    assert run.best_epoch == 1
    assert run.model_parameter_count > 0
    assert (tmp_path / "models/proposer/test-v3/checkpoint.pt").exists()
    assert (tmp_path / "models/proposer/test-v3/proposer.pt2").exists()


def test_train_proposer_supports_factorized_v4(tmp_path: Path) -> None:
    pytest.importorskip("torch")

    dataset_dir = tmp_path / "artifacts" / "datasets" / "phase4"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "train.jsonl").write_text(
        "\n".join(
            [
                json.dumps(_dataset_example_dict(sample_id="train:1", split="train")),
                json.dumps(_dataset_example_dict(sample_id="train:2", split="train")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "validation.jsonl").write_text(
        json.dumps(_dataset_example_dict(sample_id="validation:1", split="validation")) + "\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "python" / "configs" / "phase5_proposer_v4.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "seed": 3,
                "output_dir": "artifacts/phase5/test-run-v4",
                "data": {
                    "dataset_path": "artifacts/datasets/phase4",
                    "train_split": "train",
                    "validation_split": "validation",
                },
                "model": {
                    "architecture": "factorized_v4",
                    "hidden_dim": 32,
                    "hidden_layers": 1,
                    "dropout": 0.0,
                },
                "optimization": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "legality_loss_weight": 1.0,
                    "policy_loss_weight": 1.0,
                },
                "evaluation": {"legality_threshold": 0.5},
                "export": {
                    "bundle_dir": "models/proposer/test-v4",
                    "checkpoint_name": "checkpoint.pt",
                    "exported_program_name": "proposer.pt2",
                    "metadata_name": "metadata.json",
                },
            }
        ),
        encoding="utf-8",
    )

    run = train_proposer(load_proposer_train_config(config_path), repo_root=tmp_path)

    assert run.best_epoch == 1
    assert run.model_parameter_count > 0
    assert (tmp_path / "models/proposer/test-v4/checkpoint.pt").exists()
    assert (tmp_path / "models/proposer/test-v4/proposer.pt2").exists()


def test_train_proposer_supports_factorized_v5(tmp_path: Path) -> None:
    pytest.importorskip("torch")

    dataset_dir = tmp_path / "artifacts" / "datasets" / "phase4"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "train.jsonl").write_text(
        "\n".join(
            [
                json.dumps(_dataset_example_dict(sample_id="train:1", split="train")),
                json.dumps(_dataset_example_dict(sample_id="train:2", split="train")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "validation.jsonl").write_text(
        json.dumps(_dataset_example_dict(sample_id="validation:1", split="validation")) + "\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "python" / "configs" / "phase5_proposer_v5.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "seed": 3,
                "output_dir": "artifacts/phase5/test-run-v5",
                "data": {
                    "dataset_path": "artifacts/datasets/phase4",
                    "train_split": "train",
                    "validation_split": "validation",
                },
                "model": {
                    "architecture": "factorized_v5",
                    "hidden_dim": 32,
                    "hidden_layers": 1,
                    "dropout": 0.0,
                },
                "optimization": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "legality_loss_weight": 1.0,
                    "policy_loss_weight": 1.0,
                },
                "evaluation": {"legality_threshold": 0.5},
                "export": {
                    "bundle_dir": "models/proposer/test-v5",
                    "checkpoint_name": "checkpoint.pt",
                    "exported_program_name": "proposer.pt2",
                    "metadata_name": "metadata.json",
                },
            }
        ),
        encoding="utf-8",
    )

    run = train_proposer(load_proposer_train_config(config_path), repo_root=tmp_path)

    assert run.best_epoch == 1
    assert run.model_parameter_count > 0
    assert (tmp_path / "models/proposer/test-v5/checkpoint.pt").exists()
    assert (tmp_path / "models/proposer/test-v5/proposer.pt2").exists()


def test_train_proposer_supports_factorized_v6(tmp_path: Path) -> None:
    pytest.importorskip("torch")

    dataset_dir = tmp_path / "artifacts" / "datasets" / "phase4"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "train.jsonl").write_text(
        "\n".join(
            [
                json.dumps(_dataset_example_dict(sample_id="train:1", split="train")),
                json.dumps(_dataset_example_dict(sample_id="train:2", split="train")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "validation.jsonl").write_text(
        json.dumps(_dataset_example_dict(sample_id="validation:1", split="validation")) + "\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "python" / "configs" / "phase5_proposer_v6.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "seed": 3,
                "output_dir": "artifacts/phase5/test-run-v6",
                "data": {
                    "dataset_path": "artifacts/datasets/phase4",
                    "train_split": "train",
                    "validation_split": "validation",
                },
                "model": {
                    "architecture": "factorized_v6",
                    "hidden_dim": 32,
                    "hidden_layers": 1,
                    "dropout": 0.0,
                },
                "optimization": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "legality_loss_weight": 1.0,
                    "policy_loss_weight": 1.0,
                },
                "evaluation": {"legality_threshold": 0.5},
                "export": {
                    "bundle_dir": "models/proposer/test-v6",
                    "checkpoint_name": "checkpoint.pt",
                    "exported_program_name": "proposer.pt2",
                    "metadata_name": "metadata.json",
                },
            }
        ),
        encoding="utf-8",
    )

    run = train_proposer(load_proposer_train_config(config_path), repo_root=tmp_path)

    assert run.best_epoch == 1
    assert run.model_parameter_count > 0
    assert (tmp_path / "models/proposer/test-v6/checkpoint.pt").exists()
    assert (tmp_path / "models/proposer/test-v6/proposer.pt2").exists()


def test_train_proposer_supports_relational_v1(tmp_path: Path) -> None:
    pytest.importorskip("torch")

    dataset_dir = tmp_path / "artifacts" / "datasets" / "phase4"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "train.jsonl").write_text(
        "\n".join(
            [
                json.dumps(_dataset_example_dict(sample_id="train:1", split="train")),
                json.dumps(_dataset_example_dict(sample_id="train:2", split="train")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "validation.jsonl").write_text(
        json.dumps(_dataset_example_dict(sample_id="validation:1", split="validation")) + "\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "python" / "configs" / "phase5_proposer_relational_v1.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "seed": 3,
                "output_dir": "artifacts/phase5/test-run-relational-v1",
                "data": {
                    "dataset_path": "artifacts/datasets/phase4",
                    "train_split": "train",
                    "validation_split": "validation",
                },
                "model": {
                    "architecture": "relational_v1",
                    "hidden_dim": 32,
                    "hidden_layers": 1,
                    "dropout": 0.0,
                },
                "optimization": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "legality_loss_weight": 1.0,
                    "policy_loss_weight": 1.0,
                },
                "evaluation": {"legality_threshold": 0.5},
                "export": {
                    "bundle_dir": "models/proposer/test-relational-v1",
                    "checkpoint_name": "checkpoint.pt",
                    "exported_program_name": "proposer.pt2",
                    "metadata_name": "metadata.json",
                },
            }
        ),
        encoding="utf-8",
    )

    run = train_proposer(load_proposer_train_config(config_path), repo_root=tmp_path)

    assert run.best_epoch == 1
    assert run.model_parameter_count > 0
    assert (tmp_path / "models/proposer/test-relational-v1/checkpoint.pt").exists()
    assert (tmp_path / "models/proposer/test-relational-v1/proposer.pt2").exists()


def test_train_proposer_supports_symbolic_v1_without_export(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("chess")

    dataset_dir = tmp_path / "artifacts" / "datasets" / "phase4"
    dataset_dir.mkdir(parents=True)
    dataset = SimpleNamespace(
        examples=[
            DatasetExample.from_dict(_dataset_example_dict(sample_id="train:1", split="train")),
            DatasetExample.from_dict(_dataset_example_dict(sample_id="train:2", split="train")),
            DatasetExample.from_dict(
                _dataset_example_dict(sample_id="validation:1", split="validation")
            ),
        ],
        summary={"ok": True},
    )
    write_dataset_artifacts(dataset_dir, dataset, write_proposer_artifacts=True)
    materialize_symbolic_proposer_artifacts(dataset_dir)

    config_path = tmp_path / "python" / "configs" / "phase5_proposer_symbolic_v1.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "seed": 3,
                "output_dir": "artifacts/phase5/test-run-symbolic-v1",
                "data": {
                    "dataset_path": "artifacts/datasets/phase4",
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
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "legality_loss_weight": 1.0,
                    "policy_loss_weight": 1.0,
                },
                "evaluation": {"legality_threshold": 0.5},
                "export": {
                    "enabled": False,
                    "bundle_dir": "models/proposer/test-symbolic-v1",
                    "checkpoint_name": "checkpoint.pt",
                    "exported_program_name": "proposer.pt2",
                    "metadata_name": "metadata.json",
                },
            }
        ),
        encoding="utf-8",
    )

    run = train_proposer(load_proposer_train_config(config_path), repo_root=tmp_path)

    assert run.best_epoch == 1
    assert run.model_parameter_count > 0
    assert (tmp_path / "models/proposer/test-symbolic-v1/checkpoint.pt").exists()
    assert "exported_program" not in run.export_paths


def _dataset_example_dict(
    *, sample_id: str = "sample:1", split: str = "train"
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "split": split,
        "source": "synthetic",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "side_to_move": "w",
        "selected_move_uci": "e2e4",
        "selected_action_encoding": [12, 28, 0],
        "next_fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "legal_moves": [
            "b1a3",
            "b1c3",
            "g1f3",
            "g1h3",
            "a2a3",
            "a2a4",
            "b2b3",
            "b2b4",
            "c2c3",
            "c2c4",
            "d2d3",
            "d2d4",
            "e2e3",
            "e2e4",
            "f2f3",
            "f2f4",
            "g2g3",
            "g2g4",
            "h2h3",
            "h2h4",
        ],
        "legal_action_encodings": [
            [1, 16, 0],
            [1, 18, 0],
            [6, 21, 0],
            [6, 23, 0],
            [8, 16, 0],
            [8, 24, 0],
            [9, 17, 0],
            [9, 25, 0],
            [10, 18, 0],
            [10, 26, 0],
            [11, 19, 0],
            [11, 27, 0],
            [12, 20, 0],
            [12, 28, 0],
            [13, 21, 0],
            [13, 29, 0],
            [14, 22, 0],
            [14, 30, 0],
            [15, 23, 0],
            [15, 31, 0],
        ],
        "position_encoding": {
            "piece_tokens": [
                [0, 0, 3],
                [1, 0, 1],
                [2, 0, 2],
                [3, 0, 4],
                [4, 0, 5],
                [5, 0, 2],
                [6, 0, 1],
                [7, 0, 3],
                [8, 0, 0],
                [9, 0, 0],
                [10, 0, 0],
                [11, 0, 0],
                [12, 0, 0],
                [13, 0, 0],
                [14, 0, 0],
                [15, 0, 0],
                [48, 1, 0],
                [49, 1, 0],
                [50, 1, 0],
                [51, 1, 0],
                [52, 1, 0],
                [53, 1, 0],
                [54, 1, 0],
                [55, 1, 0],
                [56, 1, 3],
                [57, 1, 1],
                [58, 1, 2],
                [59, 1, 4],
                [60, 1, 5],
                [61, 1, 2],
                [62, 1, 1],
                [63, 1, 3],
            ],
            "square_tokens": [
                [0, 4],
                [1, 2],
                [2, 3],
                [3, 5],
                [4, 6],
                [5, 3],
                [6, 2],
                [7, 4],
                [8, 1],
                [9, 1],
                [10, 1],
                [11, 1],
                [12, 1],
                [13, 1],
                [14, 1],
                [15, 1],
                [16, 0],
                [17, 0],
                [18, 0],
                [19, 0],
                [20, 0],
                [21, 0],
                [22, 0],
                [23, 0],
                [24, 0],
                [25, 0],
                [26, 0],
                [27, 0],
                [28, 0],
                [29, 0],
                [30, 0],
                [31, 0],
                [32, 0],
                [33, 0],
                [34, 0],
                [35, 0],
                [36, 0],
                [37, 0],
                [38, 0],
                [39, 0],
                [40, 0],
                [41, 0],
                [42, 0],
                [43, 0],
                [44, 0],
                [45, 0],
                [46, 0],
                [47, 0],
                [48, 7],
                [49, 7],
                [50, 7],
                [51, 7],
                [52, 7],
                [53, 7],
                [54, 7],
                [55, 7],
                [56, 10],
                [57, 8],
                [58, 9],
                [59, 11],
                [60, 12],
                [61, 9],
                [62, 8],
                [63, 10],
            ],
            "rule_token": [0, 15, 64, 0, 1, 1],
        },
        "wdl_target": {"win": 1, "draw": 0, "loss": 0},
        "annotations": {
            "in_check": False,
            "is_checkmate": False,
            "is_stalemate": False,
            "has_legal_en_passant": False,
            "has_legal_castle": False,
            "has_legal_promotion": False,
            "is_low_material_endgame": False,
            "legal_move_count": 20,
            "piece_count": 32,
            "selected_move_is_capture": False,
            "selected_move_is_promotion": False,
            "selected_move_is_castle": False,
            "selected_move_is_en_passant": False,
            "selected_move_gives_check": False,
        },
        "result": "1-0",
        "metadata": {},
    }
