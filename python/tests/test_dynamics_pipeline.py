"""Regression tests for the Phase-6 latent dynamics pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from train.config import load_dynamics_train_config
from train.datasets.artifacts import (
    DynamicsTrainingExample,
    POSITION_FEATURE_SIZE,
    SYMBOLIC_PROPOSER_CANDIDATE_FEATURE_SIZE,
    dynamics_artifact_name,
    load_dynamics_examples,
)
from train.models.dynamics import LatentDynamicsModel
from train.trainers import dynamics as dynamics_trainer
from train.trainers import evaluate_dynamics_checkpoint, train_dynamics


def test_load_dynamics_train_config_accepts_phase6_schema(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 7,
                "output_dir": "artifacts/phase6/test",
                "data": {
                    "dataset_path": "artifacts/datasets/test",
                    "train_split": "train",
                    "validation_split": "validation",
                },
                "model": {
                    "architecture": "mlp_v1",
                    "latent_dim": 64,
                    "hidden_dim": 128,
                    "hidden_layers": 2,
                    "action_embedding_dim": 32,
                    "dropout": 0.1,
                },
                "optimization": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 1e-3,
                    "weight_decay": 0.0,
                    "reconstruction_loss_weight": 1.0,
                    "piece_loss_weight": 1.0,
                    "square_loss_weight": 1.0,
                    "rule_loss_weight": 1.0,
                },
                "evaluation": {"drift_horizon": 2},
                "runtime": {"torch_threads": 0, "dataloader_workers": 0},
                "export": {
                    "bundle_dir": "models/dynamics/test",
                    "checkpoint_name": "checkpoint.pt",
                    "exported_program_name": "dynamics.pt2",
                    "metadata_name": "metadata.json",
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_dynamics_train_config(config_path)

    assert config.model.latent_dim == 64
    assert config.model.architecture == "mlp_v1"
    assert config.evaluation.drift_horizon == 2


def test_load_dynamics_train_config_accepts_explicit_drift_dataset(tmp_path: Path) -> None:
    config_path = tmp_path / "config_drift.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 7,
                "output_dir": "artifacts/phase6/test-drift",
                "data": {
                    "dataset_path": "artifacts/datasets/test",
                    "train_split": "train",
                    "validation_split": "validation",
                },
                "model": {
                    "architecture": "structured_v2",
                    "latent_dim": 64,
                    "hidden_dim": 128,
                    "hidden_layers": 2,
                    "action_embedding_dim": 32,
                    "dropout": 0.1,
                },
                "optimization": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 1e-3,
                    "weight_decay": 0.0,
                    "reconstruction_loss_weight": 1.0,
                    "piece_loss_weight": 1.0,
                    "square_loss_weight": 1.0,
                    "rule_loss_weight": 2.0,
                },
                "evaluation": {
                    "drift_horizon": 2,
                    "drift_dataset_path": "artifacts/datasets/drift",
                    "drift_split": "test"
                },
                "runtime": {"torch_threads": 0, "dataloader_workers": 0},
                "export": {
                    "bundle_dir": "models/dynamics/test-drift",
                    "checkpoint_name": "checkpoint.pt",
                    "exported_program_name": "dynamics.pt2",
                    "metadata_name": "metadata.json",
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_dynamics_train_config(config_path)

    assert config.evaluation.drift_dataset_path == "artifacts/datasets/drift"
    assert config.evaluation.drift_split == "test"


def test_load_dynamics_train_config_accepts_latent_consistency_weight(tmp_path: Path) -> None:
    config_path = tmp_path / "config_latent.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 7,
                "output_dir": "artifacts/phase6/test-latent",
                "data": {
                    "dataset_path": "artifacts/datasets/test",
                    "train_split": "train",
                    "validation_split": "validation",
                },
                "model": {
                    "architecture": "structured_v2",
                    "latent_dim": 64,
                    "hidden_dim": 128,
                    "hidden_layers": 2,
                    "action_embedding_dim": 32,
                    "dropout": 0.1,
                },
                "optimization": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 1e-3,
                    "weight_decay": 0.0,
                    "reconstruction_loss_weight": 1.0,
                    "piece_loss_weight": 1.0,
                    "square_loss_weight": 1.0,
                    "rule_loss_weight": 2.0,
                    "latent_consistency_loss_weight": 0.25
                },
                "evaluation": {"drift_horizon": 2},
                "runtime": {"torch_threads": 0, "dataloader_workers": 0},
                "export": {
                    "bundle_dir": "models/dynamics/test-latent",
                    "checkpoint_name": "checkpoint.pt",
                    "exported_program_name": "dynamics.pt2",
                    "metadata_name": "metadata.json",
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_dynamics_train_config(config_path)

    assert config.optimization.latent_consistency_loss_weight == 0.25


def test_load_dynamics_train_config_accepts_edit_v1(tmp_path: Path) -> None:
    config_path = tmp_path / "config_edit.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 7,
                "output_dir": "artifacts/phase6/test-edit",
                "data": {
                    "dataset_path": "artifacts/datasets/test",
                    "train_split": "train",
                    "validation_split": "validation",
                },
                "model": {
                    "architecture": "edit_v1",
                    "latent_dim": 64,
                    "hidden_dim": 128,
                    "hidden_layers": 2,
                    "action_embedding_dim": 32,
                    "dropout": 0.1,
                },
                "optimization": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 1e-3,
                    "weight_decay": 0.0,
                    "reconstruction_loss_weight": 1.0,
                    "piece_loss_weight": 1.0,
                    "square_loss_weight": 1.0,
                    "rule_loss_weight": 2.0,
                    "delta_loss_weight": 0.1,
                    "latent_consistency_loss_weight": 0.0
                },
                "evaluation": {"drift_horizon": 2},
                "runtime": {"torch_threads": 0, "dataloader_workers": 0},
                "export": {
                    "bundle_dir": "models/dynamics/test-edit",
                    "checkpoint_name": "checkpoint.pt",
                    "exported_program_name": "dynamics.pt2",
                    "metadata_name": "metadata.json",
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_dynamics_train_config(config_path)

    assert config.model.architecture == "edit_v1"
    assert config.optimization.delta_loss_weight == 0.1


def test_load_dynamics_train_config_accepts_structured_v3(tmp_path: Path) -> None:
    config_path = tmp_path / "config_structured_v3.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 7,
                "output_dir": "artifacts/phase6/test-structured-v3",
                "data": {
                    "dataset_path": "artifacts/datasets/test",
                    "train_split": "train",
                    "validation_split": "validation",
                },
                "model": {
                    "architecture": "structured_v3",
                    "latent_dim": 64,
                    "hidden_dim": 128,
                    "hidden_layers": 2,
                    "action_embedding_dim": 32,
                    "dropout": 0.1,
                },
                "optimization": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 1e-3,
                    "weight_decay": 0.0,
                    "reconstruction_loss_weight": 1.0,
                    "piece_loss_weight": 1.0,
                    "square_loss_weight": 1.0,
                    "rule_loss_weight": 2.0,
                    "delta_loss_weight": 0.02,
                    "latent_consistency_loss_weight": 0.1
                },
                "evaluation": {"drift_horizon": 2},
                "runtime": {"torch_threads": 0, "dataloader_workers": 0},
                "export": {
                    "bundle_dir": "models/dynamics/test-structured-v3",
                    "checkpoint_name": "checkpoint.pt",
                    "exported_program_name": "dynamics.pt2",
                    "metadata_name": "metadata.json",
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_dynamics_train_config(config_path)

    assert config.model.architecture == "structured_v3"
    assert config.optimization.delta_loss_weight == 0.02


def test_load_dynamics_train_config_accepts_structured_v4(tmp_path: Path) -> None:
    config_path = tmp_path / "config_structured_v4.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 7,
                "output_dir": "artifacts/phase6/test-structured-v4",
                "data": {
                    "dataset_path": "artifacts/datasets/test",
                    "train_split": "train",
                    "validation_split": "validation",
                },
                "model": {
                    "architecture": "structured_v4",
                    "latent_dim": 64,
                    "hidden_dim": 128,
                    "hidden_layers": 2,
                    "action_embedding_dim": 32,
                    "dropout": 0.1,
                },
                "optimization": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 1e-3,
                    "weight_decay": 0.0,
                    "reconstruction_loss_weight": 1.0,
                    "piece_loss_weight": 1.0,
                    "square_loss_weight": 1.0,
                    "rule_loss_weight": 2.0,
                    "delta_loss_weight": 0.0,
                    "latent_consistency_loss_weight": 0.1,
                    "drift_supervision_loss_weight": 0.05,
                    "drift_supervision_horizon": 2
                },
                "evaluation": {"drift_horizon": 2},
                "runtime": {"torch_threads": 0, "dataloader_workers": 0},
                "export": {
                    "bundle_dir": "models/dynamics/test-structured-v4",
                    "checkpoint_name": "checkpoint.pt",
                    "exported_program_name": "dynamics.pt2",
                    "metadata_name": "metadata.json",
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_dynamics_train_config(config_path)

    assert config.model.architecture == "structured_v4"
    assert config.optimization.drift_supervision_loss_weight == 0.05


def test_load_dynamics_train_config_accepts_structured_v5(tmp_path: Path) -> None:
    config_path = tmp_path / "config_structured_v5.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 7,
                "output_dir": "artifacts/phase6/test-structured-v5",
                "data": {
                    "dataset_path": "artifacts/datasets/test",
                    "train_split": "train",
                    "validation_split": "validation",
                },
                "model": {
                    "architecture": "structured_v5",
                    "latent_dim": 64,
                    "hidden_dim": 128,
                    "hidden_layers": 2,
                    "action_embedding_dim": 32,
                    "dropout": 0.1,
                },
                "optimization": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 1e-3,
                    "weight_decay": 0.0,
                    "reconstruction_loss_weight": 1.0,
                    "piece_loss_weight": 1.0,
                    "square_loss_weight": 1.0,
                    "rule_loss_weight": 2.0,
                    "latent_consistency_loss_weight": 0.1
                },
                "evaluation": {"drift_horizon": 2},
                "runtime": {"torch_threads": 0, "dataloader_workers": 0},
                "export": {
                    "bundle_dir": "models/dynamics/test-structured-v5",
                    "checkpoint_name": "checkpoint.pt",
                    "exported_program_name": "dynamics.pt2",
                    "metadata_name": "metadata.json",
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_dynamics_train_config(config_path)

    assert config.model.architecture == "structured_v5"


def test_load_dynamics_examples_builds_from_full_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "train.jsonl").write_text(
        _dataset_jsonl(
            [
                _dataset_example_payload("sample-a", "train", next_fen="fen-a", ply=1),
                _dataset_example_payload("sample-b", "train", next_fen="fen-b", ply=2),
            ]
        ),
        encoding="utf-8",
    )

    def fake_label_records(records, *, repo_root=None, command=None):  # type: ignore[no-untyped-def]
        assert repo_root == tmp_path
        return [
            {"position_encoding": _position_encoding_payload(offset=index + 10)}
            for index, _ in enumerate(records)
        ]

    monkeypatch.setattr(
        "train.datasets.artifacts.label_records_with_oracle",
        fake_label_records,
    )

    examples = load_dynamics_examples(dataset_dir, "train", repo_root=tmp_path)

    assert len(examples) == 2
    assert examples[0].trajectory_id == "sample:1"
    assert examples[1].ply_index == 2
    assert examples[0].action_features is not None
    assert len(examples[0].action_features or []) == SYMBOLIC_PROPOSER_CANDIDATE_FEATURE_SIZE
    assert len(examples[0].next_feature_vector) == POSITION_FEATURE_SIZE


def test_train_and_evaluate_dynamics_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    dataset_dir = tmp_path / "artifacts" / "datasets" / "phase6_fixture"
    dataset_dir.mkdir(parents=True)
    _write_dynamics_artifact(
        dataset_dir / dynamics_artifact_name("train"),
        [
            _dynamics_example_payload("train-1", "train", action_index=3, offset=0, ply=1),
            _dynamics_example_payload("train-2", "train", action_index=4, offset=1, ply=2),
            _dynamics_example_payload("train-3", "train", action_index=5, offset=2, ply=10),
            _dynamics_example_payload("train-4", "train", action_index=6, offset=3, ply=11),
        ],
    )
    _write_dynamics_artifact(
        dataset_dir / dynamics_artifact_name("validation"),
        [
            _dynamics_example_payload("val-1", "validation", action_index=7, offset=4, ply=1),
            _dynamics_example_payload("val-2", "validation", action_index=8, offset=5, ply=2),
        ],
    )
    _write_dynamics_artifact(
        dataset_dir / dynamics_artifact_name("test"),
        [
            _dynamics_example_payload("test-1", "test", action_index=9, offset=6, ply=1),
            _dynamics_example_payload("test-2", "test", action_index=10, offset=7, ply=2),
        ],
    )

    config_path = tmp_path / "phase6.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 3,
                "output_dir": str((tmp_path / "artifacts" / "phase6" / "run").relative_to(tmp_path)),
                "data": {
                    "dataset_path": str(dataset_dir.relative_to(tmp_path)),
                    "train_split": "train",
                    "validation_split": "validation",
                },
                "model": {
                    "architecture": "mlp_v1",
                    "latent_dim": 32,
                    "hidden_dim": 64,
                    "hidden_layers": 2,
                    "action_embedding_dim": 16,
                    "dropout": 0.0,
                },
                "optimization": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 1e-3,
                    "weight_decay": 0.0,
                    "reconstruction_loss_weight": 1.0,
                    "piece_loss_weight": 1.0,
                    "square_loss_weight": 1.0,
                    "rule_loss_weight": 1.0,
                },
                "evaluation": {"drift_horizon": 2},
                "runtime": {"torch_threads": 1, "dataloader_workers": 0},
                "export": {
                    "bundle_dir": str((tmp_path / "models" / "dynamics" / "test").relative_to(tmp_path)),
                    "checkpoint_name": "checkpoint.pt",
                    "exported_program_name": "dynamics.pt2",
                    "metadata_name": "metadata.json",
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_dynamics_train_config(config_path)
    result = train_dynamics(config, repo_root=tmp_path)

    assert result.best_epoch == 1
    assert Path(result.summary_path).is_file()
    assert Path(result.export_paths["metadata"]).is_file()

    metrics = evaluate_dynamics_checkpoint(
        Path(result.export_paths["checkpoint"]),
        dataset_path=dataset_dir,
        split="test",
        drift_horizon=2,
        repo_root=tmp_path,
    )

    assert metrics.total_examples == 2
    assert metrics.drift_examples == 1


def test_train_and_evaluate_structured_v5_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    dataset_dir = tmp_path / "artifacts" / "datasets" / "phase6_fixture_v5"
    dataset_dir.mkdir(parents=True)
    _write_dynamics_artifact(
        dataset_dir / dynamics_artifact_name("train"),
        [
            _dynamics_example_payload("train-1", "train", action_index=3, offset=0, ply=1),
            _dynamics_example_payload("train-2", "train", action_index=4, offset=1, ply=2),
            _dynamics_example_payload("train-3", "train", action_index=5, offset=2, ply=10),
            _dynamics_example_payload("train-4", "train", action_index=6, offset=3, ply=11),
        ],
    )
    _write_dynamics_artifact(
        dataset_dir / dynamics_artifact_name("validation"),
        [
            _dynamics_example_payload("val-1", "validation", action_index=7, offset=4, ply=1),
            _dynamics_example_payload("val-2", "validation", action_index=8, offset=5, ply=2),
        ],
    )
    _write_dynamics_artifact(
        dataset_dir / dynamics_artifact_name("test"),
        [
            _dynamics_example_payload("test-1", "test", action_index=9, offset=6, ply=1),
            _dynamics_example_payload("test-2", "test", action_index=10, offset=7, ply=2),
        ],
    )

    config_path = tmp_path / "phase6_v5.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 3,
                "output_dir": str((tmp_path / "artifacts" / "phase6" / "run-v5").relative_to(tmp_path)),
                "data": {
                    "dataset_path": str(dataset_dir.relative_to(tmp_path)),
                    "train_split": "train",
                    "validation_split": "validation",
                },
                "model": {
                    "architecture": "structured_v5",
                    "latent_dim": 32,
                    "hidden_dim": 64,
                    "hidden_layers": 2,
                    "action_embedding_dim": 16,
                    "dropout": 0.0,
                },
                "optimization": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 1e-3,
                    "weight_decay": 0.0,
                    "reconstruction_loss_weight": 1.0,
                    "piece_loss_weight": 1.0,
                    "square_loss_weight": 1.0,
                    "rule_loss_weight": 1.0,
                    "latent_consistency_loss_weight": 0.1
                },
                "evaluation": {"drift_horizon": 2},
                "runtime": {"torch_threads": 1, "dataloader_workers": 0},
                "export": {
                    "bundle_dir": str((tmp_path / "models" / "dynamics" / "test-v5").relative_to(tmp_path)),
                    "checkpoint_name": "checkpoint.pt",
                    "exported_program_name": "dynamics.pt2",
                    "metadata_name": "metadata.json",
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_dynamics_train_config(config_path)
    result = train_dynamics(config, repo_root=tmp_path)
    metadata_payload = json.loads(Path(result.export_paths["metadata"]).read_text(encoding="utf-8"))

    metrics = evaluate_dynamics_checkpoint(
        Path(result.export_paths["checkpoint"]),
        dataset_path=dataset_dir,
        split="test",
        drift_horizon=2,
        repo_root=tmp_path,
    )

    assert metadata_payload["input"]["action"]["symbolic"]["feature_dim"] == 18
    assert metrics.total_examples == 2
    assert metrics.drift_examples == 1


def test_latent_dynamics_step_from_transition_input_matches_step() -> None:
    torch = pytest.importorskip("torch")
    model = LatentDynamicsModel(
        architecture="structured_v2",
        latent_dim=16,
        hidden_dim=32,
        hidden_layers=2,
        action_embedding_dim=8,
        dropout=0.0,
    )
    features = torch.tensor([_feature_vector(0), _feature_vector(1)], dtype=torch.float32)
    action_indices = torch.tensor([3, 7], dtype=torch.long)

    latent = model.encode(features)
    action_embedding = model.action_embedding(action_indices)
    transition_input = torch.cat((latent, action_embedding), dim=1)

    via_step = model.step(latent, action_indices)
    via_transition_input = model.step_from_transition_input(latent, transition_input)

    assert torch.allclose(via_step, via_transition_input)


def test_structured_v5_step_requires_symbolic_action_features() -> None:
    torch = pytest.importorskip("torch")
    model = LatentDynamicsModel(
        architecture="structured_v5",
        latent_dim=16,
        hidden_dim=32,
        hidden_layers=2,
        action_embedding_dim=8,
        dropout=0.0,
    )
    features = torch.tensor([_feature_vector(0), _feature_vector(1)], dtype=torch.float32)
    action_indices = torch.tensor([3, 7], dtype=torch.long)
    action_features = torch.tensor(
        [_action_feature_vector(3), _action_feature_vector(7)],
        dtype=torch.float32,
    )

    latent = model.encode(features)
    action_embedding = model.action_embedding(action_indices)
    via_step = model.step(latent, action_indices, action_features=action_features)
    via_action_embedding = model.step_from_action_embedding(
        latent,
        action_embedding,
        action_features=action_features,
    )
    transition_input = torch.cat(
        (
            latent,
            torch.cat((action_embedding, model.symbolic_action_projection(action_features)), dim=1),
        ),
        dim=1,
    )
    via_transition_input = model.step_from_transition_input(latent, transition_input)

    assert torch.allclose(via_step, via_action_embedding)
    assert torch.allclose(via_step, via_transition_input)

    with pytest.raises(ValueError, match="action_features"):
        model.step(latent, action_indices)


def test_run_drift_supervision_matches_manual_chain_average() -> None:
    torch = pytest.importorskip("torch")
    model = LatentDynamicsModel(
        architecture="structured_v4",
        latent_dim=16,
        hidden_dim=32,
        hidden_layers=2,
        action_embedding_dim=8,
        dropout=0.0,
    )
    chains = [
        [
            DynamicsTrainingExample(
                **_dynamics_example_payload("a-1", "train", action_index=3, offset=0, ply=1)
            ),
            DynamicsTrainingExample(
                **_dynamics_example_payload("a-2", "train", action_index=4, offset=1, ply=2)
            ),
        ],
        [
            DynamicsTrainingExample(
                **_dynamics_example_payload("b-1", "train", action_index=5, offset=10, ply=1)
            ),
            DynamicsTrainingExample(
                **_dynamics_example_payload("b-2", "train", action_index=6, offset=11, ply=2)
            ),
        ],
    ]

    result = dynamics_trainer._run_drift_supervision(  # type: ignore[attr-defined]
        model,
        chains,
        optimizer=None,
        training=False,
        weight=0.1,
    )

    manual_total = 0.0
    with torch.inference_mode():
        for chain in chains:
            latent = model.encode(torch.tensor([chain[0].feature_vector], dtype=torch.float32))
            for example in chain:
                latent = model.step(latent, torch.tensor([example.action_index], dtype=torch.long))
            predicted = model.decode(latent).next_features
            target = torch.tensor([chain[-1].next_feature_vector], dtype=torch.float32)
            manual_total += float(torch.nn.functional.mse_loss(predicted, target).item())

    assert result["count"] == len(chains)
    assert result["raw_loss"] == pytest.approx(manual_total / len(chains))


def _dataset_jsonl(records: list[dict[str, object]]) -> str:
    return "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n"


def _write_dynamics_artifact(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(_dataset_jsonl(records), encoding="utf-8")


def _dataset_example_payload(
    sample_id: str,
    split: str,
    *,
    next_fen: str,
    ply: int,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "split": split,
        "source": "fixture",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "side_to_move": "w",
        "selected_move_uci": "e2e4",
        "selected_action_encoding": [12, 28, 0],
        "next_fen": next_fen,
        "legal_moves": ["e2e4"],
        "legal_action_encodings": [[12, 28, 0]],
        "position_encoding": _position_encoding_payload(offset=ply),
        "wdl_target": None,
        "annotations": {
            "in_check": False,
            "is_checkmate": False,
            "is_stalemate": False,
            "has_legal_en_passant": False,
            "has_legal_castle": False,
            "has_legal_promotion": False,
            "is_low_material_endgame": False,
            "legal_move_count": 1,
            "piece_count": 32,
            "selected_move_is_capture": False,
            "selected_move_is_promotion": False,
            "selected_move_is_castle": False,
            "selected_move_is_en_passant": False,
            "selected_move_gives_check": False,
        },
        "result": None,
        "metadata": {"source_pgn": "sample", "game_index": 1, "ply": ply},
    }


def _dynamics_example_payload(
    sample_id: str,
    split: str,
    *,
    action_index: int,
    offset: int,
    ply: int,
) -> dict[str, object]:
    current = _feature_vector(offset)
    nxt = _feature_vector(offset + 1)
    return {
        "sample_id": sample_id,
        "split": split,
        "feature_vector": current,
        "action_index": action_index,
        "action_features": _action_feature_vector(action_index),
        "next_feature_vector": nxt,
        "is_capture": False,
        "is_promotion": False,
        "is_castle": False,
        "is_en_passant": False,
        "gives_check": False,
        "trajectory_id": f"{split}:1",
        "ply_index": ply,
    }


def _action_feature_vector(action_index: int) -> list[float]:
    return [
        float((action_index + offset) % 3 == 0)
        for offset in range(SYMBOLIC_PROPOSER_CANDIDATE_FEATURE_SIZE)
    ]


def _position_encoding_payload(*, offset: int) -> dict[str, object]:
    return {
        "piece_tokens": [[1, 0, offset % 64]],
        "square_tokens": [[square, (square + offset) % 13] for square in range(64)],
        "rule_token": [0, 15, -1, offset, 1, 0],
    }


def _feature_vector(offset: int) -> list[float]:
    position = _position_encoding_payload(offset=offset)
    flat: list[float] = [-1.0] * 96
    flat[0:3] = [1.0, 0.0, float(offset % 64)]
    for square_index, occupant_code in position["square_tokens"]:  # type: ignore[misc]
        flat.extend([float(square_index), float(occupant_code)])
    flat.extend(float(value) for value in position["rule_token"])  # type: ignore[arg-type]
    return flat
