"""Tests for proposer artifact materialization and benchmarking helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from train.datasets import materialize_proposer_artifacts, proposer_artifact_name


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_proposer_artifacts.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_proposer_artifacts", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
main = _MODULE.main


def test_materialize_proposer_artifacts_backfills_split_files(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    payload = _dataset_example_dict()
    (dataset_dir / "train.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    (dataset_dir / "validation.jsonl").write_text("", encoding="utf-8")
    (dataset_dir / "test.jsonl").write_text("", encoding="utf-8")

    written_counts = materialize_proposer_artifacts(dataset_dir)

    assert written_counts == {"test": 0, "train": 1, "validation": 0}
    proposer_path = dataset_dir / proposer_artifact_name("train")
    rendered = [
        json.loads(line)
        for line in proposer_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rendered) == 1
    assert rendered[0]["sample_id"] == payload["sample_id"]
    assert rendered[0]["selected_action_index"] is not None


def test_benchmark_main_reports_full_vs_lean_sizes(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    payload = _dataset_example_dict()
    for split in ("train", "validation", "test"):
        body = json.dumps({**payload, "sample_id": f"{split}:1", "split": split}) + "\n"
        (dataset_dir / f"{split}.jsonl").write_text(body, encoding="utf-8")

    artifact_out = tmp_path / "bench" / "artifact.json"
    exit_code = main(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--output-root",
            str(tmp_path / "out"),
            "--artifact-out",
            str(artifact_out),
            "--repeats",
            "1",
        ]
    )

    assert exit_code == 0
    rendered = json.loads(artifact_out.read_text(encoding="utf-8"))
    assert rendered["runtime"]["hostname"]
    assert rendered["full"]["train"]["example_count"] == 1
    assert rendered["lean"]["train"]["example_count"] == 1
    assert rendered["size_bytes"]["full"]["train"] > rendered["size_bytes"]["lean"]["train"]
    assert float(rendered["speedup"]["train"]) > 0.0


def test_benchmark_main_can_include_training_comparison(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    payload = _dataset_example_dict()
    for split in ("train", "validation", "test"):
        body = json.dumps({**payload, "sample_id": f"{split}:1", "split": split}) + "\n"
        (dataset_dir / f"{split}.jsonl").write_text(body, encoding="utf-8")

    config_path = tmp_path / "config.json"
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
                    "epochs": 10,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "legality_loss_weight": 1.0,
                    "policy_loss_weight": 1.0,
                },
                "evaluation": {"legality_threshold": 0.5},
                "runtime": {"torch_threads": 1, "dataloader_workers": 0},
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
    artifact_out = tmp_path / "bench" / "train-artifact.json"

    fake_run = SimpleNamespace(
        history=[
            {
                "train": {"examples_per_second": 100.0},
                "validation": {"examples_per_second": 80.0},
            }
        ],
        best_validation={"legal_set_f1": 0.1},
    )
    with patch.object(_MODULE, "train_proposer", return_value=fake_run):
        exit_code = main(
            [
                "--dataset-dir",
                str(dataset_dir),
                "--output-root",
                str(tmp_path / "out"),
                "--artifact-out",
                str(artifact_out),
                "--repeats",
                "1",
                "--train-config",
                str(config_path),
                "--train-epochs",
                "1",
            ]
        )

    assert exit_code == 0
    rendered = json.loads(artifact_out.read_text(encoding="utf-8"))
    assert rendered["training"]["full"]["train_examples_per_second"] == 100.0
    assert rendered["training"]["lean"]["validation_examples_per_second"] == 80.0


def _dataset_example_dict() -> dict[str, object]:
    return {
        "sample_id": "sample:1",
        "split": "train",
        "source": "synthetic",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "side_to_move": "w",
        "selected_move_uci": "e2e4",
        "selected_action_encoding": [12, 28, 0],
        "next_fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "legal_moves": ["e2e3", "e2e4"],
        "legal_action_encodings": [[12, 20, 0], [12, 28, 0]],
        "position_encoding": {
            "piece_tokens": [[4, 0, 5], [60, 1, 5]],
            "square_tokens": [[square, 0] for square in range(64)],
            "rule_token": [0, 0, 0, 0, 1, 0],
        },
        "wdl_target": None,
        "annotations": {
            "in_check": False,
            "is_checkmate": False,
            "is_stalemate": False,
            "has_legal_en_passant": False,
            "has_legal_castle": False,
            "has_legal_promotion": False,
            "is_low_material_endgame": True,
            "legal_move_count": 2,
            "piece_count": 2,
            "selected_move_is_capture": False,
            "selected_move_is_promotion": False,
            "selected_move_is_castle": False,
            "selected_move_is_en_passant": False,
            "selected_move_gives_check": False,
        },
        "result": None,
        "metadata": {},
    }
