"""Tests for proposer-run comparison helper."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_proposer_runs.py"
_SPEC = importlib.util.spec_from_file_location("compare_proposer_runs", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
main = _MODULE.main


def test_main_writes_architecture_and_best_metrics(tmp_path: Path) -> None:
    summary_a = tmp_path / "summary_a.json"
    verify_a = tmp_path / "verify_a.json"
    summary_b = tmp_path / "summary_b.json"
    verify_b = tmp_path / "verify_b.json"
    out = tmp_path / "nested" / "compare.json"

    summary_a.write_text(
        json.dumps(
            {
                "config": {
                    "model": {
                        "architecture": "mlp_v1",
                        "hidden_dim": 128,
                        "hidden_layers": 2,
                    },
                    "data": {"dataset_path": "/tmp/dataset-a"},
                    "optimization": {
                        "batch_size": 128,
                        "learning_rate": 0.001,
                        "legality_loss_weight": 1.0,
                        "policy_loss_weight": 1.0,
                    },
                    "runtime": {"torch_threads": 4},
                },
                "best_validation": {
                    "legal_set_f1": 0.1,
                    "legal_set_precision": 0.2,
                    "legal_set_recall": 0.3,
                    "policy_loss": 5.0,
                    "policy_top1_accuracy": 0.04,
                    "examples_per_second": 100.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    verify_a.write_text(
        json.dumps(
            {
                "legal_set_f1": 0.11,
                "legal_set_precision": 0.21,
                "legal_set_recall": 0.31,
                "policy_loss": 4.0,
                "policy_top1_accuracy": 0.05,
                "examples_per_second": 120.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary_b.write_text(
        json.dumps(
            {
                "config": {
                    "model": {
                        "architecture": "multistream_v2",
                        "hidden_dim": 128,
                        "hidden_layers": 2,
                    },
                    "data": {"dataset_path": "/tmp/dataset-b"},
                    "optimization": {
                        "batch_size": 128,
                        "learning_rate": 0.001,
                        "legality_loss_weight": 1.0,
                        "policy_loss_weight": 1.0,
                    },
                    "runtime": {"torch_threads": 4},
                },
                "best_validation": {
                    "legal_set_f1": 0.2,
                    "legal_set_precision": 0.3,
                    "legal_set_recall": 0.4,
                    "policy_loss": 6.0,
                    "policy_top1_accuracy": 0.03,
                    "examples_per_second": 90.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    verify_b.write_text(
        json.dumps(
            {
                "legal_set_f1": 0.18,
                "legal_set_precision": 0.28,
                "legal_set_recall": 0.38,
                "policy_loss": 4.5,
                "policy_top1_accuracy": 0.02,
                "examples_per_second": 95.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--run",
            f"mlp:{summary_a}:{verify_a}",
            "--run",
            f"multistream:{summary_b}:{verify_b}",
            "--artifact-out",
            str(out),
        ]
    )

    assert exit_code == 0
    rendered = json.loads(out.read_text(encoding="utf-8"))
    assert rendered["runs"][0]["config"]["architecture"] == "mlp_v1"
    assert rendered["runs"][1]["config"]["architecture"] == "multistream_v2"
    assert rendered["best_by_metric"]["validation_legal_set_f1"] == "multistream"
    assert rendered["best_by_metric"]["verify_policy_top1_accuracy"] == "mlp"
