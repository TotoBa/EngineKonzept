"""Tests for the proposer-run comparison helper."""

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


def test_main_compares_validation_and_verify_metrics(tmp_path: Path) -> None:
    summary_a = tmp_path / "a-summary.json"
    verify_a = tmp_path / "a-verify.json"
    summary_b = tmp_path / "b-summary.json"
    verify_b = tmp_path / "b-verify.json"
    summary_a.write_text(json.dumps(_summary_payload(0.01, 0.02, 128, 1.0)), encoding="utf-8")
    verify_a.write_text(json.dumps(_verify_payload(0.02, 0.03)), encoding="utf-8")
    summary_b.write_text(json.dumps(_summary_payload(0.03, 0.01, 256, 2.0)), encoding="utf-8")
    verify_b.write_text(json.dumps(_verify_payload(0.01, 0.05)), encoding="utf-8")
    artifact_out = tmp_path / "compare.json"

    exit_code = main(
        [
            "--run",
            f"run-a:{summary_a}:{verify_a}",
            "--run",
            f"run-b:{summary_b}:{verify_b}",
            "--artifact-out",
            str(artifact_out),
        ]
    )

    assert exit_code == 0
    rendered = json.loads(artifact_out.read_text(encoding="utf-8"))
    assert rendered["best_by_metric"]["validation_legal_set_f1"] == "run-b"
    assert rendered["best_by_metric"]["validation_policy_top1_accuracy"] == "run-a"
    assert rendered["best_by_metric"]["verify_policy_top1_accuracy"] == "run-b"


def _summary_payload(
    validation_legal_set_f1: float,
    validation_policy_top1_accuracy: float,
    hidden_dim: int,
    policy_loss_weight: float,
) -> dict[str, object]:
    return {
        "best_validation": {
            "legal_set_f1": validation_legal_set_f1,
            "legal_set_precision": 0.5,
            "legal_set_recall": 0.01,
            "policy_loss": 7.0,
            "policy_top1_accuracy": validation_policy_top1_accuracy,
            "examples_per_second": 1000.0,
        },
        "config": {
            "data": {"dataset_path": "artifacts/datasets/example"},
            "model": {"hidden_dim": hidden_dim, "hidden_layers": 2},
            "optimization": {
                "batch_size": 128,
                "learning_rate": 0.001,
                "legality_loss_weight": 1.0,
                "policy_loss_weight": policy_loss_weight,
            },
            "runtime": {"torch_threads": 4},
        },
    }


def _verify_payload(legal_set_f1: float, policy_top1_accuracy: float) -> dict[str, object]:
    return {
        "legal_set_f1": legal_set_f1,
        "legal_set_precision": 0.5,
        "legal_set_recall": 0.01,
        "policy_loss": 7.0,
        "policy_top1_accuracy": policy_top1_accuracy,
        "examples_per_second": 2000.0,
    }
