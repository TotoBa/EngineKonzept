"""Tests for the Phase-7 corpus-suite workflow builder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_phase7_workflow_suite.py"
)
_SPEC = importlib.util.spec_from_file_location("build_phase7_workflow_suite", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class _FakeBaseline:
    def to_dict(self) -> dict[str, float]:
        return {"reply_top1_accuracy": 0.5, "reply_top3_accuracy": 0.75}


def test_build_phase7_workflow_suite_writes_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_run_workflow_build(**kwargs):
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "example_count": kwargs["max_examples"],
            "output_dir": str(output_dir),
            "split": kwargs["split"],
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary

    monkeypatch.setattr(_MODULE, "_run_workflow_build", fake_run_workflow_build)
    monkeypatch.setattr(
        _MODULE,
        "evaluate_symbolic_opponent_baseline",
        lambda *_args, **_kwargs: _FakeBaseline(),
    )

    exit_code = _MODULE.main(
        [
            "--checkpoint",
            "models/proposer/stockfish_pgn_symbolic_v1_v1/checkpoint.pt",
            "--teacher-engine",
            "/usr/games/stockfish18",
            "--output-root",
            str(tmp_path / "suite"),
            "--tier",
            "pgn_10k",
        ]
    )

    assert exit_code == 0
    summary_path = tmp_path / "suite" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["train_paths"] == [
        str(tmp_path / "suite" / "pgn_10k_train_v1" / "opponent_head_train.jsonl")
    ]
    assert summary["validation_paths"] == [
        str(
            tmp_path
            / "suite"
            / "pgn_10k_validation_v1"
            / "opponent_head_validation.jsonl"
        )
    ]
    assert summary["verify_paths"] == [
        str(tmp_path / "suite" / "pgn_10k_verify_v1" / "opponent_head_test.jsonl")
    ]
    assert summary["tiers"]["pgn_10k"]["verify"]["symbolic_baseline"] == {
        "reply_top1_accuracy": 0.5,
        "reply_top3_accuracy": 0.75,
    }
