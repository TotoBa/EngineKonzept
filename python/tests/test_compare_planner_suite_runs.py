"""Tests for planner-suite comparison helper."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_planner_suite_runs.py"
_SPEC = importlib.util.spec_from_file_location("compare_planner_suite_runs", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
main = _MODULE.main


def test_main_accepts_multiple_additional_planners(tmp_path: Path) -> None:
    root_only = tmp_path / "root_only.json"
    symbolic_reply = tmp_path / "symbolic_reply.json"
    learned_reply = tmp_path / "learned_reply.json"
    trained_planner = tmp_path / "trained_planner.json"
    reference_planner = tmp_path / "reference_planner.json"
    extra_planner = tmp_path / "extra_planner.json"
    out = tmp_path / "nested" / "compare.json"

    root_only.write_text(json.dumps({"aggregate": {"root_top1_accuracy": 0.1}}) + "\n", encoding="utf-8")
    symbolic_reply.write_text(
        json.dumps({"aggregate": {"root_top1_accuracy": 0.2}}) + "\n",
        encoding="utf-8",
    )
    learned_reply.write_text(
        json.dumps({"aggregate": {"root_top1_accuracy": 0.3}}) + "\n",
        encoding="utf-8",
    )
    trained_planner.write_text(
        json.dumps({"root_top1_accuracy": 0.8, "teacher_root_mean_reciprocal_rank": 0.88}) + "\n",
        encoding="utf-8",
    )
    reference_planner.write_text(
        json.dumps({"root_top1_accuracy": 0.75, "teacher_root_mean_reciprocal_rank": 0.84}) + "\n",
        encoding="utf-8",
    )
    extra_planner.write_text(
        json.dumps({"root_top1_accuracy": 0.7, "teacher_root_mean_reciprocal_rank": 0.82}) + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--root-only",
            str(root_only),
            "--symbolic-reply",
            str(symbolic_reply),
            "--learned-reply",
            str(learned_reply),
            "--trained-planner",
            str(trained_planner),
            "--trained-planner-name",
            "set_v2_expanded",
            "--reference-planner",
            str(reference_planner),
            "--reference-planner-name",
            "set_v2_baseline",
            "--additional-planner",
            f"set_v5_candidate={extra_planner}",
            "--output-path",
            str(out),
        ]
    )

    assert exit_code == 0
    rendered = json.loads(out.read_text(encoding="utf-8"))
    assert rendered["runs"]["set_v2_expanded"]["root_top1_accuracy"] == 0.8
    assert rendered["runs"]["set_v2_baseline"]["teacher_root_mean_reciprocal_rank"] == 0.84
    assert rendered["runs"]["set_v5_candidate"]["root_top1_accuracy"] == 0.7
