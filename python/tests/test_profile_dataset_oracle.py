"""Tests for dataset oracle profiling helper."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from train.datasets.schema import RawPositionRecord


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "profile_dataset_oracle.py"
_SPEC = importlib.util.spec_from_file_location("profile_dataset_oracle", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_select_records = _MODULE._select_records
_top_phases = _MODULE._top_phases
main = _MODULE.main


def test_select_records_cycles_templates_for_profile_expansion() -> None:
    records = [
        RawPositionRecord(sample_id="a", fen="fen-a", source="src"),
        RawPositionRecord(sample_id="b", fen="fen-b", source="src"),
    ]

    selected = _select_records(records, target_count=5)

    assert len(selected) == 5
    assert selected[2].sample_id == "a:profile:2"
    assert selected[3].sample_id == "b:profile:3"


def test_top_phases_orders_by_share() -> None:
    profile = {
        "phases": [
            ["json_parse", {"share_of_measured": 0.1}],
            ["legal_generation", {"share_of_measured": 0.5}],
            ["annotations", {"share_of_measured": 0.2}],
        ]
    }

    top = _top_phases(profile, limit=2)

    assert [phase["name"] for phase in top] == ["legal_generation", "annotations"]


def test_main_writes_profile_artifact(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.jsonl"
    profile_path = tmp_path / "fake_profile.py"
    artifact_path = tmp_path / "nested" / "profile.json"
    input_path.write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
                "source": "synthetic",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    profile_path.write_text(
        "import json, sys\n"
        "payload = sys.stdin.read().strip().splitlines()\n"
        "json.dump({'records': len(payload), 'phases': "
        "[['legal_generation', {'seconds': 1.0, 'milliseconds_per_record': 10.0, "
        "'share_of_measured': 0.5}], "
        "['json_serialize', {'seconds': 0.2, 'milliseconds_per_record': 2.0, "
        "'share_of_measured': 0.1}]]}, sys.stdout)\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--input",
            str(input_path),
            "--source-format",
            "jsonl",
            "--records",
            "4",
            "--artifact-out",
            str(artifact_path),
            "--profile-command",
            sys.executable,
            str(profile_path),
        ]
    )

    assert exit_code == 0
    rendered = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert rendered["record_count"] == 4
    assert rendered["profile"]["records"] == 4
    assert rendered["top_phases"][0]["name"] == "legal_generation"
