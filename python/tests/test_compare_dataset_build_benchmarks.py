"""Tests for dataset-build benchmark comparison helper."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "compare_dataset_build_benchmarks.py"
)
_SPEC = importlib.util.spec_from_file_location("compare_dataset_build_benchmarks", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
main = _MODULE.main


def test_main_writes_rollup_and_notes(tmp_path: Path) -> None:
    left_10k = tmp_path / "left_10k.json"
    left_20k = tmp_path / "left_20k.json"
    right_10k = tmp_path / "right_10k.json"
    right_20k = tmp_path / "right_20k.json"
    out = tmp_path / "nested" / "compare.json"

    for path, record_count, auto_seconds, explicit_seconds, effective_batch_size in [
        (left_10k, 10240, 6.0, 5.5, 2560),
        (left_20k, 20480, 12.0, 10.0, 5120),
        (right_10k, 10240, 15.0, 15.2, 500),
        (right_20k, 20480, 31.0, 29.0, 500),
    ]:
        path.write_text(
            json.dumps(
                {
                    "input": f"/tmp/{path.name}",
                    "record_count": record_count,
                    "repeats": 2,
                    "runtime": {"hostname": path.stem},
                    "results": [
                        {
                            "name": "auto_w4",
                            "seconds": auto_seconds,
                            "oracle_schedule": {
                                "effective_batch_size": effective_batch_size,
                            },
                            "digests": {"dataset_jsonl": "same"},
                        },
                        {
                            "name": "w4_b500",
                            "seconds": explicit_seconds,
                            "oracle_schedule": {
                                "effective_batch_size": 500,
                            },
                            "digests": {"dataset_jsonl": "same"},
                        },
                    ],
                    "fastest": {"name": "w4_b500"},
                }
            )
            + "\n",
            encoding="utf-8",
        )

    exit_code = main(
        [
            "--series",
            f"local:{left_10k}:{left_20k}",
            "--series",
            f"raspberrypi:{right_10k}:{right_20k}",
            "--artifact-out",
            str(out),
        ]
    )

    assert exit_code == 0
    rendered = json.loads(out.read_text(encoding="utf-8"))
    assert rendered["10k"]["local"]["delta_seconds"] == 0.5
    assert rendered["20k"]["raspberrypi"]["delta_ratio"] == round(31.0 / 29.0, 3)
    assert rendered["10k"]["raspberrypi"]["runtime"]["hostname"] == "right_10k"
    assert any("same effective batch size" in note for note in rendered["notes"])
