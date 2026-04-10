from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_phase10_lapv1_workflow.py"
)
_SPEC = importlib.util.spec_from_file_location("build_phase10_lapv1_workflow", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

_build_expected_chunk_records = _MODULE._build_expected_chunk_records
_normalize_chunk_range = _MODULE._normalize_chunk_range


def test_normalize_chunk_range_defaults_to_full_span() -> None:
    assert _normalize_chunk_range(total_chunks=7, chunk_start=None, chunk_end=None) == (1, 7)


def test_build_expected_chunk_records_can_slice_middle_range(tmp_path: Path) -> None:
    records = _build_expected_chunk_records(
        output_dir=tmp_path / "workflow",
        canonical_split="train",
        max_examples=10,
        chunk_size=2,
        chunk_start=2,
        chunk_end=4,
    )

    assert [record["index"] for record in records] == [2, 3, 4]
    assert [record["start_index"] for record in records] == [2, 4, 6]
    assert all(record["total_chunks"] == 5 for record in records)

