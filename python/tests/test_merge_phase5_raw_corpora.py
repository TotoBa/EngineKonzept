from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "merge_phase5_raw_corpora.py"
)
_SPEC = importlib.util.spec_from_file_location("merge_phase5_raw_corpora", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

RawCorpusSourceSpec = _MODULE.RawCorpusSourceSpec
merge_phase5_raw_corpora = _MODULE.merge_phase5_raw_corpora


def test_merge_phase5_raw_corpora_deduplicates_and_keeps_verify_disjoint(tmp_path: Path) -> None:
    source_a = tmp_path / "source_a"
    source_b = tmp_path / "source_b"
    source_a.mkdir()
    source_b.mkdir()

    _write_jsonl(
        source_a / "train_raw.jsonl",
        [
            {
                "sample_id": "a:1",
                "fen": "fen_shared_train",
                "source": "source-a",
                "selected_move_uci": "e2e4",
                "result": "1-0",
                "metadata": {"tag": "a"},
            },
            {
                "sample_id": "a:2",
                "fen": "fen_only_a",
                "source": "source-a",
                "selected_move_uci": "d2d4",
                "result": "1/2-1/2",
                "metadata": {"tag": "a"},
            },
        ],
    )
    _write_jsonl(
        source_a / "verify_raw.jsonl",
        [
            {
                "sample_id": "a:v1",
                "fen": "fen_verify_a",
                "source": "source-a",
                "selected_move_uci": "g1f3",
                "result": "1-0",
                "metadata": {"tag": "a"},
            }
        ],
    )

    _write_jsonl(
        source_b / "train_raw.jsonl",
        [
            {
                "sample_id": "b:1",
                "fen": "fen_shared_train",
                "source": "source-b",
                "selected_move_uci": "c2c4",
                "result": "0-1",
                "metadata": {"tag": "b"},
            },
            {
                "sample_id": "b:2",
                "fen": "fen_train_hits_verify",
                "source": "source-b",
                "selected_move_uci": "b1c3",
                "result": "1-0",
                "metadata": {"tag": "b"},
            },
        ],
    )
    _write_jsonl(
        source_b / "verify_raw.jsonl",
        [
            {
                "sample_id": "b:v1",
                "fen": "fen_train_hits_verify",
                "source": "source-b",
                "selected_move_uci": "a2a4",
                "result": "1/2-1/2",
                "metadata": {"tag": "b"},
            }
        ],
    )

    output_dir = tmp_path / "merged"
    summary = merge_phase5_raw_corpora(
        source_specs=(
            RawCorpusSourceSpec(name="source_a", raw_dir=source_a),
            RawCorpusSourceSpec(name="source_b", raw_dir=source_b),
        ),
        output_dir=output_dir,
    )

    train_rows = _read_jsonl(output_dir / "train_raw.jsonl")
    verify_rows = _read_jsonl(output_dir / "verify_raw.jsonl")

    assert summary["train_records"] == 2
    assert summary["verify_records"] == 2
    assert summary["verify_train_overlap"] == 0
    assert summary["collisions_train_to_verify"] == 0
    assert summary["skipped_train_due_to_verify"] == 1

    train_by_fen = {row["fen"]: row for row in train_rows}
    verify_by_fen = {row["fen"]: row for row in verify_rows}

    assert set(train_by_fen) == {"fen_shared_train", "fen_only_a"}
    assert set(verify_by_fen) == {"fen_verify_a", "fen_train_hits_verify"}
    assert train_by_fen["fen_shared_train"]["source"] == "source-b"
    assert verify_by_fen["fen_train_hits_verify"]["source"] == "source-b"


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
