from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_phase5_corpus_suite.py"
_SPEC = importlib.util.spec_from_file_location("build_phase5_corpus_suite", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
build_phase5_corpus_suite = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = build_phase5_corpus_suite
_SPEC.loader.exec_module(build_phase5_corpus_suite)


def _write_split_jsonl(path: Path, count: int) -> None:
    path.write_text("".join("{}\n" for _ in range(count)), encoding="utf-8")


def _write_existing_dataset(dataset_dir: Path, *, split_counts: dict[str, int]) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "split_counts": split_counts,
        "total_examples": sum(split_counts.values()),
        "unique_fens": sum(split_counts.values()),
    }
    (dataset_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_split_jsonl(dataset_dir / "dataset.jsonl", sum(split_counts.values()))
    for split_name, count in split_counts.items():
        _write_split_jsonl(dataset_dir / f"{split_name}.jsonl", count)
        _write_split_jsonl(dataset_dir / f"proposer_{split_name}.jsonl", count)
        _write_split_jsonl(dataset_dir / f"proposer_symbolic_{split_name}.jsonl", count)
        _write_split_jsonl(dataset_dir / f"dynamics_{split_name}.jsonl", count)


def test_build_phase5_corpus_suite_validates_existing_tiers(monkeypatch, tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    verify_dir = tmp_path / "verify"
    _write_existing_dataset(train_dir, split_counts={"train": 8, "validation": 2})
    _write_existing_dataset(verify_dir, split_counts={"test": 3})

    monkeypatch.setattr(
        build_phase5_corpus_suite,
        "CURRENT_PHASE5_CORPUS_TIERS",
        {
            "mini": build_phase5_corpus_suite.ExistingDatasetTierSpec(
                name="mini",
                train_dataset_dir=train_dir.relative_to(tmp_path),
                verify_dataset_dir=verify_dir.relative_to(tmp_path),
            )
        },
    )

    artifact_out = tmp_path / "manifest.json"
    manifest = build_phase5_corpus_suite.build_phase5_corpus_suite(
        tier_names=["mini"],
        artifact_out=artifact_out,
        oracle_workers=4,
        oracle_batch_size=0,
        chunk_size=8,
        repo_root=tmp_path,
    )

    assert artifact_out.exists()
    assert manifest["tiers"]["mini"]["train_dataset"]["summary"]["split_counts"]["train"] == 8
    assert manifest["tiers"]["mini"]["verify_dataset"]["summary"]["split_counts"]["test"] == 3


def test_validate_dataset_dir_requires_current_artifacts(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "broken"
    dataset_dir.mkdir()
    (dataset_dir / "summary.json").write_text(
        json.dumps({"split_counts": {"train": 1}}, indent=2) + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "dataset.jsonl").write_text("{}\n", encoding="utf-8")
    (dataset_dir / "train.jsonl").write_text("{}\n", encoding="utf-8")

    try:
        build_phase5_corpus_suite._validate_dataset_dir(dataset_dir)
    except FileNotFoundError as exc:
        assert "proposer_train.jsonl" in str(exc)
        assert "proposer_symbolic_train.jsonl" in str(exc)
        assert "dynamics_train.jsonl" in str(exc)
    else:
        raise AssertionError("expected missing current-artifact validation to fail")
