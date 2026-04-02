"""Dataset artifact writing helpers."""

from __future__ import annotations

import json
from pathlib import Path

from train.datasets.artifacts import proposer_artifact_name, to_proposer_example
from train.datasets.builder import BuiltDataset


def write_dataset_artifacts(
    output_dir: Path,
    dataset: BuiltDataset,
    *,
    write_proposer_artifacts: bool = False,
) -> None:
    """Write the full dataset, split files, summary JSON, and optional proposer artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(output_dir / "dataset.jsonl", dataset.examples)
    for split_name in ("train", "validation", "test"):
        split_examples = [example for example in dataset.examples if example.split == split_name]
        _write_jsonl(output_dir / f"{split_name}.jsonl", split_examples)
        if write_proposer_artifacts:
            proposer_examples = [to_proposer_example(example) for example in split_examples]
            _write_jsonl(output_dir / proposer_artifact_name(split_name), proposer_examples)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(dataset.summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, examples: list[object]) -> None:
    lines = [json.dumps(example.to_dict(), sort_keys=True) for example in examples]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
