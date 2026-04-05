"""Build precomputed Phase-10 LAPv1 training artifacts from planner-head JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from train.datasets.lapv1_training import (  # noqa: E402
    lapv1_training_example_from_planner_head,
)
from train.datasets.planner_head import PlannerHeadExample  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner-head-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--log-every", type=int, default=1000)
    args = parser.parse_args()

    if args.log_every <= 0:
        raise ValueError("log-every must be positive")

    planner_head_path = _resolve_repo_path(args.planner_head_path)
    output_path = _resolve_repo_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    example_count = 0
    candidate_count_sum = 0
    curriculum_priority_sum = 0.0
    split_name = "unknown"
    with planner_head_path.open("r", encoding="utf-8") as input_handle, output_path.open(
        "w",
        encoding="utf-8",
    ) as output_handle:
        for line_number, raw_line in enumerate(input_handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            planner_example = PlannerHeadExample.from_json(
                line,
                source=f"{planner_head_path}:{line_number}",
            )
            lapv1_example = lapv1_training_example_from_planner_head(planner_example)
            output_handle.write(json.dumps(lapv1_example.to_dict(), sort_keys=True))
            output_handle.write("\n")
            example_count += 1
            candidate_count_sum += len(lapv1_example.candidate_action_indices)
            curriculum_priority_sum += float(lapv1_example.curriculum_priority)
            split_name = lapv1_example.split
            if example_count % args.log_every == 0:
                print(
                    "[lapv1-artifact] "
                    f"converted={example_count} "
                    f"source={planner_head_path.name} "
                    f"output={output_path.name}",
                    flush=True,
                )

    summary = {
        "planner_head_path": str(planner_head_path),
        "output_path": str(output_path),
        "split": split_name,
        "example_count": example_count,
        "mean_candidate_count": (
            0.0 if example_count == 0 else candidate_count_sum / example_count
        ),
        "mean_curriculum_priority": (
            0.0 if example_count == 0 else curriculum_priority_sum / example_count
        ),
    }
    summary_path = output_path.parent / "lapv1.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
