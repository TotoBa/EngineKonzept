"""Select a hard-position LAPv1 Stage-T2 subset from a precomputed lapv1 JSONL."""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from train.datasets.lapv1_training import LAPv1TrainingExample  # noqa: E402
from train.eval.lapv1_curriculum import lapv1_hardness_score  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, required=True)
    parser.add_argument("--log-every", type=int, default=10000)
    parser.add_argument("--gap-cap-cp", type=float, default=128.0)
    parser.add_argument("--value-cap-cp", type=float, default=256.0)
    parser.add_argument("--candidate-count-cap", type=int, default=8)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if args.max_examples <= 0:
        raise ValueError("max-examples must be positive")
    if args.log_every <= 0:
        raise ValueError("log-every must be positive")

    input_path = _resolve_repo_path(args.input_path)
    output_path = _resolve_repo_path(args.output_path)
    summary_path = output_path.parent / f"{output_path.stem}.summary.json"
    if args.skip_existing and output_path.exists() and summary_path.exists():
        print(
            "[lapv1-hard] "
            f"reusing existing subset input={input_path} output={output_path}",
            flush=True,
        )
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    heap: list[tuple[float, str, int]] = []
    processed_examples = 0
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            example = LAPv1TrainingExample.from_json(
                line,
                source=f"{input_path}:{line_number}",
            )
            score = lapv1_hardness_score(
                example,
                gap_cap_cp=args.gap_cap_cp,
                value_cap_cp=args.value_cap_cp,
                candidate_count_cap=args.candidate_count_cap,
            )
            record = (score, example.sample_id, line_number)
            if len(heap) < args.max_examples:
                heapq.heappush(heap, record)
            elif record > heap[0]:
                heapq.heapreplace(heap, record)
            processed_examples += 1
            if processed_examples % args.log_every == 0:
                print(
                    "[lapv1-hard] "
                    f"indexed={processed_examples} selected={len(heap)} "
                    f"input={input_path.name}",
                    flush=True,
                )

    selected_line_numbers = {line_number for _score, _sample_id, line_number in heap}
    selected_score_by_line = {
        line_number: score for score, _sample_id, line_number in heap
    }
    selected_examples = 0
    selected_score_sum = 0.0
    selected_gap_sum = 0.0
    selected_curriculum_sum = 0.0
    selected_candidate_sum = 0
    selected_split = "unknown"
    with input_path.open("r", encoding="utf-8") as input_handle, output_path.open(
        "w",
        encoding="utf-8",
    ) as output_handle:
        for line_number, raw_line in enumerate(input_handle, 1):
            if line_number not in selected_line_numbers:
                continue
            line = raw_line.strip()
            if not line:
                continue
            output_handle.write(line)
            output_handle.write("\n")
            example = LAPv1TrainingExample.from_json(
                line,
                source=f"{input_path}:{line_number}",
            )
            selected_examples += 1
            selected_score_sum += selected_score_by_line[line_number]
            selected_gap_sum += float(abs(example.teacher_top1_minus_top2_cp or 0.0))
            selected_curriculum_sum += float(example.curriculum_priority)
            selected_candidate_sum += len(example.candidate_action_indices)
            selected_split = example.split

    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "split": selected_split,
        "processed_examples": processed_examples,
        "selected_examples": selected_examples,
        "max_examples": int(args.max_examples),
        "gap_cap_cp": float(args.gap_cap_cp),
        "value_cap_cp": float(args.value_cap_cp),
        "candidate_count_cap": int(args.candidate_count_cap),
        "mean_selected_score": (
            0.0 if selected_examples == 0 else selected_score_sum / selected_examples
        ),
        "mean_selected_gap_cp": (
            0.0 if selected_examples == 0 else selected_gap_sum / selected_examples
        ),
        "mean_selected_curriculum_priority": (
            0.0
            if selected_examples == 0
            else selected_curriculum_sum / selected_examples
        ),
        "mean_selected_candidate_count": (
            0.0 if selected_examples == 0 else selected_candidate_sum / selected_examples
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
