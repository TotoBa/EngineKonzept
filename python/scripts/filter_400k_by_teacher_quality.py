"""Filter 400k planner-head artifacts by teacher-signal quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from train.datasets import (
    filter_planner_head_examples,
    load_planner_head_examples,
    write_planner_head_artifact,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path)
    parser.add_argument("--max-abs-root-value-cp", type=float, default=2000.0)
    parser.add_argument("--ambiguous-score-span-cp", type=float, default=5.0)
    parser.add_argument("--min-candidate-count", type=int, default=2)
    args = parser.parse_args(argv)

    examples = load_planner_head_examples(args.input_path)
    kept, summary = filter_planner_head_examples(
        examples,
        max_abs_root_value_cp=args.max_abs_root_value_cp,
        ambiguous_score_span_cp=args.ambiguous_score_span_cp,
        min_candidate_count=args.min_candidate_count,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_planner_head_artifact(args.output_path, kept)

    summary_payload = {
        "input_path": str(args.input_path),
        "output_path": str(args.output_path),
        "max_abs_root_value_cp": args.max_abs_root_value_cp,
        "ambiguous_score_span_cp": args.ambiguous_score_span_cp,
        "min_candidate_count": args.min_candidate_count,
        **summary.to_dict(),
    }

    if args.summary_path is not None:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(
            json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(summary_payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
