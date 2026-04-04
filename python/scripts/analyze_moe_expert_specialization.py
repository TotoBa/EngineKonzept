"""Analyze routing behavior and expert specialization for a trained MoE planner."""

from __future__ import annotations

import argparse
from pathlib import Path

from train.eval.moe_analysis import analyze_moe_expert_specialization


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a moe_v1 checkpoint")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to a planner_head_<split>.jsonl artifact",
    )
    parser.add_argument("--output-path", type=Path, required=True, help="JSON report output path")
    parser.add_argument("--max-examples", type=int, default=None, help="Optional example limit")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    args = parser.parse_args()

    analyze_moe_expert_specialization(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
