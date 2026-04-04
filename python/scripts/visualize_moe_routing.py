"""Render a saved MoE routing report as diagnostic plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train.eval.moe_analysis import visualize_moe_routing_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-path", type=Path, required=True, help="Path to the JSON report")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for plot files")
    args = parser.parse_args()

    report = json.loads(args.report_path.read_text(encoding="utf-8"))
    visualize_moe_routing_report(report=report, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
