"""Merge per-tier Phase-7 summary files into a single combined summary.json.

When building tiers in parallel (with --tier), each run overwrites the root
summary.json.  This script re-reads the per-split summary files and the
last written root summary to reconstruct a combined summary that covers
all tiers present in the output root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train.datasets import opponent_head_artifact_name


KNOWN_TIERS = ("pgn_10k", "merged_unique_122k", "unique_pi_400k")
SPLITS = (("train", "train"), ("validation", "validation"), ("verify", "test"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--teacher-engine", type=str, default="/usr/games/stockfish18")
    parser.add_argument("--nodes", type=int, default=256)
    parser.add_argument("--multipv", type=int, default=8)
    parser.add_argument("--policy-temperature-cp", type=float, default=100.0)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    summary: dict = {
        "checkpoint": args.checkpoint,
        "teacher_engine": args.teacher_engine,
        "teacher_nodes": args.nodes,
        "teacher_multipv": args.multipv,
        "policy_temperature_cp": args.policy_temperature_cp,
        "output_root": str(output_root),
        "tiers": {},
        "train_paths": [],
        "validation_paths": [],
        "verify_paths": [],
    }

    for tier_name in KNOWN_TIERS:
        tier_entry: dict = {}
        all_splits_present = True
        for split_name, canonical_split in SPLITS:
            split_dir = output_root / f"{tier_name}_{split_name}_v1"
            split_summary_path = split_dir / "summary.json"
            if not split_summary_path.exists():
                all_splits_present = False
                break
            split_summary = json.loads(split_summary_path.read_text(encoding="utf-8"))
            dataset_dir = split_summary.get("dataset_dir", "")
            opponent_head_path = str(split_dir / opponent_head_artifact_name(canonical_split))
            tier_entry[split_name] = {
                "dataset_dir": dataset_dir,
                "output_dir": str(split_dir),
                "opponent_head_path": opponent_head_path,
                "summary": split_summary,
            }
            summary_key = "verify_paths" if split_name == "verify" else f"{split_name}_paths"
            summary[summary_key].append(opponent_head_path)

        if all_splits_present:
            summary["tiers"][tier_name] = tier_entry
            print(f"  included: {tier_name}")
        else:
            print(f"  skipped (incomplete): {tier_name}")

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nMerged summary: {summary_path}")
    print(f"Tiers: {sorted(summary['tiers'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
