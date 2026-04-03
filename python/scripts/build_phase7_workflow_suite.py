"""Build the current Phase-7 workflow suite over the known Phase-5 corpus tiers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

from train.eval import evaluate_symbolic_opponent_baseline


REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW_SCRIPT = REPO_ROOT / "python" / "scripts" / "build_opponent_workflow_dataset.py"


@dataclass(frozen=True)
class WorkflowTierSpec:
    """Phase-7 workflow slice specification for one corpus tier."""

    name: str
    train_dataset_dir: Path
    verify_dataset_dir: Path
    train_examples: int
    validation_examples: int
    verify_examples: int


DEFAULT_TIERS: tuple[WorkflowTierSpec, ...] = (
    WorkflowTierSpec(
        name="pgn_10k",
        train_dataset_dir=Path("artifacts/datasets/phase5_stockfish_pgn_train_pi_10k_v1"),
        verify_dataset_dir=Path("artifacts/datasets/phase5_stockfish_pgn_verify_pi_10k_v1"),
        train_examples=1024,
        validation_examples=256,
        verify_examples=512,
    ),
    WorkflowTierSpec(
        name="merged_unique_122k",
        train_dataset_dir=Path("artifacts/datasets/phase5_stockfish_merged_unique_train_v1"),
        verify_dataset_dir=Path("artifacts/datasets/phase5_stockfish_merged_unique_verify_v1"),
        train_examples=2048,
        validation_examples=512,
        verify_examples=512,
    ),
    WorkflowTierSpec(
        name="unique_pi_400k",
        train_dataset_dir=Path("artifacts/datasets/phase5_stockfish_unique_pi_400k_train_v1"),
        verify_dataset_dir=Path("artifacts/datasets/phase5_stockfish_unique_pi_400k_verify_v1"),
        train_examples=4096,
        validation_examples=1024,
        verify_examples=386,
    ),
)

EXPANDED_TIERS: tuple[WorkflowTierSpec, ...] = (
    WorkflowTierSpec(
        name="pgn_10k",
        train_dataset_dir=Path("artifacts/datasets/phase5_stockfish_pgn_train_pi_10k_v1"),
        verify_dataset_dir=Path("artifacts/datasets/phase5_stockfish_pgn_verify_pi_10k_v1"),
        train_examples=4096,
        validation_examples=1024,
        verify_examples=512,
    ),
    WorkflowTierSpec(
        name="merged_unique_122k",
        train_dataset_dir=Path("artifacts/datasets/phase5_stockfish_merged_unique_train_v1"),
        verify_dataset_dir=Path("artifacts/datasets/phase5_stockfish_merged_unique_verify_v1"),
        train_examples=16384,
        validation_examples=4096,
        verify_examples=512,
    ),
    WorkflowTierSpec(
        name="unique_pi_400k",
        train_dataset_dir=Path("artifacts/datasets/phase5_stockfish_unique_pi_400k_train_v1"),
        verify_dataset_dir=Path("artifacts/datasets/phase5_stockfish_unique_pi_400k_verify_v1"),
        train_examples=32768,
        validation_examples=8192,
        verify_examples=386,
    ),
)

_TIER_VERSIONS: dict[str, tuple[WorkflowTierSpec, ...]] = {
    "default": DEFAULT_TIERS,
    "expanded": EXPANDED_TIERS,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-engine", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/phase7/opponent_workflow_corpus_suite_v1"),
    )
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--multipv", type=int, default=8)
    parser.add_argument("--policy-temperature-cp", type=float, default=100.0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--tier-version",
        choices=sorted(_TIER_VERSIONS),
        default="default",
        help="Tier specification version: 'default' (v1 counts) or 'expanded' (7x more data).",
    )
    parser.add_argument(
        "--tier",
        action="append",
        help="Restrict the build to one or more named corpus tiers.",
    )
    args = parser.parse_args(argv)

    tier_specs = _TIER_VERSIONS[args.tier_version]
    checkpoint_path = _resolve_repo_path(args.checkpoint)
    teacher_engine = _resolve_repo_path(args.teacher_engine)
    output_root = _resolve_repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    selected_names = set(args.tier or [tier.name for tier in tier_specs])
    selected_tiers = [tier for tier in tier_specs if tier.name in selected_names]

    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "teacher_engine": str(teacher_engine),
        "teacher_nodes": args.nodes,
        "teacher_multipv": args.multipv,
        "policy_temperature_cp": args.policy_temperature_cp,
        "top_k": args.top_k,
        "output_root": str(output_root),
        "tiers": {},
        "train_paths": [],
        "validation_paths": [],
        "verify_paths": [],
    }

    for tier in selected_tiers:
        tier_summary = _build_tier(
            tier,
            checkpoint_path=checkpoint_path,
            teacher_engine=teacher_engine,
            output_root=output_root,
            nodes=args.nodes,
            multipv=args.multipv,
            policy_temperature_cp=args.policy_temperature_cp,
            top_k=args.top_k,
        )
        summary["tiers"][tier.name] = tier_summary
        summary["train_paths"].append(tier_summary["train"]["opponent_head_path"])
        summary["validation_paths"].append(tier_summary["validation"]["opponent_head_path"])
        summary["verify_paths"].append(tier_summary["verify"]["opponent_head_path"])

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _build_tier(
    tier: WorkflowTierSpec,
    *,
    checkpoint_path: Path,
    teacher_engine: Path,
    output_root: Path,
    nodes: int,
    multipv: int,
    policy_temperature_cp: float,
    top_k: int,
) -> dict[str, Any]:
    train_dir = output_root / f"{tier.name}_train_v1"
    validation_dir = output_root / f"{tier.name}_validation_v1"
    verify_dir = output_root / f"{tier.name}_verify_v1"

    train_summary = _run_workflow_build(
        dataset_dir=_resolve_repo_path(tier.train_dataset_dir),
        split="train",
        checkpoint_path=checkpoint_path,
        teacher_engine=teacher_engine,
        output_dir=train_dir,
        max_examples=tier.train_examples,
        nodes=nodes,
        multipv=multipv,
        policy_temperature_cp=policy_temperature_cp,
        top_k=top_k,
    )
    validation_summary = _run_workflow_build(
        dataset_dir=_resolve_repo_path(tier.train_dataset_dir),
        split="validation",
        checkpoint_path=checkpoint_path,
        teacher_engine=teacher_engine,
        output_dir=validation_dir,
        max_examples=tier.validation_examples,
        nodes=nodes,
        multipv=multipv,
        policy_temperature_cp=policy_temperature_cp,
        top_k=top_k,
    )
    verify_summary = _run_workflow_build(
        dataset_dir=_resolve_repo_path(tier.verify_dataset_dir),
        split="test",
        checkpoint_path=checkpoint_path,
        teacher_engine=teacher_engine,
        output_dir=verify_dir,
        max_examples=tier.verify_examples,
        nodes=nodes,
        multipv=multipv,
        policy_temperature_cp=policy_temperature_cp,
        top_k=top_k,
    )
    symbolic_baseline = evaluate_symbolic_opponent_baseline(
        checkpoint_path,
        dataset_path=verify_dir,
        split="test",
    ).to_dict()

    return {
        "train": {
            "dataset_dir": str(_resolve_repo_path(tier.train_dataset_dir)),
            "output_dir": str(train_dir),
            "opponent_head_path": str(train_dir / "opponent_head_train.jsonl"),
            "summary": train_summary,
        },
        "validation": {
            "dataset_dir": str(_resolve_repo_path(tier.train_dataset_dir)),
            "output_dir": str(validation_dir),
            "opponent_head_path": str(validation_dir / "opponent_head_validation.jsonl"),
            "summary": validation_summary,
        },
        "verify": {
            "dataset_dir": str(_resolve_repo_path(tier.verify_dataset_dir)),
            "output_dir": str(verify_dir),
            "opponent_head_path": str(verify_dir / "opponent_head_test.jsonl"),
            "summary": verify_summary,
            "symbolic_baseline": symbolic_baseline,
        },
    }


def _run_workflow_build(
    *,
    dataset_dir: Path,
    split: str,
    checkpoint_path: Path,
    teacher_engine: Path,
    output_dir: Path,
    max_examples: int,
    nodes: int,
    multipv: int,
    policy_temperature_cp: float,
    top_k: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(_WORKFLOW_SCRIPT),
        "--dataset-dir",
        str(dataset_dir),
        "--split",
        split,
        "--checkpoint",
        str(checkpoint_path),
        "--teacher-engine",
        str(teacher_engine),
        "--output-dir",
        str(output_dir),
        "--nodes",
        str(nodes),
        "--multipv",
        str(multipv),
        "--policy-temperature-cp",
        str(policy_temperature_cp),
        "--top-k",
        str(top_k),
        "--max-examples",
        str(max_examples),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    summary_path = output_dir / "summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
