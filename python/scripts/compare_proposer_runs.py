"""Compare multiple Phase-5 proposer runs from summary and verify artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec in the form name:summary_path:verify_path",
    )
    parser.add_argument("--artifact-out", type=Path)
    args = parser.parse_args(argv)

    runs = [_load_run(spec) for spec in args.run]
    comparison = {
        "runs": runs,
        "best_by_metric": {
            "validation_legal_set_f1": _best_run_name(runs, ("validation", "legal_set_f1")),
            "validation_policy_top1_accuracy": _best_run_name(
                runs, ("validation", "policy_top1_accuracy")
            ),
            "verify_legal_set_f1": _best_run_name(runs, ("verify", "legal_set_f1")),
            "verify_policy_top1_accuracy": _best_run_name(
                runs, ("verify", "policy_top1_accuracy")
            ),
        },
    }
    rendered = json.dumps(comparison, indent=2, sort_keys=True)
    if args.artifact_out is not None:
        args.artifact_out.parent.mkdir(parents=True, exist_ok=True)
        args.artifact_out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


def _load_run(spec: str) -> dict[str, Any]:
    name, summary_path, verify_path = spec.split(":", maxsplit=2)
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    verify = json.loads(Path(verify_path).read_text(encoding="utf-8"))
    validation = dict(summary["best_validation"])
    config = dict(summary["config"])
    return {
        "name": name,
        "config": {
            "architecture": config["model"].get("architecture", "mlp_v1"),
            "dataset_path": config["data"]["dataset_path"],
            "hidden_dim": config["model"]["hidden_dim"],
            "hidden_layers": config["model"]["hidden_layers"],
            "batch_size": config["optimization"]["batch_size"],
            "learning_rate": config["optimization"]["learning_rate"],
            "legality_loss_weight": config["optimization"]["legality_loss_weight"],
            "policy_loss_weight": config["optimization"]["policy_loss_weight"],
            "torch_threads": config.get("runtime", {}).get("torch_threads", 0),
        },
        "validation": {
            "legal_set_f1": validation["legal_set_f1"],
            "legal_set_precision": validation["legal_set_precision"],
            "legal_set_recall": validation["legal_set_recall"],
            "policy_loss": validation["policy_loss"],
            "policy_top1_accuracy": validation["policy_top1_accuracy"],
            "examples_per_second": validation["examples_per_second"],
        },
        "verify": {
            "legal_set_f1": verify["legal_set_f1"],
            "legal_set_precision": verify["legal_set_precision"],
            "legal_set_recall": verify["legal_set_recall"],
            "policy_loss": verify["policy_loss"],
            "policy_top1_accuracy": verify["policy_top1_accuracy"],
            "examples_per_second": verify["examples_per_second"],
        },
        "paths": {
            "summary": summary_path,
            "verify": verify_path,
        },
    }


def _best_run_name(runs: list[dict[str, Any]], path: tuple[str, str]) -> str:
    return max(runs, key=lambda run: float(run[path[0]][path[1]]))["name"]


if __name__ == "__main__":
    raise SystemExit(main())
