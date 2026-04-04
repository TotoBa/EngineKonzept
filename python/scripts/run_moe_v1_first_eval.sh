#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG_PATH="${1:-python/configs/phase9_planner_moe_v1_10k_122k_v1.json}"
SUMMARY_PATH="${2:-artifacts/moe_v1/first_eval_summary.json}"

CHECKPOINT_PATH="$(
  PYTHONPATH=python .venv/bin/python - <<'PY' "$CONFIG_PATH"
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
bundle_dir = Path(config["export"]["bundle_dir"])
checkpoint_name = config["export"].get("checkpoint_name", "checkpoint.pt")
print(bundle_dir / checkpoint_name)
PY
)"

VALIDATION_ARGS="$(
  PYTHONPATH=python .venv/bin/python - <<'PY' "$CONFIG_PATH"
import json
import shlex
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
paths = [config["data"]["validation_path"], *config["data"].get("additional_validation_paths", [])]
print(" ".join(f"--dataset-path {shlex.quote(path)}" for path in paths))
PY
)"

TMP_DIR="${TMPDIR:-$REPO_ROOT/.tmp}"
mkdir -p "$TMP_DIR"
mkdir -p "$(dirname "$SUMMARY_PATH")"

TRAIN_LOG="$(mktemp "$TMP_DIR/moe_train_XXXX.json")"
EVAL_LOG="$(mktemp "$TMP_DIR/moe_eval_XXXX.json")"

PYTHONPATH=python .venv/bin/python python/scripts/train_planner.py --config "$CONFIG_PATH"
PYTHONPATH=python .venv/bin/python python/scripts/eval_planner.py --checkpoint "$CHECKPOINT_PATH" $VALIDATION_ARGS > "$EVAL_LOG"

PYTHONPATH=python .venv/bin/python - <<'PY' "$CONFIG_PATH" "$CHECKPOINT_PATH" "$SUMMARY_PATH" "$EVAL_LOG"
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
checkpoint_path = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
eval_log_path = Path(sys.argv[4])

config = json.loads(config_path.read_text(encoding="utf-8"))
training_summary_path = Path(config["output_dir"]) / "summary.json"
training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
eval_metrics = json.loads(eval_log_path.read_text(encoding="utf-8"))

references = {
    "set_v2_10k_122k_expanded": json.loads(
        Path("artifacts/phase8/planner_corpus_suite_set_v2_10k_122k_expanded_v1_verify.json").read_text(encoding="utf-8")
    ),
    "set_v6_10k_122k_expanded": json.loads(
        Path("artifacts/phase8/planner_corpus_suite_set_v6_10k_122k_expanded_v1_verify.json").read_text(encoding="utf-8")
    ),
}

summary = {
    "config_path": str(config_path),
    "checkpoint_path": str(checkpoint_path),
    "training_summary_path": str(training_summary_path),
    "trained_planner": {
        "name": "moe_v1_10k_122k_v1",
        "root_top1_accuracy": eval_metrics.get("root_top1_accuracy"),
        "teacher_root_mean_reciprocal_rank": eval_metrics.get("teacher_root_mean_reciprocal_rank"),
        "router_entropy": training_summary.get("best_validation", {}).get("router_entropy"),
        "load_balance_loss": training_summary.get("best_validation", {}).get("load_balance_loss"),
    },
    "references": {
        name: {
            "root_top1_accuracy": payload.get("root_top1_accuracy"),
            "teacher_root_mean_reciprocal_rank": payload.get("teacher_root_mean_reciprocal_rank"),
        }
        for name, payload in references.items()
    },
}
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

rows = [
    ("moe_v1_10k_122k_v1", summary["trained_planner"]),
    *summary["references"].items(),
]
print("name                              top1      mrr       entropy   load_balance")
for name, metrics in rows:
    top1 = metrics.get("root_top1_accuracy")
    mrr = metrics.get("teacher_root_mean_reciprocal_rank")
    entropy = metrics.get("router_entropy")
    load_balance = metrics.get("load_balance_loss")
    print(
        f"{name:<32} "
        f"{(top1 if top1 is not None else float('nan')):>7.4f}  "
        f"{(mrr if mrr is not None else float('nan')):>7.4f}  "
        f"{(entropy if entropy is not None else float('nan')):>8.4f}  "
        f"{(load_balance if load_balance is not None else float('nan')):>11.6f}"
    )
print(f"\nsummary written to {summary_path}")
PY
