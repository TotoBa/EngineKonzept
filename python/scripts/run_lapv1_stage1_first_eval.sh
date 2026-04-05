#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG_PATH="${1:-python/configs/phase10_lapv1_stage1_10k_122k_v1.json}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

PYTHONPATH=python "$PYTHON_BIN" - <<'PY' "$CONFIG_PATH"
import sys
from pathlib import Path

from train.config import resolve_repo_path
from train.trainers import count_lapv1_model_parameters, load_lapv1_train_config

repo_root = Path.cwd()
config_path = Path(sys.argv[1])
config = load_lapv1_train_config(config_path)
parameter_count = count_lapv1_model_parameters(config)
parameter_size_mb = parameter_count * 4 / (1024 * 1024)

train_paths = [resolve_repo_path(repo_root, path) for path in config.data.resolved_train_paths()]
validation_paths = [
    resolve_repo_path(repo_root, path)
    for path in config.data.resolved_validation_paths()
]
missing_paths = [
    str(path)
    for path in [*train_paths, *validation_paths]
    if not path.exists()
]

print("[lapv1-stage1-first-eval]", flush=True)
print(f"config_path={config_path}", flush=True)
print(f"stage={config.stage}", flush=True)
print(f"output_dir={resolve_repo_path(repo_root, config.output_dir)}", flush=True)
print(f"bundle_dir={resolve_repo_path(repo_root, config.export.bundle_dir)}", flush=True)
print(f"epochs={config.optimization.epochs}", flush=True)
print(f"batch_size={config.optimization.batch_size}", flush=True)
print(f"train_paths={len(train_paths)} validation_paths={len(validation_paths)}", flush=True)
print(f"parameter_count={parameter_count}", flush=True)
print(f"approx_fp32_size_mb={parameter_size_mb:.2f}", flush=True)

for path in train_paths:
    print(f"train_data_path={path}", flush=True)
for path in validation_paths:
    print(f"validation_data_path={path}", flush=True)

if missing_paths:
    print("[lapv1-stage1-first-eval] missing_data_paths_detected", flush=True)
    for path in missing_paths:
        print(f"missing_data_path={path}", flush=True)
    raise SystemExit(1)

print("[lapv1-stage1-first-eval] config loads and referenced planner-head artifacts exist", flush=True)
print("[lapv1-stage1-first-eval] no training started", flush=True)
PY
