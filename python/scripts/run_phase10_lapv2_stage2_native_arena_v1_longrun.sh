#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
CONFIG_PATH="${PHASE10_LAPV2_NATIVE_ARENA_CONFIG:-$REPO_ROOT/python/configs/phase10_lapv2_stage2_native_arena_all_sources_v1.json}"
WORKFLOW_ROOT="${PHASE10_LAPV2_NATIVE_WORKFLOW_ROOT:-/srv/schach/engine_training/phase10/lapv2_workflow_all_sources_v1}"
FULL_TRAIN_PATH="${PHASE10_LAPV2_NATIVE_FULL_TRAIN_PATH:-$WORKFLOW_ROOT/all_unique_train_v1/lapv1_train.jsonl}"
FULL_VALIDATION_PATH="${PHASE10_LAPV2_NATIVE_FULL_VALIDATION_PATH:-$WORKFLOW_ROOT/all_unique_validation_v1/lapv1_validation.jsonl}"
HARD_TRAIN_PATH="${PHASE10_LAPV2_NATIVE_HARD_TRAIN_PATH:-$WORKFLOW_ROOT/all_unique_train_hard_v1/lapv1_train_hard.jsonl}"
HARD_VALIDATION_PATH="${PHASE10_LAPV2_NATIVE_HARD_VALIDATION_PATH:-$WORKFLOW_ROOT/all_unique_validation_hard_v1/lapv1_validation_hard.jsonl}"
HARD_TRAIN_MAX_EXAMPLES="${PHASE10_LAPV2_NATIVE_HARD_TRAIN_MAX_EXAMPLES:-150000}"
HARD_VALIDATION_MAX_EXAMPLES="${PHASE10_LAPV2_NATIVE_HARD_VALIDATION_MAX_EXAMPLES:-15000}"
HARD_LOG_EVERY="${PHASE10_LAPV2_NATIVE_HARD_LOG_EVERY:-10000}"
TMPDIR="${TMPDIR:-$REPO_ROOT/.tmp}"
export TMPDIR

SKIP_EXISTING=()
for arg in "$@"; do
  if [[ "$arg" == "--skip-existing" ]]; then
    SKIP_EXISTING+=("--skip-existing")
    break
  fi
done

"$PYTHON_BIN" "$REPO_ROOT/python/scripts/build_lapv1_hard_positions_dataset.py" \
  --input-path "$FULL_TRAIN_PATH" \
  --output-path "$HARD_TRAIN_PATH" \
  --max-examples "$HARD_TRAIN_MAX_EXAMPLES" \
  --log-every "$HARD_LOG_EVERY" \
  "${SKIP_EXISTING[@]}"

"$PYTHON_BIN" "$REPO_ROOT/python/scripts/build_lapv1_hard_positions_dataset.py" \
  --input-path "$FULL_VALIDATION_PATH" \
  --output-path "$HARD_VALIDATION_PATH" \
  --max-examples "$HARD_VALIDATION_MAX_EXAMPLES" \
  --log-every "$HARD_LOG_EVERY" \
  "${SKIP_EXISTING[@]}"

exec "$PYTHON_BIN" "$REPO_ROOT/python/scripts/run_phase10_lapv1_stage1_arena_campaign.py" \
  --config "$CONFIG_PATH" \
  "$@"
