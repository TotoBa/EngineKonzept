#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
CONFIG_PATH="${PHASE10_LAPV1_ARENA_CONFIG:-$REPO_ROOT/python/configs/phase10_lapv1_stage2_fast_arena_all_unique_v2.json}"
TMPDIR="${TMPDIR:-$REPO_ROOT/.tmp}"
export TMPDIR

exec "$PYTHON_BIN" "$REPO_ROOT/python/scripts/run_phase10_lapv1_stage1_arena_campaign.py" --config "$CONFIG_PATH" "$@"
