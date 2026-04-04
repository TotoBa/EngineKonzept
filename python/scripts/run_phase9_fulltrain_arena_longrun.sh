#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
CONFIG_PATH="${PHASE9_FULLTRAIN_ARENA_CONFIG:-$REPO_ROOT/python/configs/phase9_fulltrain_then_arena_expanded_v1.json}"
TMPDIR="${TMPDIR:-$REPO_ROOT/.tmp}"
export TMPDIR

exec "$PYTHON_BIN" "$REPO_ROOT/python/scripts/run_phase9_fulltrain_arena_campaign.py" --config "$CONFIG_PATH" "$@"
