#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
CONFIG_PATH="${PHASE9_REPLAY_CAMPAIGN_CONFIG:-$REPO_ROOT/python/configs/phase9_replay_campaign_active_expanded_v1.json}"
TMPDIR="${TMPDIR:-$REPO_ROOT/.tmp}"
export TMPDIR

exec "$PYTHON_BIN" "$REPO_ROOT/python/scripts/run_phase9_replay_campaign.py" --config "$CONFIG_PATH" "$@"
