#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${1:-${REPO_ROOT}/python/configs/phase9_teacher_retrain_cycle_active_vs_vice_probe_v1.json}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

TMPDIR="${REPO_ROOT}/.tmp" "${REPO_ROOT}/.venv/bin/python" \
  "${REPO_ROOT}/python/scripts/run_selfplay_teacher_retrain_cycle.py" \
  --config "${CONFIG_PATH}"
