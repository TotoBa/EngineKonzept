#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_PATH="${1:-python/configs/phase10_lapv1_stage1_10k_122k_v1.json}"

export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
export TMPDIR="${TMPDIR:-${REPO_ROOT}/.tmp}"
mkdir -p "${TMPDIR}"

python3 python/scripts/train_lapv1.py --config "${CONFIG_PATH}"
