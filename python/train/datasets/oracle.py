"""Python wrapper around the Rust dataset oracle."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Sequence

from train.datasets.schema import RawPositionRecord

DATASET_ORACLE_ENV = "ENGINEKONZEPT_DATASET_ORACLE"


class OracleError(RuntimeError):
    """Raised when the Rust dataset oracle fails."""


def label_records_with_oracle(
    records: Sequence[RawPositionRecord],
    *,
    repo_root: Path | None = None,
    command: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Enrich raw records by sending them through the Rust dataset oracle."""
    if not records:
        return []

    resolved_repo_root = repo_root or _default_repo_root()
    resolved_command = list(command or _default_oracle_command())

    payload = "\n".join(
        json.dumps(record.to_oracle_input(), sort_keys=True) for record in records
    )
    completed = subprocess.run(
        resolved_command,
        cwd=resolved_repo_root / "rust",
        input=f"{payload}\n",
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise OracleError(completed.stderr.strip() or completed.stdout.strip())

    outputs = [
        json.loads(line)
        for line in completed.stdout.splitlines()
        if line.strip()
    ]
    if len(outputs) != len(records):
        raise OracleError(
            f"dataset oracle returned {len(outputs)} records for {len(records)} inputs"
        )
    return outputs


def _default_oracle_command() -> list[str]:
    if env_command := os.environ.get(DATASET_ORACLE_ENV):
        return shlex.split(env_command)
    return ["cargo", "run", "--quiet", "-p", "tools", "--bin", "dataset-oracle"]


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]
