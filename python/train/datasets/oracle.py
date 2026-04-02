"""Python wrapper around the Rust dataset oracle."""

from __future__ import annotations

import json
import os
import shlex
import socket
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
    resolved_command = list(command or _default_oracle_command(resolved_repo_root))

    payload = "\n".join(
        json.dumps(record.to_oracle_input(), sort_keys=True) for record in records
    )
    if _is_unix_socket_command(resolved_command):
        return _label_records_with_unix_socket(records, payload, resolved_command[0])

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


def _label_records_with_unix_socket(
    records: Sequence[RawPositionRecord],
    payload: str,
    endpoint: str,
) -> list[dict[str, Any]]:
    socket_path = _parse_unix_socket_endpoint(endpoint)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(str(socket_path))
            client.sendall(f"{payload}\n".encode("utf-8"))
            client.shutdown(socket.SHUT_WR)

            chunks: list[bytes] = []
            while True:
                chunk = client.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
    except OSError as error:
        raise OracleError(f"dataset oracle socket failed: {error}") from error

    response = b"".join(chunks).decode("utf-8")
    outputs = [json.loads(line) for line in response.splitlines() if line.strip()]
    if len(outputs) != len(records):
        raise OracleError(
            f"dataset oracle returned {len(outputs)} records for {len(records)} inputs"
        )
    return outputs


def _default_oracle_command(repo_root: Path) -> list[str]:
    if env_command := os.environ.get(DATASET_ORACLE_ENV):
        return shlex.split(env_command)
    built_binary = repo_root / "rust" / "target" / "debug" / "dataset-oracle"
    if built_binary.exists():
        return [str(built_binary)]
    return ["cargo", "run", "--quiet", "-p", "tools", "--bin", "dataset-oracle"]


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _is_unix_socket_command(command: Sequence[str]) -> bool:
    return len(command) == 1 and command[0].startswith("unix://")


def _parse_unix_socket_endpoint(endpoint: str) -> Path:
    if not endpoint.startswith("unix://"):
        raise OracleError(f"unsupported oracle endpoint: {endpoint}")
    path = endpoint.removeprefix("unix://")
    if not path:
        raise OracleError("dataset oracle unix endpoint must include a socket path")
    return Path(path)
