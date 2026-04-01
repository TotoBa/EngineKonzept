"""Benchmark subprocess vs. Unix-socket dataset oracle transports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Sequence

from train.datasets import RawPositionRecord, load_raw_records
from train.datasets.oracle import label_records_with_oracle

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-format", required=True)
    parser.add_argument("--source-name")
    parser.add_argument("--records", type=int, default=10_000)
    parser.add_argument(
        "--socket-path",
        type=Path,
        default=Path("/tmp/enginekonzept-oracle-bench.sock"),
    )
    parser.add_argument(
        "--subprocess-command",
        nargs="+",
        help="Override the one-shot oracle command used for the subprocess benchmark",
    )
    parser.add_argument(
        "--daemon-command",
        nargs="+",
        help="Override the daemon command prefix; '--socket <path>' is appended automatically",
    )
    args = parser.parse_args(argv)

    seed_records = load_raw_records(
        _resolve_repo_path(args.input),
        args.source_format,
        source_name=args.source_name,
    )
    records = expand_records(seed_records, target_count=args.records)

    subprocess_started = time.perf_counter()
    subprocess_outputs = label_records_with_oracle(
        records,
        repo_root=REPO_ROOT,
        command=args.subprocess_command,
    )
    subprocess_seconds = time.perf_counter() - subprocess_started

    daemon = start_oracle_daemon(args.socket_path, command_prefix=args.daemon_command)
    try:
        daemon_started = time.perf_counter()
        daemon_outputs = label_records_with_oracle(
            records,
            command=[f"unix://{args.socket_path}"],
        )
        daemon_seconds = time.perf_counter() - daemon_started
    finally:
        stop_oracle_daemon(daemon, args.socket_path)

    subprocess_digest = digest_outputs(subprocess_outputs)
    daemon_digest = digest_outputs(daemon_outputs)
    if subprocess_digest != daemon_digest:
        raise RuntimeError("oracle benchmark output mismatch between subprocess and daemon")

    result = {
        "record_count": len(records),
        "source_record_count": len(seed_records),
        "subprocess_seconds": round(subprocess_seconds, 6),
        "subprocess_records_per_second": round(rate(len(records), subprocess_seconds), 3),
        "daemon_seconds": round(daemon_seconds, 6),
        "daemon_records_per_second": round(rate(len(records), daemon_seconds), 3),
        "speedup": round(rate(subprocess_seconds, daemon_seconds), 3),
        "output_digest": subprocess_digest,
        "socket_path": str(args.socket_path),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def expand_records(seed_records: list[RawPositionRecord], *, target_count: int) -> list[RawPositionRecord]:
    if target_count <= 0:
        raise ValueError("--records must be positive")
    if not seed_records:
        raise ValueError("input must produce at least one raw record")

    expanded: list[RawPositionRecord] = []
    for index in range(target_count):
        template = seed_records[index % len(seed_records)]
        expanded.append(
            RawPositionRecord(
                sample_id=f"{template.sample_id}:bench:{index}",
                fen=template.fen,
                source=template.source,
                selected_move_uci=template.selected_move_uci,
                result=template.result,
                metadata=dict(template.metadata),
            )
        )
    return expanded


def start_oracle_daemon(
    socket_path: Path,
    *,
    command_prefix: Sequence[str] | None,
) -> subprocess.Popen[str]:
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        socket_path.unlink()

    resolved_command = list(command_prefix or _default_daemon_command())
    process = subprocess.Popen(
        [*resolved_command, "--socket", str(socket_path)],
        cwd=_command_cwd(resolved_command),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if socket_path.exists():
            return process
        if process.poll() is not None:
            stderr = process.stderr.read() if process.stderr is not None else ""
            raise RuntimeError(f"dataset-oracle-daemon failed to start: {stderr.strip()}")
        time.sleep(0.05)

    stop_oracle_daemon(process, socket_path)
    raise RuntimeError("dataset-oracle-daemon did not create its socket within 10s")


def stop_oracle_daemon(process: subprocess.Popen[str], socket_path: Path) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    if socket_path.exists():
        socket_path.unlink()


def digest_outputs(outputs: list[dict[str, object]]) -> str:
    payload = "\n".join(json.dumps(output, sort_keys=True) for output in outputs)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def rate(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _default_daemon_command() -> list[str]:
    built_binary = REPO_ROOT / "rust" / "target" / "debug" / "dataset-oracle-daemon"
    if built_binary.exists():
        return [str(built_binary)]
    return [
        "cargo",
        "run",
        "--quiet",
        "-p",
        "tools",
        "--bin",
        "dataset-oracle-daemon",
        "--",
    ]


def _command_cwd(command: Sequence[str]) -> Path | None:
    return REPO_ROOT / "rust" if command and command[0] == "cargo" else None


if __name__ == "__main__":
    raise SystemExit(main())
