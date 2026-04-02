"""Profile the Rust dataset oracle on a reproducible raw-record slice."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from typing import Any, Sequence

from train.datasets import SUPPORTED_SOURCE_FORMATS, load_raw_records
from train.datasets.schema import RawPositionRecord

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-format", choices=SUPPORTED_SOURCE_FORMATS, required=True)
    parser.add_argument("--source-name")
    parser.add_argument("--records", type=int, default=0)
    parser.add_argument("--artifact-out", type=Path)
    parser.add_argument(
        "--profile-command",
        nargs="+",
        help="Override the dataset-oracle-profile command",
    )
    args = parser.parse_args(argv)

    records = load_raw_records(
        _resolve_repo_path(args.input),
        args.source_format,
        source_name=args.source_name,
    )
    if args.records > 0:
        records = _select_records(records, target_count=args.records)
    if not records:
        raise ValueError("profile input must produce at least one raw record")

    profile = _run_profile(records, command=args.profile_command)
    result = {
        "input": str(_resolve_repo_path(args.input)),
        "record_count": len(records),
        "runtime": _runtime_metadata(),
        "command": list(args.profile_command or _default_profile_command()),
        "profile": profile,
        "top_phases": _top_phases(profile, limit=5),
    }

    rendered = json.dumps(result, indent=2)
    if args.artifact_out is not None:
        args.artifact_out.parent.mkdir(parents=True, exist_ok=True)
        args.artifact_out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


def _run_profile(
    records: Sequence[RawPositionRecord], *, command: Sequence[str] | None
) -> dict[str, Any]:
    payload = "\n".join(
        json.dumps(
            {
                "fen": record.fen,
                "selected_move_uci": record.selected_move_uci,
            }
        )
        for record in records
    ).encode("utf-8")
    resolved_command = list(command or _default_profile_command())
    process = subprocess.run(
        resolved_command,
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=_command_cwd(resolved_command),
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"dataset-oracle-profile failed with {process.returncode}: "
            f"{process.stderr.decode('utf-8', 'replace').strip()}"
        )
    return json.loads(process.stdout.decode("utf-8"))


def _top_phases(profile: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    phases = [
        {"name": name, **phase}
        for name, phase in profile["phases"]
        if phase["share_of_measured"] > 0.0
    ]
    phases.sort(key=lambda phase: phase["share_of_measured"], reverse=True)
    return phases[:limit]


def _select_records(
    records: Sequence[RawPositionRecord],
    *,
    target_count: int,
) -> list[RawPositionRecord]:
    if target_count <= 0:
        return list(records)
    if not records:
        return []
    if target_count <= len(records):
        return list(records[:target_count])

    expanded: list[RawPositionRecord] = []
    for index in range(target_count):
        template = records[index % len(records)]
        expanded.append(
            RawPositionRecord(
                sample_id=f"{template.sample_id}:profile:{index}",
                fen=template.fen,
                source=template.source,
                selected_move_uci=template.selected_move_uci,
                result=template.result,
                metadata=dict(template.metadata),
            )
        )
    return expanded


def _default_profile_command() -> list[str]:
    built_binary = REPO_ROOT / "rust" / "target" / "debug" / "dataset-oracle-profile"
    if built_binary.exists():
        return [str(built_binary)]
    return ["cargo", "run", "--quiet", "-p", "tools", "--bin", "dataset-oracle-profile"]


def _command_cwd(command: Sequence[str]) -> Path | None:
    return REPO_ROOT / "rust" if command and command[0] == "cargo" else None


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _runtime_metadata() -> dict[str, object]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
