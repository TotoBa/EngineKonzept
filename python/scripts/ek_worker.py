"""Worker CLI for the MySQL-backed distributed training orchestrator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from train.orchestrator.controller import OrchestratorController  # noqa: E402
from train.orchestrator.db import OrchestratorDB  # noqa: E402
from train.orchestrator.models import DEFAULT_MYSQL_PORT, MySQLConfig  # noqa: E402
from train.orchestrator.worker import (  # noqa: E402
    OrchestratorWorker,
    build_default_worker_descriptor,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_db_args(parser)
    parser.add_argument("--capabilities", required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--worker-id", default=None)
    parser.add_argument("--version", default="orchestrator-v1")
    parser.add_argument("--lease-seconds", type=int, default=300)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    db = OrchestratorDB(_db_config_from_args(args))
    controller = OrchestratorController(db=db, repo_root=REPO_ROOT)
    descriptor = build_default_worker_descriptor(
        capabilities=_parse_capabilities(str(args.capabilities)),
        scratch_root=args.scratch_root,
        version=str(args.version),
        worker_id=args.worker_id,
    )
    worker = OrchestratorWorker(
        db=db,
        controller=controller,
        descriptor=descriptor,
        repo_root=REPO_ROOT,
        log_root=args.log_root,
        lease_seconds=int(args.lease_seconds),
        heartbeat_interval_seconds=float(args.heartbeat_seconds),
    )
    if args.once:
        claimed = worker.run_once()
        print(json.dumps({"claimed": claimed, "worker_id": descriptor.worker_id}))
        return 0
    worker.run_forever(poll_interval_seconds=float(args.poll_seconds))
    return 0


def _parse_capabilities(raw_value: str) -> tuple[str, ...]:
    capabilities = tuple(
        value.strip()
        for value in raw_value.split(",")
        if value.strip()
    )
    if not capabilities:
        raise ValueError("at least one capability is required")
    return capabilities


def _add_db_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db-host", default=None)
    parser.add_argument("--db-port", type=int, default=None)
    parser.add_argument("--db-user", default=None)
    parser.add_argument("--db-password", default=None)
    parser.add_argument("--db-name", default=None)


def _db_config_from_args(args: argparse.Namespace) -> MySQLConfig:
    host = str(args.db_host or os.environ.get("EK_MYSQL_HOST") or "")
    user = str(args.db_user or os.environ.get("EK_MYSQL_USER") or "")
    password = str(args.db_password or os.environ.get("EK_MYSQL_PASSWORD") or "")
    database = str(args.db_name or os.environ.get("EK_MYSQL_DATABASE") or "")
    missing = [
        name
        for name, value in (
            ("db-host", host),
            ("db-user", user),
            ("db-password", password),
            ("db-name", database),
        )
        if not value
    ]
    if missing:
        raise ValueError("missing required DB parameters: " + ", ".join(missing))
    return MySQLConfig(
        host=host,
        user=user,
        password=password,
        database=database,
        port=int(args.db_port or os.environ.get("EK_MYSQL_PORT", DEFAULT_MYSQL_PORT)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
