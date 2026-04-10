"""Control-plane CLI for the MySQL-backed distributed training orchestrator."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_db_args(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_db = subparsers.add_parser("init-db", help="create the orchestrator schema")
    init_db.add_argument("--drop-existing", action="store_true")

    status = subparsers.add_parser("status", help="show campaigns, tasks, and workers")
    status.add_argument("--limit", type=int, default=20)

    subparsers.add_parser("requeue-expired", help="requeue tasks with expired leases")

    submit = subparsers.add_parser("submit-phase10", help="submit one Phase-10 campaign")
    submit.add_argument("--config", type=Path, required=True)
    submit.add_argument("--kind", default="phase10_native")

    args = parser.parse_args()
    db = OrchestratorDB(_db_config_from_args(args))
    controller = OrchestratorController(db=db, repo_root=REPO_ROOT)

    if args.command == "init-db":
        db.init_schema(drop_existing=bool(args.drop_existing))
        print(json.dumps({"initialized": True, "drop_existing": bool(args.drop_existing)}))
        return 0
    if args.command == "status":
        print(json.dumps(db.status_snapshot(limit=int(args.limit)), indent=2, sort_keys=True))
        return 0
    if args.command == "requeue-expired":
        print(json.dumps(db.requeue_expired_tasks(), indent=2, sort_keys=True))
        return 0
    if args.command == "submit-phase10":
        result = controller.submit_phase10_campaign(
            config_path=args.config,
            kind=str(args.kind),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    raise ValueError(f"unsupported command: {args.command}")


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
