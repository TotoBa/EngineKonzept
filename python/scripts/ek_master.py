"""Master CLI for the MySQL-backed distributed training orchestrator."""

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
from train.orchestrator.master import OrchestratorMaster, load_master_spec  # noqa: E402
from train.orchestrator.models import DEFAULT_MYSQL_PORT, MySQLConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_db_args(parser)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=None)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--until-terminal", action="store_true")
    parser.add_argument("--max-cycles", type=int, default=1000)
    args = parser.parse_args()

    db = OrchestratorDB(_db_config_from_args(args))
    controller = OrchestratorController(db=db, repo_root=REPO_ROOT)
    master = OrchestratorMaster(
        db=db,
        controller=controller,
        repo_root=REPO_ROOT,
        spec=load_master_spec(args.config),
        spec_path=args.config,
    )

    if args.once:
        print(json.dumps(master.reconcile_once(), indent=2, sort_keys=True))
        return 0
    if args.until_terminal:
        print(
            json.dumps(
                master.run_until_terminal(
                    poll_interval_seconds=args.poll_seconds,
                    max_cycles=int(args.max_cycles),
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    master.run_forever(poll_interval_seconds=args.poll_seconds)
    return 0


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
