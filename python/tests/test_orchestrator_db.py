from __future__ import annotations

from contextlib import contextmanager

import pymysql

from train.orchestrator.db import OrchestratorDB
from train.orchestrator.models import MySQLConfig


class _FakeCursor:
    def __init__(self, actions: list[object]) -> None:
        self._actions = actions
        self.rowcount = 0

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, _sql: str, _params: object | None = None) -> None:
        action = self._actions.pop(0)
        if isinstance(action, BaseException):
            raise action
        self.rowcount = int(action)


class _FakeConnection:
    def __init__(self, actions: list[object]) -> None:
        self._actions = actions
        self.commits = 0
        self.rollbacks = 0

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._actions)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


def test_requeue_expired_tasks_retries_mysql_deadlock(monkeypatch) -> None:
    db = OrchestratorDB(
        MySQLConfig(
            host="localhost",
            user="user",
            password="password",
            database="database",
        )
    )
    connections = [
        _FakeConnection(
            [pymysql.err.OperationalError(1213, "Deadlock found when trying to get lock")]
        ),
        _FakeConnection([2, 1]),
    ]

    @contextmanager
    def fake_connect():
        yield connections.pop(0)

    monkeypatch.setattr(db, "_connect", fake_connect)

    result = db.requeue_expired_tasks()

    assert result == {"requeued": 2, "failed": 1}
    assert connections == []
