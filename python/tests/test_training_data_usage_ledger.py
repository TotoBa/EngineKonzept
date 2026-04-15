from __future__ import annotations

from typing import Any

import pymysql

from train.datasets.schema import RawPositionRecord
from train.orchestrator.models import MySQLConfig
from train.orchestrator.training_data_usage_ledger import (
    CREATE_TRAINING_DATA_USAGE_TABLES,
    TRAINING_DATA_USAGE_SCHEMA_MIGRATIONS,
    MySQLLineageTrainingUsageLedger,
)


class _LedgerCursor:
    def __init__(self, connection: "_LedgerConnection") -> None:
        self._connection = connection
        self._fetchone_results = connection.fetchone_results

    def __enter__(self) -> "_LedgerCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, sql: str, params: object | None = None) -> None:
        if self._connection.execute_errors:
            raise self._connection.execute_errors.pop(0)
        self._connection.execute_calls.append((sql, params))

    def executemany(self, sql: str, rows: list[tuple[Any, ...]]) -> None:
        if self._connection.executemany_errors:
            raise self._connection.executemany_errors.pop(0)
        self._connection.executemany_calls.append((sql, list(rows)))

    def fetchone(self) -> object | None:
        if self._fetchone_results:
            return self._fetchone_results.pop(0)
        return None

    def fetchall(self) -> list[object]:
        return []


class _LedgerConnection:
    def __init__(
        self,
        *,
        fetchone_results: list[object] | None = None,
        execute_errors: list[BaseException] | None = None,
        executemany_errors: list[BaseException] | None = None,
    ) -> None:
        self.execute_calls: list[tuple[str, object | None]] = []
        self.executemany_calls: list[tuple[str, list[tuple[Any, ...]]]] = []
        self.fetchone_results = list(fetchone_results or [])
        self.execute_errors = list(execute_errors or [])
        self.executemany_errors = list(executemany_errors or [])
        self.commits = 0
        self.rollbacks = 0
        self.pings = 0
        self.open = True

    def cursor(self) -> _LedgerCursor:
        return _LedgerCursor(self)

    def ping(self, reconnect: bool = False) -> None:
        self.pings += 1
        if reconnect:
            self.open = True

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.open = False


def test_mysql_usage_ledger_ensure_schema_runs_once(monkeypatch) -> None:
    ledger = MySQLLineageTrainingUsageLedger(
        MySQLConfig(host="localhost", user="user", password="pw", database="db")
    )
    connection = _LedgerConnection()
    monkeypatch.setattr(ledger, "_connection_or_raise", lambda: connection)

    ledger.ensure_schema()
    ledger.ensure_schema()

    executed_sql = [sql.strip() for sql, _params in connection.execute_calls]
    expected_sql = [
        statement.strip()
        for statement in (
            *CREATE_TRAINING_DATA_USAGE_TABLES,
            *TRAINING_DATA_USAGE_SCHEMA_MIGRATIONS,
        )
    ]
    assert executed_sql == expected_sql
    assert connection.commits == 1


def test_mysql_usage_ledger_records_compact_counter_rows(monkeypatch) -> None:
    ledger = MySQLLineageTrainingUsageLedger(
        MySQLConfig(host="localhost", user="user", password="pw", database="db")
    )
    connection = _LedgerConnection()
    monkeypatch.setattr(ledger, "ensure_schema", lambda: None)
    monkeypatch.setattr(
        ledger,
        "generation_usage",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(ledger, "_connection_or_raise", lambda: connection)

    result = ledger.record_generation_usage(
        master_name="master",
        lineage_name="lineage",
        generation=3,
        campaign_id=7,
        model_id=8,
        merged_raw_dir="/tmp/raw",
        train_records=[
            RawPositionRecord(sample_id="train:a", fen="fen:a", source="base"),
            RawPositionRecord(sample_id="train:a-dup", fen="fen:a", source="feedback"),
            RawPositionRecord(sample_id="train:b", fen="fen:b", source="idle"),
        ],
        verify_records=[
            RawPositionRecord(sample_id="verify:a", fen="fen:v", source="base"),
        ],
    )

    assert result["already_recorded"] is False
    assert connection.commits == 1
    assert connection.rollbacks == 0
    assert len(connection.executemany_calls) == 2

    train_sql, train_rows = connection.executemany_calls[0]
    verify_sql, verify_rows = connection.executemany_calls[1]

    assert "sample_id" not in train_sql
    assert "sample_source" not in train_sql
    assert "sample_id" not in verify_sql
    assert "sample_source" not in verify_sql
    assert len(train_rows) == 2
    assert len(verify_rows) == 1
    assert all(len(row) == 8 for row in train_rows + verify_rows)


def test_mysql_usage_ledger_generation_usage_retries_after_server_gone_away(
    monkeypatch,
) -> None:
    ledger = MySQLLineageTrainingUsageLedger(
        MySQLConfig(host="localhost", user="user", password="pw", database="db")
    )
    first_connection = _LedgerConnection(
        execute_errors=[pymysql.err.OperationalError(2006, "MySQL server has gone away")]
    )
    second_connection = _LedgerConnection(
        fetchone_results=[
            {
                "master_name": "master",
                "lineage_name": "lineage",
                "generation": 4,
                "campaign_id": 9,
                "model_id": 11,
                "merged_raw_dir": "/tmp/raw",
                "train_record_count": 10,
                "verify_record_count": 2,
            }
        ]
    )
    connections = [first_connection, second_connection]

    def connection_or_raise() -> _LedgerConnection:
        if ledger._connection is None:  # type: ignore[attr-defined]
            ledger._connection = connections.pop(0)  # type: ignore[attr-defined]
        return ledger._connection  # type: ignore[attr-defined]

    monkeypatch.setattr(ledger, "ensure_schema", lambda: None)
    monkeypatch.setattr(ledger, "_connection_or_raise", connection_or_raise)
    monkeypatch.setattr("train.orchestrator.training_data_usage_ledger.time.sleep", lambda _seconds: None)

    result = ledger.generation_usage(
        master_name="master",
        lineage_name="lineage",
        generation=4,
    )

    assert result == {
        "master_name": "master",
        "lineage_name": "lineage",
        "generation": 4,
        "campaign_id": 9,
        "model_id": 11,
        "merged_raw_dir": "/tmp/raw",
        "train_record_count": 10,
        "verify_record_count": 2,
        "already_recorded": True,
    }
    assert first_connection.rollbacks == 1
    assert first_connection.open is False
    assert second_connection.pings >= 1
