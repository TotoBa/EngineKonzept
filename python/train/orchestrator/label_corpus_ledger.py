"""MySQL-backed resumable ledger for unique PGN/Stockfish corpus labeling."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any, Iterator, Protocol

import pymysql
from pymysql.cursors import DictCursor, SSDictCursor

from train.orchestrator.models import MySQLConfig


CREATE_LABEL_CORPUS_TABLES = (
    """
    CREATE TABLE IF NOT EXISTS label_corpus_namespaces (
        namespace_hash CHAR(64) NOT NULL PRIMARY KEY,
        namespace VARCHAR(1024) NOT NULL,
        created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
    ) ENGINE=InnoDB
    """,
    """
    CREATE TABLE IF NOT EXISTS label_corpus_samples (
        namespace_hash CHAR(64) NOT NULL,
        fen_hash CHAR(64) NOT NULL,
        fen TEXT NOT NULL,
        sample_split VARCHAR(16) NOT NULL,
        sample_id VARCHAR(255) NOT NULL,
        source VARCHAR(128) NOT NULL,
        result VARCHAR(16) NULL,
        metadata_json JSON NOT NULL,
        selected_move_uci VARCHAR(16) NULL,
        status VARCHAR(16) NOT NULL,
        created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
        updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
            ON UPDATE CURRENT_TIMESTAMP(6),
        PRIMARY KEY (namespace_hash, fen_hash),
        KEY idx_label_corpus_samples_status (namespace_hash, status, sample_split),
        KEY idx_label_corpus_samples_split (namespace_hash, sample_split, fen_hash)
    ) ENGINE=InnoDB
    """,
)


@dataclass(frozen=True)
class LabelCorpusSample:
    """One labeled raw row stored in the MySQL corpus ledger."""

    sample_id: str
    fen: str
    split: str
    source: str
    result: str | None
    metadata: dict[str, Any]
    selected_move_uci: str


@dataclass(frozen=True)
class ReservedLabelCorpusSample:
    """One reserved but not yet labeled sample from the MySQL ledger."""

    fen_hash: str
    fen: str
    split: str
    metadata: dict[str, Any]


class LabelCorpusLedger(Protocol):
    """Minimal storage contract used by the unique corpus builder."""

    def ensure_schema(self) -> None: ...

    def split_counts(self, namespace: str, *, labeled_only: bool = False) -> dict[str, int]: ...

    def reserve_sample(
        self,
        namespace: str,
        *,
        fen_hash: str,
        sample_id: str,
        fen: str,
        split: str,
        source: str,
        result: str | None,
        metadata: dict[str, Any],
    ) -> bool: ...

    def iter_reserved_samples(self, namespace: str) -> Iterator[ReservedLabelCorpusSample]: ...

    def load_reserved_sample(
        self,
        namespace: str,
        *,
        fen_hash: str,
    ) -> ReservedLabelCorpusSample | None: ...

    def delete_reserved_sample(self, namespace: str, *, fen_hash: str) -> None: ...

    def mark_sample_labeled(
        self,
        namespace: str,
        *,
        fen_hash: str,
        selected_move_uci: str,
        metadata: dict[str, Any],
    ) -> bool: ...

    def iter_labeled_samples(self, namespace: str, *, split: str) -> Iterator[LabelCorpusSample]: ...

    def close(self) -> None: ...


class MySQLLabelCorpusLedger:
    """Persistent MySQL-backed ledger used instead of local SQLite state."""

    def __init__(self, config: MySQLConfig) -> None:
        self._config = config
        self._connection: pymysql.connections.Connection | None = None
        self._known_namespaces: set[str] = set()

    def ensure_schema(self) -> None:
        self._execute_write_batch(CREATE_LABEL_CORPUS_TABLES)

    def split_counts(self, namespace: str, *, labeled_only: bool = False) -> dict[str, int]:
        namespace_hash = label_corpus_namespace_hash(namespace)
        sql = """
            SELECT sample_split, COUNT(*) AS count
            FROM label_corpus_samples
            WHERE namespace_hash = %s
        """
        parameters: list[Any] = [namespace_hash]
        if labeled_only:
            sql += " AND status = %s"
            parameters.append("labeled")
        else:
            sql += " AND status IN (%s, %s)"
            parameters.extend(["reserved", "labeled"])
        sql += " GROUP BY sample_split"
        counts = {"train": 0, "verify": 0}
        with self._cursor() as cursor:
            cursor.execute(sql, tuple(parameters))
            rows = cursor.fetchall()
        for row in rows:
            counts[str(row["sample_split"])] = int(row["count"])
        return counts

    def reserve_sample(
        self,
        namespace: str,
        *,
        fen_hash: str,
        sample_id: str,
        fen: str,
        split: str,
        source: str,
        result: str | None,
        metadata: dict[str, Any],
    ) -> bool:
        namespace_hash = self._ensure_namespace(namespace)

        def operation() -> bool:
            with self._cursor() as cursor:
                cursor.execute(
                    """
                    INSERT IGNORE INTO label_corpus_samples (
                        namespace_hash,
                        fen_hash,
                        fen,
                        sample_split,
                        sample_id,
                        source,
                        result,
                        metadata_json,
                        selected_move_uci,
                        status
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NULL, 'reserved')
                    """,
                    (
                        namespace_hash,
                        fen_hash,
                        fen,
                        split,
                        sample_id,
                        source,
                        result,
                        json.dumps(metadata, sort_keys=True),
                    ),
                )
                inserted = int(cursor.rowcount) > 0
            self._connection_or_raise().commit()
            return inserted

        return self._with_retry(operation)

    def iter_reserved_samples(self, namespace: str) -> Iterator[ReservedLabelCorpusSample]:
        namespace_hash = label_corpus_namespace_hash(namespace)
        with self._stream_cursor() as cursor:
            cursor.execute(
                """
                SELECT fen_hash, fen, sample_split, metadata_json
                FROM label_corpus_samples
                WHERE namespace_hash = %s
                  AND status = 'reserved'
                ORDER BY created_at ASC, fen_hash ASC
                """,
                (namespace_hash,),
            )
            while True:
                rows = cursor.fetchmany(512)
                if not rows:
                    break
                for row in rows:
                    yield ReservedLabelCorpusSample(
                        fen_hash=str(row["fen_hash"]),
                        fen=str(row["fen"]),
                        split=str(row["sample_split"]),
                        metadata=_decode_json_value(row["metadata_json"]) or {},
                    )

    def load_reserved_sample(
        self,
        namespace: str,
        *,
        fen_hash: str,
    ) -> ReservedLabelCorpusSample | None:
        namespace_hash = label_corpus_namespace_hash(namespace)
        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT fen_hash, fen, sample_split, metadata_json
                FROM label_corpus_samples
                WHERE namespace_hash = %s
                  AND fen_hash = %s
                  AND status = 'reserved'
                """,
                (namespace_hash, fen_hash),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return ReservedLabelCorpusSample(
            fen_hash=str(row["fen_hash"]),
            fen=str(row["fen"]),
            split=str(row["sample_split"]),
            metadata=_decode_json_value(row["metadata_json"]) or {},
        )

    def delete_reserved_sample(self, namespace: str, *, fen_hash: str) -> None:
        namespace_hash = label_corpus_namespace_hash(namespace)

        def operation() -> None:
            with self._cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM label_corpus_samples
                    WHERE namespace_hash = %s
                      AND fen_hash = %s
                      AND status = 'reserved'
                    """,
                    (namespace_hash, fen_hash),
                )
            self._connection_or_raise().commit()

        self._with_retry(operation)

    def mark_sample_labeled(
        self,
        namespace: str,
        *,
        fen_hash: str,
        selected_move_uci: str,
        metadata: dict[str, Any],
    ) -> bool:
        namespace_hash = label_corpus_namespace_hash(namespace)

        def operation() -> bool:
            with self._cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE label_corpus_samples
                    SET
                        selected_move_uci = %s,
                        metadata_json = %s,
                        status = 'labeled'
                    WHERE namespace_hash = %s
                      AND fen_hash = %s
                      AND status = 'reserved'
                    """,
                    (
                        selected_move_uci,
                        json.dumps(metadata, sort_keys=True),
                        namespace_hash,
                        fen_hash,
                    ),
                )
                updated = int(cursor.rowcount) > 0
            self._connection_or_raise().commit()
            return updated

        return self._with_retry(operation)

    def iter_labeled_samples(self, namespace: str, *, split: str) -> Iterator[LabelCorpusSample]:
        namespace_hash = label_corpus_namespace_hash(namespace)
        with self._stream_cursor() as cursor:
            cursor.execute(
                """
                SELECT sample_id, fen, sample_split, source, result, metadata_json, selected_move_uci
                FROM label_corpus_samples
                WHERE namespace_hash = %s
                  AND sample_split = %s
                  AND status = 'labeled'
                ORDER BY fen_hash ASC
                """,
                (namespace_hash, split),
            )
            while True:
                rows = cursor.fetchmany(512)
                if not rows:
                    break
                for row in rows:
                    selected_move_uci = row["selected_move_uci"]
                    if selected_move_uci is None:
                        continue
                    yield LabelCorpusSample(
                        sample_id=str(row["sample_id"]),
                        fen=str(row["fen"]),
                        split=str(row["sample_split"]),
                        source=str(row["source"]),
                        result=(str(row["result"]) if row["result"] is not None else None),
                        metadata=_decode_json_value(row["metadata_json"]) or {},
                        selected_move_uci=str(selected_move_uci),
                    )

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _ensure_namespace(self, namespace: str) -> str:
        namespace_hash = label_corpus_namespace_hash(namespace)
        if namespace_hash in self._known_namespaces:
            return namespace_hash

        def operation() -> None:
            with self._cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO label_corpus_namespaces (namespace_hash, namespace)
                    VALUES (%s, %s)
                    ON DUPLICATE KEY UPDATE namespace = VALUES(namespace)
                    """,
                    (namespace_hash, namespace),
                )
            self._connection_or_raise().commit()

        self._with_retry(operation)
        self._known_namespaces.add(namespace_hash)
        return namespace_hash

    def _execute_write_batch(self, statements: tuple[str, ...]) -> None:
        def operation() -> None:
            with self._cursor() as cursor:
                for statement in statements:
                    cursor.execute(statement)
            self._connection_or_raise().commit()

        self._with_retry(operation)

    def _with_retry(self, operation: Any) -> Any:
        max_retries = 3
        for attempt_index in range(max_retries):
            try:
                return operation()
            except (pymysql.err.InternalError, pymysql.err.OperationalError) as exc:
                self._rollback_quietly()
                error_code = int(exc.args[0]) if exc.args else 0
                if error_code not in {1205, 1213} or attempt_index + 1 >= max_retries:
                    raise
                time.sleep(0.05 * (attempt_index + 1))

    def _cursor(self) -> Any:
        connection = self._connection_or_raise()
        connection.ping(reconnect=True)
        return connection.cursor()

    def _stream_cursor(self) -> Any:
        connection = self._connection_or_raise()
        connection.ping(reconnect=True)
        return connection.cursor(cursor=SSDictCursor)

    def _connection_or_raise(self) -> pymysql.connections.Connection:
        if self._connection is None:
            self._connection = pymysql.connect(
                host=self._config.host,
                user=self._config.user,
                password=self._config.password,
                database=self._config.database,
                port=self._config.port,
                connect_timeout=self._config.connect_timeout_seconds,
                read_timeout=self._config.read_timeout_seconds,
                write_timeout=self._config.write_timeout_seconds,
                autocommit=False,
                cursorclass=DictCursor,
            )
        return self._connection

    def _rollback_quietly(self) -> None:
        if self._connection is None:
            return
        try:
            self._connection.rollback()
        except Exception:
            return


def label_corpus_namespace_hash(namespace: str) -> str:
    """Return the stable namespace hash used in the MySQL primary key."""
    return hashlib.sha256(namespace.encode("utf-8")).hexdigest()


def _decode_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (dict, list)):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    return json.loads(str(value))
