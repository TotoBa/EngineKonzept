"""Track per-lineage raw-sample reuse so follow-up generations can prefer fresh data."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Any, Protocol, Sequence

import pymysql
from pymysql.cursors import DictCursor

from train.datasets.schema import RawPositionRecord
from train.orchestrator.models import MySQLConfig


CREATE_TRAINING_DATA_USAGE_TABLES = (
    """
    CREATE TABLE IF NOT EXISTS lineage_generation_usage (
        master_name VARCHAR(255) NOT NULL,
        lineage_name VARCHAR(255) NOT NULL,
        generation INT NOT NULL,
        campaign_id BIGINT UNSIGNED NOT NULL,
        model_id BIGINT UNSIGNED NULL,
        merged_raw_dir VARCHAR(1024) NOT NULL,
        train_record_count INT NOT NULL DEFAULT 0,
        verify_record_count INT NOT NULL DEFAULT 0,
        applied_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
        PRIMARY KEY (master_name, lineage_name, generation),
        KEY idx_lineage_generation_usage_campaign (campaign_id)
    ) ENGINE=InnoDB
    """,
    """
    CREATE TABLE IF NOT EXISTS lineage_sample_usage (
        master_name VARCHAR(255) NOT NULL,
        lineage_name VARCHAR(255) NOT NULL,
        sample_split VARCHAR(16) NOT NULL,
        fen_hash CHAR(64) NOT NULL,
        sample_id VARCHAR(255) NOT NULL DEFAULT '',
        sample_source VARCHAR(255) NOT NULL DEFAULT '',
        usage_count INT NOT NULL DEFAULT 0,
        first_generation INT NOT NULL,
        last_generation INT NOT NULL,
        last_campaign_id BIGINT UNSIGNED NULL,
        last_model_id BIGINT UNSIGNED NULL,
        updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
            ON UPDATE CURRENT_TIMESTAMP(6),
        PRIMARY KEY (master_name, lineage_name, sample_split, fen_hash),
        KEY idx_lineage_sample_usage_rank (
            master_name,
            lineage_name,
            sample_split,
            usage_count,
            last_generation
        )
    ) ENGINE=InnoDB
    """,
)

TRAINING_DATA_USAGE_SCHEMA_MIGRATIONS = (
    """
    ALTER TABLE lineage_sample_usage
        MODIFY COLUMN sample_id VARCHAR(255) NOT NULL DEFAULT '',
        MODIFY COLUMN sample_source VARCHAR(255) NOT NULL DEFAULT ''
    """,
)

_USAGE_BATCH_SIZE = 1000
_RETRYABLE_LEDGER_ERROR_CODES = {1205, 1213, 2006, 2013, 2055}


@dataclass(frozen=True)
class LineageSampleUsageState:
    """Reuse state for one unique raw sample position within one lineage."""

    usage_count: int = 0
    last_generation: int = 0


class LineageTrainingUsageLedger(Protocol):
    """Minimal ledger contract used by the master for usage-aware data selection."""

    def ensure_schema(self) -> None: ...

    def generation_usage(
        self,
        *,
        master_name: str,
        lineage_name: str,
        generation: int,
    ) -> dict[str, Any] | None: ...

    def usage_state(
        self,
        *,
        master_name: str,
        lineage_name: str,
        split_name: str,
        fen_hashes: Sequence[str],
    ) -> dict[str, LineageSampleUsageState]: ...

    def record_generation_usage(
        self,
        *,
        master_name: str,
        lineage_name: str,
        generation: int,
        campaign_id: int,
        model_id: int | None,
        merged_raw_dir: str,
        train_records: Sequence[RawPositionRecord],
        verify_records: Sequence[RawPositionRecord],
    ) -> dict[str, Any]: ...

    def close(self) -> None: ...


class InMemoryLineageTrainingUsageLedger:
    """Simple test ledger used when the master has no real MySQL config."""

    def __init__(self) -> None:
        self._generation_rows: dict[tuple[str, str, int], dict[str, Any]] = {}
        self._sample_rows: dict[tuple[str, str, str, str], LineageSampleUsageState] = {}

    def ensure_schema(self) -> None:
        return None

    def generation_usage(
        self,
        *,
        master_name: str,
        lineage_name: str,
        generation: int,
    ) -> dict[str, Any] | None:
        row = self._generation_rows.get((master_name, lineage_name, generation))
        return dict(row) if row is not None else None

    def usage_state(
        self,
        *,
        master_name: str,
        lineage_name: str,
        split_name: str,
        fen_hashes: Sequence[str],
    ) -> dict[str, LineageSampleUsageState]:
        return {
            fen_hash: self._sample_rows[(master_name, lineage_name, split_name, fen_hash)]
            for fen_hash in fen_hashes
            if (master_name, lineage_name, split_name, fen_hash) in self._sample_rows
        }

    def record_generation_usage(
        self,
        *,
        master_name: str,
        lineage_name: str,
        generation: int,
        campaign_id: int,
        model_id: int | None,
        merged_raw_dir: str,
        train_records: Sequence[RawPositionRecord],
        verify_records: Sequence[RawPositionRecord],
    ) -> dict[str, Any]:
        existing = self.generation_usage(
            master_name=master_name,
            lineage_name=lineage_name,
            generation=generation,
        )
        if existing is not None:
            return existing
        self._apply_records(
            master_name=master_name,
            lineage_name=lineage_name,
            split_name="train",
            records=train_records,
            generation=generation,
            campaign_id=campaign_id,
            model_id=model_id,
        )
        self._apply_records(
            master_name=master_name,
            lineage_name=lineage_name,
            split_name="verify",
            records=verify_records,
            generation=generation,
            campaign_id=campaign_id,
            model_id=model_id,
        )
        row = {
            "master_name": master_name,
            "lineage_name": lineage_name,
            "generation": generation,
            "campaign_id": campaign_id,
            "model_id": model_id,
            "merged_raw_dir": merged_raw_dir,
            "train_record_count": len(train_records),
            "verify_record_count": len(verify_records),
            "already_recorded": False,
        }
        self._generation_rows[(master_name, lineage_name, generation)] = dict(row)
        return row

    def close(self) -> None:
        return None

    def _apply_records(
        self,
        *,
        master_name: str,
        lineage_name: str,
        split_name: str,
        records: Sequence[RawPositionRecord],
        generation: int,
        campaign_id: int,
        model_id: int | None,
    ) -> None:
        del campaign_id, model_id
        seen_hashes: set[str] = set()
        for record in records:
            fen_hash = lineage_training_usage_fen_hash(record.fen)
            if fen_hash in seen_hashes:
                continue
            seen_hashes.add(fen_hash)
            key = (master_name, lineage_name, split_name, fen_hash)
            current = self._sample_rows.get(key)
            self._sample_rows[key] = LineageSampleUsageState(
                usage_count=(0 if current is None else int(current.usage_count)) + 1,
                last_generation=generation,
            )


class MySQLLineageTrainingUsageLedger:
    """Persistent lineage-level reuse tracker stored in the MySQL control plane."""

    def __init__(self, config: MySQLConfig) -> None:
        self._config = config
        self._connection: pymysql.connections.Connection | None = None
        self._schema_ensured = False

    def ensure_schema(self) -> None:
        if self._schema_ensured:
            return
        def operation() -> None:
            with self._cursor() as cursor:
                for statement in CREATE_TRAINING_DATA_USAGE_TABLES:
                    cursor.execute(statement)
                for statement in TRAINING_DATA_USAGE_SCHEMA_MIGRATIONS:
                    cursor.execute(statement)
            self._connection_or_raise().commit()

        self._with_retry(operation)
        self._schema_ensured = True

    def generation_usage(
        self,
        *,
        master_name: str,
        lineage_name: str,
        generation: int,
    ) -> dict[str, Any] | None:
        self.ensure_schema()
        row = self._with_retry(
            lambda: self._generation_usage_row(
                master_name=master_name,
                lineage_name=lineage_name,
                generation=generation,
            )
        )
        if row is None:
            return None
        return {
            "master_name": str(row["master_name"]),
            "lineage_name": str(row["lineage_name"]),
            "generation": int(row["generation"]),
            "campaign_id": int(row["campaign_id"]),
            "model_id": (int(row["model_id"]) if row["model_id"] is not None else None),
            "merged_raw_dir": str(row["merged_raw_dir"]),
            "train_record_count": int(row["train_record_count"]),
            "verify_record_count": int(row["verify_record_count"]),
            "already_recorded": True,
        }

    def usage_state(
        self,
        *,
        master_name: str,
        lineage_name: str,
        split_name: str,
        fen_hashes: Sequence[str],
    ) -> dict[str, LineageSampleUsageState]:
        if not fen_hashes:
            return {}
        self.ensure_schema()
        return self._with_retry(
            lambda: self._usage_state_rows(
                master_name=master_name,
                lineage_name=lineage_name,
                split_name=split_name,
                fen_hashes=fen_hashes,
            )
        )

    def record_generation_usage(
        self,
        *,
        master_name: str,
        lineage_name: str,
        generation: int,
        campaign_id: int,
        model_id: int | None,
        merged_raw_dir: str,
        train_records: Sequence[RawPositionRecord],
        verify_records: Sequence[RawPositionRecord],
    ) -> dict[str, Any]:
        self.ensure_schema()
        def operation() -> dict[str, Any]:
            existing = self._generation_usage(
                master_name=master_name,
                lineage_name=lineage_name,
                generation=generation,
            )
            if existing is not None:
                return existing
            try:
                self._record_split_usage(
                    master_name=master_name,
                    lineage_name=lineage_name,
                    split_name="train",
                    generation=generation,
                    campaign_id=campaign_id,
                    model_id=model_id,
                    records=train_records,
                )
                self._record_split_usage(
                    master_name=master_name,
                    lineage_name=lineage_name,
                    split_name="verify",
                    generation=generation,
                    campaign_id=campaign_id,
                    model_id=model_id,
                    records=verify_records,
                )
                with self._cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO lineage_generation_usage (
                            master_name,
                            lineage_name,
                            generation,
                            campaign_id,
                            model_id,
                            merged_raw_dir,
                            train_record_count,
                            verify_record_count
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            master_name,
                            lineage_name,
                            generation,
                            campaign_id,
                            model_id,
                            merged_raw_dir,
                            len(train_records),
                            len(verify_records),
                        ),
                    )
                self._connection_or_raise().commit()
            except Exception:
                self._rollback_quietly()
                raise
            return {
                "master_name": master_name,
                "lineage_name": lineage_name,
                "generation": generation,
                "campaign_id": campaign_id,
                "model_id": model_id,
                "merged_raw_dir": merged_raw_dir,
                "train_record_count": len(train_records),
                "verify_record_count": len(verify_records),
                "already_recorded": False,
            }

        return self._with_retry(operation)

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _record_split_usage(
        self,
        *,
        master_name: str,
        lineage_name: str,
        split_name: str,
        generation: int,
        campaign_id: int,
        model_id: int | None,
        records: Sequence[RawPositionRecord],
    ) -> None:
        unique_hashes: set[str] = set()
        for record in records:
            fen_hash = lineage_training_usage_fen_hash(record.fen)
            unique_hashes.add(fen_hash)
        if not unique_hashes:
            return
        query = """
            INSERT INTO lineage_sample_usage (
                master_name,
                lineage_name,
                sample_split,
                fen_hash,
                usage_count,
                first_generation,
                last_generation,
                last_campaign_id,
                last_model_id
            )
            VALUES (%s, %s, %s, %s, 1, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                usage_count = usage_count + 1,
                last_generation = VALUES(last_generation),
                last_campaign_id = VALUES(last_campaign_id),
                last_model_id = VALUES(last_model_id)
        """
        rows = [
            (
                master_name,
                lineage_name,
                split_name,
                fen_hash,
                generation,
                generation,
                campaign_id,
                model_id,
            )
            for fen_hash in sorted(unique_hashes)
        ]
        with self._cursor() as cursor:
            for offset in range(0, len(rows), _USAGE_BATCH_SIZE):
                cursor.executemany(query, rows[offset : offset + _USAGE_BATCH_SIZE])

    def _generation_usage(
        self,
        *,
        master_name: str,
        lineage_name: str,
        generation: int,
    ) -> dict[str, Any] | None:
        row = self._generation_usage_row(
            master_name=master_name,
            lineage_name=lineage_name,
            generation=generation,
        )
        if row is None:
            return None
        return {
            "master_name": str(row["master_name"]),
            "lineage_name": str(row["lineage_name"]),
            "generation": int(row["generation"]),
            "campaign_id": int(row["campaign_id"]),
            "model_id": (int(row["model_id"]) if row["model_id"] is not None else None),
            "merged_raw_dir": str(row["merged_raw_dir"]),
            "train_record_count": int(row["train_record_count"]),
            "verify_record_count": int(row["verify_record_count"]),
            "already_recorded": True,
        }

    def _generation_usage_row(
        self,
        *,
        master_name: str,
        lineage_name: str,
        generation: int,
    ) -> dict[str, Any] | None:
        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    master_name,
                    lineage_name,
                    generation,
                    campaign_id,
                    model_id,
                    merged_raw_dir,
                    train_record_count,
                    verify_record_count
                FROM lineage_generation_usage
                WHERE master_name = %s
                  AND lineage_name = %s
                  AND generation = %s
                """,
                (master_name, lineage_name, generation),
            )
            return cursor.fetchone()

    def _usage_state_rows(
        self,
        *,
        master_name: str,
        lineage_name: str,
        split_name: str,
        fen_hashes: Sequence[str],
    ) -> dict[str, LineageSampleUsageState]:
        rows_by_hash: dict[str, LineageSampleUsageState] = {}
        with self._cursor() as cursor:
            for offset in range(0, len(fen_hashes), _USAGE_BATCH_SIZE):
                batch = list(fen_hashes[offset : offset + _USAGE_BATCH_SIZE])
                placeholders = ", ".join(["%s"] * len(batch))
                cursor.execute(
                    f"""
                    SELECT fen_hash, usage_count, last_generation
                    FROM lineage_sample_usage
                    WHERE master_name = %s
                      AND lineage_name = %s
                      AND sample_split = %s
                      AND fen_hash IN ({placeholders})
                    """,
                    (master_name, lineage_name, split_name, *batch),
                )
                for row in cursor.fetchall():
                    rows_by_hash[str(row["fen_hash"])] = LineageSampleUsageState(
                        usage_count=int(row["usage_count"]),
                        last_generation=int(row["last_generation"]),
                    )
        return rows_by_hash

    def _connection_or_raise(self) -> pymysql.connections.Connection:
        if self._connection is None or not self._connection.open:
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

    def _cursor(self):
        connection = self._connection_or_raise()
        connection.ping(reconnect=True)
        return connection.cursor()

    def _with_retry(self, operation: Any) -> Any:
        max_retries = 3
        for attempt_index in range(max_retries):
            try:
                return operation()
            except (
                pymysql.err.InterfaceError,
                pymysql.err.InternalError,
                pymysql.err.OperationalError,
            ) as exc:
                self._rollback_quietly()
                error_code = int(exc.args[0]) if exc.args else 0
                if error_code not in _RETRYABLE_LEDGER_ERROR_CODES or attempt_index + 1 >= max_retries:
                    raise
                self._reset_connection()
                time.sleep(0.05 * (attempt_index + 1))
        raise AssertionError("unreachable")

    def _rollback_quietly(self) -> None:
        if self._connection is None:
            return
        try:
            self._connection.rollback()
        except Exception:
            return

    def _reset_connection(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                pass
        self._connection = None


def lineage_training_usage_fen_hash(fen: str) -> str:
    """Return the stable ledger key for one raw position."""
    return hashlib.sha256(fen.encode("utf-8")).hexdigest()
