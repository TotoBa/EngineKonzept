"""MySQL schema and query helpers for the distributed training control plane."""

from __future__ import annotations

from contextlib import contextmanager
import json
import time
from typing import Any, Mapping, Sequence

import pymysql
from pymysql.cursors import DictCursor

from train.orchestrator.models import (
    ArtifactRef,
    CampaignRow,
    ModelRow,
    MySQLConfig,
    PlannedTask,
    TASK_STATE_FAILED,
    TASK_STATE_LEASED,
    TASK_STATE_QUEUED,
    TASK_STATE_SUCCEEDED,
    TaskResult,
    TaskRow,
)

_CREATE_TABLES = (
    """
    CREATE TABLE IF NOT EXISTS campaigns (
        id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        kind VARCHAR(64) NOT NULL,
        status VARCHAR(32) NOT NULL,
        config_path VARCHAR(1024) NOT NULL,
        active_model_id BIGINT UNSIGNED NULL,
        metadata_json JSON NULL,
        created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
        updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
            ON UPDATE CURRENT_TIMESTAMP(6),
        KEY idx_campaigns_kind_status (kind, status)
    ) ENGINE=InnoDB
    """,
    """
    CREATE TABLE IF NOT EXISTS models (
        id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
        campaign_id BIGINT UNSIGNED NOT NULL,
        parent_model_id BIGINT UNSIGNED NULL,
        generation INT NOT NULL DEFAULT 0,
        train_config_path VARCHAR(1024) NULL,
        agent_spec_path VARCHAR(1024) NULL,
        checkpoint_path VARCHAR(1024) NULL,
        bundle_path VARCHAR(1024) NULL,
        verify_json_path VARCHAR(1024) NULL,
        arena_summary_path VARCHAR(1024) NULL,
        status VARCHAR(32) NOT NULL,
        promotion_score DOUBLE NULL,
        metadata_json JSON NULL,
        created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
        KEY idx_models_campaign_status (campaign_id, status)
    ) ENGINE=InnoDB
    """,
    """
    CREATE TABLE IF NOT EXISTS tasks (
        id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
        campaign_id BIGINT UNSIGNED NOT NULL,
        model_id BIGINT UNSIGNED NULL,
        task_type VARCHAR(64) NOT NULL,
        capability VARCHAR(64) NOT NULL,
        priority INT NOT NULL DEFAULT 0,
        state VARCHAR(32) NOT NULL,
        payload_json JSON NOT NULL,
        result_json JSON NULL,
        worker_id VARCHAR(128) NULL,
        lease_until TIMESTAMP(6) NULL,
        attempt_count INT NOT NULL DEFAULT 0,
        max_attempts INT NOT NULL DEFAULT 1,
        depends_on_count INT NOT NULL DEFAULT 0,
        not_before TIMESTAMP(6) NULL,
        created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
        updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
            ON UPDATE CURRENT_TIMESTAMP(6),
        KEY idx_tasks_claim (state, capability, priority, id),
        KEY idx_tasks_campaign_state (campaign_id, state),
        KEY idx_tasks_lease (state, lease_until),
        KEY idx_tasks_worker (worker_id)
    ) ENGINE=InnoDB
    """,
    """
    CREATE TABLE IF NOT EXISTS task_dependencies (
        task_id BIGINT UNSIGNED NOT NULL,
        depends_on_task_id BIGINT UNSIGNED NOT NULL,
        PRIMARY KEY (task_id, depends_on_task_id),
        KEY idx_task_dependencies_parent (depends_on_task_id)
    ) ENGINE=InnoDB
    """,
    """
    CREATE TABLE IF NOT EXISTS task_attempts (
        id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
        task_id BIGINT UNSIGNED NOT NULL,
        worker_id VARCHAR(128) NOT NULL,
        started_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
        ended_at TIMESTAMP(6) NULL,
        exit_code INT NULL,
        stdout_path VARCHAR(1024) NULL,
        stderr_path VARCHAR(1024) NULL,
        result_summary_path VARCHAR(1024) NULL,
        details_json JSON NULL,
        KEY idx_task_attempts_task (task_id),
        KEY idx_task_attempts_worker (worker_id)
    ) ENGINE=InnoDB
    """,
    """
    CREATE TABLE IF NOT EXISTS workers (
        id VARCHAR(128) NOT NULL PRIMARY KEY,
        hostname VARCHAR(255) NOT NULL,
        capabilities_json JSON NOT NULL,
        scratch_root VARCHAR(1024) NOT NULL,
        status VARCHAR(32) NOT NULL,
        current_task_id BIGINT UNSIGNED NULL,
        lease_until TIMESTAMP(6) NULL,
        last_heartbeat_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
        version VARCHAR(64) NOT NULL,
        metadata_json JSON NULL
    ) ENGINE=InnoDB
    """,
    """
    CREATE TABLE IF NOT EXISTS artifacts (
        id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
        kind VARCHAR(64) NOT NULL,
        path VARCHAR(700) NOT NULL,
        sha256 CHAR(64) NULL,
        size_bytes BIGINT NULL,
        producer_task_id BIGINT UNSIGNED NULL,
        state VARCHAR(32) NOT NULL,
        metadata_json JSON NULL,
        created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
        UNIQUE KEY idx_artifacts_kind_path (kind, path)
    ) ENGINE=InnoDB
    """,
    """
    CREATE TABLE IF NOT EXISTS arena_matches (
        id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
        campaign_id BIGINT UNSIGNED NOT NULL,
        model_a VARCHAR(255) NOT NULL,
        model_b VARCHAR(255) NOT NULL,
        opening_id VARCHAR(255) NOT NULL,
        color_assignment VARCHAR(32) NOT NULL,
        seed BIGINT NULL,
        task_id BIGINT UNSIGNED NULL,
        result VARCHAR(16) NULL,
        game_record_path VARCHAR(1024) NULL,
        finished_at TIMESTAMP(6) NULL,
        KEY idx_arena_matches_campaign (campaign_id)
    ) ENGINE=InnoDB
    """,
)


class OrchestratorDB:
    """Thin DB wrapper over PyMySQL with explicit JSON encoding/decoding."""

    def __init__(self, config: MySQLConfig) -> None:
        self._config = config

    @property
    def config(self) -> MySQLConfig:
        return self._config

    @contextmanager
    def _connect(self) -> Any:
        connection = pymysql.connect(
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
        try:
            yield connection
        finally:
            connection.close()

    def init_schema(self, *, drop_existing: bool = False) -> None:
        """Create the control-plane schema, optionally dropping all tracked tables first."""
        with self._connect() as connection:
            try:
                with connection.cursor() as cursor:
                    if drop_existing:
                        cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
                        cursor.execute("SHOW TABLES")
                        table_names = [next(iter(row.values())) for row in cursor.fetchall()]
                        for table_name in table_names:
                            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                        cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
                    for statement in _CREATE_TABLES:
                        cursor.execute(statement)
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def insert_campaign(
        self,
        *,
        name: str,
        kind: str,
        status: str,
        config_path: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Insert one campaign row and return its id."""
        with self._connect() as connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO campaigns (name, kind, status, config_path, metadata_json)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            name,
                            kind,
                            status,
                            config_path,
                            _json_or_none(metadata),
                        ),
                    )
                    campaign_id = int(cursor.lastrowid)
                connection.commit()
                return campaign_id
            except Exception:
                connection.rollback()
                raise

    def insert_model(
        self,
        *,
        campaign_id: int,
        status: str,
        generation: int = 0,
        parent_model_id: int | None = None,
        train_config_path: str | None = None,
        agent_spec_path: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Insert one model row and return its id."""
        with self._connect() as connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO models (
                            campaign_id,
                            parent_model_id,
                            generation,
                            train_config_path,
                            agent_spec_path,
                            status,
                            metadata_json
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            campaign_id,
                            parent_model_id,
                            generation,
                            train_config_path,
                            agent_spec_path,
                            status,
                            _json_or_none(metadata),
                        ),
                    )
                    model_id = int(cursor.lastrowid)
                connection.commit()
                return model_id
            except Exception:
                connection.rollback()
                raise

    def load_campaign(self, campaign_id: int) -> CampaignRow | None:
        """Load one normalized campaign row."""
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM campaigns WHERE id = %s", (campaign_id,))
                row = cursor.fetchone()
        if row is None:
            return None
        return CampaignRow.from_db_row(_decode_campaign_row(row))

    def list_campaign_records(
        self,
        *,
        kind: str | None = None,
        status: str | None = None,
        limit: int = 1000,
    ) -> list[CampaignRow]:
        """List normalized campaign rows ordered by creation id."""
        where_clauses: list[str] = []
        parameters: list[Any] = []
        if kind is not None:
            where_clauses.append("kind = %s")
            parameters.append(kind)
        if status is not None:
            where_clauses.append("status = %s")
            parameters.append(status)
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT *
                    FROM campaigns
                    {where_sql}
                    ORDER BY id ASC
                    LIMIT %s
                    """,
                    (*parameters, limit),
                )
                rows = cursor.fetchall()
        return [CampaignRow.from_db_row(_decode_campaign_row(row)) for row in rows]

    def load_model(self, model_id: int) -> ModelRow | None:
        """Load one normalized model row."""
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM models WHERE id = %s", (model_id,))
                row = cursor.fetchone()
        if row is None:
            return None
        return ModelRow.from_db_row(_decode_model_row(row))

    def list_model_records(
        self,
        *,
        campaign_id: int | None = None,
        status: str | None = None,
        limit: int = 1000,
    ) -> list[ModelRow]:
        """List normalized model rows ordered by creation id."""
        where_clauses: list[str] = []
        parameters: list[Any] = []
        if campaign_id is not None:
            where_clauses.append("campaign_id = %s")
            parameters.append(campaign_id)
        if status is not None:
            where_clauses.append("status = %s")
            parameters.append(status)
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT *
                    FROM models
                    {where_sql}
                    ORDER BY id ASC
                    LIMIT %s
                    """,
                    (*parameters, limit),
                )
                rows = cursor.fetchall()
        return [ModelRow.from_db_row(_decode_model_row(row)) for row in rows]

    def load_task(self, task_id: int) -> TaskRow | None:
        """Load one normalized task row."""
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM tasks WHERE id = %s", (task_id,))
                row = cursor.fetchone()
        if row is None:
            return None
        return TaskRow.from_db_row(_decode_task_row(row))

    def insert_planned_tasks(
        self,
        *,
        campaign_id: int,
        model_id: int | None,
        planned_tasks: Sequence[PlannedTask],
        extra_dependency_task_ids: Sequence[int] = (),
    ) -> dict[str, int]:
        """Insert a task batch plus dependency edges and return task ids by key."""
        with self._connect() as connection:
            try:
                key_to_id: dict[str, int] = {}
                with connection.cursor() as cursor:
                    for planned in planned_tasks:
                        depends_on_count = len(planned.depends_on) + len(extra_dependency_task_ids)
                        cursor.execute(
                            """
                            INSERT INTO tasks (
                                campaign_id,
                                model_id,
                                task_type,
                                capability,
                                priority,
                                state,
                                payload_json,
                                max_attempts,
                                depends_on_count
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                campaign_id,
                                model_id,
                                planned.task_type,
                                planned.capability,
                                planned.priority,
                                TASK_STATE_QUEUED,
                                json.dumps(planned.payload, sort_keys=True),
                                planned.max_attempts,
                                depends_on_count,
                            ),
                        )
                        key_to_id[planned.key] = int(cursor.lastrowid)

                    dependency_rows: list[tuple[int, int]] = []
                    for planned in planned_tasks:
                        task_id = key_to_id[planned.key]
                        dependency_rows.extend(
                            (task_id, dependency_id) for dependency_id in extra_dependency_task_ids
                        )
                        dependency_rows.extend(
                            (task_id, key_to_id[parent_key]) for parent_key in planned.depends_on
                        )
                    if dependency_rows:
                        cursor.executemany(
                            """
                            INSERT INTO task_dependencies (task_id, depends_on_task_id)
                            VALUES (%s, %s)
                            """,
                            dependency_rows,
                        )
                connection.commit()
                return key_to_id
            except Exception:
                connection.rollback()
                raise

    def register_worker(
        self,
        *,
        worker_id: str,
        hostname: str,
        capabilities: Sequence[str],
        scratch_root: str,
        version: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Insert or refresh one worker row."""
        with self._connect() as connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO workers (
                            id,
                            hostname,
                            capabilities_json,
                            scratch_root,
                            status,
                            version,
                            metadata_json
                        )
                        VALUES (%s, %s, %s, %s, 'idle', %s, %s)
                        ON DUPLICATE KEY UPDATE
                            hostname = VALUES(hostname),
                            capabilities_json = VALUES(capabilities_json),
                            scratch_root = VALUES(scratch_root),
                            version = VALUES(version),
                            metadata_json = VALUES(metadata_json),
                            last_heartbeat_at = CURRENT_TIMESTAMP(6)
                        """,
                        (
                            worker_id,
                            hostname,
                            json.dumps(list(capabilities), sort_keys=True),
                            scratch_root,
                            version,
                            _json_or_none(metadata),
                        ),
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def heartbeat_worker(
        self,
        *,
        worker_id: str,
        status: str,
        current_task_id: int | None,
        lease_seconds: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Refresh worker liveness and current-task information."""
        with self._connect() as connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE workers
                        SET
                            status = %s,
                            current_task_id = %s,
                            lease_until = (
                                CASE
                                    WHEN %s IS NULL THEN NULL
                                    ELSE TIMESTAMPADD(SECOND, %s, UTC_TIMESTAMP(6))
                                END
                            ),
                            metadata_json = COALESCE(%s, metadata_json),
                            last_heartbeat_at = CURRENT_TIMESTAMP(6)
                        WHERE id = %s
                        """,
                        (
                            status,
                            current_task_id,
                            lease_seconds,
                            lease_seconds,
                            _json_or_none(metadata),
                            worker_id,
                        ),
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def claim_tasks(
        self,
        *,
        worker_id: str,
        capabilities: Sequence[str],
        lease_seconds: int,
        limit: int = 1,
    ) -> list[TaskRow]:
        """Atomically claim eligible queued tasks for one worker."""
        if not capabilities:
            raise ValueError("claim_tasks requires at least one capability")
        placeholders = ", ".join(["%s"] * len(capabilities))
        with self._connect() as connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
                    connection.begin()
                    cursor.execute(
                        f"""
                        SELECT *
                        FROM tasks
                        WHERE state = %s
                          AND capability IN ({placeholders})
                          AND (not_before IS NULL OR not_before <= UTC_TIMESTAMP(6))
                          AND NOT EXISTS (
                              SELECT 1
                              FROM task_dependencies dependency
                              INNER JOIN tasks parent
                                  ON parent.id = dependency.depends_on_task_id
                              WHERE dependency.task_id = tasks.id
                                AND parent.state <> %s
                          )
                        ORDER BY priority DESC, id ASC
                        LIMIT %s
                        FOR UPDATE SKIP LOCKED
                        """,
                        (
                            TASK_STATE_QUEUED,
                            *capabilities,
                            TASK_STATE_SUCCEEDED,
                            limit,
                        ),
                    )
                    rows = cursor.fetchall()
                    if not rows:
                        connection.commit()
                        return []
                    task_ids = [int(row["id"]) for row in rows]
                    update_placeholders = ", ".join(["%s"] * len(task_ids))
                    cursor.execute(
                        f"""
                        UPDATE tasks
                        SET
                            state = %s,
                            worker_id = %s,
                            lease_until = TIMESTAMPADD(SECOND, %s, UTC_TIMESTAMP(6)),
                            attempt_count = attempt_count + 1
                        WHERE id IN ({update_placeholders})
                        """,
                        (
                            TASK_STATE_LEASED,
                            worker_id,
                            lease_seconds,
                            *task_ids,
                        ),
                    )
                    connection.commit()
                normalized_rows = []
                for row in rows:
                    normalized = _decode_task_row(row)
                    normalized["attempt_count"] = int(normalized["attempt_count"]) + 1
                    normalized_rows.append(TaskRow.from_db_row(normalized))
                return normalized_rows
            except Exception:
                connection.rollback()
                raise

    def renew_task_lease(self, *, task_id: int, worker_id: str, lease_seconds: int) -> None:
        """Extend one leased task for the owning worker."""
        with self._connect() as connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE tasks
                        SET lease_until = TIMESTAMPADD(SECOND, %s, UTC_TIMESTAMP(6))
                        WHERE id = %s AND worker_id = %s AND state = %s
                        """,
                        (
                            lease_seconds,
                            task_id,
                            worker_id,
                            TASK_STATE_LEASED,
                        ),
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def requeue_expired_tasks(self) -> dict[str, int]:
        """Requeue expired leases or mark them failed after the final attempt."""
        max_retries = 3
        for attempt_index in range(max_retries):
            try:
                return self._requeue_expired_tasks_once()
            except (pymysql.err.InternalError, pymysql.err.OperationalError) as exc:
                error_code = int(exc.args[0]) if exc.args else 0
                if error_code not in {1205, 1213} or attempt_index + 1 >= max_retries:
                    raise
                time.sleep(0.05 * (attempt_index + 1))
        raise AssertionError("unreachable")

    def _requeue_expired_tasks_once(self) -> dict[str, int]:
        """Execute one best-effort expired-lease reconciliation transaction."""
        with self._connect() as connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE tasks
                        SET
                            state = %s,
                            worker_id = NULL,
                            lease_until = NULL
                        WHERE state = %s
                          AND lease_until IS NOT NULL
                          AND lease_until < UTC_TIMESTAMP(6)
                          AND attempt_count < max_attempts
                        """,
                        (
                            TASK_STATE_QUEUED,
                            TASK_STATE_LEASED,
                        ),
                    )
                    requeued = int(cursor.rowcount)
                    cursor.execute(
                        """
                        UPDATE tasks
                        SET
                            state = %s,
                            worker_id = NULL,
                            lease_until = NULL,
                            result_json = %s
                        WHERE state = %s
                          AND lease_until IS NOT NULL
                          AND lease_until < UTC_TIMESTAMP(6)
                          AND attempt_count >= max_attempts
                        """,
                        (
                            TASK_STATE_FAILED,
                            json.dumps(
                                {
                                    "failure_kind": "lease_expired",
                                    "message": "task lease expired after final attempt",
                                },
                                sort_keys=True,
                            ),
                            TASK_STATE_LEASED,
                        ),
                    )
                    failed = int(cursor.rowcount)
                connection.commit()
                return {"requeued": requeued, "failed": failed}
            except Exception:
                connection.rollback()
                raise

    def record_task_attempt_start(
        self,
        *,
        task_id: int,
        worker_id: str,
        stdout_path: str,
        stderr_path: str,
    ) -> int:
        """Create one attempt row and return its id."""
        with self._connect() as connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO task_attempts (task_id, worker_id, stdout_path, stderr_path)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (task_id, worker_id, stdout_path, stderr_path),
                    )
                    attempt_id = int(cursor.lastrowid)
                connection.commit()
                return attempt_id
            except Exception:
                connection.rollback()
                raise

    def finish_task_attempt(
        self,
        *,
        attempt_id: int,
        exit_code: int,
        result_summary_path: str | None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Close one task attempt row."""
        with self._connect() as connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE task_attempts
                        SET
                            ended_at = CURRENT_TIMESTAMP(6),
                            exit_code = %s,
                            result_summary_path = %s,
                            details_json = %s
                        WHERE id = %s
                        """,
                        (
                            exit_code,
                            result_summary_path,
                            _json_or_none(details),
                            attempt_id,
                        ),
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def mark_task_succeeded(
        self,
        *,
        task_id: int,
        result: TaskResult,
        artifacts: Sequence[ArtifactRef],
    ) -> None:
        """Persist one successful task result plus its produced artifacts."""
        with self._connect() as connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE tasks
                        SET
                            state = %s,
                            worker_id = NULL,
                            lease_until = NULL,
                            result_json = %s
                        WHERE id = %s
                        """,
                        (
                            TASK_STATE_SUCCEEDED,
                            json.dumps(result.to_dict(), sort_keys=True),
                            task_id,
                        ),
                    )
                    for artifact in artifacts:
                        cursor.execute(
                            """
                            INSERT INTO artifacts (
                                kind,
                                path,
                                sha256,
                                size_bytes,
                                producer_task_id,
                                state,
                                metadata_json
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON DUPLICATE KEY UPDATE
                                sha256 = VALUES(sha256),
                                size_bytes = VALUES(size_bytes),
                                producer_task_id = VALUES(producer_task_id),
                                state = VALUES(state),
                                metadata_json = VALUES(metadata_json)
                            """,
                            (
                                artifact.kind,
                                artifact.path,
                                artifact.sha256,
                                artifact.size_bytes,
                                task_id,
                                artifact.state,
                                _json_or_none(artifact.to_dict()),
                            ),
                        )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def mark_task_failed(
        self,
        *,
        task: TaskRow,
        error_message: str,
        retry_allowed: bool,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Persist a failed attempt, requeueing if allowed."""
        next_state = TASK_STATE_QUEUED if retry_allowed else TASK_STATE_FAILED
        with self._connect() as connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE tasks
                        SET
                            state = %s,
                            worker_id = NULL,
                            lease_until = NULL,
                            result_json = %s
                        WHERE id = %s
                        """,
                        (
                            next_state,
                            json.dumps(
                                {
                                    "error_message": error_message,
                                    "retry_allowed": retry_allowed,
                                    "details": dict(details or {}),
                                },
                                sort_keys=True,
                            ),
                            task.id,
                        ),
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def update_model_record(self, model_id: int, **fields: Any) -> None:
        """Patch mutable model fields on one row."""
        self._update_row(table="models", row_id=model_id, id_column="id", fields=fields)

    def update_campaign_record(self, campaign_id: int, **fields: Any) -> None:
        """Patch mutable campaign fields on one row."""
        self._update_row(
            table="campaigns",
            row_id=campaign_id,
            id_column="id",
            fields=fields,
        )

    def status_snapshot(self, *, limit: int = 20) -> dict[str, Any]:
        """Return a compact JSON-ready status view."""
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT state, COUNT(*) AS count
                    FROM tasks
                    GROUP BY state
                    ORDER BY state
                    """
                )
                task_counts = {str(row["state"]): int(row["count"]) for row in cursor.fetchall()}
                cursor.execute(
                    """
                    SELECT id, name, kind, status, config_path, active_model_id, updated_at
                    FROM campaigns
                    ORDER BY id DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                campaigns = [_jsonify_row(dict(row)) for row in cursor.fetchall()]
                cursor.execute(
                    """
                    SELECT id, campaign_id, task_type, capability, state, worker_id, updated_at
                    FROM tasks
                    ORDER BY id DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                tasks = [_jsonify_row(dict(row)) for row in cursor.fetchall()]
                cursor.execute(
                    """
                    SELECT id, hostname, status, current_task_id, last_heartbeat_at, version
                    FROM workers
                    ORDER BY last_heartbeat_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                workers = [_jsonify_row(dict(row)) for row in cursor.fetchall()]
        return {
            "db": self._config.safe_dict(),
            "task_counts": task_counts,
            "campaigns": campaigns,
            "tasks": tasks,
            "workers": workers,
        }

    def _update_row(
        self,
        *,
        table: str,
        row_id: int,
        id_column: str,
        fields: Mapping[str, Any],
    ) -> None:
        filtered_fields = {
            key: value for key, value in fields.items() if value is not None or key.endswith("_json")
        }
        if not filtered_fields:
            return
        assignments = ", ".join(f"{column} = %s" for column in filtered_fields)
        values = [_sql_value(column, value) for column, value in filtered_fields.items()]
        with self._connect() as connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        f"UPDATE {table} SET {assignments} WHERE {id_column} = %s",
                        (*values, row_id),
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise


def _json_or_none(payload: Mapping[str, Any] | Sequence[Any] | None) -> str | None:
    if payload is None:
        return None
    return json.dumps(payload, sort_keys=True)


def _sql_value(column: str, value: Any) -> Any:
    if column.endswith("_json") and value is not None and not isinstance(value, str):
        return json.dumps(value, sort_keys=True)
    return value


def _decode_task_row(row: Mapping[str, Any]) -> dict[str, Any]:
    decoded = dict(row)
    decoded["payload_json"] = _decode_json_column(decoded["payload_json"])
    decoded["result_json"] = _decode_json_column(decoded.get("result_json"))
    return decoded


def _decode_campaign_row(row: Mapping[str, Any]) -> dict[str, Any]:
    decoded = dict(row)
    decoded["metadata_json"] = _decode_json_column(decoded.get("metadata_json")) or {}
    return decoded


def _decode_model_row(row: Mapping[str, Any]) -> dict[str, Any]:
    decoded = dict(row)
    decoded["metadata_json"] = _decode_json_column(decoded.get("metadata_json")) or {}
    return decoded


def _decode_json_column(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    return json.loads(str(value))


def _jsonify_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): (value.isoformat(sep=" ") if hasattr(value, "isoformat") else value)
        for key, value in row.items()
    }
