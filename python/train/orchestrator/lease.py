"""Lease helpers for long-running worker tasks."""

from __future__ import annotations

from contextlib import AbstractContextManager
import threading
from typing import Any, Protocol


class LeaseClient(Protocol):
    """Small protocol so the heartbeat loop can renew leases via the DB wrapper."""

    def renew_task_lease(self, *, task_id: int, worker_id: str, lease_seconds: int) -> None:
        """Extend the current task lease."""

    def heartbeat_worker(
        self,
        *,
        worker_id: str,
        status: str,
        current_task_id: int | None,
        lease_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Refresh the worker heartbeat row."""


class LeaseHeartbeat(AbstractContextManager["LeaseHeartbeat"]):
    """Background loop that keeps a task lease alive while work is in progress."""

    def __init__(
        self,
        *,
        client: LeaseClient,
        worker_id: str,
        task_id: int,
        lease_seconds: int,
        interval_seconds: float,
    ) -> None:
        self._client = client
        self._worker_id = worker_id
        self._task_id = task_id
        self._lease_seconds = lease_seconds
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"lease-heartbeat-{task_id}")
        self._thread.daemon = True

    def __enter__(self) -> "LeaseHeartbeat":
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=max(self._interval_seconds * 2.0, 1.0))

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self._client.renew_task_lease(
                task_id=self._task_id,
                worker_id=self._worker_id,
                lease_seconds=self._lease_seconds,
            )
            self._client.heartbeat_worker(
                worker_id=self._worker_id,
                status="busy",
                current_task_id=self._task_id,
                lease_seconds=self._lease_seconds,
            )
