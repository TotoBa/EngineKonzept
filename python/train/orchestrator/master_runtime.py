"""Runtime control wrapper around the long-running orchestrator master."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import threading
import time
from typing import Any, Mapping

from train.orchestrator.master import MasterSpec, OrchestratorMaster, write_master_spec


_NAMED_SPEC_COLLECTIONS = {
    "lineages": "lineages",
    "label_jobs": "label_jobs",
    "idle_phase10_jobs": "idle_phase10_jobs",
}


class OrchestratorMasterRuntime:
    """Thread-safe runtime controls over one :class:`OrchestratorMaster`."""

    def __init__(self, *, master: OrchestratorMaster) -> None:
        self._master = master
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._loop_thread: threading.Thread | None = None
        self._paused = False
        self._cycle_count = 0
        self._last_summary = _load_json_if_exists(master.output_root / "summary.json")
        self._last_error: dict[str, Any] | None = None
        self._last_reconcile_started_at: float | None = None
        self._last_reconcile_finished_at: float | None = None
        self._last_requeue_result: dict[str, Any] | None = None

    @property
    def master(self) -> OrchestratorMaster:
        return self._master

    def runtime_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "loop_running": self._loop_thread is not None and self._loop_thread.is_alive(),
                "paused": self._paused,
                "cycle_count": self._cycle_count,
                "poll_interval_seconds": float(self._master.spec.poll_interval_seconds),
                "spec_path": str(self._master.spec_path),
                "output_root": str(self._master.output_root),
                "last_summary_available": self._last_summary is not None,
                "last_error": deepcopy(self._last_error),
                "last_requeue_result": deepcopy(self._last_requeue_result),
                "last_reconcile_started_at": _epoch_or_none(self._last_reconcile_started_at),
                "last_reconcile_finished_at": _epoch_or_none(self._last_reconcile_finished_at),
            }

    def latest_summary(self) -> dict[str, Any] | None:
        with self._lock:
            return deepcopy(self._last_summary)

    def bootstrap(self, *, limit: int = 20) -> dict[str, Any]:
        return {
            "runtime": self.runtime_status(),
            "spec": self.get_spec_dict(),
            "summary": self.latest_summary(),
            "status": self.status_snapshot(limit=limit),
        }

    def get_spec_dict(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._master.spec.to_dict())

    def status_snapshot(self, *, limit: int = 20) -> dict[str, Any]:
        return self._master.db.status_snapshot(limit=limit)

    def reconcile_once(self) -> dict[str, Any]:
        with self._lock:
            self._last_reconcile_started_at = time.time()
            try:
                summary = self._master.reconcile_once()
            except Exception as exc:
                self._last_reconcile_finished_at = time.time()
                self._last_error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "timestamp": int(self._last_reconcile_finished_at),
                }
                raise
            self._last_reconcile_finished_at = time.time()
            self._last_error = None
            self._last_summary = deepcopy(summary)
            self._cycle_count += 1
            return deepcopy(summary)

    def requeue_expired_tasks(self) -> dict[str, Any]:
        result = self._master.db.requeue_expired_tasks()
        with self._lock:
            self._last_requeue_result = {
                **dict(result),
                "timestamp": int(time.time()),
            }
            return deepcopy(self._last_requeue_result)

    def start_loop(self) -> dict[str, Any]:
        with self._lock:
            if self._loop_thread is not None and self._loop_thread.is_alive():
                self._paused = False
                self._wake_event.set()
                return self.runtime_status()
            self._paused = False
            self._stop_event = threading.Event()
            self._wake_event = threading.Event()
            thread = threading.Thread(
                target=self._loop_main,
                name=f"master-runtime:{self._master.spec.name}",
                daemon=True,
            )
            self._loop_thread = thread
            thread.start()
            return self.runtime_status()

    def stop_loop(self, *, join_timeout_seconds: float = 5.0) -> dict[str, Any]:
        with self._lock:
            thread = self._loop_thread
            self._stop_event.set()
            self._wake_event.set()
        if thread is not None:
            thread.join(timeout=join_timeout_seconds)
        with self._lock:
            if self._loop_thread is thread and thread is not None and not thread.is_alive():
                self._loop_thread = None
            return self.runtime_status()

    def pause_loop(self) -> dict[str, Any]:
        with self._lock:
            self._paused = True
            self._wake_event.set()
            return self.runtime_status()

    def resume_loop(self) -> dict[str, Any]:
        with self._lock:
            self._paused = False
            self._wake_event.set()
            return self.runtime_status()

    def replace_spec(self, spec: MasterSpec) -> dict[str, Any]:
        with self._lock:
            write_master_spec(self._master.spec_path, spec)
            self._master.replace_spec(spec)
            self._last_summary = _load_json_if_exists(self._master.output_root / "summary.json")
            self._wake_event.set()
            return self.get_spec_dict()

    def replace_spec_from_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        spec = MasterSpec.from_dict(dict(payload))
        return self.replace_spec(spec)

    def patch_spec(self, patch: Mapping[str, Any]) -> dict[str, Any]:
        current_payload = self.get_spec_dict()
        merged_payload = _merge_spec_dict(current_payload, dict(patch))
        spec = MasterSpec.from_dict(merged_payload)
        return self.replace_spec(spec)

    def update_named_spec_entry(
        self,
        *,
        collection_name: str,
        entry_name: str,
        patch: Mapping[str, Any],
    ) -> dict[str, Any]:
        if collection_name not in _NAMED_SPEC_COLLECTIONS:
            raise ValueError(f"unsupported spec collection: {collection_name}")
        payload = self.get_spec_dict()
        entries = list(payload.get(collection_name) or [])
        for index, entry in enumerate(entries):
            if str(dict(entry).get("name")) == entry_name:
                entries[index] = _merge_spec_dict(dict(entry), dict(patch))
                payload[collection_name] = entries
                spec = MasterSpec.from_dict(payload)
                self.replace_spec(spec)
                return deepcopy(entries[index])
        raise KeyError(f"{collection_name} entry not found: {entry_name}")

    def submit_phase10_campaign(self, *, config_path: Path, kind: str = "phase10_native") -> dict[str, Any]:
        return self._master.controller.submit_phase10_campaign(config_path=config_path, kind=kind)

    def submit_label_campaign(
        self,
        *,
        config_path: Path,
        kind: str = "label_pgn_corpus",
    ) -> dict[str, Any]:
        return self._master.controller.submit_label_pgn_corpus_campaign(
            config_path=config_path,
            kind=kind,
        )

    def submit_idle_phase10_campaign(
        self,
        *,
        config_path: Path,
        kind: str = "phase10_idle_artifacts",
    ) -> dict[str, Any]:
        return self._master.controller.submit_idle_phase10_artifact_campaign(
            config_path=config_path,
            kind=kind,
        )

    def _loop_main(self) -> None:
        while not self._stop_event.is_set():
            paused = False
            with self._lock:
                paused = self._paused
            if not paused:
                try:
                    self.reconcile_once()
                except Exception:
                    # Errors are surfaced via runtime_status and do not kill the control loop.
                    pass
            timeout_seconds = float(self._master.spec.poll_interval_seconds)
            self._wake_event.wait(timeout=timeout_seconds)
            self._wake_event.clear()
        with self._lock:
            current = threading.current_thread()
            if self._loop_thread is current:
                self._loop_thread = None


def _merge_spec_dict(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, patch_value in patch.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(patch_value, Mapping):
            merged[key] = _merge_spec_dict(base_value, patch_value)
        else:
            merged[key] = deepcopy(patch_value)
    return merged


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _epoch_or_none(value: float | None) -> int | None:
    if value is None:
        return None
    return int(value)
