"""Raw dataset source ingestion for Phase 4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from train.datasets.schema import RawPositionRecord

SUPPORTED_SOURCE_FORMATS = ("edge-cases", "fen-lines", "epd", "jsonl")


def load_raw_records(
    input_path: Path,
    source_format: str,
    *,
    source_name: str | None = None,
) -> list[RawPositionRecord]:
    """Load raw records from a supported source format."""
    if source_format not in SUPPORTED_SOURCE_FORMATS:
        raise ValueError(f"unsupported source format: {source_format}")

    resolved_source_name = source_name or input_path.stem
    if source_format == "edge-cases":
        return _load_name_fen_records(input_path, resolved_source_name)
    if source_format == "fen-lines":
        return _load_fen_lines(input_path, resolved_source_name)
    if source_format == "epd":
        return _load_epd_positions(input_path, resolved_source_name)
    return _load_jsonl_records(input_path, resolved_source_name)


def _load_name_fen_records(input_path: Path, source_name: str) -> list[RawPositionRecord]:
    records: list[RawPositionRecord] = []
    for line_number, raw_line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, fen = _split_name_and_fen(line, line_number, input_path)
        records.append(
            RawPositionRecord(
                sample_id=f"{source_name}:{name}",
                fen=fen,
                source=source_name,
                metadata={"name": name},
            )
        )
    return records


def _load_fen_lines(input_path: Path, source_name: str) -> list[RawPositionRecord]:
    records: list[RawPositionRecord] = []
    for line_number, raw_line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            name, fen = _split_name_and_fen(line, line_number, input_path)
            sample_id = f"{source_name}:{name}"
            metadata: dict[str, Any] = {"name": name}
        else:
            fen = line
            sample_id = f"{source_name}:{line_number}"
            metadata = {}
        records.append(
            RawPositionRecord(
                sample_id=sample_id,
                fen=fen,
                source=source_name,
                metadata=metadata,
            )
        )
    return records


def _load_epd_positions(input_path: Path, source_name: str) -> list[RawPositionRecord]:
    records: list[RawPositionRecord] = []
    for line_number, raw_line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fen = _normalize_epd_to_fen(line)
        records.append(
            RawPositionRecord(
                sample_id=f"{source_name}:{line_number}",
                fen=fen,
                source=source_name,
                metadata={"raw_epd": line},
            )
        )
    return records


def _load_jsonl_records(input_path: Path, source_name: str) -> list[RawPositionRecord]:
    records: list[RawPositionRecord] = []
    for line_number, raw_line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{input_path}:{line_number}: jsonl records must be objects")
        sample_id = str(payload.get("sample_id") or f"{source_name}:{line_number}")
        source = str(payload.get("source") or source_name)
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError(f"{input_path}:{line_number}: metadata must be an object")
        records.append(
            RawPositionRecord(
                sample_id=sample_id,
                fen=str(payload["fen"]),
                source=source,
                selected_move_uci=_optional_string(payload.get("selected_move_uci")),
                result=_optional_string(payload.get("result")),
                metadata=dict(metadata),
            )
        )
    return records


def _split_name_and_fen(line: str, line_number: int, input_path: Path) -> tuple[str, str]:
    parts = line.split("|", maxsplit=1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise ValueError(f"{input_path}:{line_number}: expected 'name|fen'")
    return parts[0].strip(), parts[1].strip()


def _normalize_epd_to_fen(line: str) -> str:
    if ";" in line:
        line = line.split(";", maxsplit=1)[0].strip()
    fields = line.split()
    if len(fields) < 4:
        raise ValueError(f"epd line must contain at least 4 fields: {line}")
    return " ".join(fields[:4] + ["0", "1"])


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
