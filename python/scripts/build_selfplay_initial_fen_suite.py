"""Build a versioned selfplay initial-position suite from existing labeled datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from train.eval.initial_fens import (
    SelfplayInitialFenEntry,
    SelfplayInitialFenSuite,
    write_selfplay_initial_fen_suite,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="TIER=PATH",
        help="One or more dataset JSONL sources grouped by tier.",
    )
    parser.add_argument(
        "--per-tier",
        type=int,
        default=2,
        help="How many curated initial positions to keep per tier.",
    )
    parser.add_argument(
        "--min-legal-moves",
        type=int,
        default=4,
        help="Exclude cramped near-terminal positions below this legal-move count.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.per_tier <= 0:
        raise ValueError("--per-tier must be positive")
    if args.min_legal_moves <= 0:
        raise ValueError("--min-legal-moves must be positive")
    if not args.dataset:
        raise ValueError("at least one --dataset TIER=PATH is required")

    dataset_specs = [_parse_dataset_spec(raw_spec) for raw_spec in args.dataset]
    entries: list[SelfplayInitialFenEntry] = []
    summary_rows: list[dict[str, object]] = []
    for tier, raw_path in dataset_specs:
        path = _resolve_repo_path(Path(raw_path))
        tier_entries = _select_tier_entries(
            tier=tier,
            path=path,
            per_tier=args.per_tier,
            min_legal_moves=args.min_legal_moves,
        )
        entries.extend(tier_entries)
        summary_rows.append(
            {
                "tier": tier,
                "source_path": str(path),
                "selected_count": len(tier_entries),
                "sample_ids": [entry.sample_id for entry in tier_entries],
                "selection_scores": [round(entry.selection_score, 3) for entry in tier_entries],
            }
        )

    suite = SelfplayInitialFenSuite(
        name=args.name,
        entries=entries,
        metadata={
            "per_tier": args.per_tier,
            "min_legal_moves": args.min_legal_moves,
            "tier_summaries": summary_rows,
        },
    )
    output_path = _resolve_repo_path(args.output_path)
    write_selfplay_initial_fen_suite(output_path, suite)

    payload = {
        "output_path": str(output_path),
        "entry_count": len(entries),
        "tiers": summary_rows,
    }
    if args.summary_path is not None:
        summary_path = _resolve_repo_path(args.summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        payload["summary_path"] = str(summary_path)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _select_tier_entries(
    *,
    tier: str,
    path: Path,
    per_tier: int,
    min_legal_moves: int,
) -> list[SelfplayInitialFenEntry]:
    scored_rows: list[tuple[float, str, dict[str, object]]] = []
    seen_fens: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number}: dataset row must be a JSON object")
        fen = str(payload["fen"])
        if fen in seen_fens:
            continue
        annotations = dict(payload.get("annotations") or {})
        if bool(annotations.get("is_checkmate")) or bool(annotations.get("is_stalemate")):
            continue
        legal_move_count = int(annotations.get("legal_move_count", 0))
        if legal_move_count < min_legal_moves:
            continue
        score, tags = _selection_score(payload)
        scored_rows.append((score, fen, {"payload": payload, "tags": tags}))
        seen_fens.add(fen)

    scored_rows.sort(
        key=lambda item: (
            -item[0],
            str(item[2]["payload"].get("sample_id", "")),
            item[1],
        )
    )
    selected_rows = scored_rows[:per_tier]
    if len(selected_rows) < per_tier:
        raise ValueError(f"{path}: only found {len(selected_rows)} valid rows for tier {tier}")
    return [
        SelfplayInitialFenEntry(
            fen=fen,
            tier=tier,
            sample_id=str(extra["payload"]["sample_id"]),
            source_path=str(path),
            result=str(extra["payload"].get("result") or "*"),
            selection_score=score,
            tags=list(extra["tags"]),
            metadata={
                "legal_move_count": int(
                    dict(extra["payload"].get("annotations") or {}).get("legal_move_count", 0)
                ),
                "side_to_move": str(extra["payload"].get("side_to_move") or ""),
            },
        )
        for score, fen, extra in selected_rows
    ]


def _selection_score(payload: dict[str, object]) -> tuple[float, list[str]]:
    annotations = dict(payload.get("annotations") or {})
    result = str(payload.get("result") or "*")
    tags: list[str] = []
    score = 0.0
    if result in {"1-0", "0-1"}:
        score += 3.0
        tags.append("decisive")
    if bool(annotations.get("in_check")):
        score += 2.5
        tags.append("in_check")
    if bool(annotations.get("selected_move_gives_check")):
        score += 1.5
        tags.append("gives_check")
    if bool(annotations.get("selected_move_is_capture")):
        score += 1.0
        tags.append("capture")
    if bool(annotations.get("has_legal_promotion")) or bool(annotations.get("selected_move_is_promotion")):
        score += 1.0
        tags.append("promotion")
    if bool(annotations.get("has_legal_castle")) or bool(annotations.get("selected_move_is_castle")):
        score += 0.75
        tags.append("castle")
    if bool(annotations.get("has_legal_en_passant")) or bool(annotations.get("selected_move_is_en_passant")):
        score += 0.75
        tags.append("en_passant")
    legal_move_count = int(annotations.get("legal_move_count", 0))
    score += max(0.0, 16.0 - min(legal_move_count, 16)) * 0.08
    piece_count = int(annotations.get("piece_count", 0))
    score += max(0.0, 28.0 - min(piece_count, 28)) * 0.03
    return score, tags


def _parse_dataset_spec(raw_spec: str) -> tuple[str, str]:
    if "=" not in raw_spec:
        raise ValueError("--dataset must use TIER=PATH")
    tier, path = raw_spec.split("=", 1)
    tier = tier.strip()
    path = path.strip()
    if not tier or not path:
        raise ValueError("--dataset must use TIER=PATH")
    return tier, path


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
