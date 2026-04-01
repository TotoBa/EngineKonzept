"""Dataset composition reporting."""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any, Sequence

from train.datasets.schema import DatasetExample


def build_summary(examples: Sequence[DatasetExample]) -> dict[str, Any]:
    """Build a JSON-friendly dataset summary."""
    split_counts = Counter(example.split for example in examples)
    source_counts = Counter(example.source for example in examples)
    wdl_counts = Counter(
        "missing" if example.wdl_target is None else _wdl_label(example)
        for example in examples
    )

    annotations = {
        "in_check": sum(example.annotations.in_check for example in examples),
        "checkmate": sum(example.annotations.is_checkmate for example in examples),
        "stalemate": sum(example.annotations.is_stalemate for example in examples),
        "has_legal_en_passant": sum(
            example.annotations.has_legal_en_passant for example in examples
        ),
        "has_legal_castle": sum(example.annotations.has_legal_castle for example in examples),
        "has_legal_promotion": sum(
            example.annotations.has_legal_promotion for example in examples
        ),
        "is_low_material_endgame": sum(
            example.annotations.is_low_material_endgame for example in examples
        ),
    }

    legal_move_counts = [example.annotations.legal_move_count for example in examples]
    piece_counts = [example.annotations.piece_count for example in examples]

    return {
        "total_examples": len(examples),
        "split_counts": dict(split_counts),
        "source_counts": dict(source_counts),
        "wdl_counts": dict(wdl_counts),
        "selected_move_count": sum(example.selected_move_uci is not None for example in examples),
        "next_state_count": sum(example.next_fen is not None for example in examples),
        "unique_fens": len({example.fen for example in examples}),
        "annotation_counts": annotations,
        "legal_move_count": _summary_stats(legal_move_counts),
        "piece_count": _summary_stats(piece_counts),
    }


def _summary_stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(mean(values), 4),
    }


def _wdl_label(example: DatasetExample) -> str:
    assert example.wdl_target is not None
    if example.wdl_target.win:
        return "win"
    if example.wdl_target.draw:
        return "draw"
    return "loss"
