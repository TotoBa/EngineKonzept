"""Dataset builder that combines raw sources, exact-rule labels, and split assignment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from train.datasets.oracle import label_records_with_oracle
from train.datasets.schema import (
    DatasetExample,
    PositionEncoding,
    RawPositionRecord,
    SplitRatios,
    TacticalAnnotations,
    WdlTarget,
)
from train.datasets.splits import assign_splits
from train.datasets.summary import build_summary


@dataclass(frozen=True)
class BuiltDataset:
    """Complete dataset build result."""

    examples: list[DatasetExample]
    summary: dict[str, object]


def build_dataset(
    records: Sequence[RawPositionRecord],
    *,
    ratios: SplitRatios | None = None,
    seed: str = "phase-4",
    repo_root: Path | None = None,
    oracle_command: Sequence[str] | None = None,
) -> BuiltDataset:
    """Build a reproducible labeled dataset from raw records."""
    resolved_ratios = ratios or SplitRatios()
    oracle_outputs = label_records_with_oracle(
        records,
        repo_root=repo_root,
        command=oracle_command,
    )
    splits = assign_splits(records, ratios=resolved_ratios, seed=seed)

    examples: list[DatasetExample] = []
    for record, split, oracle_payload in zip(records, splits, oracle_outputs, strict=True):
        annotations = TacticalAnnotations.from_oracle_dict(oracle_payload["annotations"])
        position_encoding = PositionEncoding.from_oracle_dict(oracle_payload["position_encoding"])
        wdl_target = _derive_wdl_target(
            result=record.result,
            side_to_move=str(oracle_payload["side_to_move"]),
            annotations=annotations,
        )
        selected_action = oracle_payload["selected_action_encoding"]
        examples.append(
            DatasetExample(
                sample_id=record.sample_id,
                split=split,
                source=record.source,
                fen=record.fen,
                side_to_move=str(oracle_payload["side_to_move"]),
                selected_move_uci=record.selected_move_uci,
                selected_action_encoding=None
                if selected_action is None
                else [int(value) for value in selected_action],
                next_fen=oracle_payload["next_fen"],
                legal_moves=[str(move) for move in oracle_payload["legal_moves"]],
                legal_action_encodings=[
                    [int(value) for value in action]
                    for action in oracle_payload["legal_action_encodings"]
                ],
                position_encoding=position_encoding,
                wdl_target=wdl_target,
                annotations=annotations,
                result=record.result,
                metadata=record.metadata,
            )
        )

    return BuiltDataset(examples=examples, summary=build_summary(examples))


def _derive_wdl_target(
    *,
    result: str | None,
    side_to_move: str,
    annotations: TacticalAnnotations,
) -> WdlTarget | None:
    if result == "1-0":
        return WdlTarget(win=1, draw=0, loss=0) if side_to_move == "w" else WdlTarget(
            win=0,
            draw=0,
            loss=1,
        )
    if result == "0-1":
        return WdlTarget(win=0, draw=0, loss=1) if side_to_move == "w" else WdlTarget(
            win=1,
            draw=0,
            loss=0,
        )
    if result == "1/2-1/2":
        return WdlTarget(win=0, draw=1, loss=0)
    if annotations.is_checkmate:
        return WdlTarget(win=0, draw=0, loss=1)
    if annotations.is_stalemate:
        return WdlTarget(win=0, draw=1, loss=0)
    return None
