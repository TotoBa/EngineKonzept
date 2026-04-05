"""Precomputed LAPv1 training artifacts derived from planner-head examples."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

from train.datasets.artifacts import (
    PIECE_TOKEN_CAPACITY,
    PIECE_TOKEN_PADDING_VALUE,
    PIECE_TOKEN_WIDTH,
    SQUARE_TOKEN_COUNT,
    SQUARE_TOKEN_WIDTH,
    split_position_features,
)
from train.datasets.contracts import build_state_context_v1, state_context_v1_feature_spec
from train.datasets.planner_head import PlannerHeadExample
from train.datasets.schema import DatasetExample, PositionEncoding, TacticalAnnotations

try:
    import chess
except ModuleNotFoundError:  # pragma: no cover - exercised when chess is absent
    chess = None


LAPV1_TRAINING_ARTIFACT_PREFIX = "lapv1_"
_STATE_CONTEXT_GLOBAL_DIM = len(state_context_v1_feature_spec()["global_feature_order"])


@dataclass(frozen=True)
class LAPv1TrainingExample:
    """One precomputed LAPv1 training row with exact symbolic side-inputs."""

    sample_id: str
    split: str
    piece_tokens: list[list[int]]
    square_tokens: list[list[float]]
    state_context_global: list[float]
    reachability_edges: list[list[int]]
    candidate_action_indices: list[int]
    candidate_features: list[list[float]]
    candidate_mask: list[bool]
    teacher_top1_candidate_index: int
    teacher_policy: list[float]
    teacher_root_value_cp: float
    teacher_wdl_target: int
    sharpness_target: float
    teacher_top1_minus_top2_cp: float | None
    teacher_candidate_rank_bucket_targets: list[int] | None
    curriculum_priority: float

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "piece_tokens": self.piece_tokens,
            "square_tokens": self.square_tokens,
            "state_context_global": self.state_context_global,
            "reachability_edges": self.reachability_edges,
            "candidate_action_indices": self.candidate_action_indices,
            "candidate_features": self.candidate_features,
            "candidate_mask": self.candidate_mask,
            "teacher_top1_candidate_index": self.teacher_top1_candidate_index,
            "teacher_policy": self.teacher_policy,
            "teacher_root_value_cp": self.teacher_root_value_cp,
            "teacher_wdl_target": self.teacher_wdl_target,
            "sharpness_target": self.sharpness_target,
            "teacher_top1_minus_top2_cp": self.teacher_top1_minus_top2_cp,
            "teacher_candidate_rank_bucket_targets": self.teacher_candidate_rank_bucket_targets,
            "curriculum_priority": self.curriculum_priority,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "LAPv1TrainingExample":
        return cls(
            sample_id=str(payload["sample_id"]),
            split=str(payload["split"]),
            piece_tokens=[
                [int(value) for value in row]
                for row in list(payload["piece_tokens"])
            ],
            square_tokens=[
                [float(value) for value in row]
                for row in list(payload["square_tokens"])
            ],
            state_context_global=[float(value) for value in list(payload["state_context_global"])],
            reachability_edges=[
                [int(value) for value in row]
                for row in list(payload["reachability_edges"])
            ],
            candidate_action_indices=[int(value) for value in list(payload["candidate_action_indices"])],
            candidate_features=[
                [float(value) for value in row]
                for row in list(payload["candidate_features"])
            ],
            candidate_mask=[bool(value) for value in list(payload["candidate_mask"])],
            teacher_top1_candidate_index=int(payload["teacher_top1_candidate_index"]),
            teacher_policy=[float(value) for value in list(payload["teacher_policy"])],
            teacher_root_value_cp=float(payload["teacher_root_value_cp"]),
            teacher_wdl_target=int(payload["teacher_wdl_target"]),
            sharpness_target=float(payload["sharpness_target"]),
            teacher_top1_minus_top2_cp=(
                None
                if payload.get("teacher_top1_minus_top2_cp") is None
                else float(payload["teacher_top1_minus_top2_cp"])
            ),
            teacher_candidate_rank_bucket_targets=(
                None
                if payload.get("teacher_candidate_rank_bucket_targets") is None
                else [int(value) for value in list(payload["teacher_candidate_rank_bucket_targets"])]
            ),
            curriculum_priority=float(payload["curriculum_priority"]),
        )

    @classmethod
    def from_json(cls, line: str, *, source: str = "<jsonl>") -> "LAPv1TrainingExample":
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{source}: LAPv1 training example must be a JSON object")
        return cls.from_dict(payload)


def lapv1_training_artifact_name(split: str) -> str:
    """Return the canonical Phase-10 LAPv1 artifact file name for one split."""
    return f"{LAPV1_TRAINING_ARTIFACT_PREFIX}{split}.jsonl"


def load_lapv1_training_examples(path: Path) -> list[LAPv1TrainingExample]:
    """Load precomputed LAPv1 training examples from JSONL."""
    if not path.exists():
        raise FileNotFoundError(f"LAPv1 training artifact not found: {path}")
    examples: list[LAPv1TrainingExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            examples.append(
                LAPv1TrainingExample.from_json(line, source=f"{path}:{line_number}")
            )
    return examples


def write_lapv1_training_artifact(
    path: Path,
    examples: Sequence[LAPv1TrainingExample],
) -> None:
    """Write precomputed LAPv1 training examples to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(example.to_dict(), sort_keys=True) for example in examples]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def lapv1_training_example_from_planner_head(
    example: PlannerHeadExample,
) -> LAPv1TrainingExample:
    """Precompute all symbolic LAPv1-side inputs for one planner-head row."""
    feature_sections = split_position_features(example.feature_vector)
    piece_tokens = _decode_piece_tokens(feature_sections["piece"])
    square_tokens = _decode_square_tokens(feature_sections["square"])
    state_context = build_state_context_v1(_dataset_example_for_fen(example.fen))
    global_features = state_context.feature_values[-_STATE_CONTEXT_GLOBAL_DIM :]
    reachability_edges = [
        [src, dst, piece_type]
        for src, dst, piece_type in zip(
            state_context.edge_src_square,
            state_context.edge_dst_square,
            state_context.edge_piece_type,
            strict=True,
        )
    ]
    return LAPv1TrainingExample(
        sample_id=example.sample_id,
        split=example.split,
        piece_tokens=piece_tokens,
        square_tokens=square_tokens,
        state_context_global=global_features,
        reachability_edges=reachability_edges,
        candidate_action_indices=list(example.candidate_action_indices),
        candidate_features=[list(row) for row in example.candidate_features],
        candidate_mask=[True] * len(example.candidate_action_indices),
        teacher_top1_candidate_index=example.teacher_top1_candidate_index,
        teacher_policy=_normalize_policy(example.teacher_policy),
        teacher_root_value_cp=example.teacher_root_value_cp,
        teacher_wdl_target=_wdl_target_from_cp(example.teacher_root_value_cp),
        sharpness_target=_sharpness_target(example.teacher_top1_minus_top2_cp),
        teacher_top1_minus_top2_cp=example.teacher_top1_minus_top2_cp,
        teacher_candidate_rank_bucket_targets=(
            None
            if example.teacher_candidate_rank_bucket_targets is None
            else list(example.teacher_candidate_rank_bucket_targets)
        ),
        curriculum_priority=example.curriculum_priority,
    )


def _dataset_example_for_fen(fen: str) -> DatasetExample:
    if chess is None:  # pragma: no cover
        raise RuntimeError("python-chess is required for LAPv1 example preparation.")
    board = chess.Board(fen)
    legal_moves = [move.uci() for move in board.legal_moves]
    return DatasetExample(
        sample_id=f"lapv1:{fen}",
        split="test",
        source="lapv1",
        fen=fen,
        side_to_move="w" if board.turn else "b",
        selected_move_uci=None,
        selected_action_encoding=None,
        next_fen=None,
        legal_moves=legal_moves,
        legal_action_encodings=[[0, 0, index] for index, _move in enumerate(legal_moves)],
        position_encoding=PositionEncoding(
            piece_tokens=[],
            square_tokens=[[square_index, 0] for square_index in range(64)],
            rule_token=[0, 0, -1, 0, 1, 0],
        ),
        wdl_target=None,
        annotations=TacticalAnnotations(
            in_check=board.is_check(),
            is_checkmate=board.is_checkmate(),
            is_stalemate=board.is_stalemate(),
            has_legal_en_passant=any(board.is_en_passant(move) for move in board.legal_moves),
            has_legal_castle=any(board.is_castling(move) for move in board.legal_moves),
            has_legal_promotion=any(move.promotion is not None for move in board.legal_moves),
            is_low_material_endgame=len(board.piece_map()) <= 6,
            legal_move_count=len(legal_moves),
            piece_count=len(board.piece_map()),
            selected_move_is_capture=None,
            selected_move_is_promotion=None,
            selected_move_is_castle=None,
            selected_move_is_en_passant=None,
            selected_move_gives_check=None,
        ),
        result=None,
        metadata={},
    )


def _decode_piece_tokens(values: Sequence[float]) -> list[list[int]]:
    tokens: list[list[int]] = []
    for offset in range(0, len(values), PIECE_TOKEN_WIDTH):
        row = [int(round(value)) for value in values[offset : offset + PIECE_TOKEN_WIDTH]]
        if row[0] == PIECE_TOKEN_PADDING_VALUE:
            tokens.append([-1, -1, -1])
        else:
            tokens.append(row)
    if len(tokens) != PIECE_TOKEN_CAPACITY:
        raise ValueError("piece token slice does not decode to capacity rows")
    return tokens


def _decode_square_tokens(values: Sequence[float]) -> list[list[float]]:
    tokens: list[list[float]] = []
    for offset in range(0, len(values), SQUARE_TOKEN_WIDTH):
        row = [float(value) for value in values[offset : offset + SQUARE_TOKEN_WIDTH]]
        tokens.append(row)
    if len(tokens) != SQUARE_TOKEN_COUNT:
        raise ValueError("square token slice does not decode to 64 rows")
    return tokens


def _normalize_policy(policy: Sequence[float]) -> list[float]:
    values = [max(0.0, float(value)) for value in policy]
    total = sum(values)
    if total <= 0.0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in values]


def _wdl_target_from_cp(cp_value: float) -> int:
    if cp_value > 20.0:
        return 2
    if cp_value < -20.0:
        return 0
    return 1


def _sharpness_target(gap_cp: float | None) -> float:
    if gap_cp is None:
        return 0.0
    return 1.0 if abs(gap_cp) < 20.0 else 0.0
