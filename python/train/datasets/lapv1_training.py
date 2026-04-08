"""Precomputed LAPv1 training artifacts derived from planner-head examples."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence
import warnings

from train.datasets.artifacts import (
    PIECE_TOKEN_CAPACITY,
    PIECE_TOKEN_PADDING_VALUE,
    PIECE_TOKEN_WIDTH,
    SQUARE_TOKEN_COUNT,
    SQUARE_TOKEN_WIDTH,
    split_position_features,
)
from train.datasets.contracts import build_state_context_v1, state_context_v1_feature_spec
from train.datasets.move_delta import halfka_delta, is_king_move, move_type_hash
from train.datasets.nnue_features import halfka_active_indices
from train.datasets.opponent_head import move_uci_for_action
from train.datasets.phase_features import phase_index
from train.datasets.planner_head import PlannerHeadExample
from train.datasets.schema import DatasetExample, PositionEncoding, TacticalAnnotations

try:
    import chess
except ModuleNotFoundError:  # pragma: no cover - exercised when chess is absent
    chess = None


LAPV1_TRAINING_ARTIFACT_PREFIX = "lapv1_"
_STATE_CONTEXT_GLOBAL_DIM = len(state_context_v1_feature_spec()["global_feature_order"])
_MISSING_FIELD_WARNINGS: set[str] = set()


@dataclass(frozen=True)
class LAPv1TrainingExample:
    """One precomputed LAPv1 training row with exact symbolic side-inputs."""

    sample_id: str
    split: str
    side_to_move: int
    phase_index: int
    king_sq_white: int
    king_sq_black: int
    piece_tokens: list[list[int]]
    square_tokens: list[list[float]]
    state_context_global: list[float]
    reachability_edges: list[list[int]]
    nnue_feat_white: list[int]
    nnue_feat_black: list[int]
    candidate_action_indices: list[int]
    candidate_move_types: list[int]
    candidate_delta_white_leave: list[list[int]]
    candidate_delta_white_enter: list[list[int]]
    candidate_delta_black_leave: list[list[int]]
    candidate_delta_black_enter: list[list[int]]
    candidate_is_white_king_move: list[bool]
    candidate_is_black_king_move: list[bool]
    candidate_nnue_feat_white_after_move: list[list[int]]
    candidate_nnue_feat_black_after_move: list[list[int]]
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
            "side_to_move": self.side_to_move,
            "phase_index": self.phase_index,
            "king_sq_white": self.king_sq_white,
            "king_sq_black": self.king_sq_black,
            "piece_tokens": self.piece_tokens,
            "square_tokens": self.square_tokens,
            "state_context_global": self.state_context_global,
            "reachability_edges": self.reachability_edges,
            "nnue_feat_white": self.nnue_feat_white,
            "nnue_feat_black": self.nnue_feat_black,
            "candidate_action_indices": self.candidate_action_indices,
            "candidate_move_types": self.candidate_move_types,
            "candidate_delta_white_leave": self.candidate_delta_white_leave,
            "candidate_delta_white_enter": self.candidate_delta_white_enter,
            "candidate_delta_black_leave": self.candidate_delta_black_leave,
            "candidate_delta_black_enter": self.candidate_delta_black_enter,
            "candidate_is_white_king_move": self.candidate_is_white_king_move,
            "candidate_is_black_king_move": self.candidate_is_black_king_move,
            "candidate_nnue_feat_white_after_move": self.candidate_nnue_feat_white_after_move,
            "candidate_nnue_feat_black_after_move": self.candidate_nnue_feat_black_after_move,
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
            side_to_move=_optional_int_field(payload, "side_to_move", default=0),
            phase_index=_optional_int_field(payload, "phase_index", default=0),
            king_sq_white=_optional_int_field(payload, "king_sq_white", default=-1),
            king_sq_black=_optional_int_field(payload, "king_sq_black", default=-1),
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
            nnue_feat_white=[
                int(value)
                for value in _optional_list_field(payload, "nnue_feat_white", default=[])
            ],
            nnue_feat_black=[
                int(value)
                for value in _optional_list_field(payload, "nnue_feat_black", default=[])
            ],
            candidate_action_indices=[int(value) for value in list(payload["candidate_action_indices"])],
            candidate_move_types=[
                int(value)
                for value in _optional_list_field(
                    payload,
                    "candidate_move_types",
                    default=[0] * len(list(payload["candidate_action_indices"])),
                )
            ],
            candidate_delta_white_leave=[
                [int(value) for value in row]
                for row in _optional_list_field(
                    payload,
                    "candidate_delta_white_leave",
                    default=[[] for _ in list(payload["candidate_action_indices"])],
                )
            ],
            candidate_delta_white_enter=[
                [int(value) for value in row]
                for row in _optional_list_field(
                    payload,
                    "candidate_delta_white_enter",
                    default=[[] for _ in list(payload["candidate_action_indices"])],
                )
            ],
            candidate_delta_black_leave=[
                [int(value) for value in row]
                for row in _optional_list_field(
                    payload,
                    "candidate_delta_black_leave",
                    default=[[] for _ in list(payload["candidate_action_indices"])],
                )
            ],
            candidate_delta_black_enter=[
                [int(value) for value in row]
                for row in _optional_list_field(
                    payload,
                    "candidate_delta_black_enter",
                    default=[[] for _ in list(payload["candidate_action_indices"])],
                )
            ],
            candidate_is_white_king_move=[
                bool(value)
                for value in _optional_list_field(
                    payload,
                    "candidate_is_white_king_move",
                    default=[False] * len(list(payload["candidate_action_indices"])),
                )
            ],
            candidate_is_black_king_move=[
                bool(value)
                for value in _optional_list_field(
                    payload,
                    "candidate_is_black_king_move",
                    default=[False] * len(list(payload["candidate_action_indices"])),
                )
            ],
            candidate_nnue_feat_white_after_move=[
                [int(value) for value in row]
                for row in _optional_list_field(
                    payload,
                    "candidate_nnue_feat_white_after_move",
                    default=[[] for _ in list(payload["candidate_action_indices"])],
                )
            ],
            candidate_nnue_feat_black_after_move=[
                [int(value) for value in row]
                for row in _optional_list_field(
                    payload,
                    "candidate_nnue_feat_black_after_move",
                    default=[[] for _ in list(payload["candidate_action_indices"])],
                )
            ],
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
    dataset_example = _dataset_example_for_fen(example.fen)
    board = chess.Board(example.fen)
    feature_sections = split_position_features(example.feature_vector)
    piece_tokens = _decode_piece_tokens(feature_sections["piece"])
    square_tokens = _decode_square_tokens(feature_sections["square"])
    state_context = build_state_context_v1(dataset_example)
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
    candidate_move_types: list[int] = []
    candidate_delta_white_leave: list[list[int]] = []
    candidate_delta_white_enter: list[list[int]] = []
    candidate_delta_black_leave: list[list[int]] = []
    candidate_delta_black_enter: list[list[int]] = []
    candidate_is_white_king_move: list[bool] = []
    candidate_is_black_king_move: list[bool] = []
    candidate_nnue_feat_white_after_move: list[list[int]] = []
    candidate_nnue_feat_black_after_move: list[list[int]] = []
    for action_index in example.candidate_action_indices:
        move = chess.Move.from_uci(move_uci_for_action(dataset_example, int(action_index)))
        white_leave, white_enter = halfka_delta(board, move, "w")
        black_leave, black_enter = halfka_delta(board, move, "b")
        white_king_move = is_king_move(board, move, "w")
        black_king_move = is_king_move(board, move, "b")
        candidate_move_types.append(move_type_hash(board, move))
        candidate_delta_white_leave.append(white_leave)
        candidate_delta_white_enter.append(white_enter)
        candidate_delta_black_leave.append(black_leave)
        candidate_delta_black_enter.append(black_enter)
        candidate_is_white_king_move.append(white_king_move)
        candidate_is_black_king_move.append(black_king_move)
        if white_king_move or black_king_move:
            after_board = board.copy(stack=False)
            after_board.push(move)
            candidate_nnue_feat_white_after_move.append(halfka_active_indices(after_board, "w"))
            candidate_nnue_feat_black_after_move.append(halfka_active_indices(after_board, "b"))
        else:
            candidate_nnue_feat_white_after_move.append([])
            candidate_nnue_feat_black_after_move.append([])
    return LAPv1TrainingExample(
        sample_id=example.sample_id,
        split=example.split,
        side_to_move=0 if board.turn == chess.WHITE else 1,
        phase_index=phase_index(board),
        king_sq_white=int(board.king(chess.WHITE) if board.king(chess.WHITE) is not None else -1),
        king_sq_black=int(board.king(chess.BLACK) if board.king(chess.BLACK) is not None else -1),
        piece_tokens=piece_tokens,
        square_tokens=square_tokens,
        state_context_global=global_features,
        reachability_edges=reachability_edges,
        nnue_feat_white=halfka_active_indices(board, "w"),
        nnue_feat_black=halfka_active_indices(board, "b"),
        candidate_action_indices=list(example.candidate_action_indices),
        candidate_move_types=candidate_move_types,
        candidate_delta_white_leave=candidate_delta_white_leave,
        candidate_delta_white_enter=candidate_delta_white_enter,
        candidate_delta_black_leave=candidate_delta_black_leave,
        candidate_delta_black_enter=candidate_delta_black_enter,
        candidate_is_white_king_move=candidate_is_white_king_move,
        candidate_is_black_king_move=candidate_is_black_king_move,
        candidate_nnue_feat_white_after_move=candidate_nnue_feat_white_after_move,
        candidate_nnue_feat_black_after_move=candidate_nnue_feat_black_after_move,
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


def _optional_int_field(payload: dict[str, object], field_name: str, *, default: int) -> int:
    if field_name not in payload:
        _warn_missing_field_once(field_name)
        return default
    return int(payload[field_name])


def _optional_list_field(
    payload: dict[str, object],
    field_name: str,
    *,
    default: list[object],
) -> list[object]:
    if field_name not in payload:
        _warn_missing_field_once(field_name)
        return list(default)
    return list(payload[field_name])


def _warn_missing_field_once(field_name: str) -> None:
    if field_name in _MISSING_FIELD_WARNINGS:
        return
    _MISSING_FIELD_WARNINGS.add(field_name)
    warnings.warn(
        f"LAPv1 training artifact is missing '{field_name}'; using compatibility default.",
        RuntimeWarning,
        stacklevel=2,
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
