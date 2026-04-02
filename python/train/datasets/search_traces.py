"""Offline search-trace artifacts over exact symbolic legal candidates."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from train.action_space import flatten_action
from train.datasets.artifacts import build_symbolic_proposer_example
from train.datasets.schema import DatasetExample, SUPPORTED_SPLITS
from train.datasets.search_teacher import (
    build_search_teacher_example_from_analysis,
)

try:
    import chess
    import chess.engine
except ModuleNotFoundError:  # pragma: no cover - exercised when chess is absent
    chess = None

SEARCH_TRACES_ARTIFACT_PREFIX = "search_traces_"


@dataclass(frozen=True)
class SearchTraceExample:
    """Offline search-trace record aligned to exact legal candidates."""

    sample_id: str
    split: str
    fen: str
    feature_vector: list[float]
    candidate_context_version: int
    global_context_version: int
    global_features: list[float]
    candidate_action_indices: list[int]
    candidate_features: list[list[float]]
    teacher_engine: str
    teacher_nodes: int | None
    teacher_depth: int | None
    teacher_movetime_ms: int | None
    teacher_multipv: int
    teacher_coverage_ratio: float
    teacher_root_value_cp: float
    teacher_root_value_mate: int | None
    teacher_candidate_scores_cp: list[float]
    teacher_top_k_action_indices: list[int]
    principal_variation_uci: list[str]
    principal_variation_action_indices: list[int]
    best_reply_uci: str | None
    best_reply_action_index: int | None
    pv_length: int
    top1_minus_top2_cp: float | None

    def to_dict(self) -> dict[str, object]:
        """Return the JSON representation."""
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "fen": self.fen,
            "feature_vector": self.feature_vector,
            "candidate_context_version": self.candidate_context_version,
            "global_context_version": self.global_context_version,
            "global_features": self.global_features,
            "candidate_action_indices": self.candidate_action_indices,
            "candidate_features": self.candidate_features,
            "teacher_engine": self.teacher_engine,
            "teacher_nodes": self.teacher_nodes,
            "teacher_depth": self.teacher_depth,
            "teacher_movetime_ms": self.teacher_movetime_ms,
            "teacher_multipv": self.teacher_multipv,
            "teacher_coverage_ratio": self.teacher_coverage_ratio,
            "teacher_root_value_cp": self.teacher_root_value_cp,
            "teacher_root_value_mate": self.teacher_root_value_mate,
            "teacher_candidate_scores_cp": self.teacher_candidate_scores_cp,
            "teacher_top_k_action_indices": self.teacher_top_k_action_indices,
            "principal_variation_uci": self.principal_variation_uci,
            "principal_variation_action_indices": self.principal_variation_action_indices,
            "best_reply_uci": self.best_reply_uci,
            "best_reply_action_index": self.best_reply_action_index,
            "pv_length": self.pv_length,
            "top1_minus_top2_cp": self.top1_minus_top2_cp,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SearchTraceExample":
        """Construct the search-trace example from JSON."""
        split = str(payload["split"])
        if split not in SUPPORTED_SPLITS:
            raise ValueError(f"unsupported split: {split}")
        return cls(
            sample_id=str(payload["sample_id"]),
            split=split,
            fen=str(payload["fen"]),
            feature_vector=[float(value) for value in list(payload["feature_vector"])],
            candidate_context_version=int(payload["candidate_context_version"]),
            global_context_version=int(payload["global_context_version"]),
            global_features=[float(value) for value in list(payload["global_features"])],
            candidate_action_indices=[int(value) for value in list(payload["candidate_action_indices"])],
            candidate_features=[
                [float(value) for value in row] for row in list(payload["candidate_features"])
            ],
            teacher_engine=str(payload["teacher_engine"]),
            teacher_nodes=_optional_int(payload.get("teacher_nodes")),
            teacher_depth=_optional_int(payload.get("teacher_depth")),
            teacher_movetime_ms=_optional_int(payload.get("teacher_movetime_ms")),
            teacher_multipv=int(payload["teacher_multipv"]),
            teacher_coverage_ratio=float(payload["teacher_coverage_ratio"]),
            teacher_root_value_cp=float(payload["teacher_root_value_cp"]),
            teacher_root_value_mate=_optional_int(payload.get("teacher_root_value_mate")),
            teacher_candidate_scores_cp=[
                float(value) for value in list(payload["teacher_candidate_scores_cp"])
            ],
            teacher_top_k_action_indices=[
                int(value) for value in list(payload["teacher_top_k_action_indices"])
            ],
            principal_variation_uci=[str(value) for value in list(payload["principal_variation_uci"])],
            principal_variation_action_indices=[
                int(value) for value in list(payload["principal_variation_action_indices"])
            ],
            best_reply_uci=_optional_str(payload.get("best_reply_uci")),
            best_reply_action_index=_optional_int(payload.get("best_reply_action_index")),
            pv_length=int(payload["pv_length"]),
            top1_minus_top2_cp=_optional_float(payload.get("top1_minus_top2_cp")),
        )

    @classmethod
    def from_json(cls, line: str, *, source: str = "<jsonl>") -> "SearchTraceExample":
        """Parse the example from one JSON line."""
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{source}: search trace example must be a JSON object")
        return cls.from_dict(payload)


def search_traces_artifact_name(split: str) -> str:
    """Return the canonical search-trace artifact filename for one split."""
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"unsupported split: {split}")
    return f"{SEARCH_TRACES_ARTIFACT_PREFIX}{split}.jsonl"


def load_search_trace_examples(path: Path) -> list[SearchTraceExample]:
    """Load search-trace examples from JSONL."""
    if not path.exists():
        raise FileNotFoundError(f"search trace artifact not found: {path}")

    examples: list[SearchTraceExample] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        examples.append(SearchTraceExample.from_json(line, source=f"{path}:{line_number}"))
    return examples


def write_search_trace_artifact(path: Path, examples: Sequence[SearchTraceExample]) -> None:
    """Write search-trace examples as JSONL."""
    lines = [json.dumps(example.to_dict(), sort_keys=True) for example in examples]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_search_trace_examples(
    examples: Sequence[DatasetExample],
    *,
    teacher_engine_path: Path,
    nodes: int | None,
    depth: int | None,
    movetime_ms: int | None,
    multipv: int,
    policy_temperature_cp: float,
    max_examples: int | None = None,
) -> list[SearchTraceExample]:
    """Materialize offline search traces with a UCI alpha-beta teacher."""
    if chess is None:  # pragma: no cover - exercised when chess is absent
        raise RuntimeError(
            "python-chess is required for search-trace workflows. Install the 'train' extra."
        )
    if nodes is None and depth is None and movetime_ms is None:
        raise ValueError("one of nodes, depth, or movetime_ms must be set")

    selected_examples = list(examples[:max_examples] if max_examples is not None else examples)
    if not selected_examples:
        return []

    limit = chess.engine.Limit(
        nodes=nodes,
        depth=depth,
        time=None if movetime_ms is None else movetime_ms / 1000.0,
    )
    teacher_name = str(teacher_engine_path)

    built: list[SearchTraceExample] = []
    with chess.engine.SimpleEngine.popen_uci(str(teacher_engine_path)) as engine:
        for example in selected_examples:
            board = chess.Board(example.fen)
            symbolic_example = build_symbolic_proposer_example(
                example,
                candidate_context_version=2,
                global_context_version=1,
            )
            effective_multipv = (
                len(symbolic_example.candidate_action_indices)
                if multipv <= 0
                else min(multipv, len(symbolic_example.candidate_action_indices))
            )
            infos = engine.analyse(board, limit, multipv=effective_multipv)
            analysis_list = infos if isinstance(infos, list) else [infos]
            built.append(
                build_search_trace_example_from_analysis(
                    example,
                    symbolic_example=symbolic_example,
                    analysis_list=analysis_list,
                    teacher_engine=teacher_name,
                    nodes=nodes,
                    depth=depth,
                    movetime_ms=movetime_ms,
                    effective_multipv=effective_multipv,
                    policy_temperature_cp=policy_temperature_cp,
                )
            )
    return built


def build_search_trace_example_from_analysis(
    example: DatasetExample,
    *,
    symbolic_example: Any,
    analysis_list: Sequence[dict[str, Any]],
    teacher_engine: str,
    nodes: int | None,
    depth: int | None,
    movetime_ms: int | None,
    effective_multipv: int,
    policy_temperature_cp: float,
) -> SearchTraceExample:
    """Build one search-trace example from exact candidate context and teacher analysis."""
    if chess is None:  # pragma: no cover - exercised when chess is absent
        raise RuntimeError(
            "python-chess is required for search-trace workflows. Install the 'train' extra."
        )

    teacher_example = build_search_teacher_example_from_analysis(
        example,
        symbolic_example=symbolic_example,
        analysis_list=analysis_list,
        teacher_engine=teacher_engine,
        nodes=nodes,
        depth=depth,
        movetime_ms=movetime_ms,
        effective_multipv=effective_multipv,
        policy_temperature_cp=policy_temperature_cp,
    )

    principal_variation_uci: list[str] = []
    if analysis_list:
        principal_variation_uci = [
            move.uci() for move in list(analysis_list[0].get("pv") or [])
        ]
    principal_variation_action_indices = _encode_pv_action_indices(
        example.fen,
        principal_variation_uci,
    )
    best_reply_uci = principal_variation_uci[1] if len(principal_variation_uci) >= 2 else None
    best_reply_action_index = (
        principal_variation_action_indices[1]
        if len(principal_variation_action_indices) >= 2
        else None
    )
    top1_minus_top2_cp = _top1_minus_top2_gap(teacher_example.teacher_candidate_scores_cp)

    return SearchTraceExample(
        sample_id=teacher_example.sample_id,
        split=teacher_example.split,
        fen=teacher_example.fen,
        feature_vector=list(teacher_example.feature_vector),
        candidate_context_version=teacher_example.candidate_context_version,
        global_context_version=teacher_example.global_context_version,
        global_features=list(teacher_example.global_features),
        candidate_action_indices=list(teacher_example.candidate_action_indices),
        candidate_features=[list(row) for row in teacher_example.candidate_features],
        teacher_engine=teacher_example.teacher_engine,
        teacher_nodes=teacher_example.teacher_nodes,
        teacher_depth=teacher_example.teacher_depth,
        teacher_movetime_ms=teacher_example.teacher_movetime_ms,
        teacher_multipv=teacher_example.teacher_multipv,
        teacher_coverage_ratio=teacher_example.teacher_coverage_ratio,
        teacher_root_value_cp=teacher_example.teacher_root_value_cp,
        teacher_root_value_mate=teacher_example.teacher_root_value_mate,
        teacher_candidate_scores_cp=list(teacher_example.teacher_candidate_scores_cp),
        teacher_top_k_action_indices=list(teacher_example.teacher_top_k_action_indices),
        principal_variation_uci=principal_variation_uci,
        principal_variation_action_indices=principal_variation_action_indices,
        best_reply_uci=best_reply_uci,
        best_reply_action_index=best_reply_action_index,
        pv_length=len(principal_variation_uci),
        top1_minus_top2_cp=top1_minus_top2_cp,
    )


def _encode_pv_action_indices(fen: str, pv_uci: Sequence[str]) -> list[int]:
    board = chess.Board(fen)
    action_indices: list[int] = []
    for move_uci in pv_uci:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            raise ValueError(f"illegal PV move for trace: {move_uci}")
        action_indices.append(_flatten_move(move))
        board.push(move)
    return action_indices


def _flatten_move(move: Any) -> int:
    promotion_index = {
        None: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
    }[move.promotion]
    return flatten_action([move.from_square, move.to_square, promotion_index])


def _top1_minus_top2_gap(scores: Sequence[float]) -> float | None:
    if len(scores) < 2:
        return None
    ordered = sorted((float(value) for value in scores), reverse=True)
    return ordered[0] - ordered[1]


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
