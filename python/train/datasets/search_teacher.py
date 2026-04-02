"""Offline alpha-beta teacher artifacts over exact symbolic legal candidates."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

from train.action_space import flatten_action
from train.datasets.artifacts import build_symbolic_proposer_example
from train.datasets.schema import DatasetExample, SUPPORTED_SPLITS

try:
    import chess
    import chess.engine
except ModuleNotFoundError:  # pragma: no cover - exercised when chess is absent
    chess = None

SEARCH_TEACHER_ARTIFACT_PREFIX = "search_teacher_"


@dataclass(frozen=True)
class SearchTeacherExample:
    """Offline search-teacher labels aligned to exact legal candidates."""

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
    teacher_policy: list[float]
    teacher_top_k_action_indices: list[int]
    teacher_pv_uci: list[str]

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
            "teacher_policy": self.teacher_policy,
            "teacher_top_k_action_indices": self.teacher_top_k_action_indices,
            "teacher_pv_uci": self.teacher_pv_uci,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SearchTeacherExample":
        """Construct the search-teacher example from JSON."""
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
            teacher_policy=[float(value) for value in list(payload["teacher_policy"])],
            teacher_top_k_action_indices=[
                int(value) for value in list(payload["teacher_top_k_action_indices"])
            ],
            teacher_pv_uci=[str(value) for value in list(payload["teacher_pv_uci"])],
        )

    @classmethod
    def from_json(cls, line: str, *, source: str = "<jsonl>") -> "SearchTeacherExample":
        """Parse the example from one JSON line."""
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{source}: search teacher example must be a JSON object")
        return cls.from_dict(payload)


def search_teacher_artifact_name(split: str) -> str:
    """Return the canonical search-teacher artifact filename for one split."""
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"unsupported split: {split}")
    return f"{SEARCH_TEACHER_ARTIFACT_PREFIX}{split}.jsonl"


def load_search_teacher_examples(path: Path) -> list[SearchTeacherExample]:
    """Load search-teacher examples from JSONL."""
    if not path.exists():
        raise FileNotFoundError(f"search teacher artifact not found: {path}")

    examples: list[SearchTeacherExample] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        examples.append(SearchTeacherExample.from_json(line, source=f"{path}:{line_number}"))
    return examples


def write_search_teacher_artifact(path: Path, examples: Sequence[SearchTeacherExample]) -> None:
    """Write search-teacher examples as JSONL."""
    lines = [json.dumps(example.to_dict(), sort_keys=True) for example in examples]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_search_teacher_examples(
    examples: Sequence[DatasetExample],
    *,
    teacher_engine_path: Path,
    nodes: int | None,
    depth: int | None,
    movetime_ms: int | None,
    multipv: int,
    policy_temperature_cp: float,
    max_examples: int | None = None,
) -> list[SearchTeacherExample]:
    """Materialize offline search-teacher labels with a UCI alpha-beta teacher."""
    if chess is None:  # pragma: no cover - exercised when chess is absent
        raise RuntimeError(
            "python-chess is required for search-teacher workflows. Install the 'train' extra."
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

    built: list[SearchTeacherExample] = []
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
                build_search_teacher_example_from_analysis(
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


def build_search_teacher_example_from_analysis(
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
) -> SearchTeacherExample:
    """Build one search-teacher example from exact candidate context and teacher analysis."""
    if chess is None:  # pragma: no cover - exercised when chess is absent
        raise RuntimeError(
            "python-chess is required for search-teacher workflows. Install the 'train' extra."
        )
    if policy_temperature_cp <= 0.0:
        raise ValueError("policy_temperature_cp must be positive")

    legal_action_by_uci = {
        move_uci: flatten_action(action)
        for move_uci, action in zip(example.legal_moves, example.legal_action_encodings, strict=True)
    }
    candidate_index_by_action = {
        action_index: index
        for index, action_index in enumerate(symbolic_example.candidate_action_indices)
    }

    observed_scores: dict[int, float] = {}
    top_k_action_indices: list[int] = []
    root_value_cp = 0.0
    root_value_mate: int | None = None
    root_pv_uci: list[str] = []

    for info_index, info in enumerate(analysis_list):
        pv = info.get("pv") or []
        if not pv:
            continue
        first_move = pv[0]
        move_uci = first_move.uci()
        if move_uci not in legal_action_by_uci:
            raise ValueError(f"{example.sample_id}: teacher proposed illegal root move {move_uci}")
        action_index = legal_action_by_uci[move_uci]
        top_k_action_indices.append(action_index)
        score_cp = _pov_score_cp(info["score"], chess.Board(example.fen).turn)
        observed_scores[action_index] = score_cp

        if info_index == 0:
            root_value_cp = score_cp
            root_value_mate = _pov_score_mate(info["score"], chess.Board(example.fen).turn)
            root_pv_uci = [candidate.uci() for candidate in pv]

    if not observed_scores:
        raise ValueError(f"{example.sample_id}: teacher analysis returned no PV entries")

    floor_score = min(observed_scores.values()) - 100.0
    candidate_scores = [floor_score] * len(symbolic_example.candidate_action_indices)
    for action_index, score_cp in observed_scores.items():
        candidate_scores[candidate_index_by_action[action_index]] = score_cp

    teacher_policy = _softmax_scores(candidate_scores, temperature_cp=policy_temperature_cp)
    coverage_ratio = len(observed_scores) / max(len(symbolic_example.candidate_action_indices), 1)

    return SearchTeacherExample(
        sample_id=example.sample_id,
        split=example.split,
        fen=example.fen,
        feature_vector=list(symbolic_example.feature_vector),
        candidate_context_version=int(symbolic_example.candidate_context_version),
        global_context_version=int(symbolic_example.global_context_version),
        global_features=list(symbolic_example.global_features),
        candidate_action_indices=list(symbolic_example.candidate_action_indices),
        candidate_features=[list(row) for row in symbolic_example.candidate_features],
        teacher_engine=teacher_engine,
        teacher_nodes=nodes,
        teacher_depth=depth,
        teacher_movetime_ms=movetime_ms,
        teacher_multipv=effective_multipv,
        teacher_coverage_ratio=coverage_ratio,
        teacher_root_value_cp=root_value_cp,
        teacher_root_value_mate=root_value_mate,
        teacher_candidate_scores_cp=candidate_scores,
        teacher_policy=teacher_policy,
        teacher_top_k_action_indices=top_k_action_indices,
        teacher_pv_uci=root_pv_uci,
    )


def _softmax_scores(scores: Sequence[float], *, temperature_cp: float) -> list[float]:
    scaled = [score / temperature_cp for score in scores]
    max_scaled = max(scaled)
    exps = [math.exp(value - max_scaled) for value in scaled]
    total = sum(exps)
    return [value / total for value in exps]


def _pov_score_cp(score: Any, turn: Any) -> float:
    pov = score.pov(turn)
    value = pov.score(mate_score=100_000)
    if value is None:
        return 0.0
    return float(value)


def _pov_score_mate(score: Any, turn: Any) -> int | None:
    pov = score.pov(turn)
    mate = pov.mate()
    return None if mate is None else int(mate)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)
