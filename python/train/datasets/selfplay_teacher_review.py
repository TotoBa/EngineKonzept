"""Post-game teacher reviews over completed selfplay sessions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from train.datasets.artifacts import build_symbolic_proposer_example
from train.datasets.oracle import label_records_with_oracle
from train.datasets.replay_buffer import load_arena_session_paths
from train.datasets.schema import SUPPORTED_SPLITS, RawPositionRecord
from train.datasets.search_teacher import (
    SearchTeacherExample,
    build_search_teacher_example_from_analysis,
)
from train.datasets.opponent_head import dataset_example_from_oracle_payload
from train.eval.selfplay import SelfplayGameRecord, SelfplayMoveRecord, SelfplaySessionRecord

try:
    import chess
    import chess.engine
except ModuleNotFoundError:  # pragma: no cover - exercised when chess is absent
    chess = None


SELFPLAY_TEACHER_REVIEW_ARTIFACT_PREFIX = "selfplay_teacher_review_"
DEFAULT_SELFPLAY_MISTAKE_DEADZONE_CP = 8.0
DEFAULT_SELFPLAY_MISTAKE_PRIORITY_SCALE_CP = 64.0
DEFAULT_SELFPLAY_MAX_PRIORITY = 4.0


@dataclass(frozen=True)
class SelfplayTeacherReviewExample:
    """Teacher-reviewed supervision for one non-external selfplay decision."""

    sample_id: str
    split: str
    agent_name: str
    game_id: str
    ply_index: int
    side_to_move: str
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
    selected_action_index: int
    selected_move_uci: str
    selected_candidate_index: int
    selected_score_cp: float
    selected_is_teacher_top1: bool
    game_result: str
    outcome_pov: str
    termination_reason: str
    mistake_deadzone_cp: float
    mistake_raw_cp: float
    mistake_cp: float
    mistake_priority: float

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "agent_name": self.agent_name,
            "game_id": self.game_id,
            "ply_index": self.ply_index,
            "side_to_move": self.side_to_move,
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
            "selected_action_index": self.selected_action_index,
            "selected_move_uci": self.selected_move_uci,
            "selected_candidate_index": self.selected_candidate_index,
            "selected_score_cp": self.selected_score_cp,
            "selected_is_teacher_top1": self.selected_is_teacher_top1,
            "game_result": self.game_result,
            "outcome_pov": self.outcome_pov,
            "termination_reason": self.termination_reason,
            "mistake_deadzone_cp": self.mistake_deadzone_cp,
            "mistake_raw_cp": self.mistake_raw_cp,
            "mistake_cp": self.mistake_cp,
            "mistake_priority": self.mistake_priority,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SelfplayTeacherReviewExample":
        split = str(payload["split"])
        if split not in SUPPORTED_SPLITS:
            raise ValueError(f"unsupported split: {split}")
        candidate_action_indices = [int(value) for value in list(payload["candidate_action_indices"])]
        candidate_features = [
            [float(value) for value in row] for row in list(payload["candidate_features"])
        ]
        teacher_candidate_scores_cp = [
            float(value) for value in list(payload["teacher_candidate_scores_cp"])
        ]
        teacher_policy = [float(value) for value in list(payload["teacher_policy"])]
        expected_length = len(candidate_action_indices)
        for name, values in (
            ("candidate_features", candidate_features),
            ("teacher_candidate_scores_cp", teacher_candidate_scores_cp),
            ("teacher_policy", teacher_policy),
        ):
            if len(values) != expected_length:
                raise ValueError(
                    f"{name} must have the same length as candidate_action_indices"
                )
        selected_candidate_index = int(payload["selected_candidate_index"])
        if not 0 <= selected_candidate_index < expected_length:
            raise ValueError("selected_candidate_index out of range")
        return cls(
            sample_id=str(payload["sample_id"]),
            split=split,
            agent_name=str(payload["agent_name"]),
            game_id=str(payload["game_id"]),
            ply_index=int(payload["ply_index"]),
            side_to_move=str(payload["side_to_move"]),
            fen=str(payload["fen"]),
            feature_vector=[float(value) for value in list(payload["feature_vector"])],
            candidate_context_version=int(payload["candidate_context_version"]),
            global_context_version=int(payload["global_context_version"]),
            global_features=[float(value) for value in list(payload["global_features"])],
            candidate_action_indices=candidate_action_indices,
            candidate_features=candidate_features,
            teacher_engine=str(payload["teacher_engine"]),
            teacher_nodes=_optional_int(payload.get("teacher_nodes")),
            teacher_depth=_optional_int(payload.get("teacher_depth")),
            teacher_movetime_ms=_optional_int(payload.get("teacher_movetime_ms")),
            teacher_multipv=int(payload["teacher_multipv"]),
            teacher_coverage_ratio=float(payload["teacher_coverage_ratio"]),
            teacher_root_value_cp=float(payload["teacher_root_value_cp"]),
            teacher_root_value_mate=_optional_int(payload.get("teacher_root_value_mate")),
            teacher_candidate_scores_cp=teacher_candidate_scores_cp,
            teacher_policy=teacher_policy,
            teacher_top_k_action_indices=[
                int(value) for value in list(payload["teacher_top_k_action_indices"])
            ],
            teacher_pv_uci=[str(value) for value in list(payload["teacher_pv_uci"])],
            selected_action_index=int(payload["selected_action_index"]),
            selected_move_uci=str(payload["selected_move_uci"]),
            selected_candidate_index=selected_candidate_index,
            selected_score_cp=float(payload["selected_score_cp"]),
            selected_is_teacher_top1=bool(payload["selected_is_teacher_top1"]),
            game_result=str(payload["game_result"]),
            outcome_pov=str(payload["outcome_pov"]),
            termination_reason=str(payload["termination_reason"]),
            mistake_deadzone_cp=float(payload["mistake_deadzone_cp"]),
            mistake_raw_cp=float(payload["mistake_raw_cp"]),
            mistake_cp=float(payload["mistake_cp"]),
            mistake_priority=float(payload["mistake_priority"]),
        )

    @classmethod
    def from_json(cls, line: str, *, source: str = "<jsonl>") -> "SelfplayTeacherReviewExample":
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{source}: selfplay teacher review example must be a JSON object")
        return cls.from_dict(payload)


def selfplay_teacher_review_artifact_name(split: str) -> str:
    """Return the canonical selfplay teacher-review artifact filename for one split."""
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"unsupported split: {split}")
    return f"{SELFPLAY_TEACHER_REVIEW_ARTIFACT_PREFIX}{split}.jsonl"


def load_selfplay_teacher_review_examples(path: Path) -> list[SelfplayTeacherReviewExample]:
    """Load selfplay teacher-review examples from JSONL."""
    if not path.exists():
        raise FileNotFoundError(f"selfplay teacher-review artifact not found: {path}")
    examples: list[SelfplayTeacherReviewExample] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        examples.append(
            SelfplayTeacherReviewExample.from_json(line, source=f"{path}:{line_number}")
        )
    return examples


def write_selfplay_teacher_review_artifact(
    path: Path,
    examples: Sequence[SelfplayTeacherReviewExample],
) -> None:
    """Write selfplay teacher-review examples as JSONL."""
    lines = [json.dumps(example.to_dict(), sort_keys=True) for example in examples]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def selfplay_teacher_review_summary(
    examples: Sequence[SelfplayTeacherReviewExample],
) -> dict[str, object]:
    """Compute a compact summary for teacher-reviewed selfplay examples."""
    if not examples:
        return {
            "example_count": 0,
            "mistake_count": 0,
            "teacher_top1_match_count": 0,
            "mean_mistake_cp": 0.0,
            "mean_mistake_priority": 0.0,
            "outcome_counts": {},
            "termination_counts": {},
        }
    outcome_counts: dict[str, int] = {}
    termination_counts: dict[str, int] = {}
    mistake_count = 0
    teacher_top1_match_count = 0
    for example in examples:
        outcome_counts[example.outcome_pov] = outcome_counts.get(example.outcome_pov, 0) + 1
        termination_counts[example.termination_reason] = (
            termination_counts.get(example.termination_reason, 0) + 1
        )
        if example.mistake_cp > 0.0:
            mistake_count += 1
        if example.selected_is_teacher_top1:
            teacher_top1_match_count += 1
    return {
        "example_count": len(examples),
        "mistake_count": mistake_count,
        "teacher_top1_match_count": teacher_top1_match_count,
        "mean_mistake_cp": round(
            sum(example.mistake_cp for example in examples) / len(examples),
            6,
        ),
        "mean_mistake_priority": round(
            sum(example.mistake_priority for example in examples) / len(examples),
            6,
        ),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "termination_counts": dict(sorted(termination_counts.items())),
    }


def build_selfplay_teacher_review_examples(
    *,
    arena_summary_path: Path,
    trainable_agent_names: Sequence[str],
    repo_root: Path,
    teacher_engine_path: Path,
    split: str,
    nodes: int | None,
    depth: int | None,
    movetime_ms: int | None,
    multipv: int,
    policy_temperature_cp: float,
    mistake_deadzone_cp: float = DEFAULT_SELFPLAY_MISTAKE_DEADZONE_CP,
    mistake_priority_scale_cp: float = DEFAULT_SELFPLAY_MISTAKE_PRIORITY_SCALE_CP,
    max_mistake_priority: float = DEFAULT_SELFPLAY_MAX_PRIORITY,
    max_examples_per_agent: int | None = None,
) -> dict[str, list[SelfplayTeacherReviewExample]]:
    """Build per-agent post-game teacher reviews from completed arena sessions."""
    if chess is None:  # pragma: no cover - exercised when chess is absent
        raise RuntimeError(
            "python-chess is required for selfplay teacher reviews. Install the 'train' extra."
        )
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"unsupported split: {split}")
    if nodes is None and depth is None and movetime_ms is None:
        raise ValueError("one of nodes, depth, or movetime_ms must be set")
    if policy_temperature_cp <= 0.0:
        raise ValueError("policy_temperature_cp must be positive")
    if max_examples_per_agent is not None and max_examples_per_agent <= 0:
        raise ValueError("max_examples_per_agent must be positive when provided")

    session_paths = load_arena_session_paths(arena_summary_path)
    sessions = [
        SelfplaySessionRecord.from_json(path.read_text(encoding="utf-8"))
        for path in session_paths
    ]
    selected_moves = _collect_trainable_moves(
        sessions,
        session_paths=session_paths,
        trainable_agent_names=set(trainable_agent_names),
        max_examples_per_agent=max_examples_per_agent,
    )
    built_by_agent = {agent_name: [] for agent_name in sorted(set(trainable_agent_names))}
    if not selected_moves:
        return built_by_agent

    records = [
        RawPositionRecord(
            sample_id=move_record["sample_id"],
            fen=move_record["fen"],
            source="selfplay_teacher_review",
        )
        for move_record in selected_moves
    ]
    payloads = label_records_with_oracle(records, repo_root=repo_root)
    dataset_examples = [
        dataset_example_from_oracle_payload(
            sample_id=move_record["sample_id"],
            split=split,
            source="selfplay_teacher_review",
            fen=move_record["fen"],
            payload=payload,
        )
        for move_record, payload in zip(selected_moves, payloads, strict=True)
    ]

    limit = chess.engine.Limit(
        nodes=nodes,
        depth=depth,
        time=None if movetime_ms is None else movetime_ms / 1000.0,
    )
    teacher_name = str(teacher_engine_path)
    with chess.engine.SimpleEngine.popen_uci(str(teacher_engine_path)) as engine:
        for move_record, dataset_example in zip(selected_moves, dataset_examples, strict=True):
            symbolic_example = build_symbolic_proposer_example(
                dataset_example,
                candidate_context_version=2,
                global_context_version=1,
            )
            effective_multipv = (
                len(symbolic_example.candidate_action_indices)
                if multipv <= 0
                else min(multipv, len(symbolic_example.candidate_action_indices))
            )
            board = chess.Board(dataset_example.fen)
            infos = engine.analyse(board, limit, multipv=effective_multipv)
            analysis_list = infos if isinstance(infos, list) else [infos]
            teacher_example = build_search_teacher_example_from_analysis(
                dataset_example,
                symbolic_example=symbolic_example,
                analysis_list=analysis_list,
                teacher_engine=teacher_name,
                nodes=nodes,
                depth=depth,
                movetime_ms=movetime_ms,
                effective_multipv=effective_multipv,
                policy_temperature_cp=policy_temperature_cp,
            )
            built_by_agent[move_record["agent_name"]].append(
                build_selfplay_teacher_review_example_from_teacher(
                    teacher_example,
                    agent_name=move_record["agent_name"],
                    game_id=move_record["game_id"],
                    ply_index=move_record["ply_index"],
                    side_to_move=move_record["side_to_move"],
                    selected_action_index=move_record["action_index"],
                    selected_move_uci=move_record["move_uci"],
                    game_result=move_record["game_result"],
                    outcome_pov=move_record["outcome_pov"],
                    termination_reason=move_record["termination_reason"],
                    mistake_deadzone_cp=mistake_deadzone_cp,
                    mistake_priority_scale_cp=mistake_priority_scale_cp,
                    max_mistake_priority=max_mistake_priority,
                )
            )
    return built_by_agent


def build_selfplay_teacher_review_example_from_teacher(
    teacher_example: SearchTeacherExample,
    *,
    agent_name: str,
    game_id: str,
    ply_index: int,
    side_to_move: str,
    selected_action_index: int,
    selected_move_uci: str,
    game_result: str,
    outcome_pov: str,
    termination_reason: str,
    mistake_deadzone_cp: float = DEFAULT_SELFPLAY_MISTAKE_DEADZONE_CP,
    mistake_priority_scale_cp: float = DEFAULT_SELFPLAY_MISTAKE_PRIORITY_SCALE_CP,
    max_mistake_priority: float = DEFAULT_SELFPLAY_MAX_PRIORITY,
) -> SelfplayTeacherReviewExample:
    """Fuse teacher labels with the actual played move and mistake severity."""
    if selected_action_index not in teacher_example.candidate_action_indices:
        raise ValueError(
            f"{teacher_example.sample_id}: selected action {selected_action_index} is not in the teacher candidate set"
        )
    selected_candidate_index = teacher_example.candidate_action_indices.index(selected_action_index)
    selected_score_cp = float(
        teacher_example.teacher_candidate_scores_cp[selected_candidate_index]
    )
    teacher_top1_action_index = int(teacher_example.teacher_top_k_action_indices[0])
    mistake_raw_cp = max(0.0, float(teacher_example.teacher_root_value_cp) - selected_score_cp)
    mistake_cp = max(0.0, mistake_raw_cp - mistake_deadzone_cp)
    return SelfplayTeacherReviewExample(
        sample_id=teacher_example.sample_id,
        split=teacher_example.split,
        agent_name=agent_name,
        game_id=game_id,
        ply_index=ply_index,
        side_to_move=side_to_move,
        fen=teacher_example.fen,
        feature_vector=list(teacher_example.feature_vector),
        candidate_context_version=int(teacher_example.candidate_context_version),
        global_context_version=int(teacher_example.global_context_version),
        global_features=list(teacher_example.global_features),
        candidate_action_indices=list(teacher_example.candidate_action_indices),
        candidate_features=[list(row) for row in teacher_example.candidate_features],
        teacher_engine=teacher_example.teacher_engine,
        teacher_nodes=teacher_example.teacher_nodes,
        teacher_depth=teacher_example.teacher_depth,
        teacher_movetime_ms=teacher_example.teacher_movetime_ms,
        teacher_multipv=teacher_example.teacher_multipv,
        teacher_coverage_ratio=float(teacher_example.teacher_coverage_ratio),
        teacher_root_value_cp=float(teacher_example.teacher_root_value_cp),
        teacher_root_value_mate=teacher_example.teacher_root_value_mate,
        teacher_candidate_scores_cp=list(teacher_example.teacher_candidate_scores_cp),
        teacher_policy=list(teacher_example.teacher_policy),
        teacher_top_k_action_indices=list(teacher_example.teacher_top_k_action_indices),
        teacher_pv_uci=list(teacher_example.teacher_pv_uci),
        selected_action_index=selected_action_index,
        selected_move_uci=selected_move_uci,
        selected_candidate_index=selected_candidate_index,
        selected_score_cp=selected_score_cp,
        selected_is_teacher_top1=(selected_action_index == teacher_top1_action_index),
        game_result=game_result,
        outcome_pov=outcome_pov,
        termination_reason=termination_reason,
        mistake_deadzone_cp=float(mistake_deadzone_cp),
        mistake_raw_cp=round(mistake_raw_cp, 6),
        mistake_cp=round(mistake_cp, 6),
        mistake_priority=build_selfplay_mistake_priority(
            mistake_cp,
            scale_cp=mistake_priority_scale_cp,
            max_priority=max_mistake_priority,
        ),
    )


def build_selfplay_mistake_priority(
    mistake_cp: float,
    *,
    scale_cp: float = DEFAULT_SELFPLAY_MISTAKE_PRIORITY_SCALE_CP,
    max_priority: float = DEFAULT_SELFPLAY_MAX_PRIORITY,
) -> float:
    """Map centipawn regret to a clipped curriculum priority."""
    if scale_cp <= 0.0:
        raise ValueError("scale_cp must be positive")
    if max_priority <= 0.0:
        raise ValueError("max_priority must be positive")
    return round(min(max_priority, max(0.0, float(mistake_cp)) / scale_cp), 6)


def _collect_trainable_moves(
    sessions: Sequence[SelfplaySessionRecord],
    *,
    session_paths: Sequence[Path],
    trainable_agent_names: set[str],
    max_examples_per_agent: int | None,
) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    counts_by_agent = {agent_name: 0 for agent_name in sorted(trainable_agent_names)}
    for session_path, session in zip(session_paths, sessions, strict=True):
        session_label = session_path.stem
        for game in session.games:
            for move in game.moves:
                agent_name = _agent_name_for_move(game, move)
                if agent_name not in trainable_agent_names:
                    continue
                if (
                    max_examples_per_agent is not None
                    and counts_by_agent[agent_name] >= max_examples_per_agent
                ):
                    continue
                counts_by_agent[agent_name] += 1
                collected.append(
                    {
                        "sample_id": f"{session_label}:{game.game_id}:{move.ply_index}:{agent_name}",
                        "agent_name": agent_name,
                        "game_id": f"{session_label}:{game.game_id}",
                        "ply_index": move.ply_index,
                        "side_to_move": move.side_to_move,
                        "fen": move.fen,
                        "move_uci": move.move_uci,
                        "action_index": move.action_index,
                        "game_result": game.result,
                        "outcome_pov": _outcome_pov(game.result, move.side_to_move),
                        "termination_reason": game.termination_reason,
                    }
                )
    return collected


def _agent_name_for_move(game: SelfplayGameRecord, move: SelfplayMoveRecord) -> str:
    return game.white_agent if move.side_to_move == "w" else game.black_agent


def _outcome_pov(game_result: str, side_to_move: str) -> str:
    if game_result == "1-0":
        return "win" if side_to_move == "w" else "loss"
    if game_result == "0-1":
        return "win" if side_to_move == "b" else "loss"
    if game_result == "1/2-1/2":
        return "draw"
    if game_result == "*":
        return "unfinished"
    raise ValueError(f"unsupported game_result: {game_result}")


def _optional_int(value: object | None) -> int | None:
    if value is None:
        return None
    return int(value)
