"""Planner replay artifacts derived from exact selfplay replay buffers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

from train.datasets.replay_buffer import ReplayBufferEntry
from train.datasets.schema import SUPPORTED_SPLITS


PLANNER_REPLAY_ARTIFACT_PREFIX = "planner_replay_"
PLANNER_REPLAY_VALUE_SCALE_CP = 256.0


@dataclass(frozen=True)
class PlannerReplayExample:
    """One replay-derived planner supervision row."""

    sample_id: str
    split: str
    fen: str
    side_to_move: str
    selected_action_index: int
    selected_move_uci: str
    selector_name: str
    outcome_pov: str
    termination_reason: str
    replay_priority: float
    root_value_cp: float
    proposer_score: float
    planner_score: float
    reply_peak_probability: float
    pressure: float
    uncertainty: float

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "fen": self.fen,
            "side_to_move": self.side_to_move,
            "selected_action_index": self.selected_action_index,
            "selected_move_uci": self.selected_move_uci,
            "selector_name": self.selector_name,
            "outcome_pov": self.outcome_pov,
            "termination_reason": self.termination_reason,
            "replay_priority": self.replay_priority,
            "root_value_cp": self.root_value_cp,
            "proposer_score": self.proposer_score,
            "planner_score": self.planner_score,
            "reply_peak_probability": self.reply_peak_probability,
            "pressure": self.pressure,
            "uncertainty": self.uncertainty,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "PlannerReplayExample":
        split = str(payload["split"])
        if split not in SUPPORTED_SPLITS:
            raise ValueError(f"unsupported split: {split}")
        return cls(
            sample_id=str(payload["sample_id"]),
            split=split,
            fen=str(payload["fen"]),
            side_to_move=str(payload["side_to_move"]),
            selected_action_index=int(payload["selected_action_index"]),
            selected_move_uci=str(payload["selected_move_uci"]),
            selector_name=str(payload["selector_name"]),
            outcome_pov=str(payload["outcome_pov"]),
            termination_reason=str(payload["termination_reason"]),
            replay_priority=float(payload["replay_priority"]),
            root_value_cp=float(payload["root_value_cp"]),
            proposer_score=float(payload["proposer_score"]),
            planner_score=float(payload["planner_score"]),
            reply_peak_probability=float(payload["reply_peak_probability"]),
            pressure=float(payload["pressure"]),
            uncertainty=float(payload["uncertainty"]),
        )

    @classmethod
    def from_json(cls, line: str, *, source: str = "<jsonl>") -> "PlannerReplayExample":
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{source}: planner replay example must be a JSON object")
        return cls.from_dict(payload)


def planner_replay_artifact_name(split: str) -> str:
    """Return the canonical planner replay filename for one split."""
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"unsupported split: {split}")
    return f"{PLANNER_REPLAY_ARTIFACT_PREFIX}{split}.jsonl"


def build_planner_replay_examples(
    entries: Sequence[ReplayBufferEntry],
    *,
    split: str,
    include_unfinished: bool = False,
    max_examples: int | None = None,
) -> list[PlannerReplayExample]:
    """Convert replay-buffer rows into replay-derived planner supervision rows."""
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"unsupported split: {split}")
    selected_entries = entries[:max_examples] if max_examples is not None else entries
    built: list[PlannerReplayExample] = []
    for entry in selected_entries:
        if not include_unfinished and entry.outcome_pov == "unfinished":
            continue
        built.append(
            PlannerReplayExample(
                sample_id=entry.sample_id,
                split=split,
                fen=entry.fen,
                side_to_move=entry.side_to_move,
                selected_action_index=entry.action_index,
                selected_move_uci=entry.move_uci,
                selector_name=entry.selector_name,
                outcome_pov=entry.outcome_pov,
                termination_reason=entry.termination_reason,
                replay_priority=_replay_priority(entry),
                root_value_cp=_root_value_cp(entry.outcome_pov),
                proposer_score=entry.proposer_score,
                planner_score=entry.planner_score,
                reply_peak_probability=entry.reply_peak_probability,
                pressure=entry.pressure,
                uncertainty=entry.uncertainty,
            )
        )
    return built


def load_planner_replay_examples(path: Path) -> list[PlannerReplayExample]:
    """Load planner replay examples from JSONL."""
    if not path.exists():
        raise FileNotFoundError(f"planner replay artifact not found: {path}")
    examples: list[PlannerReplayExample] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        examples.append(PlannerReplayExample.from_json(line, source=f"{path}:{line_number}"))
    return examples


def write_planner_replay_artifact(path: Path, examples: Sequence[PlannerReplayExample]) -> None:
    """Write planner replay examples as JSONL."""
    lines = [json.dumps(example.to_dict(), sort_keys=True) for example in examples]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def planner_replay_summary(examples: Sequence[PlannerReplayExample]) -> dict[str, object]:
    """Compute a compact summary for replay-derived planner supervision."""
    outcome_counts: dict[str, int] = {}
    termination_counts: dict[str, int] = {}
    selector_counts: dict[str, int] = {}
    for example in examples:
        outcome_counts[example.outcome_pov] = outcome_counts.get(example.outcome_pov, 0) + 1
        termination_counts[example.termination_reason] = (
            termination_counts.get(example.termination_reason, 0) + 1
        )
        selector_counts[example.selector_name] = selector_counts.get(example.selector_name, 0) + 1
    return {
        "example_count": len(examples),
        "mean_replay_priority": round(
            sum(example.replay_priority for example in examples) / len(examples), 6
        )
        if examples
        else 0.0,
        "mean_root_value_cp": round(
            sum(example.root_value_cp for example in examples) / len(examples), 6
        )
        if examples
        else 0.0,
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "termination_counts": dict(sorted(termination_counts.items())),
        "selector_counts": dict(sorted(selector_counts.items())),
    }


def _root_value_cp(outcome_pov: str) -> float:
    if outcome_pov == "win":
        return PLANNER_REPLAY_VALUE_SCALE_CP
    if outcome_pov == "loss":
        return -PLANNER_REPLAY_VALUE_SCALE_CP
    if outcome_pov in {"draw", "unfinished"}:
        return 0.0
    raise ValueError(f"unsupported outcome_pov: {outcome_pov}")


def _replay_priority(entry: ReplayBufferEntry) -> float:
    base = 1.0 if entry.outcome_pov in {"win", "loss"} else 0.5
    if entry.outcome_pov == "unfinished":
        base = 0.0
    return round(base + max(entry.pressure, 0.0) + max(entry.reply_peak_probability, 0.0), 6)
