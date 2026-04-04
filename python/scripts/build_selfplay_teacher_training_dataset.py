"""Build per-agent post-game selfplay teacher reviews and planner-head trainsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train.datasets import (
    build_planner_head_examples_from_selfplay_teacher_reviews,
    build_selfplay_teacher_review_examples,
    planner_head_artifact_name,
    selfplay_teacher_review_artifact_name,
    selfplay_teacher_review_summary,
    write_planner_head_artifact,
    write_selfplay_teacher_review_artifact,
)
from train.eval.agent_spec import load_selfplay_agent_spec
from train.eval.arena import SelfplayArenaSpec


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena-summary", type=Path, required=True)
    parser.add_argument("--arena-spec", type=Path, required=True)
    parser.add_argument("--teacher-engine", type=Path, default=Path("/usr/games/stockfish18"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--movetime-ms", type=int)
    parser.add_argument("--multipv", type=int, default=0)
    parser.add_argument("--policy-temperature-cp", type=float, default=64.0)
    parser.add_argument("--mistake-deadzone-cp", type=float, default=8.0)
    parser.add_argument("--mistake-priority-scale-cp", type=float, default=64.0)
    parser.add_argument("--max-mistake-priority", type=float, default=4.0)
    parser.add_argument("--max-examples-per-agent", type=int)
    parser.add_argument("--max-head-examples-per-agent", type=int)
    parser.add_argument("--include-non-mistakes", action="store_true")
    args = parser.parse_args()

    arena_summary_path = _resolve_repo_path(args.arena_summary)
    arena_spec_path = _resolve_repo_path(args.arena_spec)
    output_root = _resolve_repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    arena_spec = SelfplayArenaSpec.from_json(arena_spec_path.read_text(encoding="utf-8"))
    agent_specs = {
        name: load_selfplay_agent_spec(_resolve_repo_path(Path(path)))
        for name, path in arena_spec.agent_specs.items()
    }
    planner_agent_specs = {
        name: spec for name, spec in agent_specs.items() if spec.agent_kind == "planner"
    }
    if not planner_agent_specs:
        raise ValueError("arena spec contains no planner agents to review")

    reviews_by_agent = build_selfplay_teacher_review_examples(
        arena_summary_path=arena_summary_path,
        trainable_agent_names=tuple(planner_agent_specs),
        repo_root=REPO_ROOT,
        teacher_engine_path=_resolve_repo_path(args.teacher_engine),
        split=args.split,
        nodes=args.nodes,
        depth=args.depth,
        movetime_ms=args.movetime_ms,
        multipv=args.multipv,
        policy_temperature_cp=args.policy_temperature_cp,
        mistake_deadzone_cp=args.mistake_deadzone_cp,
        mistake_priority_scale_cp=args.mistake_priority_scale_cp,
        max_mistake_priority=args.max_mistake_priority,
        max_examples_per_agent=args.max_examples_per_agent,
    )

    agent_summaries: dict[str, object] = {}
    for agent_name, agent_spec in sorted(planner_agent_specs.items()):
        agent_root = output_root / agent_name
        agent_root.mkdir(parents=True, exist_ok=True)
        review_examples = reviews_by_agent.get(agent_name, [])
        review_path = agent_root / selfplay_teacher_review_artifact_name(args.split)
        write_selfplay_teacher_review_artifact(review_path, review_examples)

        proposer_checkpoint = agent_spec.proposer_checkpoint
        if proposer_checkpoint is None:
            raise ValueError(f"{agent_name}: planner agent is missing proposer_checkpoint")
        planner_head_examples = build_planner_head_examples_from_selfplay_teacher_reviews(
            review_examples=review_examples,
            proposer_checkpoint=_resolve_repo_path(Path(proposer_checkpoint)),
            dynamics_checkpoint=(
                _resolve_repo_path(Path(agent_spec.dynamics_checkpoint))
                if agent_spec.dynamics_checkpoint is not None
                else None
            ),
            opponent_mode=agent_spec.opponent_mode,
            opponent_checkpoint=(
                _resolve_repo_path(Path(agent_spec.opponent_checkpoint))
                if agent_spec.opponent_checkpoint is not None
                else None
            ),
            root_top_k=agent_spec.root_top_k,
            max_examples=args.max_head_examples_per_agent,
            include_non_mistakes=bool(args.include_non_mistakes),
            repo_root=REPO_ROOT,
        )
        planner_head_path = agent_root / planner_head_artifact_name(args.split)
        write_planner_head_artifact(planner_head_path, planner_head_examples)

        agent_summary = {
            "agent_spec_path": str(_resolve_repo_path(Path(arena_spec.agent_specs[agent_name]))),
            "review_path": str(review_path),
            "review_summary": selfplay_teacher_review_summary(review_examples),
            "planner_head_path": str(planner_head_path),
            "planner_head_example_count": len(planner_head_examples),
        }
        (agent_root / "summary.json").write_text(
            json.dumps(agent_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        agent_summaries[agent_name] = agent_summary

    summary = {
        "arena_summary_path": str(arena_summary_path),
        "arena_spec_path": str(arena_spec_path),
        "teacher_engine": str(_resolve_repo_path(args.teacher_engine)),
        "split": args.split,
        "depth": args.depth,
        "nodes": args.nodes,
        "movetime_ms": args.movetime_ms,
        "multipv": args.multipv,
        "policy_temperature_cp": args.policy_temperature_cp,
        "mistake_deadzone_cp": args.mistake_deadzone_cp,
        "mistake_priority_scale_cp": args.mistake_priority_scale_cp,
        "max_mistake_priority": args.max_mistake_priority,
        "agents": agent_summaries,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
