"""Run a first small exact selfplay session over the current learned stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train.eval.planner_runtime import build_planner_runtime
from train.eval.selfplay import STARTING_FEN, run_selfplay_session


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposer-checkpoint", type=Path, required=True)
    parser.add_argument("--planner-checkpoint", type=Path)
    parser.add_argument("--opponent-checkpoint", type=Path)
    parser.add_argument("--dynamics-checkpoint", type=Path)
    parser.add_argument("--black-proposer-checkpoint", type=Path)
    parser.add_argument("--black-planner-checkpoint", type=Path)
    parser.add_argument("--black-opponent-checkpoint", type=Path)
    parser.add_argument("--black-dynamics-checkpoint", type=Path)
    parser.add_argument("--opponent-mode", choices=("none", "symbolic", "learned"), default="symbolic")
    parser.add_argument("--black-opponent-mode", choices=("none", "symbolic", "learned"))
    parser.add_argument("--root-top-k", type=int, default=4)
    parser.add_argument("--black-root-top-k", type=int)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--max-plies", type=int, default=32)
    parser.add_argument("--initial-fen", action="append", default=[])
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    white_agent = build_planner_runtime(
        name="white",
        proposer_checkpoint=_resolve_repo_path(args.proposer_checkpoint),
        planner_checkpoint=_optional_repo_path(args.planner_checkpoint),
        opponent_checkpoint=_optional_repo_path(args.opponent_checkpoint),
        dynamics_checkpoint=_optional_repo_path(args.dynamics_checkpoint),
        opponent_mode=args.opponent_mode,
        root_top_k=args.root_top_k,
        repo_root=REPO_ROOT,
    )
    black_agent = build_planner_runtime(
        name="black",
        proposer_checkpoint=_resolve_repo_path(
            args.black_proposer_checkpoint or args.proposer_checkpoint
        ),
        planner_checkpoint=_optional_repo_path(
            args.black_planner_checkpoint or args.planner_checkpoint
        ),
        opponent_checkpoint=_optional_repo_path(
            args.black_opponent_checkpoint or args.opponent_checkpoint
        ),
        dynamics_checkpoint=_optional_repo_path(
            args.black_dynamics_checkpoint or args.dynamics_checkpoint
        ),
        opponent_mode=args.black_opponent_mode or args.opponent_mode,
        root_top_k=args.black_root_top_k or args.root_top_k,
        repo_root=REPO_ROOT,
    )
    initial_fens = args.initial_fen or [STARTING_FEN]
    session = run_selfplay_session(
        white_agent=white_agent,
        black_agent=black_agent,
        repo_root=REPO_ROOT,
        games=args.games,
        initial_fens=[_normalize_initial_fen(fen) for fen in initial_fens],
        max_plies=args.max_plies,
    )
    payload = session.to_dict()
    output_path = _resolve_repo_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _optional_repo_path(path: Path | None) -> Path | None:
    return None if path is None else _resolve_repo_path(path)


def _normalize_initial_fen(value: str) -> str:
    if value == "startpos":
        return STARTING_FEN
    return value


if __name__ == "__main__":
    raise SystemExit(main())
