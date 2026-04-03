"""Run one versioned Phase-9 replay campaign end to end."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train.eval.campaign import (
    load_selfplay_replay_campaign_spec,
    run_selfplay_replay_campaign,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--games-per-matchup", type=int)
    parser.add_argument("--max-plies", type=int)
    parser.add_argument("--max-replay-examples", type=int)
    parser.add_argument("--max-replay-head-examples", type=int)
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    spec = load_selfplay_replay_campaign_spec(_resolve_repo_path(args.config))
    summary = run_selfplay_replay_campaign(
        spec=spec,
        repo_root=REPO_ROOT,
        games_per_matchup_override=args.games_per_matchup,
        max_plies_override=args.max_plies,
        max_replay_examples=args.max_replay_examples,
        max_replay_head_examples=args.max_replay_head_examples,
        selected_runs=args.run or None,
        skip_existing=bool(args.skip_existing),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
