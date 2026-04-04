"""Run one versioned Phase-9 evolution campaign end to end."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train.eval.evolution_campaign import (
    PlannerEvolutionCampaignSpec,
    load_planner_evolution_campaign_spec,
    run_planner_evolution_campaign,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    spec = load_planner_evolution_campaign_spec(_resolve_repo_path(args.config))
    if args.output_root is not None:
        payload = spec.to_dict()
        payload["output_root"] = str(_resolve_repo_path(args.output_root))
        spec = PlannerEvolutionCampaignSpec.from_dict(payload)
    summary = run_planner_evolution_campaign(
        spec=spec,
        repo_root=REPO_ROOT,
        skip_existing=bool(args.skip_existing),
        iterations_override=args.iterations,
        selected_runs=args.run or None,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
