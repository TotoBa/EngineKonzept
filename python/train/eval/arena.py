"""Small exact arena harness over versioned selfplay agent specs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable

from train.eval.agent_spec import load_selfplay_agent_spec
from train.eval.planner_runtime import build_planner_runtime_from_spec
from train.eval.selfplay import STARTING_FEN, SelfplaySessionRecord, run_selfplay_session


SELFPLAY_ARENA_SPEC_VERSION = 1


@dataclass(frozen=True)
class SelfplayArenaMatchupSpec:
    """One ordered white-vs-black selfplay matchup inside an arena suite."""

    white_agent: str
    black_agent: str
    games: int = 1
    max_plies: int | None = None
    initial_fens: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.white_agent:
            raise ValueError("arena matchup white_agent must be non-empty")
        if not self.black_agent:
            raise ValueError("arena matchup black_agent must be non-empty")
        if self.games <= 0:
            raise ValueError("arena matchup games must be positive")
        if self.max_plies is not None and self.max_plies <= 0:
            raise ValueError("arena matchup max_plies must be positive when provided")

    def to_dict(self) -> dict[str, object]:
        return {
            "white_agent": self.white_agent,
            "black_agent": self.black_agent,
            "games": self.games,
            "max_plies": self.max_plies,
            "initial_fens": list(self.initial_fens),
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SelfplayArenaMatchupSpec":
        return cls(
            white_agent=str(payload["white_agent"]),
            black_agent=str(payload["black_agent"]),
            games=int(payload.get("games", 1)),
            max_plies=_optional_int(payload.get("max_plies")),
            initial_fens=[str(value) for value in list(payload.get("initial_fens") or [])],
            tags=[str(value) for value in list(payload.get("tags") or [])],
        )


@dataclass(frozen=True)
class SelfplayArenaSpec:
    """Versioned suite spec for checkpoint-vs-checkpoint selfplay arenas."""

    name: str
    agent_specs: dict[str, str]
    schedule_mode: str = "explicit"
    matchups: list[SelfplayArenaMatchupSpec] = field(default_factory=list)
    default_games: int = 1
    default_max_plies: int = 32
    default_initial_fens: list[str] = field(default_factory=lambda: [STARTING_FEN])
    round_robin_swap_colors: bool = True
    include_self_matches: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    spec_version: int = SELFPLAY_ARENA_SPEC_VERSION

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("arena spec name must be non-empty")
        if self.spec_version != SELFPLAY_ARENA_SPEC_VERSION:
            raise ValueError(f"unsupported selfplay arena spec version: {self.spec_version}")
        if not self.agent_specs:
            raise ValueError("arena spec must list at least one agent spec")
        if self.schedule_mode not in {"explicit", "round_robin"}:
            raise ValueError("arena schedule_mode must be 'explicit' or 'round_robin'")
        if self.default_games <= 0:
            raise ValueError("arena default_games must be positive")
        if self.default_max_plies <= 0:
            raise ValueError("arena default_max_plies must be positive")
        if self.schedule_mode == "explicit":
            if not self.matchups:
                raise ValueError("arena explicit schedule requires at least one matchup")
            for matchup in self.matchups:
                if matchup.white_agent not in self.agent_specs:
                    raise ValueError(f"unknown white agent in arena spec: {matchup.white_agent}")
                if matchup.black_agent not in self.agent_specs:
                    raise ValueError(f"unknown black agent in arena spec: {matchup.black_agent}")
        elif len(self.agent_specs) < 2 and not self.include_self_matches:
            raise ValueError("round_robin arena needs at least two agents unless self matches are allowed")

    def expanded_matchups(self) -> list[SelfplayArenaMatchupSpec]:
        """Return the concrete ordered matchup list for this arena."""
        if self.schedule_mode == "explicit":
            return [
                SelfplayArenaMatchupSpec(
                    white_agent=matchup.white_agent,
                    black_agent=matchup.black_agent,
                    games=matchup.games,
                    max_plies=matchup.max_plies or self.default_max_plies,
                    initial_fens=list(matchup.initial_fens or self.default_initial_fens),
                    tags=list(matchup.tags),
                )
                for matchup in self.matchups
            ]

        agent_names = list(self.agent_specs)
        matchups: list[SelfplayArenaMatchupSpec] = []
        for white_agent in agent_names:
            for black_agent in agent_names:
                if white_agent == black_agent and not self.include_self_matches:
                    continue
                if not self.round_robin_swap_colors and white_agent > black_agent:
                    continue
                matchups.append(
                    SelfplayArenaMatchupSpec(
                        white_agent=white_agent,
                        black_agent=black_agent,
                        games=self.default_games,
                        max_plies=self.default_max_plies,
                        initial_fens=list(self.default_initial_fens),
                    )
                )
        return matchups

    def to_dict(self) -> dict[str, object]:
        return {
            "spec_version": self.spec_version,
            "name": self.name,
            "agent_specs": dict(self.agent_specs),
            "schedule_mode": self.schedule_mode,
            "matchups": [matchup.to_dict() for matchup in self.matchups],
            "default_games": self.default_games,
            "default_max_plies": self.default_max_plies,
            "default_initial_fens": list(self.default_initial_fens),
            "round_robin_swap_colors": self.round_robin_swap_colors,
            "include_self_matches": self.include_self_matches,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SelfplayArenaSpec":
        return cls(
            spec_version=int(payload.get("spec_version", SELFPLAY_ARENA_SPEC_VERSION)),
            name=str(payload["name"]),
            agent_specs={
                str(name): str(path)
                for name, path in dict(payload["agent_specs"]).items()
            },
            schedule_mode=str(payload.get("schedule_mode", "explicit")),
            matchups=[
                SelfplayArenaMatchupSpec.from_dict(dict(matchup))
                for matchup in list(payload.get("matchups") or [])
            ],
            default_games=int(payload.get("default_games", 1)),
            default_max_plies=int(payload.get("default_max_plies", 32)),
            default_initial_fens=[
                _normalize_initial_fen(str(value))
                for value in list(payload.get("default_initial_fens") or [STARTING_FEN])
            ],
            round_robin_swap_colors=bool(payload.get("round_robin_swap_colors", True)),
            include_self_matches=bool(payload.get("include_self_matches", False)),
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def from_json(cls, raw_json: str) -> "SelfplayArenaSpec":
        payload = json.loads(raw_json)
        if not isinstance(payload, dict):
            raise ValueError("selfplay arena spec must be a JSON object")
        return cls.from_dict(payload)


@dataclass(frozen=True)
class ArenaMatchupResult:
    """Compact aggregate result for one arena matchup session."""

    name: str
    white_agent: str
    black_agent: str
    session_path: str
    game_count: int
    mean_move_count: float
    white_score: float
    black_score: float
    result_counts: dict[str, int]
    termination_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "white_agent": self.white_agent,
            "black_agent": self.black_agent,
            "session_path": self.session_path,
            "game_count": self.game_count,
            "mean_move_count": self.mean_move_count,
            "white_score": round(self.white_score, 3),
            "black_score": round(self.black_score, 3),
            "result_counts": self.result_counts,
            "termination_counts": self.termination_counts,
        }


AgentBuilder = Callable[[str, Path, Path], Any]
OracleLoader = Callable[[str], Any]


def load_selfplay_arena_spec(path: Path) -> SelfplayArenaSpec:
    """Load a versioned arena spec from JSON."""
    return SelfplayArenaSpec.from_json(path.read_text(encoding="utf-8"))


def write_selfplay_arena_spec(path: Path, spec: SelfplayArenaSpec) -> None:
    """Write a versioned arena spec as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_selfplay_arena(
    *,
    spec: SelfplayArenaSpec,
    repo_root: Path,
    output_root: Path,
    agent_builder: AgentBuilder | None = None,
    oracle_loader: OracleLoader | None = None,
) -> dict[str, object]:
    """Run a full arena suite and return the aggregate summary payload."""
    output_root.mkdir(parents=True, exist_ok=True)
    sessions_root = output_root / "sessions"
    sessions_root.mkdir(parents=True, exist_ok=True)

    builder = agent_builder or _default_agent_builder
    agents = {
        agent_name: builder(agent_name, _resolve_repo_path(repo_root, spec_path), repo_root)
        for agent_name, spec_path in spec.agent_specs.items()
    }

    matchup_results: list[ArenaMatchupResult] = []
    standings: dict[str, dict[str, object]] = {
        agent_name: {
            "games": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "unfinished": 0,
            "score": 0.0,
        }
        for agent_name in spec.agent_specs
    }

    for matchup_index, matchup in enumerate(spec.expanded_matchups(), start=1):
        session = run_selfplay_session(
            white_agent=agents[matchup.white_agent],
            black_agent=agents[matchup.black_agent],
            repo_root=repo_root,
            games=matchup.games,
            initial_fens=[_normalize_initial_fen(fen) for fen in matchup.initial_fens],
            max_plies=matchup.max_plies or spec.default_max_plies,
            oracle_loader=oracle_loader,
        )
        session_path = sessions_root / (
            f"{matchup_index:02d}_{matchup.white_agent}_vs_{matchup.black_agent}.json"
        )
        session_path.write_text(
            json.dumps(session.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result = _summarize_matchup(
            session=session,
            white_agent=matchup.white_agent,
            black_agent=matchup.black_agent,
            session_path=session_path,
        )
        matchup_results.append(result)
        _update_standings(standings, result)

    all_game_counts = [result.game_count for result in matchup_results]
    aggregate = {
        "matchup_count": len(matchup_results),
        "game_count": sum(all_game_counts),
        "mean_games_per_matchup": round(
            sum(all_game_counts) / len(all_game_counts), 3
        ) if all_game_counts else 0.0,
    }
    summary = {
        "arena_name": spec.name,
        "arena_spec_version": spec.spec_version,
        "schedule_mode": spec.schedule_mode,
        "metadata": spec.metadata,
        "aggregate": aggregate,
        "standings": standings,
        "matchups": [result.to_dict() for result in matchup_results],
    }
    return summary


def _default_agent_builder(agent_name: str, spec_path: Path, repo_root: Path) -> Any:
    spec = load_selfplay_agent_spec(spec_path)
    if spec.name != agent_name:
        spec = SelfplayAgentProxySpec(spec=spec, name=agent_name).materialize()
    return build_planner_runtime_from_spec(spec, repo_root=repo_root)


@dataclass(frozen=True)
class SelfplayAgentProxySpec:
    """Small helper to keep arena aliases stable without editing checkpoint specs."""

    spec: Any
    name: str

    def materialize(self) -> Any:
        return type(self.spec)(
            name=self.name,
            proposer_checkpoint=self.spec.proposer_checkpoint,
            planner_checkpoint=self.spec.planner_checkpoint,
            opponent_checkpoint=self.spec.opponent_checkpoint,
            dynamics_checkpoint=self.spec.dynamics_checkpoint,
            opponent_mode=self.spec.opponent_mode,
            root_top_k=self.spec.root_top_k,
            tags=list(self.spec.tags),
            metadata=dict(self.spec.metadata),
            spec_version=self.spec.spec_version,
        )


def _summarize_matchup(
    *,
    session: SelfplaySessionRecord,
    white_agent: str,
    black_agent: str,
    session_path: Path,
) -> ArenaMatchupResult:
    result_counts = Counter(game.result for game in session.games)
    termination_counts = Counter(game.termination_reason for game in session.games)
    white_score = 0.0
    black_score = 0.0
    move_counts: list[int] = []
    for game in session.games:
        white_points, black_points = _score_game_result(game.result)
        white_score += white_points
        black_score += black_points
        move_counts.append(game.move_count)
    return ArenaMatchupResult(
        name=f"{white_agent}_vs_{black_agent}",
        white_agent=white_agent,
        black_agent=black_agent,
        session_path=str(session_path),
        game_count=len(session.games),
        mean_move_count=round(sum(move_counts) / len(move_counts), 3) if move_counts else 0.0,
        white_score=white_score,
        black_score=black_score,
        result_counts=dict(sorted(result_counts.items())),
        termination_counts=dict(sorted(termination_counts.items())),
    )


def _update_standings(
    standings: dict[str, dict[str, object]],
    result: ArenaMatchupResult,
) -> None:
    white = standings[result.white_agent]
    black = standings[result.black_agent]
    white["games"] = int(white["games"]) + result.game_count
    black["games"] = int(black["games"]) + result.game_count
    white["score"] = round(float(white["score"]) + result.white_score, 3)
    black["score"] = round(float(black["score"]) + result.black_score, 3)

    white_wins = result.result_counts.get("1-0", 0)
    black_wins = result.result_counts.get("0-1", 0)
    draws = result.result_counts.get("1/2-1/2", 0)
    unfinished = result.result_counts.get("*", 0)

    white["wins"] = int(white["wins"]) + white_wins
    white["losses"] = int(white["losses"]) + black_wins
    white["draws"] = int(white["draws"]) + draws
    white["unfinished"] = int(white["unfinished"]) + unfinished

    black["wins"] = int(black["wins"]) + black_wins
    black["losses"] = int(black["losses"]) + white_wins
    black["draws"] = int(black["draws"]) + draws
    black["unfinished"] = int(black["unfinished"]) + unfinished


def _score_game_result(result: str) -> tuple[float, float]:
    if result == "1-0":
        return 1.0, 0.0
    if result == "0-1":
        return 0.0, 1.0
    if result in {"1/2-1/2", "*"}:
        return 0.5, 0.5
    raise ValueError(f"unsupported game result: {result}")


def _resolve_repo_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else repo_root / path


def _normalize_initial_fen(value: str) -> str:
    return STARTING_FEN if value == "startpos" else value


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)
