"""Small exact selfplay loop over the symbolic proposer and bounded planner contracts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import TYPE_CHECKING, Any, Callable, Protocol, Sequence

if TYPE_CHECKING:
    from train.datasets.schema import DatasetExample
    from train.eval.planner_runtime import PlannerRootDecision
else:
    DatasetExample = Any
    PlannerRootDecision = Any


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class SelfplayAgent(Protocol):
    """Small protocol so future architecture changes can swap move selectors cleanly."""

    name: str

    def select_move(self, example: DatasetExample) -> PlannerRootDecision:
        """Choose one legal move from the current exact position contract."""


OracleLoader = Callable[[str], DatasetExample]


@dataclass(frozen=True)
class SelfplayMaxPliesAdjudicationSpec:
    """Optional engine-based adjudication to reduce unresolved max-plies terminations."""

    engine_path: str
    score_threshold_pawns: float = 0.3
    extension_step_plies: int = 16
    max_extensions: int = 4
    nodes: int | None = 64
    depth: int | None = None
    movetime_ms: int | None = None

    def __post_init__(self) -> None:
        if not self.engine_path:
            raise ValueError("adjudication engine_path must be non-empty")
        if self.score_threshold_pawns < 0.0:
            raise ValueError("adjudication score_threshold_pawns must be non-negative")
        if self.extension_step_plies <= 0:
            raise ValueError("adjudication extension_step_plies must be positive")
        if self.max_extensions < 0:
            raise ValueError("adjudication max_extensions must be non-negative")
        if self.nodes is None and self.depth is None and self.movetime_ms is None:
            raise ValueError("one of adjudication nodes, depth, or movetime_ms must be set")
        if self.nodes is not None and self.nodes <= 0:
            raise ValueError("adjudication nodes must be positive when provided")
        if self.depth is not None and self.depth <= 0:
            raise ValueError("adjudication depth must be positive when provided")
        if self.movetime_ms is not None and self.movetime_ms <= 0:
            raise ValueError("adjudication movetime_ms must be positive when provided")

    def to_dict(self) -> dict[str, object]:
        return {
            "engine_path": self.engine_path,
            "score_threshold_pawns": self.score_threshold_pawns,
            "extension_step_plies": self.extension_step_plies,
            "max_extensions": self.max_extensions,
            "nodes": self.nodes,
            "depth": self.depth,
            "movetime_ms": self.movetime_ms,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SelfplayMaxPliesAdjudicationSpec":
        return cls(
            engine_path=str(payload["engine_path"]),
            score_threshold_pawns=float(payload.get("score_threshold_pawns", 0.3)),
            extension_step_plies=int(payload.get("extension_step_plies", 16)),
            max_extensions=int(payload.get("max_extensions", 4)),
            nodes=_optional_int(payload.get("nodes")),
            depth=_optional_int(payload.get("depth")),
            movetime_ms=_optional_int(payload.get("movetime_ms")),
        )


@dataclass(frozen=True)
class SelfplayAdjudicationOutcome:
    """One max-plies adjudication outcome for the current exact position."""

    should_continue: bool
    result: str | None
    termination_reason: str
    engine_path: str
    score_cp_white: float
    score_threshold_pawns: float

    def to_dict(self) -> dict[str, object]:
        return {
            "should_continue": self.should_continue,
            "result": self.result,
            "termination_reason": self.termination_reason,
            "engine_path": self.engine_path,
            "score_cp_white": round(self.score_cp_white, 3),
            "score_pawns_white": round(self.score_cp_white / 100.0, 4),
            "score_threshold_pawns": round(self.score_threshold_pawns, 4),
        }


class SelfplayAdjudicator(Protocol):
    """Minimal protocol so future adjudicators can swap in without changing selfplay."""

    def adjudicate(self, fen: str) -> SelfplayAdjudicationOutcome:
        """Return either a decisive adjudication or a signal to keep playing."""


class StockfishMaxPliesAdjudicator:
    """Engine-based adjudicator that only runs when a game hits the max-plies boundary."""

    def __init__(self, spec: SelfplayMaxPliesAdjudicationSpec) -> None:
        self.spec = spec
        self._process = subprocess.Popen(
            [spec.engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._handshake()

    def close(self) -> None:
        if self._process.poll() is not None:
            return
        try:
            self._send_line("quit")
        except BrokenPipeError:
            pass
        try:
            self._process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5.0)

    def __enter__(self) -> "StockfishMaxPliesAdjudicator":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def adjudicate(self, fen: str) -> SelfplayAdjudicationOutcome:
        self._send_line("ucinewgame")
        self._send_line(f"position fen {fen}")
        self._send_line(self._go_command())
        score_cp_side_to_move: float | None = None
        while True:
            line = self._read_line()
            if line.startswith("info "):
                parsed = _parse_uci_score_cp(line)
                if parsed is not None:
                    score_cp_side_to_move = parsed
                continue
            if line.startswith("bestmove "):
                break
        if score_cp_side_to_move is None:
            raise ValueError("selfplay adjudication analysis returned no score")
        score_cp_white = _score_cp_white(score_cp_side_to_move, fen)
        threshold_cp = self.spec.score_threshold_pawns * 100.0
        if abs(score_cp_white) <= threshold_cp:
            return SelfplayAdjudicationOutcome(
                should_continue=True,
                result=None,
                termination_reason="engine_adjudication_neutral",
                engine_path=self.spec.engine_path,
                score_cp_white=score_cp_white,
                score_threshold_pawns=self.spec.score_threshold_pawns,
            )
        return SelfplayAdjudicationOutcome(
            should_continue=False,
            result="1-0" if score_cp_white > 0.0 else "0-1",
            termination_reason=(
                "engine_adjudication_white_advantage"
                if score_cp_white > 0.0
                else "engine_adjudication_black_advantage"
            ),
            engine_path=self.spec.engine_path,
            score_cp_white=score_cp_white,
            score_threshold_pawns=self.spec.score_threshold_pawns,
        )

    def _go_command(self) -> str:
        parts = ["go"]
        if self.spec.nodes is not None:
            parts.extend(["nodes", str(self.spec.nodes)])
        if self.spec.depth is not None:
            parts.extend(["depth", str(self.spec.depth)])
        if self.spec.movetime_ms is not None:
            parts.extend(["movetime", str(self.spec.movetime_ms)])
        return " ".join(parts)

    def _handshake(self) -> None:
        self._send_line("uci")
        self._read_until("uciok")
        self._send_line("setoption name Threads value 1")
        self._send_line("isready")
        self._read_until("readyok")

    def _send_line(self, line: str) -> None:
        if self._process.stdin is None:
            raise RuntimeError("stockfish adjudicator stdin is not available")
        self._process.stdin.write(line + "\n")
        self._process.stdin.flush()

    def _read_line(self) -> str:
        if self._process.stdout is None:
            raise RuntimeError("stockfish adjudicator stdout is not available")
        raw_line = self._process.stdout.readline()
        if raw_line == "":
            raise RuntimeError("stockfish adjudicator ended unexpectedly")
        return raw_line.strip()

    def _read_until(self, expected: str) -> None:
        while True:
            line = self._read_line()
            if line == expected:
                return


@dataclass(frozen=True)
class SelfplayMoveRecord:
    """One selected move inside a selfplay game."""

    ply_index: int
    side_to_move: str
    fen: str
    move_uci: str
    action_index: int
    selector_name: str
    legal_candidate_count: int
    considered_candidate_count: int
    proposer_score: float
    planner_score: float
    reply_peak_probability: float
    pressure: float
    uncertainty: float
    next_fen: str

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "ply_index": self.ply_index,
            "side_to_move": self.side_to_move,
            "fen": self.fen,
            "move_uci": self.move_uci,
            "action_index": self.action_index,
            "selector_name": self.selector_name,
            "legal_candidate_count": self.legal_candidate_count,
            "considered_candidate_count": self.considered_candidate_count,
            "proposer_score": round(self.proposer_score, 6),
            "planner_score": round(self.planner_score, 6),
            "reply_peak_probability": round(self.reply_peak_probability, 6),
            "pressure": round(self.pressure, 6),
            "uncertainty": round(self.uncertainty, 6),
            "next_fen": self.next_fen,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SelfplayMoveRecord":
        return cls(
            ply_index=int(payload["ply_index"]),
            side_to_move=str(payload["side_to_move"]),
            fen=str(payload["fen"]),
            move_uci=str(payload["move_uci"]),
            action_index=int(payload["action_index"]),
            selector_name=str(payload["selector_name"]),
            legal_candidate_count=int(payload["legal_candidate_count"]),
            considered_candidate_count=int(payload["considered_candidate_count"]),
            proposer_score=float(payload["proposer_score"]),
            planner_score=float(payload["planner_score"]),
            reply_peak_probability=float(payload["reply_peak_probability"]),
            pressure=float(payload["pressure"]),
            uncertainty=float(payload["uncertainty"]),
            next_fen=str(payload["next_fen"]),
        )


@dataclass(frozen=True)
class SelfplayGameRecord:
    """One complete selfplay game or bounded probe episode."""

    game_id: str
    initial_fen: str
    final_fen: str
    result: str
    termination_reason: str
    move_count: int
    white_agent: str
    black_agent: str
    moves: list[SelfplayMoveRecord]
    adjudication: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "game_id": self.game_id,
            "initial_fen": self.initial_fen,
            "final_fen": self.final_fen,
            "result": self.result,
            "termination_reason": self.termination_reason,
            "move_count": self.move_count,
            "white_agent": self.white_agent,
            "black_agent": self.black_agent,
            "moves": [move.to_dict() for move in self.moves],
        }
        if self.adjudication is not None:
            payload["adjudication"] = self.adjudication
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SelfplayGameRecord":
        return cls(
            game_id=str(payload["game_id"]),
            initial_fen=str(payload["initial_fen"]),
            final_fen=str(payload["final_fen"]),
            result=str(payload["result"]),
            termination_reason=str(payload["termination_reason"]),
            move_count=int(payload["move_count"]),
            white_agent=str(payload["white_agent"]),
            black_agent=str(payload["black_agent"]),
            moves=[
                SelfplayMoveRecord.from_dict(dict(move))
                for move in list(payload["moves"])
            ],
            adjudication=(
                dict(payload["adjudication"])
                if isinstance(payload.get("adjudication"), dict)
                else None
            ),
        )


@dataclass(frozen=True)
class SelfplaySessionRecord:
    """Aggregate report for a first small selfplay session."""

    games: list[SelfplayGameRecord]

    def to_dict(self) -> dict[str, object]:
        termination_counts = Counter(game.termination_reason for game in self.games)
        result_counts = Counter(game.result for game in self.games)
        move_counts = [game.move_count for game in self.games]
        aggregate = {
            "game_count": len(self.games),
            "mean_move_count": round(sum(move_counts) / len(move_counts), 3) if move_counts else 0.0,
            "termination_counts": dict(sorted(termination_counts.items())),
            "result_counts": dict(sorted(result_counts.items())),
        }
        return {
            "aggregate": aggregate,
            "games": [game.to_dict() for game in self.games],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SelfplaySessionRecord":
        return cls(
            games=[
                SelfplayGameRecord.from_dict(dict(game))
                for game in list(payload["games"])
            ]
        )

    @classmethod
    def from_json(cls, raw_json: str) -> "SelfplaySessionRecord":
        payload = json.loads(raw_json)
        if not isinstance(payload, dict):
            raise ValueError("selfplay session must be a JSON object")
        return cls.from_dict(payload)


def play_selfplay_game(
    *,
    game_id: str,
    white_agent: SelfplayAgent,
    black_agent: SelfplayAgent,
    repo_root: Path,
    initial_fen: str = STARTING_FEN,
    max_plies: int = 80,
    oracle_loader: OracleLoader | None = None,
    adjudicator: SelfplayAdjudicator | None = None,
    adjudication_spec: SelfplayMaxPliesAdjudicationSpec | None = None,
) -> SelfplayGameRecord:
    """Play one small exact selfplay game with legal termination checks."""
    if max_plies <= 0:
        raise ValueError("max_plies must be positive")
    if adjudication_spec is None and adjudicator is not None:
        raise ValueError("adjudication_spec is required when adjudicator is provided")

    loader = oracle_loader or _build_oracle_loader(repo_root)
    current_fen = initial_fen
    moves: list[SelfplayMoveRecord] = []
    repetition_counts: Counter[str] = Counter({_repetition_key(initial_fen): 1})
    current_max_plies = max_plies
    adjudication_extensions_used = 0

    while True:
        example = loader(current_fen)
        if example.annotations.is_checkmate:
            return SelfplayGameRecord(
                game_id=game_id,
                initial_fen=initial_fen,
                final_fen=current_fen,
                result=_checkmate_result(example.side_to_move),
                termination_reason="checkmate",
                move_count=len(moves),
                white_agent=white_agent.name,
                black_agent=black_agent.name,
                moves=moves,
            )
        if example.annotations.is_stalemate:
            return _drawn_game(
                game_id=game_id,
                initial_fen=initial_fen,
                final_fen=current_fen,
                termination_reason="stalemate",
                white_agent=white_agent.name,
                black_agent=black_agent.name,
                moves=moves,
            )
        if repetition_counts[_repetition_key(current_fen)] >= 3:
            return _drawn_game(
                game_id=game_id,
                initial_fen=initial_fen,
                final_fen=current_fen,
                termination_reason="threefold_repetition",
                white_agent=white_agent.name,
                black_agent=black_agent.name,
                moves=moves,
            )
        if _halfmove_clock(current_fen) >= 100:
            return _drawn_game(
                game_id=game_id,
                initial_fen=initial_fen,
                final_fen=current_fen,
                termination_reason="fifty_move_rule",
                white_agent=white_agent.name,
                black_agent=black_agent.name,
                moves=moves,
            )
        if len(moves) >= current_max_plies:
            if adjudicator is not None and adjudication_spec is not None:
                outcome = adjudicator.adjudicate(current_fen)
                adjudication_payload = {
                    **outcome.to_dict(),
                    "current_max_plies": current_max_plies,
                    "extensions_used": adjudication_extensions_used,
                    "max_extensions": adjudication_spec.max_extensions,
                }
                if (
                    outcome.should_continue
                    and adjudication_extensions_used < adjudication_spec.max_extensions
                ):
                    adjudication_extensions_used += 1
                    current_max_plies += adjudication_spec.extension_step_plies
                    continue
                if outcome.should_continue:
                    return _drawn_game(
                        game_id=game_id,
                        initial_fen=initial_fen,
                        final_fen=current_fen,
                        termination_reason="engine_adjudication_draw",
                        white_agent=white_agent.name,
                        black_agent=black_agent.name,
                        moves=moves,
                        adjudication=adjudication_payload,
                    )
                return SelfplayGameRecord(
                    game_id=game_id,
                    initial_fen=initial_fen,
                    final_fen=current_fen,
                    result=outcome.result or "*",
                    termination_reason=outcome.termination_reason,
                    move_count=len(moves),
                    white_agent=white_agent.name,
                    black_agent=black_agent.name,
                    moves=moves,
                    adjudication=adjudication_payload,
                )
            return SelfplayGameRecord(
                game_id=game_id,
                initial_fen=initial_fen,
                final_fen=current_fen,
                result="*",
                termination_reason="max_plies",
                move_count=len(moves),
                white_agent=white_agent.name,
                black_agent=black_agent.name,
                moves=moves,
            )

        agent = white_agent if example.side_to_move == "w" else black_agent
        decision = agent.select_move(example)
        moves.append(
            SelfplayMoveRecord(
                ply_index=len(moves),
                side_to_move=example.side_to_move,
                fen=current_fen,
                move_uci=decision.move_uci,
                action_index=decision.action_index,
                selector_name=decision.selector_name,
                legal_candidate_count=decision.legal_candidate_count,
                considered_candidate_count=decision.considered_candidate_count,
                proposer_score=decision.proposer_score,
                planner_score=decision.planner_score,
                reply_peak_probability=decision.reply_peak_probability,
                pressure=decision.pressure,
                uncertainty=decision.uncertainty,
                next_fen=decision.next_fen,
            )
        )
        current_fen = decision.next_fen
        repetition_counts[_repetition_key(current_fen)] += 1


def run_selfplay_session(
    *,
    white_agent: SelfplayAgent,
    black_agent: SelfplayAgent,
    repo_root: Path,
    games: int,
    initial_fens: Sequence[str] | None = None,
    max_plies: int = 80,
    oracle_loader: OracleLoader | None = None,
    adjudicator: SelfplayAdjudicator | None = None,
    adjudication_spec: SelfplayMaxPliesAdjudicationSpec | None = None,
) -> SelfplaySessionRecord:
    """Run a small reproducible selfplay session."""
    if games <= 0:
        raise ValueError("games must be positive")
    if initial_fens is None:
        initial_fen_list = [STARTING_FEN]
    else:
        initial_fen_list = [str(fen) for fen in initial_fens]
        if not initial_fen_list:
            raise ValueError("initial_fens must not be empty")

    built_games = [
        play_selfplay_game(
            game_id=f"game_{index + 1:04d}",
            white_agent=white_agent,
            black_agent=black_agent,
            repo_root=repo_root,
            initial_fen=initial_fen_list[index % len(initial_fen_list)],
            max_plies=max_plies,
            oracle_loader=oracle_loader,
            adjudicator=adjudicator,
            adjudication_spec=adjudication_spec,
        )
        for index in range(games)
    ]
    return SelfplaySessionRecord(games=built_games)


def _build_oracle_loader(repo_root: Path) -> OracleLoader:
    def _loader(fen: str) -> DatasetExample:
        from train.datasets.opponent_head import dataset_example_from_oracle_payload
        from train.datasets.oracle import label_records_with_oracle
        from train.datasets.schema import RawPositionRecord

        payload = label_records_with_oracle(
            [RawPositionRecord(sample_id=f"selfplay:{fen}", fen=fen, source="selfplay")],
            repo_root=repo_root,
        )[0]
        return dataset_example_from_oracle_payload(
            sample_id=f"selfplay:{fen}",
            split="test",
            source="selfplay",
            fen=fen,
            payload=payload,
        )

    return _loader


def _repetition_key(fen: str) -> str:
    return " ".join(fen.split()[:4])


def _halfmove_clock(fen: str) -> int:
    parts = fen.split()
    if len(parts) < 5:
        raise ValueError(f"invalid FEN for selfplay: {fen}")
    return int(parts[4])


def _checkmate_result(side_to_move: str) -> str:
    if side_to_move == "w":
        return "0-1"
    if side_to_move == "b":
        return "1-0"
    raise ValueError(f"unsupported side_to_move: {side_to_move}")


def _drawn_game(
    *,
    game_id: str,
    initial_fen: str,
    final_fen: str,
    termination_reason: str,
    white_agent: str,
    black_agent: str,
    moves: list[SelfplayMoveRecord],
    adjudication: dict[str, object] | None = None,
) -> SelfplayGameRecord:
    return SelfplayGameRecord(
        game_id=game_id,
        initial_fen=initial_fen,
        final_fen=final_fen,
        result="1/2-1/2",
        termination_reason=termination_reason,
        move_count=len(moves),
        white_agent=white_agent,
        black_agent=black_agent,
        moves=moves,
        adjudication=adjudication,
    )


def open_max_plies_adjudicator(
    spec: SelfplayMaxPliesAdjudicationSpec | None,
) -> StockfishMaxPliesAdjudicator | None:
    """Open the configured max-plies adjudicator, if any."""
    if spec is None:
        return None
    return StockfishMaxPliesAdjudicator(spec)


def _parse_uci_score_cp(line: str) -> float | None:
    tokens = line.split()
    try:
        score_index = tokens.index("score")
    except ValueError:
        return None
    if score_index + 2 >= len(tokens):
        return None
    score_kind = tokens[score_index + 1]
    score_value = tokens[score_index + 2]
    if score_kind == "cp":
        return float(int(score_value))
    if score_kind == "mate":
        mate_value = int(score_value)
        return 100_000.0 if mate_value > 0 else -100_000.0
    return None


def _score_cp_white(score_cp_side_to_move: float, fen: str) -> float:
    side_to_move = fen.split()[1]
    if side_to_move == "w":
        return score_cp_side_to_move
    if side_to_move == "b":
        return -score_cp_side_to_move
    raise ValueError(f"invalid FEN side-to-move in selfplay adjudication: {fen}")


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)
