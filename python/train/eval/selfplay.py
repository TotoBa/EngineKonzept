"""Small exact selfplay loop over the symbolic proposer and bounded planner contracts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Protocol, Sequence

from train.datasets import dataset_example_from_oracle_payload
from train.datasets.oracle import label_records_with_oracle
from train.datasets.schema import DatasetExample, RawPositionRecord
from train.eval.planner_runtime import PlannerRootDecision


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class SelfplayAgent(Protocol):
    """Small protocol so future architecture changes can swap move selectors cleanly."""

    name: str

    def select_move(self, example: DatasetExample) -> PlannerRootDecision:
        """Choose one legal move from the current exact position contract."""


OracleLoader = Callable[[str], DatasetExample]


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

    def to_dict(self) -> dict[str, object]:
        return {
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
) -> SelfplayGameRecord:
    """Play one small exact selfplay game with legal termination checks."""
    if max_plies <= 0:
        raise ValueError("max_plies must be positive")

    loader = oracle_loader or _build_oracle_loader(repo_root)
    current_fen = initial_fen
    moves: list[SelfplayMoveRecord] = []
    repetition_counts: Counter[str] = Counter({_repetition_key(initial_fen): 1})

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
        if len(moves) >= max_plies:
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
        )
        for index in range(games)
    ]
    return SelfplaySessionRecord(games=built_games)


def _build_oracle_loader(repo_root: Path) -> OracleLoader:
    def _loader(fen: str) -> DatasetExample:
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
    )
