"""Offline UCI-engine agents for bounded arena benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Callable

from train.action_space import flatten_action
from train.datasets import dataset_example_from_oracle_payload
from train.datasets.oracle import label_records_with_oracle
from train.datasets.schema import DatasetExample, RawPositionRecord
from train.eval.agent_spec import SelfplayAgentSpec
from train.eval.planner_runtime import PlannerRootDecision


SelectedMoveLabeler = Callable[[DatasetExample, str, Path], DatasetExample]


@dataclass
class ExternalUciEngineAgent:
    """Exact legal-move selector backed by an external UCI engine process."""

    name: str
    engine_path: Path
    repo_root: Path
    nodes: int | None = None
    depth: int | None = None
    movetime_ms: int | None = None
    threads: int = 1
    hash_mb: int | None = 16
    engine_options: dict[str, str] | None = None
    label_selected_move: SelectedMoveLabeler = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.nodes is None and self.depth is None and self.movetime_ms is None:
            raise ValueError("external engine agent requires one of nodes, depth, or movetime_ms")
        if self.threads <= 0:
            raise ValueError("external engine agent threads must be positive")
        if self.label_selected_move is None:
            self.label_selected_move = _label_selected_move
        self._process = subprocess.Popen(
            [str(self.engine_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._option_names = self._handshake()

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is None or process.poll() is not None:
            return
        try:
            self._send_line("quit")
        except BrokenPipeError:
            pass
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)

    def __del__(self) -> None:  # pragma: no cover - best-effort resource cleanup
        try:
            self.close()
        except Exception:
            return

    def select_move(self, example: DatasetExample) -> PlannerRootDecision:
        if not example.legal_moves:
            raise ValueError(f"{example.sample_id}: external engine cannot select from empty legal set")
        self._send_line("ucinewgame")
        self._send_line(f"position fen {example.fen}")
        self._send_line(self._go_command())

        bestmove_uci: str | None = None
        while True:
            line = self._read_line()
            if line.startswith("bestmove "):
                tokens = line.split()
                if len(tokens) < 2:
                    raise ValueError(f"invalid bestmove line from {self.engine_path}: {line}")
                bestmove_uci = tokens[1]
                break
        if bestmove_uci is None or bestmove_uci == "(none)":
            raise ValueError(f"{self.engine_path}: engine produced no move for {example.sample_id}")
        if bestmove_uci not in example.legal_moves:
            raise ValueError(
                f"{self.engine_path}: illegal engine move {bestmove_uci} for {example.sample_id}"
            )

        selected_example = self.label_selected_move(example, bestmove_uci, self.repo_root)
        assert selected_example.next_fen is not None
        action_index = _action_index_for_move(example, bestmove_uci)
        return PlannerRootDecision(
            move_uci=bestmove_uci,
            action_index=action_index,
            next_fen=str(selected_example.next_fen),
            selector_name=self.name,
            legal_candidate_count=len(example.legal_moves),
            considered_candidate_count=len(example.legal_moves),
            proposer_score=0.0,
            planner_score=0.0,
            reply_peak_probability=0.0,
            pressure=0.0,
            uncertainty=0.0,
        )

    def _go_command(self) -> str:
        parts = ["go"]
        if self.nodes is not None:
            parts.extend(["nodes", str(self.nodes)])
        if self.depth is not None:
            parts.extend(["depth", str(self.depth)])
        if self.movetime_ms is not None:
            parts.extend(["movetime", str(self.movetime_ms)])
        return " ".join(parts)

    def _handshake(self) -> set[str]:
        option_names: set[str] = set()
        self._send_line("uci")
        while True:
            line = self._read_line()
            if line == "uciok":
                break
            option_name = _parse_uci_option_name(line)
            if option_name is not None:
                option_names.add(option_name.lower())
        self._set_option_if_available(option_names, "Threads", str(self.threads))
        if self.hash_mb is not None:
            self._set_option_if_available(option_names, "Hash", str(self.hash_mb))
        self._set_option_if_available(option_names, "Ponder", "false")
        self._set_option_if_available(option_names, "OwnBook", "false")
        self._set_option_if_available(option_names, "Book", "false")
        for option_name, option_value in sorted((self.engine_options or {}).items()):
            if option_name.lower() not in option_names:
                raise ValueError(
                    f"{self.engine_path}: unsupported configured UCI option {option_name!r}"
                )
            self._send_line(f"setoption name {option_name} value {option_value}")
        self._send_line("isready")
        self._read_until("readyok")
        return option_names

    def _set_option_if_available(self, option_names: set[str], option_name: str, value: str) -> None:
        if option_name.lower() not in option_names:
            return
        self._send_line(f"setoption name {option_name} value {value}")

    def _send_line(self, line: str) -> None:
        if self._process.stdin is None:
            raise RuntimeError(f"{self.engine_path}: stdin is unavailable")
        self._process.stdin.write(line + "\n")
        self._process.stdin.flush()

    def _read_line(self) -> str:
        if self._process.stdout is None:
            raise RuntimeError(f"{self.engine_path}: stdout is unavailable")
        raw_line = self._process.stdout.readline()
        if raw_line == "":
            raise RuntimeError(f"{self.engine_path}: process ended unexpectedly")
        return raw_line.strip()

    def _read_until(self, expected: str) -> None:
        while True:
            if self._read_line() == expected:
                return


def build_external_engine_agent_from_spec(
    spec: SelfplayAgentSpec,
    *,
    repo_root: Path,
) -> ExternalUciEngineAgent:
    """Build one offline UCI-engine agent from a versioned selfplay-agent spec."""
    if spec.agent_kind != "uci_engine":
        raise ValueError(f"expected uci_engine spec, got {spec.agent_kind}")
    if spec.external_engine_path is None:
        raise ValueError("uci_engine spec requires external_engine_path")
    engine_path = _resolve_repo_path(repo_root, spec.external_engine_path)
    return ExternalUciEngineAgent(
        name=spec.name,
        engine_path=engine_path,
        repo_root=repo_root,
        nodes=spec.external_engine_nodes,
        depth=spec.external_engine_depth,
        movetime_ms=spec.external_engine_movetime_ms,
        threads=spec.external_engine_threads,
        hash_mb=spec.external_engine_hash_mb,
        engine_options=dict(spec.external_engine_options),
    )


def _resolve_repo_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else repo_root / path


def _parse_uci_option_name(line: str) -> str | None:
    tokens = line.split()
    if len(tokens) < 5 or tokens[0] != "option" or tokens[1] != "name":
        return None
    try:
        type_index = tokens.index("type", 2)
    except ValueError:
        return None
    return " ".join(tokens[2:type_index]).strip() or None


def _action_index_for_move(example: DatasetExample, move_uci: str) -> int:
    for legal_move, action in zip(example.legal_moves, example.legal_action_encodings, strict=True):
        if legal_move == move_uci:
            return flatten_action(action)
    raise ValueError(f"{example.sample_id}: no exact action encoding for move {move_uci}")


def _label_selected_move(
    example: DatasetExample,
    move_uci: str,
    repo_root: Path,
) -> DatasetExample:
    payload = label_records_with_oracle(
        [
            RawPositionRecord(
                sample_id=f"{example.sample_id}:external_engine_selected",
                fen=example.fen,
                source="selfplay",
                selected_move_uci=move_uci,
            )
        ],
        repo_root=repo_root,
    )[0]
    return dataset_example_from_oracle_payload(
        sample_id=f"{example.sample_id}:external_engine_selected",
        split=example.split,
        source="selfplay",
        fen=example.fen,
        payload=payload,
    )
