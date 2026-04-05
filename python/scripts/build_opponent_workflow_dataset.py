"""Build the full offline Phase-7 workflow stack for one dataset split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any, Sequence

import chess
import chess.engine

from train.datasets import load_split_examples
from train.datasets.artifacts import build_symbolic_proposer_example
from train.datasets.opponent_head import (
    build_opponent_head_examples,
    opponent_head_artifact_name,
    write_opponent_head_artifact,
)
from train.datasets.search_curriculum import (
    build_search_curriculum_examples,
    search_curriculum_artifact_name,
    write_search_curriculum_artifact,
)
from train.datasets.search_disagreements import (
    build_search_disagreement_examples,
    search_disagreements_artifact_name,
    write_search_disagreement_artifact,
)
from train.datasets.search_teacher import (
    build_search_teacher_example_from_analysis,
    search_teacher_artifact_name,
    write_search_teacher_artifact,
)
from train.datasets.search_traces import (
    build_search_trace_example_from_analysis,
    search_traces_artifact_name,
    write_search_trace_artifact,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-engine", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--multipv", type=int, default=8)
    parser.add_argument("--policy-temperature-cp", type=float, default=100.0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-examples", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=0)
    args = parser.parse_args(argv)

    dataset_dir = _resolve_repo_path(args.dataset_dir)
    checkpoint_path = _resolve_repo_path(args.checkpoint)
    teacher_engine = _resolve_repo_path(args.teacher_engine)
    output_dir = _resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_examples = load_split_examples(dataset_dir, args.split)
    if not dataset_examples:
        raise ValueError(f"dataset split is empty: {args.split}")
    selected_examples = list(dataset_examples[: args.max_examples])

    teacher_examples, trace_examples = _build_teacher_and_trace_examples(
        selected_examples,
        teacher_engine_path=teacher_engine,
        nodes=args.nodes,
        multipv=args.multipv,
        policy_temperature_cp=args.policy_temperature_cp,
        log_every=args.log_every,
        split=args.split,
    )
    _log(
        f"[workflow:{args.split}] building disagreements/curriculum/opponent "
        f"for {len(selected_examples)} examples"
    )
    disagreement_examples = build_search_disagreement_examples(
        selected_examples,
        teacher_examples,
        checkpoint_path=checkpoint_path,
        top_k=args.top_k,
    )
    curriculum_examples = build_search_curriculum_examples(
        trace_examples,
        disagreement_examples,
    )
    opponent_examples = build_opponent_head_examples(
        selected_examples,
        trace_examples,
        curriculum_examples,
        repo_root=REPO_ROOT,
    )

    teacher_path = output_dir / search_teacher_artifact_name(args.split)
    trace_path = output_dir / search_traces_artifact_name(args.split)
    disagreement_path = output_dir / search_disagreements_artifact_name(args.split)
    curriculum_path = output_dir / search_curriculum_artifact_name(args.split)
    opponent_path = output_dir / opponent_head_artifact_name(args.split)

    write_search_teacher_artifact(teacher_path, teacher_examples)
    write_search_trace_artifact(trace_path, trace_examples)
    write_search_disagreement_artifact(disagreement_path, disagreement_examples)
    write_search_curriculum_artifact(curriculum_path, curriculum_examples)
    write_opponent_head_artifact(opponent_path, opponent_examples)

    summary = {
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint_path),
        "teacher_engine": str(teacher_engine),
        "teacher_nodes": args.nodes,
        "teacher_multipv": args.multipv,
        "example_count": len(selected_examples),
        "reply_supervised_count": sum(
            1 for example in opponent_examples if example.teacher_reply_action_index is not None
        ),
        "teacher_coverage_ratio": (
            sum(example.teacher_coverage_ratio for example in teacher_examples) / len(teacher_examples)
            if teacher_examples
            else 0.0
        ),
        "curriculum_priority_mean": (
            sum(example.curriculum_priority for example in curriculum_examples)
            / len(curriculum_examples)
            if curriculum_examples
            else 0.0
        ),
        "disagreement_rate": (
            sum(1 for example in disagreement_examples if example.top1_disagrees)
            / len(disagreement_examples)
            if disagreement_examples
            else 0.0
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _build_teacher_and_trace_examples(
    examples: Sequence[Any],
    *,
    teacher_engine_path: Path,
    nodes: int,
    multipv: int,
    policy_temperature_cp: float,
    log_every: int,
    split: str,
) -> tuple[list[Any], list[Any]]:
    teacher_examples: list[Any] = []
    trace_examples: list[Any] = []
    with _UciTeacher(teacher_engine_path, multipv=multipv) as teacher:
        total = len(examples)
        for index, example in enumerate(examples, start=1):
            symbolic_example = build_symbolic_proposer_example(
                example,
                candidate_context_version=2,
                global_context_version=1,
            )
            effective_multipv = (
                len(symbolic_example.candidate_action_indices)
                if multipv <= 0
                else min(multipv, len(symbolic_example.candidate_action_indices))
            )
            analysis_list = teacher.analyse(
                example.fen,
                nodes=nodes,
                multipv=effective_multipv,
            )
            teacher_examples.append(
                build_search_teacher_example_from_analysis(
                    example,
                    symbolic_example=symbolic_example,
                    analysis_list=analysis_list,
                    teacher_engine=str(teacher_engine_path),
                    nodes=nodes,
                    depth=None,
                    movetime_ms=None,
                    effective_multipv=effective_multipv,
                    policy_temperature_cp=policy_temperature_cp,
                )
            )
            trace_examples.append(
                build_search_trace_example_from_analysis(
                    example,
                    symbolic_example=symbolic_example,
                    analysis_list=analysis_list,
                    teacher_engine=str(teacher_engine_path),
                    nodes=nodes,
                    depth=None,
                    movetime_ms=None,
                    effective_multipv=effective_multipv,
                    policy_temperature_cp=policy_temperature_cp,
                )
            )
            if log_every > 0 and (index % log_every == 0 or index == total):
                _log(
                    f"[workflow:{split}] analysed {index}/{total} positions "
                    f"with teacher={teacher_engine_path}"
                )
    return teacher_examples, trace_examples


class _UciTeacher:
    def __init__(self, engine_path: Path, *, multipv: int) -> None:
        self.engine_path = engine_path
        self.multipv = max(1, multipv)
        self._process: subprocess.Popen[str] | None = None

    def __enter__(self) -> "_UciTeacher":
        self._process = subprocess.Popen(
            [str(self.engine_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._expect("uciok")
        self._send(f"setoption name MultiPV value {self.multipv}")
        self._send("isready")
        self._expect("readyok")
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._process is None:
            return
        try:
            self._send("quit")
        finally:
            self._process.wait(timeout=5)
            self._process = None

    def analyse(self, fen: str, *, nodes: int, multipv: int) -> list[dict[str, Any]]:
        board = chess.Board(fen)
        self._send(f"setoption name MultiPV value {max(1, multipv)}")
        self._send("isready")
        self._expect("readyok")
        self._send(f"position fen {fen}")
        self._send(f"go nodes {nodes}")
        by_multipv: dict[int, dict[str, Any]] = {}
        for raw_line in self._stdout():
            line = raw_line.strip()
            if line.startswith("info "):
                parsed = _parse_analysis_line(line, turn=board.turn)
                if parsed is not None:
                    key, info = parsed
                    by_multipv[key] = info
            elif line.startswith("bestmove "):
                break
        return [by_multipv[key] for key in sorted(by_multipv)]

    def _send(self, command: str) -> None:
        stdin = self._stdin()
        stdin.write(command + "\n")
        stdin.flush()

    def _expect(self, sentinel: str) -> None:
        for raw_line in self._stdout():
            if raw_line.strip() == sentinel:
                return
        raise RuntimeError(f"UCI teacher closed before {sentinel!r}")

    def _stdin(self) -> Any:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("UCI teacher stdin is unavailable")
        return self._process.stdin

    def _stdout(self) -> Any:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("UCI teacher stdout is unavailable")
        return self._process.stdout


def _parse_analysis_line(line: str, *, turn: bool) -> tuple[int, dict[str, Any]] | None:
    if " score " not in line or " pv " not in line:
        return None
    tokens = line.split()
    if not tokens or tokens[0] != "info":
        return None

    multipv = 1
    score: chess.engine.PovScore | None = None
    pv_index: int | None = None
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "multipv" and index + 1 < len(tokens):
            multipv = int(tokens[index + 1])
            index += 2
            continue
        if token == "score" and index + 2 < len(tokens):
            kind = tokens[index + 1]
            value = int(tokens[index + 2])
            if kind == "cp":
                score = chess.engine.PovScore(chess.engine.Cp(value), turn)
            elif kind == "mate":
                score = chess.engine.PovScore(chess.engine.Mate(value), turn)
            index += 3
            continue
        if token == "pv":
            pv_index = index + 1
            break
        index += 1
    if score is None or pv_index is None or pv_index >= len(tokens):
        return None
    pv = [chess.Move.from_uci(move_uci) for move_uci in tokens[pv_index:]]
    if not pv:
        return None
    return multipv, {"score": score, "pv": pv}


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _log(message: str) -> None:
    print(message, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
