from __future__ import annotations

from pathlib import Path

from train.datasets.schema import DatasetExample, PositionEncoding, TacticalAnnotations
from train.eval.external_engine import ExternalUciEngineAgent


def _make_example() -> DatasetExample:
    return DatasetExample(
        sample_id="engine_test",
        split="test",
        source="arena_test",
        fen="8/8/8/8/8/8/8/K6k w - - 0 1",
        side_to_move="w",
        selected_move_uci=None,
        selected_action_encoding=None,
        next_fen=None,
        legal_moves=["a1a2", "a1b1"],
        legal_action_encodings=[[0, 8, 0], [0, 1, 0]],
        position_encoding=PositionEncoding(
            piece_tokens=[],
            square_tokens=[[index, 0] for index in range(64)],
            rule_token=[0, 0, -1, 0, 1, 0],
        ),
        wdl_target=None,
        annotations=TacticalAnnotations(
            in_check=False,
            is_checkmate=False,
            is_stalemate=False,
            has_legal_en_passant=False,
            has_legal_castle=False,
            has_legal_promotion=False,
            is_low_material_endgame=True,
            legal_move_count=2,
            piece_count=2,
            selected_move_is_capture=None,
            selected_move_is_promotion=None,
            selected_move_is_castle=None,
            selected_move_is_en_passant=None,
            selected_move_gives_check=None,
        ),
        result=None,
        metadata={},
    )


def _write_fake_uci_engine(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import sys

while True:
    line = sys.stdin.readline()
    if not line:
        break
    command = line.strip()
    if command == "uci":
        print("id name FakeUci")
        print("id author Test")
        print("option name Threads type spin default 1 min 1 max 1")
        print("option name Hash type spin default 16 min 1 max 64")
        print("uciok")
        sys.stdout.flush()
    elif command == "isready":
        print("readyok")
        sys.stdout.flush()
    elif command.startswith("position fen "):
        continue
    elif command.startswith("go "):
        print("bestmove a1a2")
        sys.stdout.flush()
    elif command == "quit":
        break
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_external_engine_agent_selects_and_labels_move(tmp_path: Path) -> None:
    engine_path = tmp_path / "fake_uci.py"
    _write_fake_uci_engine(engine_path)
    example = _make_example()

    def _label_selected_move(_example: DatasetExample, move_uci: str, _repo_root: Path) -> DatasetExample:
        assert move_uci == "a1a2"
        return DatasetExample(
            sample_id="engine_test:next",
            split="test",
            source="arena_test",
            fen=example.fen,
            side_to_move=example.side_to_move,
            selected_move_uci=move_uci,
            selected_action_encoding=[0, 8, 0],
            next_fen="8/8/8/8/8/8/K7/7k b - - 1 1",
            legal_moves=example.legal_moves,
            legal_action_encodings=example.legal_action_encodings,
            position_encoding=example.position_encoding,
            wdl_target=None,
            annotations=example.annotations,
            result=None,
            metadata={},
        )

    agent = ExternalUciEngineAgent(
        name="fake_uci",
        engine_path=engine_path,
        repo_root=tmp_path,
        nodes=32,
        threads=1,
        hash_mb=16,
        label_selected_move=_label_selected_move,
    )
    try:
        decision = agent.select_move(example)
    finally:
        agent.close()

    assert decision.move_uci == "a1a2"
    assert decision.action_index == 40
    assert decision.next_fen == "8/8/8/8/8/8/K7/7k b - - 1 1"
    assert decision.considered_candidate_count == 2
