"""Check StateContextV1 against shared Python/Rust golden vectors."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

from train.datasets.contracts import build_state_context_v1
from train.datasets.schema import DatasetExample, PositionEncoding, TacticalAnnotations

try:
    import chess
except ModuleNotFoundError as exc:  # pragma: no cover - exercised when chess is absent
    raise RuntimeError(
        "python-chess is required for StateContextV1 golden checks. Install the 'train' extra."
    ) from exc


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    golden_path = repo_root / "artifacts" / "golden" / "state_context_v1_golden.json"
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    examples = list(golden["examples"])

    python_mismatches = _python_mismatches(examples)
    rust_mismatches = _rust_mismatches(examples, repo_root=repo_root)

    if python_mismatches or rust_mismatches:
        print("state_context_v1 golden drift detected:")
        for fen in python_mismatches:
            print(f"  python mismatch: {fen}")
        for fen in rust_mismatches:
            print(f"  rust mismatch: {fen}")
        return 1

    print(f"checked {len(examples)} StateContextV1 goldens: exact match")
    return 0


def _python_mismatches(examples: list[dict[str, Any]]) -> list[str]:
    mismatches: list[str] = []
    for example in examples:
        fen = str(example["fen"])
        actual = build_state_context_v1(_dataset_example_for_fen(fen)).to_dict()
        expected = {
            "contract_name": "StateContext",
            "version": 1,
            "feature_values": list(example["feature_values"]),
            "edge_src_square": list(example["edge_src_square"]),
            "edge_dst_square": list(example["edge_dst_square"]),
            "edge_piece_type": list(example["edge_piece_type"]),
        }
        if actual != expected:
            mismatches.append(fen)
    return mismatches


def _rust_mismatches(examples: list[dict[str, Any]], *, repo_root: Path) -> list[str]:
    input_payload = "".join(f"{example['fen']}\n" for example in examples)
    completed = subprocess.run(
        ["cargo", "run", "--quiet", "-p", "tools", "--bin", "state_context_dump"],
        cwd=repo_root / "rust",
        input=input_payload,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())

    outputs = [
        json.loads(line)
        for line in completed.stdout.splitlines()
        if line.strip()
    ]
    if len(outputs) != len(examples):
        raise RuntimeError(
            f"state_context_dump returned {len(outputs)} records for {len(examples)} inputs"
        )

    mismatches: list[str] = []
    for example, actual in zip(examples, outputs, strict=True):
        expected = {
            "fen": str(example["fen"]),
            "feature_values": list(example["feature_values"]),
            "edge_src_square": list(example["edge_src_square"]),
            "edge_dst_square": list(example["edge_dst_square"]),
            "edge_piece_type": list(example["edge_piece_type"]),
        }
        if actual != expected:
            mismatches.append(str(example["fen"]))
    return mismatches


def _dataset_example_for_fen(fen: str) -> DatasetExample:
    board = chess.Board(fen)
    legal_moves = [move.uci() for move in board.legal_moves]
    return DatasetExample(
        sample_id="state_context_golden",
        split="test",
        source="state_context_golden",
        fen=fen,
        side_to_move="w" if board.turn else "b",
        selected_move_uci=None,
        selected_action_encoding=None,
        next_fen=None,
        legal_moves=legal_moves,
        legal_action_encodings=[[0, 0, index] for index, _move in enumerate(legal_moves)],
        position_encoding=PositionEncoding(
            piece_tokens=[],
            square_tokens=[[square_index, 0] for square_index in range(64)],
            rule_token=[0, 0, -1, 0, 1, 0],
        ),
        wdl_target=None,
        annotations=TacticalAnnotations(
            in_check=board.is_check(),
            is_checkmate=board.is_checkmate(),
            is_stalemate=board.is_stalemate(),
            has_legal_en_passant=any(board.is_en_passant(move) for move in board.legal_moves),
            has_legal_castle=any(board.is_castling(move) for move in board.legal_moves),
            has_legal_promotion=any(move.promotion is not None for move in board.legal_moves),
            is_low_material_endgame=len(board.piece_map()) <= 6,
            legal_move_count=len(legal_moves),
            piece_count=len(board.piece_map()),
            selected_move_is_capture=None,
            selected_move_is_promotion=None,
            selected_move_is_castle=None,
            selected_move_is_en_passant=None,
            selected_move_gives_check=None,
        ),
        result=None,
        metadata={},
    )


if __name__ == "__main__":
    raise SystemExit(main())
