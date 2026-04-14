"""Tests for dataset oracle transport modes."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from unittest.mock import patch

from train.datasets.oracle import _default_oracle_command, label_records_with_oracle
from train.datasets.schema import RawPositionRecord


def test_label_records_with_unix_socket_endpoint() -> None:
    socket_path = Path("/tmp/enginekonzept-oracle.sock")
    responses = [
        {
            "annotations": {
                "has_legal_castle": False,
                "has_legal_en_passant": False,
                "has_legal_promotion": False,
                "in_check": False,
                "is_checkmate": False,
                "is_low_material_endgame": True,
                "is_stalemate": False,
                "legal_move_count": 3,
                "piece_count": 2,
                "selected_move_gives_check": None,
                "selected_move_is_capture": None,
                "selected_move_is_castle": None,
                "selected_move_is_en_passant": None,
                "selected_move_is_promotion": None,
            },
            "fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "legal_action_encodings": [[0, 1, 0]],
            "legal_moves": ["e1d1"],
            "next_fen": None,
            "position_encoding": {
                "piece_tokens": [[4, 0, 5], [60, 1, 5]],
                "rule_token": [0, 0, 0, 0, 1, 0],
                "square_tokens": [[index, 0] for index in range(64)],
            },
            "selected_action_encoding": None,
            "selected_move_uci": None,
            "side_to_move": "w",
        }
    ]

    sent_payloads: list[bytes] = []

    class FakeSocket:
        def __enter__(self) -> "FakeSocket":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def connect(self, path: str) -> None:
            assert path == str(socket_path)

        def sendall(self, payload: bytes) -> None:
            sent_payloads.append(payload)

        def shutdown(self, how: int) -> None:
            assert how == socket.SHUT_WR

        def recv(self, size: int) -> bytes:
            assert size == 65536
            if not hasattr(self, "_sent_once"):
                self._sent_once = True
                return (json.dumps(responses[0]) + "\n").encode("utf-8")
            return b""

    with patch("train.datasets.oracle.socket.socket", return_value=FakeSocket()):
        outputs = label_records_with_oracle(
            [
                RawPositionRecord(
                    sample_id="quiet",
                    fen="4k3/8/8/8/8/8/8/4K3 w - - 0 1",
                    source="test",
                )
            ],
            command=[f"unix://{socket_path}"],
        )

    assert outputs == responses
    assert len(sent_payloads) == 1
    assert b'"fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1"' in sent_payloads[0]


def test_label_records_with_subprocess_command_roundtrips(tmp_path: Path) -> None:
    record = RawPositionRecord(
        sample_id="quiet",
        fen="4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        source="test",
    )

    script = tmp_path / "oracle_echo.py"
    script.write_text(
        "\n".join(
            [
                "import json, sys",
                "for line in sys.stdin:",
                "    if not line.strip():",
                "        continue",
                "    req = json.loads(line)",
                "    print(json.dumps({'fen': req['fen']}))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    outputs = label_records_with_oracle([record], command=["python3", str(script)])

    assert outputs == [{"fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1"}]


def test_default_oracle_command_prefers_built_binary(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    binary = repo_root / "rust" / "target" / "debug" / "dataset-oracle"
    binary.parent.mkdir(parents=True)
    binary.write_text("", encoding="utf-8")

    with patch.dict(os.environ, {}, clear=True):
        command = _default_oracle_command(repo_root)

    assert command == [str(binary)]


def test_default_oracle_command_honors_env_override(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"

    with patch.dict(os.environ, {"ENGINEKONZEPT_DATASET_ORACLE": "python3 oracle.py"}):
        command = _default_oracle_command(repo_root)

    assert command == ["python3", "oracle.py"]


def test_default_oracle_command_falls_back_to_home_cargo_bin(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    cargo_bin = tmp_path / "home" / ".cargo" / "bin" / "cargo"
    cargo_bin.parent.mkdir(parents=True)
    cargo_bin.write_text("", encoding="utf-8")

    with (
        patch.dict(os.environ, {}, clear=True),
        patch("train.datasets.oracle.shutil.which", return_value=None),
        patch("train.datasets.oracle.Path.home", return_value=tmp_path / "home"),
    ):
        command = _default_oracle_command(repo_root)

    assert command == [
        str(cargo_bin),
        "run",
        "--quiet",
        "-p",
        "tools",
        "--bin",
        "dataset-oracle",
    ]
