"""Tests for the Phase-7 workflow orchestration script."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_opponent_workflow_dataset.py"
)
_SPEC = importlib.util.spec_from_file_location("build_opponent_workflow_dataset", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_parse_analysis_line_cp_score() -> None:
    chess = pytest.importorskip("chess")

    parsed = _MODULE._parse_analysis_line(
        "info depth 10 multipv 2 score cp 34 pv e2e4 e7e5 g1f3",
        turn=chess.WHITE,
    )

    assert parsed is not None
    multipv, payload = parsed
    assert multipv == 2
    assert payload["score"].pov(chess.WHITE).score(mate_score=100_000) == 34
    assert [move.uci() for move in payload["pv"]] == ["e2e4", "e7e5", "g1f3"]


def test_parse_analysis_line_ignores_non_pv_info() -> None:
    assert _MODULE._parse_analysis_line("info depth 8 nodes 1024 nps 100000", turn=True) is None
