"""Tests for offline proposer-vs-teacher disagreement workflows."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from train.datasets import (
    DatasetExample,
    build_search_disagreement_examples,
    build_search_teacher_example_from_analysis,
    build_symbolic_proposer_example,
)
from train.datasets.search_disagreements import SearchDisagreementExample
from train.models.proposer import LegalityPolicyProposer


def test_build_search_disagreement_examples_scores_symbolic_checkpoint(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    chess = pytest.importorskip("chess")

    example = DatasetExample.from_dict(_dataset_example_dict())
    symbolic_teacher_example = build_symbolic_proposer_example(
        example,
        candidate_context_version=2,
        global_context_version=1,
    )
    teacher_example = build_search_teacher_example_from_analysis(
        example,
        symbolic_example=symbolic_teacher_example,
        analysis_list=[
            {
                "score": chess.engine.PovScore(chess.engine.Cp(80), chess.WHITE),
                "pv": [chess.Move.from_uci("e2e4"), chess.Move.from_uci("e7e5")],
            },
            {
                "score": chess.engine.PovScore(chess.engine.Cp(20), chess.WHITE),
                "pv": [chess.Move.from_uci("d2d4"), chess.Move.from_uci("d7d5")],
            },
        ],
        teacher_engine="/usr/games/stockfish18",
        nodes=128,
        depth=None,
        movetime_ms=None,
        effective_multipv=2,
        policy_temperature_cp=100.0,
    )

    checkpoint_path = _write_checkpoint(
        tmp_path,
        architecture="symbolic_v1",
        hidden_dim=32,
        hidden_layers=1,
    )

    built = build_search_disagreement_examples(
        [example],
        [teacher_example],
        checkpoint_path=checkpoint_path,
        top_k=4,
    )

    assert len(built) == 1
    disagreement = built[0]
    assert disagreement.sample_id == example.sample_id
    assert disagreement.candidate_context_version == 1
    assert disagreement.global_context_version == 1
    assert disagreement.teacher_top1_action_index == teacher_example.teacher_top_k_action_indices[0]
    assert len(disagreement.proposer_candidate_scores) == len(disagreement.candidate_action_indices)
    assert len(disagreement.proposer_policy) == len(disagreement.candidate_action_indices)
    assert abs(sum(disagreement.proposer_policy) - 1.0) < 1e-6
    assert len(disagreement.proposer_top_k_action_indices) == 4
    assert 1 <= disagreement.teacher_rank_of_proposer_top1 <= len(
        disagreement.candidate_action_indices
    )
    assert 1 <= disagreement.proposer_rank_of_teacher_top1 <= len(
        disagreement.candidate_action_indices
    )
    assert disagreement.top1_disagrees == (
        disagreement.teacher_top1_action_index != disagreement.proposer_top1_action_index
    )
    assert disagreement.teacher_top1_minus_top2_cp is not None
    assert disagreement.proposer_top1_minus_top2_logit is not None


def test_search_disagreement_example_roundtrips_json() -> None:
    example = SearchDisagreementExample.from_dict(
        {
            "sample_id": "sample-1",
            "split": "validation",
            "fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "feature_vector": [0.0, 1.0],
            "candidate_context_version": 1,
            "global_context_version": 1,
            "global_features": [0.0],
            "candidate_action_indices": [1, 2],
            "candidate_features": [[0.0], [1.0]],
            "teacher_engine": "/usr/games/stockfish18",
            "teacher_nodes": 64,
            "teacher_depth": None,
            "teacher_movetime_ms": None,
            "teacher_multipv": 2,
            "teacher_coverage_ratio": 1.0,
            "teacher_root_value_cp": 20.0,
            "teacher_root_value_mate": None,
            "teacher_candidate_scores_cp": [20.0, 10.0],
            "teacher_policy": [0.7, 0.3],
            "teacher_top_k_action_indices": [1, 2],
            "proposer_checkpoint": "models/proposer/symbolic/checkpoint.pt",
            "proposer_candidate_scores": [0.2, -0.1],
            "proposer_policy": [0.57, 0.43],
            "proposer_top_k_action_indices": [1, 2],
            "teacher_top1_action_index": 1,
            "proposer_top1_action_index": 1,
            "teacher_rank_of_proposer_top1": 1,
            "proposer_rank_of_teacher_top1": 1,
            "top1_disagrees": False,
            "teacher_top1_minus_top2_cp": 10.0,
            "proposer_top1_minus_top2_logit": 0.3,
            "teacher_top1_advantage_cp": 0.0,
            "policy_l1_distance": 0.26,
        }
    )

    roundtrip = SearchDisagreementExample.from_json(json.dumps(example.to_dict()))

    assert roundtrip == example


def test_build_search_disagreement_examples_rejects_non_symbolic_checkpoint(
    tmp_path: Path,
) -> None:
    chess = pytest.importorskip("chess")

    example = DatasetExample.from_dict(_dataset_example_dict())
    symbolic_teacher_example = build_symbolic_proposer_example(
        example,
        candidate_context_version=2,
        global_context_version=1,
    )
    teacher_example = build_search_teacher_example_from_analysis(
        example,
        symbolic_example=symbolic_teacher_example,
        analysis_list=[
            {
                "score": chess.engine.PovScore(chess.engine.Cp(40), chess.WHITE),
                "pv": [chess.Move.from_uci("e2e4")],
            }
        ],
        teacher_engine="/usr/games/stockfish18",
        nodes=32,
        depth=None,
        movetime_ms=None,
        effective_multipv=1,
        policy_temperature_cp=100.0,
    )
    checkpoint_path = _write_checkpoint(
        tmp_path,
        architecture="mlp_v1",
        hidden_dim=32,
        hidden_layers=1,
    )

    with pytest.raises(ValueError, match="symbolic_v1"):
        build_search_disagreement_examples(
            [example],
            [teacher_example],
            checkpoint_path=checkpoint_path,
        )


def _write_checkpoint(
    tmp_path: Path,
    *,
    architecture: str,
    hidden_dim: int,
    hidden_layers: int,
) -> Path:
    torch = pytest.importorskip("torch")

    torch.manual_seed(0)
    model = LegalityPolicyProposer(
        architecture=architecture,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        dropout=0.0,
    )
    checkpoint_path = tmp_path / f"{architecture}_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "training_config": {
                "seed": 0,
                "output_dir": "artifacts/tests",
                "data": {
                    "dataset_path": "artifacts/datasets/phase5_stockfish_pgn_train_pi_10k_v1",
                    "train_split": "train",
                    "validation_split": "validation",
                },
                "model": {
                    "architecture": architecture,
                    "hidden_dim": hidden_dim,
                    "hidden_layers": hidden_layers,
                    "dropout": 0.0,
                },
                "optimization": {
                    "epochs": 1,
                    "batch_size": 4,
                    "learning_rate": 1e-3,
                    "weight_decay": 0.0,
                    "legality_loss_weight": 1.0,
                    "policy_loss_weight": 1.0,
                },
                "evaluation": {
                    "legality_threshold": 0.5,
                    "checkpoint_selection": "legality_first",
                    "selection_policy_weight": 1.0,
                },
                "runtime": {
                    "torch_threads": 0,
                    "dataloader_workers": 0,
                },
                "export": {
                    "bundle_dir": "models/proposer/test",
                    "enabled": False,
                    "checkpoint_name": "checkpoint.pt",
                    "exported_program_name": "proposer.pt2",
                    "runtime_weights_name": "symbolic_runtime.bin",
                    "metadata_name": "metadata.json",
                },
            },
        },
        checkpoint_path,
    )
    return checkpoint_path


def _dataset_example_dict() -> dict[str, object]:
    return {
        "sample_id": "sample-1",
        "split": "train",
        "source": "fixture",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "side_to_move": "w",
        "selected_move_uci": "e2e4",
        "selected_action_encoding": [12, 28, 0],
        "next_fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "legal_moves": [
            "b1a3",
            "b1c3",
            "g1f3",
            "g1h3",
            "a2a3",
            "a2a4",
            "b2b3",
            "b2b4",
            "c2c3",
            "c2c4",
            "d2d3",
            "d2d4",
            "e2e3",
            "e2e4",
            "f2f3",
            "f2f4",
            "g2g3",
            "g2g4",
            "h2h3",
            "h2h4",
        ],
        "legal_action_encodings": [
            [1, 16, 0],
            [1, 18, 0],
            [6, 21, 0],
            [6, 23, 0],
            [8, 16, 0],
            [8, 24, 0],
            [9, 17, 0],
            [9, 25, 0],
            [10, 18, 0],
            [10, 26, 0],
            [11, 19, 0],
            [11, 27, 0],
            [12, 20, 0],
            [12, 28, 0],
            [13, 21, 0],
            [13, 29, 0],
            [14, 22, 0],
            [14, 30, 0],
            [15, 23, 0],
            [15, 31, 0],
        ],
        "position_encoding": {
            "piece_tokens": [[0, 0, 3]],
            "square_tokens": [[square, 0] for square in range(64)],
            "rule_token": [0, 15, -1, 0, 1, 0],
        },
        "wdl_target": {"win": 1, "draw": 0, "loss": 0},
        "annotations": {
            "in_check": False,
            "is_checkmate": False,
            "is_stalemate": False,
            "has_legal_en_passant": False,
            "has_legal_castle": False,
            "has_legal_promotion": False,
            "is_low_material_endgame": False,
            "legal_move_count": 20,
            "piece_count": 32,
            "selected_move_is_capture": False,
            "selected_move_is_promotion": False,
            "selected_move_is_castle": False,
            "selected_move_is_en_passant": False,
            "selected_move_gives_check": False,
        },
        "result": "1-0",
        "metadata": {},
    }
