from __future__ import annotations

from pathlib import Path

import chess
import pytest

from train.config import (
    PlannerDataConfig,
    PlannerEvaluationConfig,
    PlannerExportConfig,
    PlannerRuntimeConfig,
)
from train.datasets.oracle import label_records_with_oracle
from train.datasets.schema import DatasetExample, RawPositionRecord
from train.datasets import dataset_example_from_oracle_payload
from train.eval.agent_spec import SelfplayAgentSpec
from train.eval.lapv1_runtime import build_lapv1_runtime_from_spec
from train.models.lapv1 import LAPV1_MODEL_NAME, LAPv1Config, LAPv1Model
from train.trainers import LAPv1Stage2Config
from train.trainers.lapv1 import LAPv1OptimizationConfig, LAPv1TrainConfig


pytest.importorskip("torch")

import torch


def test_lapv1_runtime_selects_legal_move_and_emits_trace(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    checkpoint_path = _write_untrained_lapv1_checkpoint(
        tmp_path / "lapv1_stage2.pt",
        max_inner_steps=2,
    )
    spec = SelfplayAgentSpec(
        name="lapv1_runtime_test",
        agent_kind="lapv1",
        lapv1_checkpoint=str(checkpoint_path),
        state_context_version=1,
        deliberation_max_inner_steps=2,
        deliberation_q_threshold=0.35,
    )
    runtime = build_lapv1_runtime_from_spec(spec, repo_root=repo_root)
    runtime.label_selected_move = _label_selected_move

    example = _oracle_example(repo_root, chess.STARTING_FEN)
    decision = runtime.select_move(example)

    assert decision.move_uci in example.legal_moves
    assert decision.selector_name == "lapv1_runtime_test"
    assert runtime.last_deliberation_trace is not None
    assert 0 < len(runtime.last_deliberation_trace.steps) <= 2


def test_lapv1_runtime_single_legal_move_returns_step0_trace(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    checkpoint_path = _write_untrained_lapv1_checkpoint(
        tmp_path / "lapv1_stage2_single.pt",
        max_inner_steps=2,
    )
    spec = SelfplayAgentSpec(
        name="lapv1_runtime_single",
        agent_kind="lapv1",
        lapv1_checkpoint=str(checkpoint_path),
        state_context_version=1,
        deliberation_max_inner_steps=2,
    )
    runtime = build_lapv1_runtime_from_spec(spec, repo_root=repo_root)
    runtime.label_selected_move = _label_selected_move

    example = _oracle_example(
        repo_root,
        "r5r1/3bk3/1n2p3/p1pP3P/P7/2P2N2/nP1N1q1P/4RK2 w - - 0 43",
    )
    decision = runtime.select_move(example)

    assert decision.move_uci == "f1f2"
    assert runtime.last_deliberation_trace is not None
    assert runtime.last_deliberation_trace.steps == []


def _write_untrained_lapv1_checkpoint(path: Path, *, max_inner_steps: int) -> Path:
    config = LAPv1TrainConfig(
        seed=17,
        output_dir=str(path.parent / "lapv1_runtime_out"),
        stage="T2",
        data=PlannerDataConfig(
            train_path="train.jsonl",
            validation_path="validation.jsonl",
        ),
        model=LAPv1Config.from_mapping(
            {
                "deliberation": {
                    "max_inner_steps": max_inner_steps,
                    "min_inner_steps": min(2, max_inner_steps),
                    "memory_slots": 4,
                    "rollback_buffer_size": 4,
                },
                "opponent_head": {
                    "architecture": "set_v2",
                    "hidden_dim": 64,
                    "hidden_layers": 1,
                    "action_embedding_dim": 16,
                    "dropout": 0.0,
                },
                "value_head": {"hidden_dim": 1024},
                "policy_head": {
                    "hidden_dim": 512,
                    "action_embedding_dim": 32,
                    "feedforward_dim": 1024,
                },
                "state_embedder": {"feedforward_dim": 1024},
                "intention_encoder": {"feedforward_dim": 1024},
            }
        ),
        optimization=LAPv1OptimizationConfig(
            epochs=2,
            batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            value_wdl_weight=1.0,
            value_cp_weight=0.25,
            sharpness_weight=0.1,
            sharpness_target_loss_weight=0.1,
            policy_ce_weight=1.0,
            policy_kl_weight=0.25,
            policy_margin_weight=0.1,
            policy_rank_weight=0.1,
            intention_aux_weight=0.05,
            deliberation_monotonicity_weight=0.05,
        ),
        evaluation=PlannerEvaluationConfig(top_k=3),
        runtime=PlannerRuntimeConfig(
            torch_threads=1,
            dataloader_workers=0,
        ),
        export=PlannerExportConfig(
            bundle_dir=str(path.parent / "bundle"),
        ),
        stage2=LAPv1Stage2Config(max_inner_steps_schedule=(max_inner_steps,)),
    )
    model = LAPv1Model(config.model)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": LAPV1_MODEL_NAME,
            "model_state_dict": model.state_dict(),
            "training_config": config.to_dict(),
        },
        path,
    )
    return path


def _oracle_example(repo_root: Path, fen: str) -> DatasetExample:
    payload = label_records_with_oracle(
        [RawPositionRecord(sample_id="lapv1_runtime_test", fen=fen, source="lapv1_test")],
        repo_root=repo_root,
    )[0]
    return dataset_example_from_oracle_payload(
        sample_id="lapv1_runtime_test",
        split="test",
        source="lapv1_test",
        fen=fen,
        payload=payload,
    )


def _label_selected_move(example: DatasetExample, move_uci: str, _repo_root: Path) -> DatasetExample:
    board = chess.Board(example.fen)
    board.push(chess.Move.from_uci(move_uci))
    return DatasetExample(
        sample_id=f"{example.sample_id}:selected",
        split=example.split,
        source=example.source,
        fen=example.fen,
        side_to_move=example.side_to_move,
        selected_move_uci=move_uci,
        selected_action_encoding=next(
            action
            for legal_move, action in zip(
                example.legal_moves,
                example.legal_action_encodings,
                strict=True,
            )
            if legal_move == move_uci
        ),
        next_fen=board.fen(),
        legal_moves=example.legal_moves,
        legal_action_encodings=example.legal_action_encodings,
        position_encoding=example.position_encoding,
        wdl_target=example.wdl_target,
        annotations=example.annotations,
        result=example.result,
        metadata=dict(example.metadata),
    )
