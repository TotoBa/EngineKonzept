from __future__ import annotations

from pathlib import Path

import pytest

from train.config import (
    PlannerDataConfig,
    PlannerEvaluationConfig,
    PlannerExportConfig,
    PlannerRuntimeConfig,
)
from train.eval.agent_spec import SelfplayAgentSpec, write_selfplay_agent_spec
from train.eval.distributed_selfplay import (
    rebuild_phase10_pre_verify_selfplay_summary,
    run_phase10_pre_verify_selfplay_shard,
)
from train.eval.initial_fens import (
    SelfplayInitialFenEntry,
    SelfplayInitialFenSuite,
    write_selfplay_initial_fen_suite,
)
from train.eval.phase10_campaign import Phase10Lapv1ArenaCampaignSpec


pytest.importorskip("torch")

import torch
from train.models.lapv1 import LAPV1_MODEL_NAME, LAPv1Config, LAPv1Model
from train.trainers import LAPv1Stage2Config
from train.trainers.lapv1 import LAPv1OptimizationConfig, LAPv1TrainConfig


def test_distributed_pre_verify_selfplay_writes_shards_and_summary(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    checkpoint_path = _write_untrained_lapv2_checkpoint(tmp_path / "lapv2_selfplay.pt")
    agent_spec_path = tmp_path / "lapv2_selfplay.json"
    write_selfplay_agent_spec(
        agent_spec_path,
        SelfplayAgentSpec(
            name="lapv2_selfplay_test",
            agent_kind="lapv1",
            lapv1_checkpoint=str(checkpoint_path),
            state_context_version=1,
            deliberation_max_inner_steps=1,
            deliberation_q_threshold=0.3,
            tags=["lapv2", "test"],
        ),
    )
    opening_suite_path = tmp_path / "openings.json"
    write_selfplay_initial_fen_suite(
        opening_suite_path,
        SelfplayInitialFenSuite(
            name="selfplay_openings_test",
            entries=[
                SelfplayInitialFenEntry(
                    fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                    tier="opening",
                    sample_id="startpos",
                    source_path="test:startpos",
                    result="*",
                    selection_score=1.0,
                ),
                SelfplayInitialFenEntry(
                    fen="rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
                    tier="opening",
                    sample_id="queen_pawn",
                    source_path="test:queen_pawn",
                    result="*",
                    selection_score=2.0,
                ),
            ],
        ),
    )
    spec = Phase10Lapv1ArenaCampaignSpec(
        name="phase10_pre_verify_selfplay_test",
        output_root=str(tmp_path / "campaign"),
        merged_raw_dir=str(tmp_path / "raw"),
        train_dataset_dir=str(tmp_path / "datasets" / "train"),
        verify_dataset_dir=str(tmp_path / "datasets" / "verify"),
        phase5_source_name="test_source",
        phase5_seed="seed",
        workflow_output_root=str(tmp_path / "workflow"),
        proposer_checkpoint="models/proposer/checkpoint.pt",
        lapv1_config_path="python/configs/test_lapv2.json",
        lapv1_agent_spec_path=str(agent_spec_path),
        lapv1_verify_output_path=str(tmp_path / "workflow" / "verify.jsonl"),
        reference_arena_summary_path=str(tmp_path / "refs" / "arena_summary.json"),
        initial_fen_suite_path=str(opening_suite_path),
        teacher_nodes=64,
        workflow_chunk_size=4,
        pre_verify_selfplay_games=4,
        pre_verify_selfplay_games_per_task=2,
        pre_verify_selfplay_max_plies=2,
        pre_verify_selfplay_opening_selection_seed=7,
    )
    output_root = tmp_path / "campaign" / "pre_verify_selfplay"

    shard1 = run_phase10_pre_verify_selfplay_shard(
        spec=spec,
        repo_root=repo_root,
        agent_spec_path=agent_spec_path,
        agent_name="lapv2_selfplay_test",
        output_root=output_root,
        shard_index=1,
        starting_game_index=0,
        games=2,
        max_plies=2,
    )
    shard2 = run_phase10_pre_verify_selfplay_shard(
        spec=spec,
        repo_root=repo_root,
        agent_spec_path=agent_spec_path,
        agent_name="lapv2_selfplay_test",
        output_root=output_root,
        shard_index=2,
        starting_game_index=2,
        games=2,
        max_plies=2,
    )
    summary = rebuild_phase10_pre_verify_selfplay_summary(
        output_root=output_root,
        agent_name="lapv2_selfplay_test",
        agent_spec_path=agent_spec_path,
    )

    assert Path(str(shard1["session_path"])).exists()
    assert Path(str(shard2["session_path"])).exists()
    assert summary["aggregate"]["game_count"] == 4
    assert summary["aggregate"]["session_count"] == 2
    assert summary["aggregate"]["termination_counts"]["max_plies"] == 4
    assert shard1["selected_openings"][0]["sample_id"] in {"startpos", "queen_pawn"}
    assert Path(output_root / "summary.json").exists()


def _write_untrained_lapv2_checkpoint(path: Path) -> Path:
    config = LAPv1TrainConfig(
        seed=17,
        output_dir=str(path.parent / "lapv2_out"),
        stage="T2",
        data=PlannerDataConfig(
            train_path="train.jsonl",
            validation_path="validation.jsonl",
        ),
        model=LAPv1Config.from_mapping(
            {
                "deliberation": {
                    "max_inner_steps": 1,
                    "min_inner_steps": 1,
                    "memory_slots": 4,
                    "rollback_buffer_size": 4,
                },
                "lapv2": {"enabled": True, "shared_opponent_readout": True},
                "opponent_head": {
                    "architecture": "set_v2",
                    "hidden_dim": 64,
                    "hidden_layers": 1,
                    "action_embedding_dim": 16,
                    "dropout": 0.0,
                },
                "value_head": {"hidden_dim": 256},
                "policy_head": {
                    "hidden_dim": 128,
                    "action_embedding_dim": 32,
                    "feedforward_dim": 256,
                },
                "state_embedder": {"feedforward_dim": 512},
                "intention_encoder": {"feedforward_dim": 512},
            }
        ),
        optimization=LAPv1OptimizationConfig(
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
        ),
        evaluation=PlannerEvaluationConfig(top_k=3),
        runtime=PlannerRuntimeConfig(
            torch_threads=1,
            dataloader_workers=0,
        ),
        export=PlannerExportConfig(
            bundle_dir=str(path.parent / "bundle"),
        ),
        stage2=LAPv1Stage2Config(max_inner_steps_schedule=(1,)),
    )
    model = LAPv1Model(config.model)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": LAPV1_MODEL_NAME,
            "model_state_dict": dict(model.state_dict()),
            "training_config": config.to_dict(),
        },
        path,
    )
    return path
