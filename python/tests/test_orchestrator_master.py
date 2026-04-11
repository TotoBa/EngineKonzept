from __future__ import annotations

import json
from pathlib import Path

import chess

from train.eval.agent_spec import SelfplayAgentSpec, write_selfplay_agent_spec
from train.orchestrator.master import (
    LabelPgnCorpusJobSpec,
    MasterSpec,
    OrchestratorMaster,
    Phase10LineageSpec,
    PromotionThresholds,
)
from train.orchestrator.models import CampaignRow, ModelRow


class _StubDB:
    def __init__(
        self,
        *,
        campaigns: list[CampaignRow] | None = None,
        models: list[ModelRow] | None = None,
    ) -> None:
        self._campaigns = list(campaigns or [])
        self._models = list(models or [])
        self.updated_models: list[tuple[int, dict[str, object]]] = []
        self.updated_campaigns: list[tuple[int, dict[str, object]]] = []

    def list_campaign_records(self, *, limit: int = 1000, **_: object) -> list[CampaignRow]:
        return list(self._campaigns)[:limit]

    def list_model_records(self, *, limit: int = 1000, **_: object) -> list[ModelRow]:
        return list(self._models)[:limit]

    def update_model_record(self, model_id: int, **fields: object) -> None:
        self.updated_models.append((model_id, dict(fields)))

    def update_campaign_record(self, campaign_id: int, **fields: object) -> None:
        self.updated_campaigns.append((campaign_id, dict(fields)))


class _StubController:
    def __init__(self) -> None:
        self.phase10_submissions: list[dict[str, object]] = []
        self.label_submissions: list[dict[str, object]] = []

    def submit_phase10_campaign(self, **kwargs: object) -> dict[str, object]:
        self.phase10_submissions.append(dict(kwargs))
        return {
            "campaign_id": 1000 + len(self.phase10_submissions),
            "config_path": str(kwargs["config_path"]),
        }

    def submit_label_pgn_corpus_campaign(self, **kwargs: object) -> dict[str, object]:
        self.label_submissions.append(dict(kwargs))
        return {
            "campaign_id": 2000 + len(self.label_submissions),
            "config_path": str(kwargs["config_path"]),
        }


def _planner_spec(name: str) -> SelfplayAgentSpec:
    return SelfplayAgentSpec(
        name=name,
        agent_kind="planner",
        proposer_checkpoint="models/proposer/test.pt",
        planner_checkpoint="models/planner/test.pt",
    )


def _lapv1_spec(name: str) -> SelfplayAgentSpec:
    return SelfplayAgentSpec(
        name=name,
        agent_kind="lapv1",
        lapv1_checkpoint="models/lapv1/test.pt",
    )


def _uci_engine_spec(name: str, *, engine_path: str = "/usr/games/vice") -> SelfplayAgentSpec:
    return SelfplayAgentSpec(
        name=name,
        agent_kind="uci_engine",
        opponent_mode="none",
        root_top_k=1,
        external_engine_path=engine_path,
        external_engine_depth=4,
    )


def _write_seed_phase10_config(
    tmp_path: Path,
    *,
    reference_agents: list[dict[str, str]] | None = None,
    reference_active_agent_specs_dir: Path | None = None,
    benchmark_agent_specs: dict[str, str] | None = None,
) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    payload = json.loads(
        (
            repo_root
            / "python"
            / "configs"
            / "phase10_lapv2_stage2_native_arena_all_sources_v1.json"
        ).read_text(encoding="utf-8")
    )
    payload["output_root"] = str(tmp_path / "seed_output")
    payload["merged_raw_dir"] = str(tmp_path / "seed_raw")
    payload["train_dataset_dir"] = str(tmp_path / "seed_train")
    payload["verify_dataset_dir"] = str(tmp_path / "seed_verify")
    payload["workflow_output_root"] = str(tmp_path / "seed_workflow")
    payload["reference_agents"] = list(reference_agents or [])
    payload["reference_active_agent_specs_dir"] = (
        str(reference_active_agent_specs_dir)
        if reference_active_agent_specs_dir is not None
        else None
    )
    payload["benchmark_agent_specs"] = dict(benchmark_agent_specs or {})
    config_path = tmp_path / "seed_phase10.json"
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return config_path


def _write_completed_raw_snapshot(
    work_dir: Path,
    *,
    train_rows: list[dict[str, object]],
    verify_rows: list[dict[str, object]],
) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "progress.json").write_text(
        json.dumps(
            {
                "completed": True,
                "counts": {"train": len(train_rows), "verify": len(verify_rows)},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (work_dir / "train_raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in train_rows),
        encoding="utf-8",
    )
    (work_dir / "verify_raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in verify_rows),
        encoding="utf-8",
    )
    (work_dir / "summary.json").write_text(
        json.dumps(
            {
                "train_raw_path": str(work_dir / "train_raw.jsonl"),
                "verify_raw_path": str(work_dir / "verify_raw.jsonl"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _session_payload(
    *,
    white_agent: str,
    black_agent: str,
    moves: list[str],
    result: str = "1-0",
    termination_reason: str = "checkmate",
    game_id: str = "game_0001",
) -> dict[str, object]:
    board = chess.Board()
    move_payloads: list[dict[str, object]] = []
    for ply_index, move_uci in enumerate(moves):
        fen = board.fen()
        move = chess.Move.from_uci(move_uci)
        board.push(move)
        move_payloads.append(
            {
                "ply_index": ply_index,
                "side_to_move": "w" if ply_index % 2 == 0 else "b",
                "fen": fen,
                "move_uci": move_uci,
                "action_index": ply_index,
                "selector_name": "test",
                "legal_candidate_count": 1,
                "considered_candidate_count": 1,
                "proposer_score": 0.0,
                "planner_score": 0.0,
                "reply_peak_probability": 0.0,
                "pressure": 0.0,
                "uncertainty": 0.0,
                "next_fen": board.fen(),
            }
        )
    return {
        "games": [
            {
                "game_id": game_id,
                "initial_fen": chess.STARTING_FEN,
                "final_fen": board.fen(),
                "result": result,
                "termination_reason": termination_reason,
                "move_count": len(moves),
                "white_agent": white_agent,
                "black_agent": black_agent,
                "moves": move_payloads,
            }
        ]
    }


def _install_completed_generation(
    *,
    db: _StubDB,
    master_name: str,
    lineage_name: str,
    paths: dict[str, Path],
    tmp_path: Path,
    arena_summary: dict[str, object],
    verify_top1: float = 0.2,
    verify_top3: float = 0.8,
    selfplay_sessions: list[dict[str, object]] | None = None,
    arena_sessions: list[dict[str, object]] | None = None,
) -> Path:
    campaign_payload = json.loads(paths["phase10_config_path"].read_text(encoding="utf-8"))
    campaign_output_root = Path(str(campaign_payload["output_root"]))
    arena_summary_path = campaign_output_root / "arena" / "summary.json"
    arena_summary_path.parent.mkdir(parents=True, exist_ok=True)
    arena_summary_path.write_text(
        json.dumps(arena_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if arena_sessions:
        sessions_root = campaign_output_root / "arena" / "sessions"
        sessions_root.mkdir(parents=True, exist_ok=True)
        for index, payload in enumerate(arena_sessions, start=1):
            (sessions_root / f"{index:02d}_arena.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    if selfplay_sessions:
        sessions_root = campaign_output_root / "pre_verify_selfplay" / "sessions"
        sessions_root.mkdir(parents=True, exist_ok=True)
        for index, payload in enumerate(selfplay_sessions, start=1):
            (sessions_root / f"{index:02d}_selfplay.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    summary_path = campaign_output_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "lapv1_verify_metrics": {
                    "root_top1_accuracy": verify_top1,
                    "root_top3_accuracy": verify_top3,
                },
                "arena_summary_path": str(arena_summary_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "trained" / "checkpoint.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"checkpoint")
    db._campaigns = [
        CampaignRow(
            id=1,
            name=f"{master_name}_{lineage_name}_g0001",
            kind="phase10_master",
            status="succeeded",
            config_path=str(paths["phase10_config_path"]),
            active_model_id=1,
            metadata={
                "master_name": master_name,
                "master_job_type": "phase10_lineage",
                "job_name": lineage_name,
                "generation": 1,
            },
            created_at=None,
            updated_at=None,
        )
    ]
    db._models = [
        ModelRow(
            id=1,
            campaign_id=1,
            parent_model_id=None,
            generation=1,
            train_config_path=str(paths["train_config_path"]),
            agent_spec_path=str(paths["agent_spec_path"]),
            checkpoint_path=str(checkpoint_path),
            bundle_path=None,
            verify_json_path=None,
            arena_summary_path=str(arena_summary_path),
            status="completed",
            promotion_score=None,
            metadata={
                "master_name": master_name,
                "lineage_name": lineage_name,
                "generation": 1,
            },
            created_at=None,
        )
    ]
    return checkpoint_path


def test_master_submits_label_job_before_dependent_lineage(tmp_path: Path) -> None:
    controller = _StubController()
    master = OrchestratorMaster(
        db=_StubDB(),
        controller=controller,
        repo_root=Path(__file__).resolve().parents[2],
        spec=MasterSpec(
            name="master_label_submit_test",
            output_root=str(tmp_path / "master"),
            label_jobs=(
                LabelPgnCorpusJobSpec(
                    name="pgn_label",
                    pgn_root="/srv/schach/PGN_DATA/pgn",
                    work_dir=str(tmp_path / "label_work"),
                    target_train_records=8,
                    target_verify_records=2,
                    max_games=4,
                ),
            ),
            lineages=(
                Phase10LineageSpec(
                    name="lapv2_lineage",
                    seed_phase10_config_path="python/configs/phase10_lapv2_stage2_native_arena_all_sources_v1.json",
                    output_root=str(tmp_path / "lineage"),
                    label_job_name="pgn_label",
                ),
            ),
        ),
        spec_path=tmp_path / "master.json",
    )

    summary = master.reconcile_once()

    assert summary["label_jobs"]["pgn_label"]["status"] == "submitted"
    assert summary["lineages"]["lapv2_lineage"]["status"] == "waiting_for_label"
    assert len(controller.label_submissions) == 1
    assert controller.phase10_submissions == []


def test_master_materializes_first_generation_from_completed_label_snapshot(tmp_path: Path) -> None:
    label_work_dir = tmp_path / "label_work"
    label_work_dir.mkdir(parents=True)
    (label_work_dir / "progress.json").write_text(
        """
{
  "completed": true,
  "counts": {"train": 8, "verify": 2}
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (label_work_dir / "train_raw.jsonl").write_text("{}\n", encoding="utf-8")
    (label_work_dir / "verify_raw.jsonl").write_text("{}\n", encoding="utf-8")

    controller = _StubController()
    master = OrchestratorMaster(
        db=_StubDB(),
        controller=controller,
        repo_root=Path(__file__).resolve().parents[2],
        spec=MasterSpec(
            name="master_first_generation_test",
            output_root=str(tmp_path / "master"),
            label_jobs=(
                LabelPgnCorpusJobSpec(
                    name="pgn_label",
                    pgn_root="/srv/schach/PGN_DATA/pgn",
                    work_dir=str(label_work_dir),
                    target_train_records=8,
                    target_verify_records=2,
                ),
            ),
            lineages=(
                Phase10LineageSpec(
                    name="lapv2_lineage",
                    seed_phase10_config_path="python/configs/phase10_lapv2_stage2_native_arena_all_sources_v1.json",
                    output_root=str(tmp_path / "lineage"),
                    label_job_name="pgn_label",
                ),
            ),
        ),
        spec_path=tmp_path / "master.json",
    )

    summary = master.reconcile_once()

    assert summary["label_jobs"]["pgn_label"]["status"] == "completed_external"
    assert summary["lineages"]["lapv2_lineage"]["status"] == "submitted_generation"
    assert len(controller.phase10_submissions) == 1

    generation_config_path = Path(str(controller.phase10_submissions[0]["config_path"]))
    generation_payload = json.loads(generation_config_path.read_text(encoding="utf-8"))
    train_payload = json.loads(
        Path(generation_payload["lapv1_config_path"]).read_text(encoding="utf-8")
    )

    assert generation_payload["merged_raw_dir"] == str(label_work_dir)
    assert generation_payload["name"] == "master_first_generation_test_lapv2_lineage_g0001"
    assert train_payload["output_dir"].startswith(str(tmp_path / "lineage" / "generation_0001"))
    assert "initial_checkpoint" not in train_payload


def test_master_rewrites_nested_stage2_workflow_paths_into_generation_output(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    seed_config_path = _write_seed_phase10_config(tmp_path)
    master = OrchestratorMaster(
        db=_StubDB(),
        controller=_StubController(),
        repo_root=repo_root,
        spec=MasterSpec(
            name="master_stage2_rewrite_test",
            output_root=str(tmp_path / "master"),
            lineages=(
                Phase10LineageSpec(
                    name="lapv2_lineage",
                    seed_phase10_config_path=str(seed_config_path),
                    output_root=str(tmp_path / "lineage"),
                ),
            ),
        ),
        spec_path=tmp_path / "master.json",
    )

    paths = master._materialize_generation_configs(  # type: ignore[attr-defined]
        lineage=master.spec.lineages[0],
        generation=1,
        parent_model=None,
        warm_start_checkpoint=None,
        label_work_dir=None,
    )
    generation_payload = json.loads(paths["phase10_config_path"].read_text(encoding="utf-8"))
    train_payload = json.loads(paths["train_config_path"].read_text(encoding="utf-8"))
    workflow_root = str(Path(str(generation_payload["workflow_output_root"])))

    assert train_payload["data"]["train_path"].startswith(workflow_root)
    assert train_payload["data"]["validation_path"].startswith(workflow_root)
    assert all(
        str(path).startswith(workflow_root)
        for path in train_payload["stage2"]["selection_validation_paths"]
    )
    assert all(
        str(path).startswith(workflow_root)
        for phase in train_payload["stage2"]["phases"]
        for path in [*phase["train_paths"], *phase["validation_paths"]]
    )


def test_master_can_bootstrap_generation_one_from_seed_artifacts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    seed_config_path = _write_seed_phase10_config(tmp_path)
    seed_payload = json.loads(seed_config_path.read_text(encoding="utf-8"))
    seed_workflow_root = Path(str(seed_payload["workflow_output_root"]))
    seed_train_dir = Path(str(seed_payload["train_dataset_dir"]))
    seed_verify_dir = Path(str(seed_payload["verify_dataset_dir"]))
    (seed_workflow_root / "summary.json").parent.mkdir(parents=True, exist_ok=True)
    (seed_workflow_root / "summary.json").write_text("{}\n", encoding="utf-8")
    (seed_train_dir / "summary.json").parent.mkdir(parents=True, exist_ok=True)
    (seed_train_dir / "summary.json").write_text("{}\n", encoding="utf-8")
    (seed_verify_dir / "summary.json").parent.mkdir(parents=True, exist_ok=True)
    (seed_verify_dir / "summary.json").write_text("{}\n", encoding="utf-8")

    master = OrchestratorMaster(
        db=_StubDB(),
        controller=_StubController(),
        repo_root=repo_root,
        spec=MasterSpec(
            name="master_stage2_seed_bootstrap_test",
            output_root=str(tmp_path / "master"),
            lineages=(
                Phase10LineageSpec(
                    name="lapv2_lineage",
                    seed_phase10_config_path=str(seed_config_path),
                    output_root=str(tmp_path / "lineage"),
                    bootstrap_generation_from_seed_artifacts=True,
                ),
            ),
        ),
        spec_path=tmp_path / "master.json",
    )

    paths = master._materialize_generation_configs(  # type: ignore[attr-defined]
        lineage=master.spec.lineages[0],
        generation=1,
        parent_model=None,
        warm_start_checkpoint=None,
        label_work_dir=None,
    )
    generation_payload = json.loads(paths["phase10_config_path"].read_text(encoding="utf-8"))
    train_payload = json.loads(paths["train_config_path"].read_text(encoding="utf-8"))

    assert generation_payload["reuse_existing_artifacts"] is True
    assert generation_payload["workflow_output_root"] == str(seed_workflow_root)
    assert generation_payload["train_dataset_dir"] == str(seed_train_dir)
    assert generation_payload["verify_dataset_dir"] == str(seed_verify_dir)
    assert "generation_0001/outputs/workflow" not in train_payload["data"]["train_path"]
    assert "generation_0001/outputs/workflow" not in train_payload["data"]["validation_path"]


def test_master_accepts_generation_and_submits_followup_with_warm_start(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    controller = _StubController()
    lineage = Phase10LineageSpec(
        name="lapv2_lineage",
        seed_phase10_config_path="python/configs/phase10_lapv2_stage2_native_arena_all_sources_v1.json",
        output_root=str(tmp_path / "lineage"),
        max_generations=2,
        on_accept="continue_training",
        promotion_thresholds=PromotionThresholds(
            min_verify_top1_accuracy=0.1,
            min_arena_score_rate=0.5,
        ),
    )
    db = _StubDB()
    master = OrchestratorMaster(
        db=db,
        controller=controller,
        repo_root=repo_root,
        spec=MasterSpec(
            name="master_followup_test",
            output_root=str(tmp_path / "master"),
            lineages=(lineage,),
        ),
        spec_path=tmp_path / "master.json",
    )

    paths = master._materialize_generation_configs(  # type: ignore[attr-defined]
        lineage=lineage,
        generation=1,
        parent_model=None,
        warm_start_checkpoint=None,
        label_work_dir=None,
    )
    campaign_payload = json.loads(paths["phase10_config_path"].read_text(encoding="utf-8"))
    campaign_output_root = Path(campaign_payload["output_root"])
    arena_summary_path = campaign_output_root / "arena" / "summary.json"
    arena_summary_path.parent.mkdir(parents=True, exist_ok=True)
    arena_summary_path.write_text(
        """
{
  "metadata": {"lapv1_agent_names": ["master_followup_test_lapv2_lineage_g0001"]},
  "standings": {
    "master_followup_test_lapv2_lineage_g0001": {"games": 4, "score": 2.5},
    "stockfish_d1": {"games": 4, "score": 1.5}
  }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    summary_path = campaign_output_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "lapv1_verify_metrics": {
                    "root_top1_accuracy": 0.2,
                    "root_top3_accuracy": 0.7,
                },
                "arena_summary_path": str(arena_summary_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "trained" / "checkpoint.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"checkpoint")
    db._campaigns = [
        CampaignRow(
            id=1,
            name="master_followup_test_lapv2_lineage_g0001",
            kind="phase10_master",
            status="succeeded",
            config_path=str(paths["phase10_config_path"]),
            active_model_id=1,
            metadata={
                "master_name": "master_followup_test",
                "master_job_type": "phase10_lineage",
                "job_name": "lapv2_lineage",
                "generation": 1,
            },
            created_at=None,
            updated_at=None,
        )
    ]
    db._models = [
        ModelRow(
            id=1,
            campaign_id=1,
            parent_model_id=None,
            generation=1,
            train_config_path=str(paths["train_config_path"]),
            agent_spec_path=str(paths["agent_spec_path"]),
            checkpoint_path=str(checkpoint_path),
            bundle_path=None,
            verify_json_path=None,
            arena_summary_path=str(arena_summary_path),
            status="completed",
            promotion_score=None,
            metadata={
                "master_name": "master_followup_test",
                "lineage_name": "lapv2_lineage",
                "generation": 1,
            },
            created_at=None,
        )
    ]

    summary = master.reconcile_once()

    assert summary["lineages"]["lapv2_lineage"]["accepted"] is True
    assert summary["lineages"]["lapv2_lineage"]["status"] == "accepted_submitted_next"
    assert len(controller.phase10_submissions) == 1
    assert db.updated_models[-1][1]["status"] == "accepted"

    next_config_path = Path(str(controller.phase10_submissions[0]["config_path"]))
    next_payload = json.loads(next_config_path.read_text(encoding="utf-8"))
    next_train_payload = json.loads(
        Path(next_payload["lapv1_config_path"]).read_text(encoding="utf-8")
    )
    assert next_train_payload["initial_checkpoint"] == str(checkpoint_path)


def test_master_rejects_generation_and_stops_when_thresholds_fail(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    controller = _StubController()
    lineage = Phase10LineageSpec(
        name="lapv2_lineage",
        seed_phase10_config_path="python/configs/phase10_lapv2_stage2_native_arena_all_sources_v1.json",
        output_root=str(tmp_path / "lineage"),
        max_generations=2,
        on_reject="stop",
        promotion_thresholds=PromotionThresholds(
            min_verify_top1_accuracy=0.9,
            min_arena_score_rate=0.9,
        ),
    )
    db = _StubDB()
    master = OrchestratorMaster(
        db=db,
        controller=controller,
        repo_root=repo_root,
        spec=MasterSpec(
            name="master_reject_test",
            output_root=str(tmp_path / "master"),
            lineages=(lineage,),
        ),
        spec_path=tmp_path / "master.json",
    )
    paths = master._materialize_generation_configs(  # type: ignore[attr-defined]
        lineage=lineage,
        generation=1,
        parent_model=None,
        warm_start_checkpoint=None,
        label_work_dir=None,
    )
    campaign_payload = json.loads(paths["phase10_config_path"].read_text(encoding="utf-8"))
    campaign_output_root = Path(campaign_payload["output_root"])
    arena_summary_path = campaign_output_root / "arena" / "summary.json"
    arena_summary_path.parent.mkdir(parents=True, exist_ok=True)
    arena_summary_path.write_text(
        """
{
  "metadata": {"lapv1_agent_names": ["master_reject_test_lapv2_lineage_g0001"]},
  "standings": {
    "master_reject_test_lapv2_lineage_g0001": {"games": 4, "score": 1.0}
  }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    summary_path = campaign_output_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "lapv1_verify_metrics": {
                    "root_top1_accuracy": 0.1,
                    "root_top3_accuracy": 0.3,
                },
                "arena_summary_path": str(arena_summary_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    db._campaigns = [
        CampaignRow(
            id=1,
            name="master_reject_test_lapv2_lineage_g0001",
            kind="phase10_master",
            status="succeeded",
            config_path=str(paths["phase10_config_path"]),
            active_model_id=1,
            metadata={
                "master_name": "master_reject_test",
                "master_job_type": "phase10_lineage",
                "job_name": "lapv2_lineage",
                "generation": 1,
            },
            created_at=None,
            updated_at=None,
        )
    ]
    db._models = [
        ModelRow(
            id=1,
            campaign_id=1,
            parent_model_id=None,
            generation=1,
            train_config_path=str(paths["train_config_path"]),
            agent_spec_path=str(paths["agent_spec_path"]),
            checkpoint_path=str(tmp_path / "trained" / "checkpoint.pt"),
            bundle_path=None,
            verify_json_path=None,
            arena_summary_path=str(arena_summary_path),
            status="completed",
            promotion_score=None,
            metadata={
                "master_name": "master_reject_test",
                "lineage_name": "lapv2_lineage",
                "generation": 1,
            },
            created_at=None,
        )
    ]

    summary = master.reconcile_once()

    assert summary["lineages"]["lapv2_lineage"]["accepted"] is False
    assert summary["lineages"]["lapv2_lineage"]["status"] == "rejected_stop"
    assert controller.phase10_submissions == []
    assert db.updated_models[-1][1]["status"] == "rejected"


def test_master_materializes_historical_vice_and_stockfish_parallel(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    active_dir = tmp_path / "active_specs"
    active_dir.mkdir()
    write_selfplay_agent_spec(active_dir / "historical_planner_v1.json", _planner_spec("historical_planner_v1"))
    write_selfplay_agent_spec(active_dir / "historical_lapv1.json", _lapv1_spec("historical_lapv1"))
    vice_spec_path = tmp_path / "vice_v2.json"
    write_selfplay_agent_spec(vice_spec_path, _uci_engine_spec("vice_v2"))
    seed_config_path = _write_seed_phase10_config(
        tmp_path,
        reference_active_agent_specs_dir=active_dir,
        benchmark_agent_specs={"vice_v2": str(vice_spec_path)},
    )
    master = OrchestratorMaster(
        db=_StubDB(),
        controller=_StubController(),
        repo_root=repo_root,
        spec=MasterSpec(
            name="master_parallel_arena_test",
            output_root=str(tmp_path / "master"),
            lineages=(
                Phase10LineageSpec(
                    name="lapv2_lineage",
                    seed_phase10_config_path=str(seed_config_path),
                    output_root=str(tmp_path / "lineage"),
                ),
            ),
        ),
        spec_path=tmp_path / "master.json",
    )

    paths = master._materialize_generation_configs(  # type: ignore[attr-defined]
        lineage=master.spec.lineages[0],
        generation=1,
        parent_model=None,
        warm_start_checkpoint=None,
        label_work_dir=None,
    )
    campaign_payload = json.loads(paths["phase10_config_path"].read_text(encoding="utf-8"))
    benchmark_specs = dict(campaign_payload["benchmark_agent_specs"])

    assert set(benchmark_specs) == {"historical_planner_v1", "vice_v2", "stockfish18_skill_00"}
    assert "historical_lapv1" not in benchmark_specs
    stockfish_payload = json.loads(
        Path(benchmark_specs["stockfish18_skill_00"]).read_text(encoding="utf-8")
    )
    assert stockfish_payload["external_engine_options"]["Skill Level"] == "0"


def test_master_submits_combined_feedback_label_job_for_selfplay_and_arena(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    controller = _StubController()
    lineage = Phase10LineageSpec(
        name="lapv2_lineage",
        seed_phase10_config_path="python/configs/phase10_lapv2_stage2_native_arena_all_sources_v1.json",
        output_root=str(tmp_path / "lineage"),
        max_generations=2,
        on_accept="continue_training",
        promotion_thresholds=PromotionThresholds(
            min_verify_top1_accuracy=0.1,
            min_arena_score_rate=0.5,
        ),
    )
    db = _StubDB()
    master = OrchestratorMaster(
        db=db,
        controller=controller,
        repo_root=repo_root,
        spec=MasterSpec(
            name="master_feedback_submit_test",
            output_root=str(tmp_path / "master"),
            lineages=(lineage,),
        ),
        spec_path=tmp_path / "master.json",
    )
    paths = master._materialize_generation_configs(  # type: ignore[attr-defined]
        lineage=lineage,
        generation=1,
        parent_model=None,
        warm_start_checkpoint=None,
        label_work_dir=None,
    )
    _install_completed_generation(
        db=db,
        master_name="master_feedback_submit_test",
        lineage_name="lapv2_lineage",
        paths=paths,
        tmp_path=tmp_path,
        arena_summary={
            "metadata": {"lapv1_agent_names": ["tracked_lap"]},
            "standings": {
                "tracked_lap": {"games": 4, "score": 2.5},
                "vice_v2": {"games": 2, "score": 0.5},
            },
            "matchups": [
                {
                    "white_agent": "tracked_lap",
                    "black_agent": "vice_v2",
                    "game_count": 2,
                    "white_score": 1.5,
                    "black_score": 0.5,
                    "result_counts": {"1-0": 1, "1/2-1/2": 1},
                    "termination_counts": {"checkmate": 1, "stalemate": 1},
                }
            ],
        },
        selfplay_sessions=[
            _session_payload(
                white_agent="tracked_lap",
                black_agent="tracked_lap",
                moves=["e2e4", "e7e5", "g1f3", "b8c6"],
                game_id="selfplay_0001",
            )
        ],
        arena_sessions=[
            _session_payload(
                white_agent="tracked_lap",
                black_agent="vice_v2",
                moves=["e2e4", "c7c5", "g1f3", "d7d6"],
                game_id="arena_0001",
            )
        ],
    )

    summary = master.reconcile_once()

    assert summary["lineages"]["lapv2_lineage"]["status"] == "waiting_for_feedback_submitted"
    assert len(controller.label_submissions) == 1
    label_config_path = Path(str(controller.label_submissions[0]["config_path"]))
    label_payload = json.loads(label_config_path.read_text(encoding="utf-8"))
    assert label_payload["complete_at_eof"] is True
    pgn_root = Path(str(label_payload["pgn_root"]))
    assert sorted(path.relative_to(pgn_root).as_posix() for path in pgn_root.rglob("*.pgn")) == [
        "arena/01_arena.pgn",
        "selfplay/01_selfplay.pgn",
    ]
    assert controller.phase10_submissions == []


def test_master_merges_feedback_corpus_before_next_generation(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    controller = _StubController()
    base_label_work_dir = tmp_path / "label_work"
    _write_completed_raw_snapshot(
        base_label_work_dir,
        train_rows=[
            {
                "sample_id": "base:train:1",
                "fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
                "source": "base",
                "selected_move_uci": "e1e2",
                "result": "1-0",
                "metadata": {},
            }
        ],
        verify_rows=[],
    )
    lineage = Phase10LineageSpec(
        name="lapv2_lineage",
        seed_phase10_config_path="python/configs/phase10_lapv2_stage2_native_arena_all_sources_v1.json",
        output_root=str(tmp_path / "lineage"),
        label_job_name="base_label",
        max_generations=2,
        on_accept="continue_training",
        promotion_thresholds=PromotionThresholds(
            min_verify_top1_accuracy=0.1,
            min_arena_score_rate=0.5,
        ),
    )
    db = _StubDB()
    master = OrchestratorMaster(
        db=db,
        controller=controller,
        repo_root=repo_root,
        spec=MasterSpec(
            name="master_feedback_merge_test",
            output_root=str(tmp_path / "master"),
            label_jobs=(
                LabelPgnCorpusJobSpec(
                    name="base_label",
                    pgn_root="/srv/schach/PGN_DATA/pgn",
                    work_dir=str(base_label_work_dir),
                    target_train_records=1,
                    target_verify_records=0,
                ),
            ),
            lineages=(lineage,),
        ),
        spec_path=tmp_path / "master.json",
    )
    paths = master._materialize_generation_configs(  # type: ignore[attr-defined]
        lineage=lineage,
        generation=1,
        parent_model=None,
        warm_start_checkpoint=None,
        label_work_dir=base_label_work_dir,
    )
    _install_completed_generation(
        db=db,
        master_name="master_feedback_merge_test",
        lineage_name="lapv2_lineage",
        paths=paths,
        tmp_path=tmp_path,
        arena_summary={
            "metadata": {"lapv1_agent_names": ["tracked_lap"]},
            "standings": {
                "tracked_lap": {"games": 2, "score": 1.5},
                "vice_v2": {"games": 2, "score": 0.5},
            },
            "matchups": [
                {
                    "white_agent": "tracked_lap",
                    "black_agent": "vice_v2",
                    "game_count": 2,
                    "white_score": 1.5,
                    "black_score": 0.5,
                    "result_counts": {"1-0": 1, "1/2-1/2": 1},
                    "termination_counts": {"checkmate": 1, "stalemate": 1},
                }
            ],
        },
    )
    feedback_work_dir = Path(lineage.output_root) / "feedback" / "generation_0001" / "label_work"
    _write_completed_raw_snapshot(
        feedback_work_dir,
        train_rows=[
            {
                "sample_id": "feedback:train:1",
                "fen": "4k3/8/8/8/8/8/8/3K4 w - - 0 1",
                "source": "feedback",
                "selected_move_uci": "d1d2",
                "result": "0-1",
                "metadata": {},
            }
        ],
        verify_rows=[],
    )

    summary = master.reconcile_once()

    assert summary["lineages"]["lapv2_lineage"]["status"] == "accepted_submitted_next"
    next_payload = json.loads(
        Path(str(controller.phase10_submissions[0]["config_path"])).read_text(encoding="utf-8")
    )
    merged_raw_dir = Path(str(next_payload["merged_raw_dir"]))
    assert merged_raw_dir != base_label_work_dir
    selection_summary = json.loads(
        (merged_raw_dir / "selection_summary.json").read_text(encoding="utf-8")
    )
    assert selection_summary["train_records"] == 2
    train_rows = [
        json.loads(line)
        for line in (merged_raw_dir / "train_raw.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["source"] for row in train_rows} == {"base", "feedback"}


def test_master_advances_stockfish_in_parallel_while_vice_stays_active(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    vice_spec_path = tmp_path / "vice_v2.json"
    write_selfplay_agent_spec(vice_spec_path, _uci_engine_spec("vice_v2"))
    seed_config_path = _write_seed_phase10_config(
        tmp_path,
        benchmark_agent_specs={"vice_v2": str(vice_spec_path)},
    )
    controller = _StubController()
    lineage = Phase10LineageSpec(
        name="lapv2_lineage",
        seed_phase10_config_path=str(seed_config_path),
        output_root=str(tmp_path / "lineage"),
        max_generations=2,
        on_accept="continue_training",
        promotion_thresholds=PromotionThresholds(
            min_verify_top1_accuracy=0.1,
            min_arena_score_rate=0.5,
        ),
    )
    db = _StubDB()
    master = OrchestratorMaster(
        db=db,
        controller=controller,
        repo_root=repo_root,
        spec=MasterSpec(
            name="master_stockfish_parallel_test",
            output_root=str(tmp_path / "master"),
            lineages=(lineage,),
        ),
        spec_path=tmp_path / "master.json",
    )
    paths = master._materialize_generation_configs(  # type: ignore[attr-defined]
        lineage=lineage,
        generation=1,
        parent_model=None,
        warm_start_checkpoint=None,
        label_work_dir=None,
    )
    _install_completed_generation(
        db=db,
        master_name="master_stockfish_parallel_test",
        lineage_name="lapv2_lineage",
        paths=paths,
        tmp_path=tmp_path,
        arena_summary={
            "metadata": {"lapv1_agent_names": ["tracked_lap"]},
            "standings": {
                "tracked_lap": {"games": 4, "score": 3.0},
                "vice_v2": {"games": 2, "score": 1.0},
                "stockfish18_skill_00": {"games": 2, "score": 0.0},
            },
            "matchups": [
                {
                    "white_agent": "tracked_lap",
                    "black_agent": "vice_v2",
                    "game_count": 2,
                    "white_score": 1.0,
                    "black_score": 1.0,
                    "result_counts": {"1-0": 1, "0-1": 1},
                    "termination_counts": {"checkmate": 2},
                },
                {
                    "white_agent": "tracked_lap",
                    "black_agent": "stockfish18_skill_00",
                    "game_count": 2,
                    "white_score": 2.0,
                    "black_score": 0.0,
                    "result_counts": {"1-0": 2},
                    "termination_counts": {"checkmate": 2},
                },
            ],
        },
    )

    summary = master.reconcile_once()

    assert summary["lineages"]["lapv2_lineage"]["status"] == "accepted_submitted_next"
    next_payload = json.loads(
        Path(str(controller.phase10_submissions[0]["config_path"])).read_text(encoding="utf-8")
    )
    assert set(dict(next_payload["benchmark_agent_specs"])) == {"vice_v2", "stockfish18_skill_01"}
    opponent_results = dict(summary["lineages"]["lapv2_lineage"]["opponent_results"])
    assert opponent_results["vice_v2"]["safely_beaten"] is False
    assert opponent_results["stockfish18_skill_00"]["safely_beaten"] is True
