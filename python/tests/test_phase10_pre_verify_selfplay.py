from __future__ import annotations

from pathlib import Path

from train.eval.agent_spec import SelfplayAgentSpec, write_selfplay_agent_spec
from train.eval.phase10_campaign import (
    Phase10Lapv1AgentVariantSpec,
    Phase10Lapv1ArenaCampaignSpec,
    materialize_resolved_lapv1_agent_specs,
    select_pre_verify_lapv1_agent,
)


def test_select_pre_verify_lapv1_agent_prefers_primary_runtime_candidate(tmp_path: Path) -> None:
    base_spec_path = tmp_path / "lapv2_base.json"
    write_selfplay_agent_spec(
        base_spec_path,
        SelfplayAgentSpec(
            name="lapv2_stage2_native_all_sources_v1",
            agent_kind="lapv1",
            lapv1_checkpoint="models/lapv2/stage2_native_all_sources_v1/bundle/checkpoint.pt",
            state_context_version=1,
            deliberation_max_inner_steps=4,
            deliberation_q_threshold=0.3,
        ),
    )
    spec = Phase10Lapv1ArenaCampaignSpec(
        name="phase10_pre_verify_select_test",
        output_root=str(tmp_path / "campaign"),
        merged_raw_dir=str(tmp_path / "raw"),
        train_dataset_dir=str(tmp_path / "datasets" / "train"),
        verify_dataset_dir=str(tmp_path / "datasets" / "verify"),
        phase5_source_name="test_source",
        phase5_seed="seed",
        workflow_output_root=str(tmp_path / "workflow"),
        proposer_checkpoint="models/proposer/checkpoint.pt",
        lapv1_config_path="python/configs/test_lapv2.json",
        lapv1_agent_spec_path=str(base_spec_path),
        lapv1_agent_variants=(
            Phase10Lapv1AgentVariantSpec(
                name="lapv2_inner0",
                deliberation_max_inner_steps=0,
                metadata={"variant_role": "root_only_control"},
            ),
            Phase10Lapv1AgentVariantSpec(
                name="lapv2_inner1",
                deliberation_max_inner_steps=1,
                deliberation_q_threshold=0.3,
                metadata={"variant_role": "primary_runtime_candidate"},
            ),
        ),
        lapv1_verify_output_path=str(tmp_path / "workflow" / "verify.jsonl"),
        reference_arena_summary_path=str(tmp_path / "refs" / "arena_summary.json"),
        initial_fen_suite_path=str(tmp_path / "openings.json"),
    )
    resolved = materialize_resolved_lapv1_agent_specs(
        spec=spec,
        tracked_lapv1_agent_path=base_spec_path,
        output_root=tmp_path / "campaign" / "pre_verify_selfplay",
    )

    agent_name, agent_path = select_pre_verify_lapv1_agent(
        spec,
        resolved_agent_paths=resolved,
        repo_root=tmp_path,
    )

    assert agent_name == "lapv2_inner1"
    assert agent_path == resolved["lapv2_inner1"]


def test_select_pre_verify_lapv1_agent_honors_explicit_variant_override(tmp_path: Path) -> None:
    base_spec_path = tmp_path / "lapv2_base.json"
    write_selfplay_agent_spec(
        base_spec_path,
        SelfplayAgentSpec(
            name="lapv2_stage2_native_all_sources_v1",
            agent_kind="lapv1",
            lapv1_checkpoint="models/lapv2/stage2_native_all_sources_v1/bundle/checkpoint.pt",
            state_context_version=1,
        ),
    )
    spec = Phase10Lapv1ArenaCampaignSpec(
        name="phase10_pre_verify_select_override_test",
        output_root=str(tmp_path / "campaign"),
        merged_raw_dir=str(tmp_path / "raw"),
        train_dataset_dir=str(tmp_path / "datasets" / "train"),
        verify_dataset_dir=str(tmp_path / "datasets" / "verify"),
        phase5_source_name="test_source",
        phase5_seed="seed",
        workflow_output_root=str(tmp_path / "workflow"),
        proposer_checkpoint="models/proposer/checkpoint.pt",
        lapv1_config_path="python/configs/test_lapv2.json",
        lapv1_agent_spec_path=str(base_spec_path),
        lapv1_agent_variants=(
            Phase10Lapv1AgentVariantSpec(name="lapv2_inner0", deliberation_max_inner_steps=0),
            Phase10Lapv1AgentVariantSpec(name="lapv2_inner1", deliberation_max_inner_steps=1),
        ),
        pre_verify_selfplay_agent_variant_name="lapv2_inner0",
        lapv1_verify_output_path=str(tmp_path / "workflow" / "verify.jsonl"),
        reference_arena_summary_path=str(tmp_path / "refs" / "arena_summary.json"),
        initial_fen_suite_path=str(tmp_path / "openings.json"),
    )
    resolved = materialize_resolved_lapv1_agent_specs(
        spec=spec,
        tracked_lapv1_agent_path=base_spec_path,
        output_root=tmp_path / "campaign" / "pre_verify_selfplay",
    )

    agent_name, agent_path = select_pre_verify_lapv1_agent(
        spec,
        resolved_agent_paths=resolved,
        repo_root=tmp_path,
    )

    assert agent_name == "lapv2_inner0"
    assert agent_path == resolved["lapv2_inner0"]
