"""Tests for latent planner-head workflow materialization."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from train.datasets.contracts import candidate_context_feature_dim, project_candidate_context_to_v1
from train.datasets.planner_head import (
    PlannerHeadExample,
    load_planner_head_examples,
    materialize_planner_latent_features,
    write_planner_head_artifact,
)


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "materialize_planner_latent_workflow.py"
)
_SPEC = importlib.util.spec_from_file_location("materialize_planner_latent_workflow", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
main = _MODULE.main


def test_project_candidate_context_to_v1_derives_minor_major_capture() -> None:
    row_v2 = [0.0] * candidate_context_feature_dim(2)
    row_v2[0] = 1.0
    row_v2[5] = 1.0
    row_v2[9] = 1.0
    row_v2[15] = 1.0
    row_v2[19] = 1.0

    projected = project_candidate_context_to_v1(row_v2, version=2)

    assert len(projected) == candidate_context_feature_dim(1)
    assert projected[0] == 1.0
    assert projected[5] == 1.0
    assert projected[9] == 1.0
    assert projected[15] == 1.0
    assert projected[17] == 1.0


def test_materialize_planner_latent_features_uses_projected_action_features() -> None:
    example = _planner_example()

    calls: list[dict[str, object]] = []

    def _predictor(_model: object, *, feature_vector: list[float], action_index: int, action_features: list[float], transition_features: list[float]) -> list[float]:
        calls.append(
            {
                "feature_vector": feature_vector,
                "action_index": action_index,
                "action_features": action_features,
                "transition_features": transition_features,
            }
        )
        return [float(action_index), float(sum(action_features))]

    materialized = materialize_planner_latent_features(
        [example],
        dynamics_model=object(),
        predictor=_predictor,
    )

    assert len(materialized) == 1
    rendered = materialized[0]
    assert rendered.latent_state_version == 1
    assert rendered.latent_features == [
        [11.0, 4.0],
        [17.0, 5.0],
    ]
    assert calls[0]["action_features"] == project_candidate_context_to_v1(
        example.candidate_features[0],
        version=2,
    )
    assert calls[1]["transition_features"] == example.transition_features[1]


def test_materialize_planner_latent_workflow_main_writes_augmented_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_root = tmp_path / "input_workflow"
    output_root = tmp_path / "output_workflow"
    train_dir = input_root / "tier_train_v1"
    validation_dir = input_root / "tier_validation_v1"
    verify_dir = input_root / "tier_verify_v1"
    train_dir.mkdir(parents=True)
    validation_dir.mkdir(parents=True)
    verify_dir.mkdir(parents=True)
    train_path = train_dir / "planner_head_train.jsonl"
    validation_path = validation_dir / "planner_head_validation.jsonl"
    verify_path = verify_dir / "planner_head_test.jsonl"
    write_planner_head_artifact(train_path, [_planner_example()])
    write_planner_head_artifact(validation_path, [_planner_example()])
    write_planner_head_artifact(verify_path, [_planner_example()])

    summary_path = input_root / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "tiers": {
                    "tier": {
                        "train": {
                            "dataset_dir": str(tmp_path / "dataset"),
                            "workflow_dir": str(tmp_path / "phase7"),
                            "planner_head_path": str(train_path),
                        },
                        "validation": {
                            "dataset_dir": str(tmp_path / "dataset"),
                            "workflow_dir": str(tmp_path / "phase7"),
                            "planner_head_path": str(validation_path),
                        },
                        "verify": {
                            "dataset_dir": str(tmp_path / "dataset"),
                            "workflow_dir": str(tmp_path / "phase7"),
                            "planner_head_path": str(verify_path),
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    class _Model:
        pass

    monkeypatch.setattr(_MODULE, "load_dynamics_checkpoint", lambda _path: (_Model(), object()))
    monkeypatch.setattr(
        _MODULE,
        "materialize_planner_latent_features",
        lambda examples, dynamics_model: [
            PlannerHeadExample(
                **{
                    **example.to_dict(),
                    "latent_state_version": 1,
                    "latent_features": [[1.0, 2.0] for _ in example.candidate_action_indices],
                }
            )
            for example in examples
        ],
    )

    exit_code = main(
        [
            "--workflow-summary",
            str(summary_path),
            "--dynamics-checkpoint",
            str(tmp_path / "dynamics.pt"),
            "--output-root",
            str(output_root),
            "--tier",
            "tier",
        ]
    )

    assert exit_code == 0
    rendered = load_planner_head_examples(output_root / "tier_train_v1" / "planner_head_train.jsonl")
    assert rendered[0].latent_state_version == 1
    assert rendered[0].latent_features == [[1.0, 2.0], [1.0, 2.0]]


def _planner_example() -> PlannerHeadExample:
    candidate_dim_v2 = candidate_context_feature_dim(2)
    transition_dim = candidate_dim_v2 + 10

    candidate_a = [0.0] * candidate_dim_v2
    candidate_a[0] = 1.0
    candidate_a[5] = 1.0
    candidate_a[9] = 1.0
    candidate_a[15] = 1.0

    candidate_b = [0.0] * candidate_dim_v2
    candidate_b[1] = 1.0
    candidate_b[6] = 1.0
    candidate_b[10] = 1.0
    candidate_b[15] = 1.0
    candidate_b[18] = 1.0

    return PlannerHeadExample(
        sample_id="sample-1",
        split="train",
        fen="4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        feature_vector=[0.1, 0.2, 0.3],
        candidate_context_version=2,
        global_context_version=1,
        global_features=[0.0] * 9,
        candidate_action_indices=[11, 17],
        candidate_features=[candidate_a, candidate_b],
        proposer_scores=[0.6, 0.4],
        transition_context_version=1,
        transition_features=[
            [0.5] * transition_dim,
            [1.5] * transition_dim,
        ],
        reply_peak_probabilities=[0.2, 0.3],
        pressures=[0.1, 0.4],
        uncertainties=[0.8, 0.7],
        curriculum_bucket_labels=["teacher_top1"],
        curriculum_priority=1.0,
        teacher_top1_action_index=11,
        teacher_top1_candidate_index=0,
        teacher_policy=[0.7, 0.3],
        teacher_root_value_cp=25.0,
        teacher_top1_minus_top2_cp=15.0,
    )
