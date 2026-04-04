from __future__ import annotations

import json

from train.datasets.planner_head import PlannerHeadExample, write_planner_head_artifact
from train.datasets.planner_quality import filter_planner_head_examples


def _planner_example(
    *,
    sample_id: str,
    candidate_count: int,
    teacher_root_value_cp: float,
    teacher_candidate_scores_cp: list[float] | None,
) -> PlannerHeadExample:
    candidate_action_indices = list(range(candidate_count))
    return PlannerHeadExample(
        sample_id=sample_id,
        split="train",
        fen="startpos",
        feature_vector=[0.0, 1.0],
        candidate_context_version=2,
        global_context_version=1,
        global_features=[0.0],
        candidate_action_indices=candidate_action_indices,
        candidate_features=[[0.0] for _ in candidate_action_indices],
        proposer_scores=[0.0 for _ in candidate_action_indices],
        transition_context_version=1,
        transition_features=[[0.0] for _ in candidate_action_indices],
        reply_peak_probabilities=[0.0 for _ in candidate_action_indices],
        pressures=[0.0 for _ in candidate_action_indices],
        uncertainties=[0.0 for _ in candidate_action_indices],
        curriculum_bucket_labels=["test"],
        curriculum_priority=0.0,
        teacher_top1_action_index=0,
        teacher_top1_candidate_index=0,
        teacher_policy=[1.0 / candidate_count for _ in candidate_action_indices],
        teacher_root_value_cp=teacher_root_value_cp,
        teacher_top1_minus_top2_cp=20.0,
        teacher_candidate_scores_cp=teacher_candidate_scores_cp,
    )


def test_filter_planner_head_examples_drops_expected_low_quality_rows() -> None:
    examples = [
        _planner_example(
            sample_id="keep",
            candidate_count=3,
            teacher_root_value_cp=80.0,
            teacher_candidate_scores_cp=[80.0, 10.0, -30.0],
        ),
        _planner_example(
            sample_id="ambiguous",
            candidate_count=3,
            teacher_root_value_cp=40.0,
            teacher_candidate_scores_cp=[10.0, 8.0, 7.0],
        ),
        _planner_example(
            sample_id="trivial",
            candidate_count=1,
            teacher_root_value_cp=40.0,
            teacher_candidate_scores_cp=[40.0],
        ),
        _planner_example(
            sample_id="extreme",
            candidate_count=3,
            teacher_root_value_cp=2501.0,
            teacher_candidate_scores_cp=[80.0, 20.0, -10.0],
        ),
    ]

    kept, summary = filter_planner_head_examples(examples)

    assert [example.sample_id for example in kept] == ["keep"]
    assert summary.total_examples == 4
    assert summary.kept_examples == 1
    assert summary.dropped_examples == 3
    assert summary.dropped_ambiguous_scores == 1
    assert summary.dropped_too_few_candidates == 1
    assert summary.dropped_nan_or_extreme_root_value == 1


def test_filter_preserves_planner_head_schema_exactly(tmp_path) -> None:
    example = _planner_example(
        sample_id="keep",
        candidate_count=3,
        teacher_root_value_cp=120.0,
        teacher_candidate_scores_cp=[120.0, 40.0, -20.0],
    )
    dropped = _planner_example(
        sample_id="drop",
        candidate_count=3,
        teacher_root_value_cp=10.0,
        teacher_candidate_scores_cp=[10.0, 9.0, 8.0],
    )

    kept, _ = filter_planner_head_examples([example, dropped])
    output_path = tmp_path / "planner_head_train.jsonl"
    write_planner_head_artifact(output_path, kept)

    payload = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert set(payload.keys()) == set(example.to_dict().keys())
    assert PlannerHeadExample.from_dict(payload) == example
