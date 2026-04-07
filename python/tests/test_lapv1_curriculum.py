from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from train.datasets.artifacts import (
    SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE,
    pack_position_features,
)
from train.datasets.lapv1_training import (
    lapv1_training_example_from_planner_head,
)
from train.datasets.planner_head import PlannerHeadExample
from train.datasets.schema import PositionEncoding
from train.eval.lapv1_curriculum import lapv1_hardness_score


def test_lapv1_hardness_score_prefers_sharp_close_choices() -> None:
    hard_example = lapv1_training_example_from_planner_head(
        _planner_example("hard", teacher_index=0, teacher_cp=5.0, teacher_gap=2.0)
    )
    easy_example = lapv1_training_example_from_planner_head(
        _planner_example("easy", teacher_index=0, teacher_cp=200.0, teacher_gap=120.0)
    )

    assert lapv1_hardness_score(hard_example) > lapv1_hardness_score(easy_example)


def test_build_lapv1_hard_positions_dataset_selects_highest_scores(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "lapv1_train.jsonl"
    output_path = tmp_path / "lapv1_train_hard.jsonl"
    examples = [
        lapv1_training_example_from_planner_head(
            _planner_example("easy", teacher_index=0, teacher_cp=220.0, teacher_gap=120.0)
        ),
        lapv1_training_example_from_planner_head(
            _planner_example("hard-a", teacher_index=1, teacher_cp=0.0, teacher_gap=2.0)
        ),
        lapv1_training_example_from_planner_head(
            _planner_example("hard-b", teacher_index=0, teacher_cp=8.0, teacher_gap=4.0)
        ),
    ]
    source_path.write_text(
        "".join(json.dumps(example.to_dict(), sort_keys=True) + "\n" for example in examples),
        encoding="utf-8",
    )

    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_lapv1_hard_positions_dataset.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--input-path",
            str(source_path),
            "--output-path",
            str(output_path),
            "--max-examples",
            "2",
            "--log-every",
            "1",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    selected_lines = [
        json.loads(line)["sample_id"]
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert selected_lines == ["hard-a", "hard-b"]


def _planner_example(
    sample_id: str,
    *,
    teacher_index: int,
    teacher_cp: float,
    teacher_gap: float,
) -> PlannerHeadExample:
    feature_vector = pack_position_features(
        PositionEncoding(
            piece_tokens=[[4, 0, 5], [60, 1, 5], [0, 0, 3], [63, 1, 3]],
            square_tokens=[[square_index, 0] for square_index in range(64)],
            rule_token=[0, 0, -1, 0, 1, 0],
        )
    )
    candidate_features = [
        [1.0] + [0.0] * 34,
        [0.0, 1.0] + [0.0] * 33,
    ]
    teacher_policy = [0.0, 0.0]
    teacher_policy[teacher_index] = 1.0
    return PlannerHeadExample(
        sample_id=sample_id,
        split="train",
        fen="4k2r/8/8/8/8/8/8/R3K3 w - - 0 1",
        feature_vector=feature_vector,
        candidate_context_version=2,
        global_context_version=1,
        global_features=[0.0] * SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE,
        candidate_action_indices=[1, 2],
        candidate_features=candidate_features,
        proposer_scores=[0.1, 0.2],
        transition_context_version=1,
        transition_features=[[0.0] * 45, [0.0] * 45],
        reply_peak_probabilities=[0.5, 0.4],
        pressures=[0.2, 0.3],
        uncertainties=[0.3, 0.4],
        curriculum_bucket_labels=["lapv1_curriculum_test"],
        curriculum_priority=1.0,
        teacher_top1_action_index=teacher_index + 1,
        teacher_top1_candidate_index=teacher_index,
        teacher_policy=teacher_policy,
        teacher_root_value_cp=teacher_cp,
        teacher_top1_minus_top2_cp=teacher_gap,
        teacher_candidate_scores_cp=[teacher_cp, teacher_cp - teacher_gap],
        teacher_candidate_score_delta_targets_cp=[0.0, -teacher_gap],
        teacher_rank_bucket_version=1,
        teacher_candidate_rank_bucket_targets=[0, 1],
        latent_state_version=None,
        latent_features=None,
    )
