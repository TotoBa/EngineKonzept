"""Offline analysis helpers for MoE planner routing and expert specialization."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import chess

from train.config import PlannerTrainConfig
from train.datasets.planner_head import load_planner_head_examples
from train.models.planner import PLANNER_MODEL_NAME
from train.models.proposer import torch_is_available
from train.trainers.planner import _build_planner_model, _collate_planner_batch

if torch_is_available():
    import torch
else:  # pragma: no cover - exercised when torch is absent
    torch = None


PHASE_LABELS = ("opening", "middlegame", "endgame")
TACTICAL_LEVEL_LABELS = ("quiet", "forcing", "tactical")
DIFFICULTY_LABELS = ("easy", "medium", "hard", "unknown")


def load_moe_planner_checkpoint(checkpoint_path: Path) -> tuple[Any, PlannerTrainConfig]:
    """Load a trained MoE planner checkpoint for offline analysis."""
    if torch is None:
        raise RuntimeError("PyTorch is required for MoE checkpoint analysis.")
    payload = torch.load(checkpoint_path, map_location="cpu")
    if payload.get("model_name") != PLANNER_MODEL_NAME:
        raise ValueError(
            f"{checkpoint_path}: unsupported planner model name {payload.get('model_name')!r}"
        )
    config = PlannerTrainConfig.from_dict(dict(payload["training_config"]))
    if config.model.architecture != "moe_v1":
        raise ValueError(
            f"{checkpoint_path}: expected a moe_v1 planner checkpoint, got {config.model.architecture!r}"
        )
    model = _build_planner_model(config)
    model.load_state_dict(dict(payload["model_state_dict"]))
    model.eval()
    return model, config


def analyze_moe_expert_specialization(
    *,
    checkpoint_path: Path,
    dataset_path: Path,
    output_path: Path | None = None,
    max_examples: int | None = None,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Analyze MoE routing and expert specialization over planner-head examples."""
    if torch is None:
        raise RuntimeError("PyTorch is required for MoE checkpoint analysis.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model, config = load_moe_planner_checkpoint(checkpoint_path)
    examples = load_planner_head_examples(dataset_path)
    if max_examples is not None:
        examples = examples[:max_examples]
    if not examples:
        raise ValueError("dataset_path did not contain any planner-head examples")

    num_experts = int(config.moe.num_experts) if config.moe is not None else 0
    phase_stats = _init_bucket_stats(PHASE_LABELS, num_experts=num_experts)
    tactical_stats = _init_bucket_stats(TACTICAL_LEVEL_LABELS, num_experts=num_experts)
    difficulty_stats = _init_bucket_stats(DIFFICULTY_LABELS, num_experts=num_experts)
    router_entropy_values: list[float] = []
    complexity_scores: list[float] = []
    example_records: list[dict[str, Any]] = []
    agreement_examples = 0
    full_ranking_agreements = 0
    top1_agreements = 0

    with torch.inference_mode():
        for start_index in range(0, len(examples), batch_size):
            batch_examples = examples[start_index : start_index + batch_size]
            batch = _collate_planner_batch(
                batch_examples,
                latent_feature_dim=config.model.latent_feature_dim,
            )
            outputs = model(
                batch["root_features"],
                batch["global_features"],
                batch["candidate_action_indices"],
                batch["candidate_features"],
                batch["proposer_scores"],
                batch["transition_features"],
                batch["latent_features"],
                batch["reply_peak_probabilities"],
                batch["pressures"],
                batch["uncertainties"],
                batch["candidate_mask"],
            )
            for row_index, example in enumerate(batch_examples):
                valid_count = len(example.candidate_action_indices)
                sparse_weights = outputs["sparse_router_weights"][row_index].detach().cpu()
                dense_weights = outputs["router_weights"][row_index].detach().cpu()
                router_entropy = float(outputs["router_entropy"].detach().cpu().item())
                router_entropy_values.append(router_entropy)
                complexity_score = None
                if outputs.get("complexity_score") is not None:
                    complexity_score = float(
                        outputs["complexity_score"][row_index].detach().cpu().item()
                    )
                    complexity_scores.append(complexity_score)
                selected_experts = [
                    int(index)
                    for index, value in enumerate(sparse_weights.tolist())
                    if value > 0.0
                ]
                phase_label = _classify_phase(example.fen)
                tactical_level = _classify_tactical_level(example.fen)
                difficulty_bucket = _classify_difficulty(example.teacher_top1_minus_top2_cp)

                _update_bucket_stats(phase_stats, phase_label, sparse_weights)
                _update_bucket_stats(tactical_stats, tactical_level, sparse_weights)
                _update_bucket_stats(difficulty_stats, difficulty_bucket, sparse_weights)

                top1_agreement = None
                full_ranking_agreement = None
                expert_candidate_logits = outputs["expert_candidate_logits"][row_index, :, :valid_count]
                top_two_experts = [
                    int(index)
                    for index in torch.argsort(sparse_weights, descending=True).tolist()
                    if float(sparse_weights[index].item()) > 0.0
                ][:2]
                if len(top_two_experts) == 2:
                    agreement_examples += 1
                    first_ranking = torch.argsort(
                        expert_candidate_logits[top_two_experts[0]],
                        descending=True,
                    )
                    second_ranking = torch.argsort(
                        expert_candidate_logits[top_two_experts[1]],
                        descending=True,
                    )
                    top1_agreement = bool(first_ranking[0].item() == second_ranking[0].item())
                    full_ranking_agreement = bool(torch.equal(first_ranking, second_ranking))
                    top1_agreements += int(top1_agreement)
                    full_ranking_agreements += int(full_ranking_agreement)

                example_records.append(
                    {
                        "sample_id": example.sample_id,
                        "fen": example.fen,
                        "phase": phase_label,
                        "tactical_level": tactical_level,
                        "difficulty_bucket": difficulty_bucket,
                        "difficulty_gap_cp": (
                            None
                            if example.teacher_top1_minus_top2_cp is None
                            else float(example.teacher_top1_minus_top2_cp)
                        ),
                        "selected_experts": selected_experts,
                        "router_weights": [float(value) for value in dense_weights.tolist()],
                        "sparse_router_weights": [float(value) for value in sparse_weights.tolist()],
                        "router_entropy": router_entropy,
                        "complexity_score": complexity_score,
                        "top1_agreement": top1_agreement,
                        "full_ranking_agreement": full_ranking_agreement,
                    }
                )

    report = {
        "checkpoint_path": str(checkpoint_path),
        "dataset_path": str(dataset_path),
        "model_architecture": config.model.architecture,
        "num_experts": num_experts,
        "example_count": len(example_records),
        "phase_labels": list(PHASE_LABELS),
        "tactical_level_labels": list(TACTICAL_LEVEL_LABELS),
        "difficulty_labels": list(DIFFICULTY_LABELS),
        "expert_activation_by_phase": _finalize_bucket_stats(phase_stats),
        "expert_activation_by_tactical_level": _finalize_bucket_stats(tactical_stats),
        "expert_activation_by_difficulty": _finalize_bucket_stats(difficulty_stats),
        "router_entropy_distribution": _summarize_distribution(router_entropy_values),
        "complexity_score_distribution": _summarize_distribution(complexity_scores),
        "expert_agreement_examples": agreement_examples,
        "expert_agreement_rate": _ratio(full_ranking_agreements, agreement_examples),
        "expert_top1_agreement_rate": _ratio(top1_agreements, agreement_examples),
        "example_records": example_records,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def visualize_moe_routing_report(
    *,
    report: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    """Render the MoE specialization report as plots on disk."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    heatmap_path = output_dir / "expert_phase_heatmap.png"
    phase_rows = [
        report["expert_activation_by_phase"][phase_label]["mean_activation_weights"]
        for phase_label in report["phase_labels"]
    ]
    figure, axis = plt.subplots(figsize=(8, 3.5))
    image = axis.imshow(phase_rows, aspect="auto", cmap="viridis")
    axis.set_title("Expert Activation by Game Phase")
    axis.set_xlabel("Expert")
    axis.set_ylabel("Phase")
    axis.set_xticks(range(report["num_experts"]))
    axis.set_yticks(range(len(report["phase_labels"])))
    axis.set_yticklabels(report["phase_labels"])
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(heatmap_path)
    plt.close(figure)
    output_paths.append(heatmap_path)

    entropy_path = output_dir / "router_entropy_histogram.png"
    entropy_values = report["router_entropy_distribution"]["values"]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.hist(entropy_values, bins=min(20, max(5, len(entropy_values))), color="#3a6ea5")
    axis.set_title("Router Entropy Distribution")
    axis.set_xlabel("Router Entropy")
    axis.set_ylabel("Examples")
    figure.tight_layout()
    figure.savefig(entropy_path)
    plt.close(figure)
    output_paths.append(entropy_path)

    scatter_path = output_dir / "complexity_vs_difficulty.png"
    figure, axis = plt.subplots(figsize=(6, 4))
    scatter_examples = [
        record
        for record in report["example_records"]
        if record["complexity_score"] is not None and record["difficulty_gap_cp"] is not None
    ]
    if scatter_examples:
        axis.scatter(
            [float(record["complexity_score"]) for record in scatter_examples],
            [float(record["difficulty_gap_cp"]) for record in scatter_examples],
            alpha=0.7,
            color="#b94e48",
        )
    else:
        axis.text(0.5, 0.5, "No complexity scores available", ha="center", va="center")
    axis.set_title("Complexity Score vs Teacher Gap")
    axis.set_xlabel("Complexity Score")
    axis.set_ylabel("Teacher Top1-Top2 Gap (cp, lower = harder)")
    figure.tight_layout()
    figure.savefig(scatter_path)
    plt.close(figure)
    output_paths.append(scatter_path)

    return output_paths


def _init_bucket_stats(labels: Sequence[str], *, num_experts: int) -> dict[str, dict[str, Any]]:
    return {
        label: {
            "examples": 0,
            "activation_totals": [0.0] * num_experts,
            "selected_totals": [0] * num_experts,
        }
        for label in labels
    }


def _update_bucket_stats(
    bucket_stats: dict[str, dict[str, Any]],
    label: str,
    sparse_router_weights: Any,
) -> None:
    stats = bucket_stats[label]
    stats["examples"] += 1
    for expert_index, value in enumerate(sparse_router_weights.tolist()):
        stats["activation_totals"][expert_index] += float(value)
        if float(value) > 0.0:
            stats["selected_totals"][expert_index] += 1


def _finalize_bucket_stats(bucket_stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    finalized: dict[str, dict[str, Any]] = {}
    for label, stats in bucket_stats.items():
        example_count = int(stats["examples"])
        finalized[label] = {
            "examples": example_count,
            "mean_activation_weights": [
                _ratio(float(total), example_count) for total in stats["activation_totals"]
            ],
            "selection_rates": [
                _ratio(float(total), example_count) for total in stats["selected_totals"]
            ],
        }
    return finalized


def _summarize_distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "values": [],
        }
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "values": [float(value) for value in values],
    }


def _classify_phase(fen: str) -> str:
    board = chess.Board(fen)
    non_king_piece_count = sum(
        1 for piece in board.piece_map().values() if piece.piece_type != chess.KING
    )
    if non_king_piece_count >= 24:
        return "opening"
    if non_king_piece_count >= 12:
        return "middlegame"
    return "endgame"


def _classify_tactical_level(fen: str) -> str:
    board = chess.Board(fen)
    forcing_moves = 0
    for move in board.legal_moves:
        if board.is_capture(move) or board.gives_check(move):
            forcing_moves += 1
    if forcing_moves == 0:
        return "quiet"
    if forcing_moves <= 3:
        return "forcing"
    return "tactical"


def _classify_difficulty(gap_cp: float | None) -> str:
    if gap_cp is None or math.isnan(gap_cp):
        return "unknown"
    if gap_cp >= 100.0:
        return "easy"
    if gap_cp < 20.0:
        return "hard"
    return "medium"


def _ratio(numerator: float, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)
