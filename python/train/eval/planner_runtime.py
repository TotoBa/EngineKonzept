"""Runtime-style bounded planner helpers for exact-candidate scoring loops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from train.datasets import (
    build_selected_move_action_features,
    build_symbolic_proposer_example,
    build_transition_context_features,
    dataset_example_from_oracle_payload,
    move_uci_for_action,
)
from train.datasets.oracle import label_records_with_oracle
from train.datasets.schema import DatasetExample, RawPositionRecord
from train.eval.dynamics import load_dynamics_checkpoint, predict_dynamics_latent
from train.eval.opponent import load_opponent_head_checkpoint, score_opponent_candidates
from train.eval.symbolic_proposer import (
    load_symbolic_proposer_checkpoint,
    score_symbolic_candidates,
)
from train.models.planner import PlannerHeadModel
from train.models.proposer import torch_is_available

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised when torch is absent
    torch = None

if TYPE_CHECKING:
    from train.config import PlannerTrainConfig, ProposerTrainConfig
    from train.config import DynamicsTrainConfig, OpponentTrainConfig


_CAPTURE_FEATURE_INDEX = 0
_PROMOTION_FEATURE_INDEX = 1
_EN_PASSANT_FEATURE_INDEX = 3
_GIVES_CHECK_FEATURE_INDEX = 4


@dataclass(frozen=True)
class PlannerRootDecision:
    """One exact move choice from a bounded planner-style runtime pass."""

    move_uci: str
    action_index: int
    next_fen: str
    selector_name: str
    legal_candidate_count: int
    considered_candidate_count: int
    proposer_score: float
    planner_score: float
    reply_peak_probability: float
    pressure: float
    uncertainty: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "move_uci": self.move_uci,
            "action_index": self.action_index,
            "next_fen": self.next_fen,
            "selector_name": self.selector_name,
            "legal_candidate_count": self.legal_candidate_count,
            "considered_candidate_count": self.considered_candidate_count,
            "proposer_score": round(self.proposer_score, 6),
            "planner_score": round(self.planner_score, 6),
            "reply_peak_probability": round(self.reply_peak_probability, 6),
            "pressure": round(self.pressure, 6),
            "uncertainty": round(self.uncertainty, 6),
        }


@dataclass(frozen=True)
class LoadedPlannerRuntime:
    """Loaded exact-candidate runtime stack for selfplay and planner probes."""

    name: str
    proposer_model: Any
    proposer_config: "ProposerTrainConfig"
    planner_model: Any | None
    planner_config: "PlannerTrainConfig | None"
    opponent_model: Any | None
    opponent_config: "OpponentTrainConfig | None"
    dynamics_model: Any | None
    dynamics_config: "DynamicsTrainConfig | None"
    opponent_mode: str
    root_top_k: int
    repo_root: Path

    def select_move(self, example: DatasetExample) -> PlannerRootDecision:
        """Select one exact legal move for the current position."""
        if not example.legal_moves:
            raise ValueError(f"{example.sample_id}: cannot select a move from an empty legal set")

        root_symbolic = build_symbolic_proposer_example(
            example,
            candidate_context_version=2,
            global_context_version=1,
        )
        root_scores, _root_policy = score_symbolic_candidates(
            self.proposer_model,
            feature_vector=root_symbolic.feature_vector,
            candidate_action_indices=root_symbolic.candidate_action_indices,
            candidate_features=root_symbolic.candidate_features,
            global_features=root_symbolic.global_features,
            candidate_context_version=root_symbolic.candidate_context_version,
        )
        ranked_indices = sorted(
            range(len(root_symbolic.candidate_action_indices)),
            key=lambda index: (-root_scores[index], index),
        )
        if self.planner_model is None:
            selected_index = ranked_indices[0]
            action_index = int(root_symbolic.candidate_action_indices[selected_index])
            move_uci = move_uci_for_action(example, action_index)
            selected_example = _label_selected_move(
                example,
                move_uci=move_uci,
                repo_root=self.repo_root,
            )
            assert selected_example.next_fen is not None
            return PlannerRootDecision(
                move_uci=move_uci,
                action_index=action_index,
                next_fen=str(selected_example.next_fen),
                selector_name=self.name,
                legal_candidate_count=len(root_symbolic.candidate_action_indices),
                considered_candidate_count=1,
                proposer_score=float(root_scores[selected_index]),
                planner_score=float(root_scores[selected_index]),
                reply_peak_probability=0.0,
                pressure=0.0,
                uncertainty=0.0,
            )

        considered_indices = ranked_indices[: min(self.root_top_k, len(ranked_indices))]
        candidate_rows = _build_candidate_rows(
            example,
            root_symbolic=root_symbolic,
            root_scores=root_scores,
            considered_indices=considered_indices,
            proposer_model=self.proposer_model,
            opponent_model=self.opponent_model,
            opponent_mode=self.opponent_mode,
            dynamics_model=self.dynamics_model,
            planner_config=self.planner_config,
            repo_root=self.repo_root,
        )
        planner_scores = _score_planner_rows(
            planner_model=self.planner_model,
            planner_config=self.planner_config,
            root_symbolic=root_symbolic,
            candidate_rows=candidate_rows,
        )
        selected_row = max(
            zip(candidate_rows, planner_scores, strict=True),
            key=lambda item: (item[1], item[0].action_index),
        )
        row, planner_score = selected_row
        return PlannerRootDecision(
            move_uci=row.move_uci,
            action_index=row.action_index,
            next_fen=row.next_fen,
            selector_name=self.name,
            legal_candidate_count=len(root_symbolic.candidate_action_indices),
            considered_candidate_count=len(candidate_rows),
            proposer_score=row.proposer_score,
            planner_score=float(planner_score),
            reply_peak_probability=row.reply_peak_probability,
            pressure=row.pressure,
            uncertainty=row.uncertainty,
        )


@dataclass(frozen=True)
class _PlannerCandidateRow:
    move_uci: str
    action_index: int
    next_fen: str
    candidate_features: list[float]
    proposer_score: float
    transition_features: list[float]
    latent_features: list[float]
    reply_peak_probability: float
    pressure: float
    uncertainty: float


def load_planner_head_checkpoint(
    checkpoint_path: Path,
) -> tuple[PlannerHeadModel, "PlannerTrainConfig"]:
    """Load a trained planner-head checkpoint for bounded runtime selection."""
    if torch is None or not torch_is_available():  # pragma: no cover - torch absent
        raise RuntimeError(
            "PyTorch is required for planner runtime evaluation. Install the 'train' extra or torch."
        )
    from train.config import PlannerTrainConfig

    payload = torch.load(checkpoint_path, map_location="cpu")
    config = PlannerTrainConfig.from_dict(dict(payload["training_config"]))
    model = PlannerHeadModel(
        architecture=config.model.architecture,
        hidden_dim=config.model.hidden_dim,
        hidden_layers=config.model.hidden_layers,
        action_embedding_dim=config.model.action_embedding_dim,
        latent_feature_dim=config.model.latent_feature_dim,
        deliberation_steps=config.model.deliberation_steps,
        memory_slots=config.model.memory_slots,
        dropout=config.model.dropout,
        enable_candidate_rank_head=config.optimization.teacher_rank_loss_weight > 0.0,
    )
    model.load_state_dict(dict(payload["model_state_dict"]))
    model.eval()
    return model, config


def build_planner_runtime(
    *,
    name: str,
    proposer_checkpoint: Path,
    repo_root: Path,
    planner_checkpoint: Path | None = None,
    opponent_checkpoint: Path | None = None,
    dynamics_checkpoint: Path | None = None,
    opponent_mode: str = "symbolic",
    root_top_k: int = 4,
) -> LoadedPlannerRuntime:
    """Load an exact-candidate runtime stack for bounded planning or proposer-only play."""
    if opponent_mode not in {"none", "symbolic", "learned"}:
        raise ValueError("opponent_mode must be 'none', 'symbolic', or 'learned'")
    if root_top_k <= 0:
        raise ValueError("root_top_k must be positive")

    proposer_model, proposer_config = load_symbolic_proposer_checkpoint(proposer_checkpoint)
    planner_model = None
    planner_config = None
    opponent_model = None
    opponent_config = None
    dynamics_model = None
    dynamics_config = None

    if planner_checkpoint is not None:
        planner_model, planner_config = load_planner_head_checkpoint(planner_checkpoint)
    elif opponent_checkpoint is not None or dynamics_checkpoint is not None:
        raise ValueError(
            "opponent_checkpoint and dynamics_checkpoint require planner_checkpoint"
        )

    if opponent_mode == "learned":
        if opponent_checkpoint is None:
            raise ValueError("opponent_checkpoint is required when opponent_mode='learned'")
        opponent_model, opponent_config = load_opponent_head_checkpoint(opponent_checkpoint)
    elif opponent_checkpoint is not None:
        opponent_model, opponent_config = load_opponent_head_checkpoint(opponent_checkpoint)

    if dynamics_checkpoint is not None:
        dynamics_model, dynamics_config = load_dynamics_checkpoint(dynamics_checkpoint)

    if planner_config is not None and planner_config.model.latent_feature_dim > 0 and dynamics_model is None:
        raise ValueError(
            "planner checkpoint expects latent_features but no dynamics checkpoint was provided"
        )

    return LoadedPlannerRuntime(
        name=name,
        proposer_model=proposer_model,
        proposer_config=proposer_config,
        planner_model=planner_model,
        planner_config=planner_config,
        opponent_model=opponent_model,
        opponent_config=opponent_config,
        dynamics_model=dynamics_model,
        dynamics_config=dynamics_config,
        opponent_mode=opponent_mode,
        root_top_k=root_top_k,
        repo_root=repo_root,
    )


def _build_candidate_rows(
    example: DatasetExample,
    *,
    root_symbolic: Any,
    root_scores: Sequence[float],
    considered_indices: Sequence[int],
    proposer_model: Any,
    opponent_model: Any | None,
    opponent_mode: str,
    dynamics_model: Any | None,
    planner_config: "PlannerTrainConfig | None",
    repo_root: Path,
) -> list[_PlannerCandidateRow]:
    root_records = [
        RawPositionRecord(
            sample_id=f"{example.sample_id}:selfplay_root:{root_symbolic.candidate_action_indices[index]}",
            fen=example.fen,
            source="selfplay",
            selected_move_uci=move_uci_for_action(
                example,
                int(root_symbolic.candidate_action_indices[index]),
            ),
        )
        for index in considered_indices
    ]
    root_payloads = label_records_with_oracle(root_records, repo_root=repo_root)
    root_selected_examples = [
        dataset_example_from_oracle_payload(
            sample_id=record.sample_id,
            split=example.split,
            source="selfplay",
            fen=example.fen,
            payload=payload,
        )
        for record, payload in zip(root_records, root_payloads, strict=True)
    ]
    successor_records = [
        RawPositionRecord(
            sample_id=f"{example.sample_id}:selfplay_successor:{index}",
            fen=str(root_selected_example.next_fen),
            source="selfplay",
        )
        for index, root_selected_example in enumerate(root_selected_examples)
    ]
    successor_payloads = label_records_with_oracle(successor_records, repo_root=repo_root)

    rows: list[_PlannerCandidateRow] = []
    latent_dim = 0 if planner_config is None else planner_config.model.latent_feature_dim
    for index, root_selected_example, successor_payload in zip(
        considered_indices,
        root_selected_examples,
        successor_payloads,
        strict=True,
    ):
        action_index = int(root_symbolic.candidate_action_indices[index])
        move_uci = move_uci_for_action(example, action_index)
        assert root_selected_example.next_fen is not None
        successor_example = dataset_example_from_oracle_payload(
            sample_id=f"{example.sample_id}:selfplay_successor_example:{action_index}",
            split=example.split,
            source="selfplay",
            fen=str(root_selected_example.next_fen),
            payload=successor_payload,
        )
        successor_symbolic = build_symbolic_proposer_example(
            successor_example,
            candidate_context_version=2,
            global_context_version=1,
        )
        transition_features = build_transition_context_features(root_selected_example, version=1)
        action_features = build_selected_move_action_features(
            root_selected_example,
            candidate_context_version=1,
        )
        if latent_dim > 0:
            assert dynamics_model is not None
            latent_features = predict_dynamics_latent(
                dynamics_model,
                feature_vector=root_symbolic.feature_vector,
                action_index=action_index,
                action_features=action_features,
                transition_features=transition_features,
            )
        else:
            latent_features = []
        reply_peak_probability, pressure, uncertainty = _reply_signals(
            proposer_model=proposer_model,
            opponent_model=opponent_model,
            opponent_mode=opponent_mode,
            root_feature_vector=root_symbolic.feature_vector,
            successor_symbolic=successor_symbolic,
            action_index=action_index,
            transition_features=transition_features,
        )
        rows.append(
            _PlannerCandidateRow(
                move_uci=move_uci,
                action_index=action_index,
                next_fen=str(root_selected_example.next_fen),
                candidate_features=list(root_symbolic.candidate_features[index]),
                proposer_score=float(root_scores[index]),
                transition_features=transition_features,
                latent_features=latent_features,
                reply_peak_probability=reply_peak_probability,
                pressure=pressure,
                uncertainty=uncertainty,
            )
        )
    return rows


def _label_selected_move(
    example: DatasetExample,
    *,
    move_uci: str,
    repo_root: Path,
) -> DatasetExample:
    payload = label_records_with_oracle(
        [
            RawPositionRecord(
                sample_id=f"{example.sample_id}:selfplay_selected",
                fen=example.fen,
                source="selfplay",
                selected_move_uci=move_uci,
            )
        ],
        repo_root=repo_root,
    )[0]
    return dataset_example_from_oracle_payload(
        sample_id=f"{example.sample_id}:selfplay_selected",
        split=example.split,
        source="selfplay",
        fen=example.fen,
        payload=payload,
    )


def _reply_signals(
    *,
    proposer_model: Any,
    opponent_model: Any | None,
    opponent_mode: str,
    root_feature_vector: Sequence[float],
    successor_symbolic: Any,
    action_index: int,
    transition_features: Sequence[float],
) -> tuple[float, float, float]:
    if not successor_symbolic.candidate_action_indices:
        return 0.0, 0.0, 0.0
    if opponent_mode == "none":
        return 0.0, 0.0, 0.0
    if opponent_mode == "symbolic":
        reply_scores, reply_policy = score_symbolic_candidates(
            proposer_model,
            feature_vector=successor_symbolic.feature_vector,
            candidate_action_indices=successor_symbolic.candidate_action_indices,
            candidate_features=successor_symbolic.candidate_features,
            global_features=successor_symbolic.global_features,
            candidate_context_version=successor_symbolic.candidate_context_version,
        )
        best_reply_index = max(
            range(len(reply_scores)),
            key=lambda index: (reply_scores[index], -index),
        )
        best_features = successor_symbolic.candidate_features[best_reply_index]
        reply_peak_probability = float(max(reply_policy))
        pressure = _pressure_from_candidate_features(best_features)
        uncertainty = 1.0 - reply_peak_probability
        return reply_peak_probability, pressure, uncertainty

    assert opponent_mode == "learned"
    if opponent_model is None:
        raise ValueError("learned opponent mode requires a loaded opponent model")
    _reply_scores, reply_policy, pressure, uncertainty = score_opponent_candidates(
        opponent_model,
        root_feature_vector=root_feature_vector,
        next_feature_vector=successor_symbolic.feature_vector,
        chosen_action_index=action_index,
        transition_features=transition_features,
        reply_candidate_action_indices=successor_symbolic.candidate_action_indices,
        reply_candidate_features=successor_symbolic.candidate_features,
        reply_global_features=successor_symbolic.global_features,
    )
    return float(max(reply_policy, default=0.0)), float(pressure), float(uncertainty)


def _pressure_from_candidate_features(candidate_features: Sequence[float]) -> float:
    if not candidate_features:
        return 0.0
    if bool(candidate_features[_GIVES_CHECK_FEATURE_INDEX]):
        return 1.0
    if bool(candidate_features[_PROMOTION_FEATURE_INDEX]):
        return 0.75
    if bool(candidate_features[_CAPTURE_FEATURE_INDEX]) or bool(
        candidate_features[_EN_PASSANT_FEATURE_INDEX]
    ):
        return 0.5
    return 0.0


def _score_planner_rows(
    *,
    planner_model: PlannerHeadModel,
    planner_config: "PlannerTrainConfig | None",
    root_symbolic: Any,
    candidate_rows: Sequence[_PlannerCandidateRow],
) -> list[float]:
    if torch is None or not torch_is_available():  # pragma: no cover - torch absent
        raise RuntimeError(
            "PyTorch is required for planner runtime evaluation. Install the 'train' extra or torch."
        )
    if planner_config is None:
        raise ValueError("planner_config is required when planner_model is loaded")

    latent_dim = planner_config.model.latent_feature_dim
    root_features = torch.tensor([list(root_symbolic.feature_vector)], dtype=torch.float32)
    global_features = torch.tensor([list(root_symbolic.global_features)], dtype=torch.float32)
    candidate_action_indices = torch.tensor(
        [[row.action_index for row in candidate_rows]],
        dtype=torch.long,
    )
    candidate_features = torch.tensor(
        [[row.candidate_features for row in candidate_rows]],
        dtype=torch.float32,
    )
    proposer_scores = torch.tensor(
        [[row.proposer_score for row in candidate_rows]],
        dtype=torch.float32,
    )
    transition_features = torch.tensor(
        [[row.transition_features for row in candidate_rows]],
        dtype=torch.float32,
    )
    if latent_dim > 0:
        latent_features = torch.tensor(
            [[row.latent_features for row in candidate_rows]],
            dtype=torch.float32,
        )
    else:
        latent_features = torch.zeros((1, len(candidate_rows), 0), dtype=torch.float32)
    reply_peak_probabilities = torch.tensor(
        [[row.reply_peak_probability for row in candidate_rows]],
        dtype=torch.float32,
    )
    pressures = torch.tensor([[row.pressure for row in candidate_rows]], dtype=torch.float32)
    uncertainties = torch.tensor([[row.uncertainty for row in candidate_rows]], dtype=torch.float32)
    candidate_mask = torch.ones((1, len(candidate_rows)), dtype=torch.bool)

    with torch.inference_mode():
        outputs = planner_model(
            root_features,
            global_features,
            candidate_action_indices,
            candidate_features,
            proposer_scores,
            transition_features,
            latent_features,
            reply_peak_probabilities,
            pressures,
            uncertainties,
            candidate_mask,
        )
    logits = outputs["logits"][0, : len(candidate_rows)].tolist()
    return [float(value) for value in logits]
