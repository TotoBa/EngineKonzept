"""Runtime-style LAPv1 move selection over exact legal candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import chess

from train.datasets import (
    build_symbolic_proposer_example,
    dataset_example_from_oracle_payload,
    move_uci_for_action,
)
from train.datasets.artifacts import (
    PIECE_TOKEN_CAPACITY,
    PIECE_TOKEN_PADDING_VALUE,
    PIECE_TOKEN_WIDTH,
    SQUARE_TOKEN_COUNT,
    SQUARE_TOKEN_WIDTH,
    split_position_features,
)
from train.datasets.contracts import build_state_context_v1
from train.datasets.move_delta import halfka_delta, is_king_move, move_type_hash
from train.datasets.nnue_features import halfka_active_indices
from train.datasets.oracle import label_records_with_oracle
from train.datasets.phase_features import phase_index as detect_phase_index
from train.datasets.schema import DatasetExample, RawPositionRecord
from train.eval.agent_spec import SelfplayAgentSpec, load_selfplay_agent_spec
from train.eval.planner_runtime import PlannerRootDecision
from train.models.dual_accumulator import pack_sparse_feature_lists
from train.models.lapv1 import LAPV1_MODEL_NAME, LAPv1Model
from train.models.proposer import torch_is_available
from train.trainers.lapv1 import LAPv1TrainConfig, _load_lapv1_model_state

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised when torch is absent
    torch = None


_STATE_CONTEXT_GLOBAL_DIM = 11
SelectedMoveLabeler = Callable[[DatasetExample, str, Path], DatasetExample]


@dataclass
class LoadedLAPv1Runtime:
    """Loaded LAPv1 runtime adapter that matches the selfplay agent protocol."""

    name: str
    model: Any
    training_config: LAPv1TrainConfig
    repo_root: Path
    state_context_version: int
    label_selected_move: SelectedMoveLabeler = field(default_factory=lambda: _label_selected_move)
    last_deliberation_trace: Any | None = None
    last_value_cp: float = 0.0
    last_sigma_value: float = 0.0

    def select_move(self, example: DatasetExample) -> PlannerRootDecision:
        """Select one legal move from the current exact position contract."""
        if not example.legal_moves:
            raise ValueError(f"{example.sample_id}: cannot select a move from an empty legal set")
        if self.state_context_version != 1:
            raise ValueError(f"unsupported state_context_version: {self.state_context_version}")
        if torch is None or not torch_is_available():  # pragma: no cover
            raise RuntimeError(
                "PyTorch is required for LAPv1 runtime evaluation. Install the 'train' extra or torch."
            )

        root_symbolic = build_symbolic_proposer_example(
            example,
            candidate_context_version=2,
            global_context_version=1,
        )
        feature_sections = split_position_features(root_symbolic.feature_vector)
        piece_tokens = torch.tensor(
            [_decode_piece_tokens(feature_sections["piece"])],
            dtype=torch.long,
        )
        square_tokens = torch.tensor(
            [_decode_square_tokens(feature_sections["square"])],
            dtype=torch.float32,
        )
        state_context = build_state_context_v1(example)
        state_context_global = torch.tensor(
            [state_context.feature_values[-_STATE_CONTEXT_GLOBAL_DIM:]],
            dtype=torch.float32,
        )
        reachability_edges = torch.full(
            (1, max(1, len(state_context.edge_src_square)), 3),
            -1,
            dtype=torch.long,
        )
        if state_context.edge_src_square:
            reachability_edges[0, : len(state_context.edge_src_square), :] = torch.tensor(
                list(
                    zip(
                        state_context.edge_src_square,
                        state_context.edge_dst_square,
                        state_context.edge_piece_type,
                        strict=True,
                    )
                ),
                dtype=torch.long,
            )
        candidate_action_indices = torch.tensor(
            [root_symbolic.candidate_action_indices],
            dtype=torch.long,
        )
        candidate_features = torch.tensor(
            [root_symbolic.candidate_features],
            dtype=torch.float32,
        )
        candidate_mask = torch.ones(
            (1, len(root_symbolic.candidate_action_indices)),
            dtype=torch.bool,
        )
        candidate_move_uci = [
            move_uci_for_action(example, int(action_index))
            for action_index in root_symbolic.candidate_action_indices
        ]
        candidate_uci = [candidate_move_uci]
        board = chess.Board(example.fen)
        candidate_move_types = torch.tensor(
            [[move_type_hash(board, chess.Move.from_uci(move_uci)) for move_uci in candidate_move_uci]],
            dtype=torch.long,
        )
        candidate_delta_white_leave: list[list[int]] = []
        candidate_delta_white_enter: list[list[int]] = []
        candidate_delta_black_leave: list[list[int]] = []
        candidate_delta_black_enter: list[list[int]] = []
        candidate_nnue_feat_white_after_move: list[list[int]] = []
        candidate_nnue_feat_black_after_move: list[list[int]] = []
        candidate_has_king_move_values: list[bool] = []
        for move_uci in candidate_move_uci:
            move = chess.Move.from_uci(move_uci)
            candidate_delta_white_leave.append(halfka_delta(board, move, "w")[0])
            candidate_delta_white_enter.append(halfka_delta(board, move, "w")[1])
            candidate_delta_black_leave.append(halfka_delta(board, move, "b")[0])
            candidate_delta_black_enter.append(halfka_delta(board, move, "b")[1])
            has_king_move = is_king_move(board, move, "w") or is_king_move(board, move, "b")
            candidate_has_king_move_values.append(has_king_move)
            if has_king_move:
                after_board = board.copy(stack=False)
                after_board.push(move)
                candidate_nnue_feat_white_after_move.append(halfka_active_indices(after_board, "w"))
                candidate_nnue_feat_black_after_move.append(halfka_active_indices(after_board, "b"))
            else:
                candidate_nnue_feat_white_after_move.append([])
                candidate_nnue_feat_black_after_move.append([])
        phase_tensor = torch.tensor(
            [detect_phase_index(board)],
            dtype=torch.long,
        )
        side_to_move = torch.tensor([0 if board.turn == chess.WHITE else 1], dtype=torch.long)
        nnue_feat_white_indices, nnue_feat_white_offsets = pack_sparse_feature_lists(
            [halfka_active_indices(board, "w")]
        )
        nnue_feat_black_indices, nnue_feat_black_offsets = pack_sparse_feature_lists(
            [halfka_active_indices(board, "b")]
        )
        candidate_delta_white_leave_indices, candidate_delta_white_leave_offsets = (
            pack_sparse_feature_lists(candidate_delta_white_leave)
        )
        candidate_delta_white_enter_indices, candidate_delta_white_enter_offsets = (
            pack_sparse_feature_lists(candidate_delta_white_enter)
        )
        candidate_delta_black_leave_indices, candidate_delta_black_leave_offsets = (
            pack_sparse_feature_lists(candidate_delta_black_leave)
        )
        candidate_delta_black_enter_indices, candidate_delta_black_enter_offsets = (
            pack_sparse_feature_lists(candidate_delta_black_enter)
        )
        candidate_nnue_feat_white_after_move_indices, candidate_nnue_feat_white_after_move_offsets = (
            pack_sparse_feature_lists(candidate_nnue_feat_white_after_move)
        )
        candidate_nnue_feat_black_after_move_indices, candidate_nnue_feat_black_after_move_offsets = (
            pack_sparse_feature_lists(candidate_nnue_feat_black_after_move)
        )
        candidate_has_king_move = torch.tensor(
            [candidate_has_king_move_values],
            dtype=torch.bool,
        )

        with torch.inference_mode():
            outputs = self.model(
                piece_tokens,
                square_tokens,
                state_context_global,
                reachability_edges,
                candidate_features,
                candidate_action_indices,
                candidate_mask,
                phase_index=phase_tensor,
                side_to_move=side_to_move,
                nnue_feat_white_indices=nnue_feat_white_indices,
                nnue_feat_white_offsets=nnue_feat_white_offsets,
                nnue_feat_black_indices=nnue_feat_black_indices,
                nnue_feat_black_offsets=nnue_feat_black_offsets,
                candidate_move_types=candidate_move_types,
                candidate_delta_white_leave_indices=candidate_delta_white_leave_indices,
                candidate_delta_white_leave_offsets=candidate_delta_white_leave_offsets,
                candidate_delta_white_enter_indices=candidate_delta_white_enter_indices,
                candidate_delta_white_enter_offsets=candidate_delta_white_enter_offsets,
                candidate_delta_black_leave_indices=candidate_delta_black_leave_indices,
                candidate_delta_black_leave_offsets=candidate_delta_black_leave_offsets,
                candidate_delta_black_enter_indices=candidate_delta_black_enter_indices,
                candidate_delta_black_enter_offsets=candidate_delta_black_enter_offsets,
                candidate_nnue_feat_white_after_move_indices=candidate_nnue_feat_white_after_move_indices,
                candidate_nnue_feat_white_after_move_offsets=candidate_nnue_feat_white_after_move_offsets,
                candidate_nnue_feat_black_after_move_indices=candidate_nnue_feat_black_after_move_indices,
                candidate_nnue_feat_black_after_move_offsets=candidate_nnue_feat_black_after_move_offsets,
                candidate_has_king_move=candidate_has_king_move,
                candidate_uci=candidate_uci,
                single_legal_move=len(root_symbolic.candidate_action_indices) == 1,
            )
            initial_policy_logits = outputs["initial_policy_logits"]

        final_policy_logits = outputs["final_policy_logits"][0]
        selected_candidate_index = int(torch.argmax(final_policy_logits).item())
        action_index = int(outputs["refined_top1_action_index"][0].item())
        move_uci = move_uci_for_action(example, action_index)
        selected_example = self.label_selected_move(example, move_uci, self.repo_root)
        assert selected_example.next_fen is not None

        self.last_deliberation_trace = outputs["deliberation_trace"]
        self.last_value_cp = float(outputs["final_value"]["cp_score"][0, 0].item())
        self.last_sigma_value = float(outputs["final_value"]["sigma_value"][0, 0].item())

        return PlannerRootDecision(
            move_uci=move_uci,
            action_index=action_index,
            next_fen=str(selected_example.next_fen),
            selector_name=self.name,
            legal_candidate_count=len(root_symbolic.candidate_action_indices),
            considered_candidate_count=len(root_symbolic.candidate_action_indices),
            proposer_score=float(initial_policy_logits[0, selected_candidate_index].item()),
            planner_score=float(final_policy_logits[selected_candidate_index].item()),
            reply_peak_probability=0.0,
            pressure=0.0,
            uncertainty=self.last_sigma_value,
        )


def load_lapv1_checkpoint(
    checkpoint_path: Path,
) -> tuple[Any, LAPv1TrainConfig]:
    """Load one LAPv1 checkpoint and its saved training config."""
    if torch is None or not torch_is_available():  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for LAPv1 runtime evaluation. Install the 'train' extra or torch."
        )
    payload = torch.load(checkpoint_path, map_location="cpu")
    if payload.get("model_name") != LAPV1_MODEL_NAME:
        raise ValueError(
            f"{checkpoint_path}: unsupported LAPv1 model name {payload.get('model_name')!r}"
        )
    training_config = LAPv1TrainConfig.from_dict(dict(payload["training_config"]))
    model = LAPv1Model(training_config.model)
    _load_lapv1_model_state(
        model,
        dict(payload["model_state_dict"]),
        checkpoint_path=checkpoint_path,
    )
    model.eval()
    return model, training_config


def build_lapv1_runtime(
    *,
    name: str,
    lapv1_checkpoint: Path,
    repo_root: Path,
    state_context_version: int = 1,
    deliberation_max_inner_steps: int | None = None,
    deliberation_q_threshold: float | None = None,
    label_selected_move: SelectedMoveLabeler | None = None,
) -> LoadedLAPv1Runtime:
    """Build one LAPv1 runtime adapter from a checkpoint."""
    model, training_config = load_lapv1_checkpoint(lapv1_checkpoint)
    if deliberation_max_inner_steps is not None:
        model.deliberation_loop.max_inner_steps = deliberation_max_inner_steps
        model.deliberation_loop.min_inner_steps = min(
            training_config.model.deliberation.min_inner_steps,
            deliberation_max_inner_steps,
        )
    if deliberation_q_threshold is not None:
        model.deliberation_loop.q_threshold = deliberation_q_threshold
    return LoadedLAPv1Runtime(
        name=name,
        model=model,
        training_config=training_config,
        repo_root=repo_root,
        state_context_version=state_context_version,
        label_selected_move=_label_selected_move if label_selected_move is None else label_selected_move,
    )


def build_lapv1_runtime_from_spec(
    spec: SelfplayAgentSpec,
    *,
    repo_root: Path,
) -> LoadedLAPv1Runtime:
    """Build one LAPv1 runtime adapter from a versioned selfplay spec."""
    if spec.agent_kind != "lapv1":
        raise ValueError(f"expected lapv1 spec, got {spec.agent_kind}")
    if spec.lapv1_checkpoint is None:
        raise ValueError("lapv1 spec requires lapv1_checkpoint")
    return build_lapv1_runtime(
        name=spec.name,
        lapv1_checkpoint=_resolve_repo_path(repo_root, spec.lapv1_checkpoint),
        repo_root=repo_root,
        state_context_version=spec.state_context_version,
        deliberation_max_inner_steps=spec.deliberation_max_inner_steps,
        deliberation_q_threshold=spec.deliberation_q_threshold,
    )


def load_lapv1_runtime_from_spec_path(
    spec_path: Path,
    *,
    repo_root: Path,
) -> LoadedLAPv1Runtime:
    """Load one LAPv1 runtime adapter from a JSON spec file."""
    spec = load_selfplay_agent_spec(spec_path)
    return build_lapv1_runtime_from_spec(spec, repo_root=repo_root)


def _decode_piece_tokens(values: Sequence[float]) -> list[list[int]]:
    tokens: list[list[int]] = []
    for offset in range(0, len(values), PIECE_TOKEN_WIDTH):
        row = [int(round(value)) for value in values[offset : offset + PIECE_TOKEN_WIDTH]]
        if row[0] == PIECE_TOKEN_PADDING_VALUE:
            tokens.append([-1, -1, -1])
        else:
            tokens.append(row)
    if len(tokens) != PIECE_TOKEN_CAPACITY:
        raise ValueError("piece token slice does not decode to capacity rows")
    return tokens


def _decode_square_tokens(values: Sequence[float]) -> list[list[float]]:
    tokens: list[list[float]] = []
    for offset in range(0, len(values), SQUARE_TOKEN_WIDTH):
        row = [float(value) for value in values[offset : offset + SQUARE_TOKEN_WIDTH]]
        tokens.append(row)
    if len(tokens) != SQUARE_TOKEN_COUNT:
        raise ValueError("square token slice does not decode to 64 rows")
    return tokens


def _resolve_repo_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else repo_root / path


def _label_selected_move(
    example: DatasetExample,
    move_uci: str,
    repo_root: Path,
) -> DatasetExample:
    payload = label_records_with_oracle(
        [
            RawPositionRecord(
                sample_id=f"{example.sample_id}:lapv1_selected",
                fen=example.fen,
                source="selfplay",
                selected_move_uci=move_uci,
            )
        ],
        repo_root=repo_root,
    )[0]
    return dataset_example_from_oracle_payload(
        sample_id=f"{example.sample_id}:lapv1_selected",
        split=example.split,
        source="selfplay",
        fen=example.fen,
        payload=payload,
    )
