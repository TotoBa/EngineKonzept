"""Dual-accumulator helpers for shared FT-based LAPv2 heads."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from train.models.feature_transformer import FeatureTransformer
from train.models.phase_moe import PhaseMoE

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised when torch is absent
    torch = None


def pack_sparse_feature_lists(
    feature_lists: Sequence[Sequence[int]],
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Pack sparse feature rows into `EmbeddingBag(indices, offsets)` inputs."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required to pack sparse feature lists.")
    offsets: list[int] = []
    indices: list[int] = []
    cursor = 0
    for feature_list in feature_lists:
        offsets.append(cursor)
        row = [int(value) for value in feature_list]
        indices.extend(row)
        cursor += len(row)
    return (
        torch.tensor(indices, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
    )


class DualAccumulatorBuilder:
    """Build the white/black accumulator pair from batch sparse inputs."""

    def __call__(
        self,
        ft: FeatureTransformer | PhaseMoE,
        batch: Mapping[str, "torch.Tensor"],
        *,
        phase_idx: "torch.Tensor | None" = None,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        if torch is None:  # pragma: no cover
            raise RuntimeError("torch is required to build dual accumulators.")
        if isinstance(ft, PhaseMoE):
            if phase_idx is None:
                raise ValueError("phase_idx is required when FT is phase-routed")
            a_white = _build_phase_routed_accumulator(
                ft,
                indices=batch["nnue_feat_white_indices"],
                offsets=batch["nnue_feat_white_offsets"],
                phase_idx=phase_idx,
            )
            a_black = _build_phase_routed_accumulator(
                ft,
                indices=batch["nnue_feat_black_indices"],
                offsets=batch["nnue_feat_black_offsets"],
                phase_idx=phase_idx,
            )
        else:
            a_white = ft.build(
                batch["nnue_feat_white_indices"],
                batch["nnue_feat_white_offsets"],
            )
            a_black = ft.build(
                batch["nnue_feat_black_indices"],
                batch["nnue_feat_black_offsets"],
            )
        return a_white, a_black


class IncrementalAccumulator:
    """Maintain one FT accumulator pair across sparse candidate deltas."""

    def __init__(self, ft: FeatureTransformer) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("torch is required to maintain incremental accumulators.")
        self.ft = ft
        self.a_white: torch.Tensor | None = None
        self.a_black: torch.Tensor | None = None
        self.dirty_white = False
        self.dirty_black = False
        self._full_rebuild_fn: Any | None = None

    def init_from_position(
        self,
        batch_pos: Mapping[str, Any],
        *,
        full_rebuild_fn: Any | None = None,
    ) -> None:
        white_lists = batch_pos.get("nnue_feat_white")
        black_lists = batch_pos.get("nnue_feat_black")
        if white_lists is not None and black_lists is not None:
            white_indices, white_offsets = pack_sparse_feature_lists([white_lists])
            black_indices, black_offsets = pack_sparse_feature_lists([black_lists])
        else:
            white_indices = batch_pos["nnue_feat_white_indices"]
            white_offsets = batch_pos["nnue_feat_white_offsets"]
            black_indices = batch_pos["nnue_feat_black_indices"]
            black_offsets = batch_pos["nnue_feat_black_offsets"]
        self.a_white = self.ft.build(white_indices, white_offsets)
        self.a_black = self.ft.build(black_indices, black_offsets)
        self.dirty_white = False
        self.dirty_black = False
        self._full_rebuild_fn = full_rebuild_fn

    def apply_move(
        self,
        leave_w: Sequence[int],
        enter_w: Sequence[int],
        leave_b: Sequence[int],
        enter_b: Sequence[int],
        *,
        is_king_w: bool,
        is_king_b: bool,
        full_rebuild_fn: Any | None = None,
    ) -> None:
        if self.a_white is None or self.a_black is None:
            raise RuntimeError("IncrementalAccumulator must be initialized before apply_move")
        if full_rebuild_fn is not None:
            self._full_rebuild_fn = full_rebuild_fn
        if is_king_w:
            self.dirty_white = True
        else:
            self.a_white = _apply_sparse_delta(self.ft, self.a_white, leave_w, enter_w)
        if is_king_b:
            self.dirty_black = True
        else:
            self.a_black = _apply_sparse_delta(self.ft, self.a_black, leave_b, enter_b)

    def get(
        self,
        perspective: str,
        *,
        full_rebuild_fn: Any | None = None,
    ) -> "torch.Tensor":
        if self.a_white is None or self.a_black is None:
            raise RuntimeError("IncrementalAccumulator must be initialized before get")
        normalized = perspective.lower()
        if normalized not in {"w", "b"}:
            raise ValueError("perspective must be 'w' or 'b'")
        rebuild = full_rebuild_fn or self._full_rebuild_fn
        if normalized == "w":
            if self.dirty_white:
                self._rebuild(full_rebuild_fn=rebuild)
            assert self.a_white is not None
            return self.a_white
        if self.dirty_black:
            self._rebuild(full_rebuild_fn=rebuild)
        assert self.a_black is not None
        return self.a_black

    def _rebuild(self, *, full_rebuild_fn: Any | None) -> None:
        if full_rebuild_fn is None:
            raise RuntimeError("full_rebuild_fn is required when an accumulator is dirty")
        rebuilt = full_rebuild_fn()
        if not isinstance(rebuilt, tuple) or len(rebuilt) != 2:
            raise ValueError("full_rebuild_fn must return (a_white, a_black)")
        self.a_white, self.a_black = rebuilt
        self.dirty_white = False
        self.dirty_black = False


def _apply_sparse_delta(
    ft: FeatureTransformer,
    accumulator: "torch.Tensor",
    leave_indices: Sequence[int],
    enter_indices: Sequence[int],
) -> "torch.Tensor":
    updated = accumulator
    if leave_indices:
        leave_rows = ft.gather_rows(torch.tensor(list(leave_indices), dtype=torch.long))
        updated = updated - leave_rows.sum(dim=0, keepdim=True)
    if enter_indices:
        enter_rows = ft.gather_rows(torch.tensor(list(enter_indices), dtype=torch.long))
        updated = updated + enter_rows.sum(dim=0, keepdim=True)
    return updated


def _build_phase_routed_accumulator(
    ft: PhaseMoE,
    *,
    indices: "torch.Tensor",
    offsets: "torch.Tensor",
    phase_idx: "torch.Tensor",
) -> "torch.Tensor":
    accumulator_dim = ft.experts[0].accumulator_dim
    merged = torch.zeros(
        (int(phase_idx.shape[0]), accumulator_dim),
        dtype=ft.experts[0].ft.weight.dtype,
        device=ft.experts[0].ft.weight.device,
    )
    for phase, expert in enumerate(ft.experts):
        mask = phase_idx == phase
        if not bool(mask.any().item()):
            continue
        phase_indices, phase_offsets = _slice_sparse_rows(indices, offsets, mask)
        merged[mask] = expert.build(phase_indices, phase_offsets)
    return merged


def _slice_sparse_rows(
    indices: "torch.Tensor",
    offsets: "torch.Tensor",
    mask: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor"]:
    selected_rows = mask.nonzero(as_tuple=False).flatten().tolist()
    if not selected_rows:
        return (
            torch.zeros((0,), dtype=torch.long, device=indices.device),
            torch.zeros((0,), dtype=torch.long, device=offsets.device),
        )
    rebuilt_indices: list[int] = []
    rebuilt_offsets: list[int] = []
    total_values = int(indices.shape[0])
    cursor = 0
    offset_values = offsets.tolist()
    for row_index in selected_rows:
        start = int(offset_values[row_index])
        end = (
            int(offset_values[row_index + 1])
            if row_index + 1 < len(offset_values)
            else total_values
        )
        rebuilt_offsets.append(cursor)
        row_values = indices[start:end].tolist()
        rebuilt_indices.extend(int(value) for value in row_values)
        cursor += len(row_values)
    return (
        torch.tensor(rebuilt_indices, dtype=torch.long, device=indices.device),
        torch.tensor(rebuilt_offsets, dtype=torch.long, device=offsets.device),
    )
