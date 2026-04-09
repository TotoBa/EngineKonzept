from __future__ import annotations

import pytest

from train.datasets.contracts import candidate_context_feature_dim
from train.models.lapv1 import _OpponentReplySignalProjector
from train.models.opponent import OpponentHeadPrediction
from train.models.opponent_readout import OpponentReadout


pytest.importorskip("torch")

import torch
from torch import nn


def test_opponent_readout_output_shapes() -> None:
    readout = OpponentReadout(
        state_dim=32,
        global_dim=11,
        action_embedding_dim=8,
        hidden_dim=32,
    )
    prediction = readout(
        torch.randn(4, 32),
        torch.tensor([1, 7, 11, 3], dtype=torch.long),
        torch.randn(4, 11),
        torch.tensor(
            [
                [5, 8, 0, 0],
                [9, 4, 2, 0],
                [7, 6, 3, 1],
                [12, 15, 17, 19],
            ],
            dtype=torch.long,
        ),
        torch.randn(4, 4, candidate_context_feature_dim(2)),
        torch.tensor(
            [
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, True],
                [True, False, False, False],
            ],
            dtype=torch.bool,
        ),
    )

    assert tuple(prediction.reply_logits.shape) == (4, 4)
    assert tuple(prediction.pressure.shape) == (4,)
    assert tuple(prediction.uncertainty.shape) == (4,)
    assert torch.isfinite(prediction.reply_logits[:, :2]).all()
    assert torch.isfinite(prediction.pressure).all()
    assert torch.isfinite(prediction.uncertainty).all()


def test_reply_signal_identical_api_to_v1() -> None:
    batch_size = 2
    selected_count = 3
    candidate_count = 4
    base_reply_logits = torch.tensor(
        [
            [0.1, 0.7, 0.2, -0.3],
            [0.4, 0.3, 0.8, -0.2],
            [0.5, 0.1, 0.0, -0.7],
            [0.2, 0.6, 0.4, -0.1],
            [0.9, 0.2, 0.3, -0.4],
            [0.5, 0.4, 0.1, -0.8],
        ],
        dtype=torch.float32,
    )
    base_pressure = torch.tensor([0.1, 0.2, 0.05, 0.3, 0.25, 0.15], dtype=torch.float32)
    base_uncertainty = torch.tensor(
        [0.2, 0.1, 0.15, 0.05, 0.12, 0.18],
        dtype=torch.float32,
    )

    class _FakeLegacyHead(nn.Module):
        def forward(
            self,
            root_features: torch.Tensor,
            next_features: torch.Tensor,
            chosen_action_indices: torch.Tensor,
            transition_features: torch.Tensor,
            reply_global_features: torch.Tensor,
            reply_candidate_action_indices: torch.Tensor,
            reply_candidate_features: torch.Tensor,
            reply_candidate_mask: torch.Tensor,
        ) -> OpponentHeadPrediction:
            del (
                root_features,
                next_features,
                chosen_action_indices,
                transition_features,
                reply_global_features,
                reply_candidate_action_indices,
                reply_candidate_features,
                reply_candidate_mask,
            )
            return OpponentHeadPrediction(
                reply_logits=base_reply_logits,
                pressure=base_pressure,
                uncertainty=base_uncertainty,
            )

    class _FakeSharedReadout(nn.Module):
        def forward(
            self,
            h_root: torch.Tensor,
            selected_action_indices: torch.Tensor,
            reply_global_features: torch.Tensor,
            reply_candidate_action_indices: torch.Tensor,
            reply_candidate_features: torch.Tensor,
            reply_candidate_mask: torch.Tensor,
        ) -> OpponentHeadPrediction:
            del (
                h_root,
                selected_action_indices,
                reply_global_features,
                reply_candidate_action_indices,
                reply_candidate_features,
                reply_candidate_mask,
            )
            return OpponentHeadPrediction(
                reply_logits=base_reply_logits,
                pressure=base_pressure,
                uncertainty=base_uncertainty,
            )

    projector_v1 = _OpponentReplySignalProjector(
        state_dim=32,
        global_dim=11,
        opponent_head=_FakeLegacyHead(),
    )
    projector_v2 = _OpponentReplySignalProjector(
        state_dim=32,
        global_dim=11,
        opponent_readout=_FakeSharedReadout(),
    )
    transitioned_latents = torch.randn(batch_size, selected_count, 32)
    signal_v1 = projector_v1(
        transitioned_latents,
        z_t=torch.randn(batch_size, 32),
        selected_action_indices=torch.tensor(
            [[1, 7, 11], [4, 5, 6]],
            dtype=torch.long,
        ),
        candidate_action_indices=torch.tensor(
            [[9, 3, 2, 0], [8, 1, 4, 0]],
            dtype=torch.long,
        ),
        candidate_features=torch.randn(
            batch_size,
            candidate_count,
            candidate_context_feature_dim(2),
        ),
        candidate_mask=torch.tensor(
            [[True, True, True, False], [True, True, True, False]],
            dtype=torch.bool,
        ),
        global_features=torch.randn(batch_size, 11),
    )
    signal_v2 = projector_v2(
        transitioned_latents,
        z_t=torch.randn(batch_size, 32),
        selected_action_indices=torch.tensor(
            [[1, 7, 11], [4, 5, 6]],
            dtype=torch.long,
        ),
        candidate_action_indices=torch.tensor(
            [[9, 3, 2, 0], [8, 1, 4, 0]],
            dtype=torch.long,
        ),
        candidate_features=torch.randn(
            batch_size,
            candidate_count,
            candidate_context_feature_dim(2),
        ),
        candidate_mask=torch.tensor(
            [[True, True, True, False], [True, True, True, False]],
            dtype=torch.bool,
        ),
        global_features=torch.randn(batch_size, 11),
    )

    assert torch.allclose(signal_v1, signal_v2, atol=1e-6)
