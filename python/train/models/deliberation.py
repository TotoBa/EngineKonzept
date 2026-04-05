"""Model-only bounded deliberation loop for LAPv1."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from train.action_space import ACTION_SPACE_SIZE

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised when torch is absent
    torch = None
    nn = None


DELIBERATION_MODEL_NAME = "lapv1_deliberation"


@dataclass(frozen=True)
class DeliberationTraceStep:
    """Inspectable one-step record for bounded planner refinement."""

    step: int
    selected_candidates: list[list[int]]
    top1_action_index: list[int]
    top1_value_cp: list[float]
    sharpness: list[float]
    uncertainty: list[float]
    pv_scratch_uci: list[list[str]]
    rollback_fired: bool


@dataclass(frozen=True)
class DeliberationTrace:
    """Container for the full bounded refinement trace."""

    steps: list[DeliberationTraceStep]


if torch is not None and nn is not None:

    class DeliberationCell(nn.Module):
        """GRU-like recurrent update over root state, memory, and candidate scores."""

        def __init__(
            self,
            *,
            state_dim: int = 512,
            memory_dim: int = 256,
            candidate_update_scale: float = 0.1,
        ) -> None:
            super().__init__()
            if state_dim <= 0:
                raise ValueError("state_dim must be positive")
            if memory_dim <= 0:
                raise ValueError("memory_dim must be positive")
            self.state_dim = state_dim
            self.memory_dim = memory_dim
            self.candidate_update_scale = candidate_update_scale
            self.memory_projection = nn.Linear(memory_dim, state_dim)
            self.state_cell = nn.GRUCell(state_dim * 3, state_dim)
            self.memory_update = nn.Linear(state_dim + memory_dim, memory_dim)
            self.state_norm = nn.LayerNorm(state_dim)

        def forward(
            self,
            z_t: torch.Tensor,
            M_t: torch.Tensor,
            C_t: torch.Tensor,
            refined_reply_signals: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Update latent root, rolling memory slots, and candidate scores."""
            memory_summary = M_t.mean(dim=1)
            candidate_summary = C_t.mean(dim=1, keepdim=True)
            reply_summary = refined_reply_signals.mean(dim=1, keepdim=True)
            state_input = torch.cat(
                [
                    z_t,
                    self.memory_projection(memory_summary),
                    torch.cat(
                        [
                            candidate_summary.expand(-1, self.state_dim // 2),
                            reply_summary.expand(-1, self.state_dim // 2),
                        ],
                        dim=1,
                    ),
                ],
                dim=1,
            )
            z_next = self.state_norm(self.state_cell(state_input, z_t))
            new_memory_slot = torch.tanh(
                self.memory_update(torch.cat([memory_summary, z_next], dim=1))
            ).unsqueeze(1)
            M_next = torch.cat([new_memory_slot, M_t[:, :-1, :]], dim=1)
            C_next = C_t + refined_reply_signals * self.candidate_update_scale
            return z_next, M_next, C_next


    class CandidateSelector(nn.Module):
        """Pick the bounded top-K candidates to refine at one deliberation step."""

        def __init__(self, *, top_k: int = 3) -> None:
            super().__init__()
            if top_k <= 0:
                raise ValueError("top_k must be positive")
            self.top_k = top_k

        def forward(
            self,
            z_t: torch.Tensor,
            C_t: torch.Tensor,
            sigma_t: torch.Tensor,
            candidate_mask: torch.Tensor,
        ) -> torch.Tensor:
            """Return per-batch candidate indices ordered by current score."""
            del z_t, sigma_t
            masked_scores = C_t.masked_fill(~candidate_mask, float("-inf"))
            selected_count = min(self.top_k, masked_scores.shape[1])
            return torch.topk(masked_scores, k=selected_count, dim=1).indices


    class LatentTransition(nn.Module):
        """Bounded learned forward projection for one selected action."""

        def __init__(
            self,
            *,
            state_dim: int = 512,
            action_embedding_dim: int = 64,
            hidden_dim: int = 512,
        ) -> None:
            super().__init__()
            if state_dim <= 0:
                raise ValueError("state_dim must be positive")
            if action_embedding_dim <= 0:
                raise ValueError("action_embedding_dim must be positive")
            if hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive")
            self.action_embedding = nn.Embedding(ACTION_SPACE_SIZE, action_embedding_dim)
            self.network = nn.Sequential(
                nn.Linear(state_dim + action_embedding_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, state_dim),
            )

        def forward(
            self,
            z_t: torch.Tensor,
            action_indices: torch.Tensor,
        ) -> torch.Tensor:
            """Project z_t forward for each selected action independently."""
            action_embeddings = self.action_embedding(action_indices)
            expanded_state = z_t.unsqueeze(1).expand(-1, action_indices.shape[1], -1)
            return self.network(torch.cat([expanded_state, action_embeddings], dim=2))


    class _DefaultValueProjector(nn.Module):
        def __init__(self, *, state_dim: int, memory_dim: int) -> None:
            super().__init__()
            self.memory_projection = nn.Linear(memory_dim, state_dim)
            self.score_head = nn.Linear(state_dim, 1)
            self.uncertainty_head = nn.Linear(state_dim, 1)

        def forward(self, z_t: torch.Tensor, M_t: torch.Tensor, C_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            del C_t
            memory_summary = self.memory_projection(M_t.mean(dim=1))
            latent = z_t + memory_summary
            return (
                self.score_head(latent).squeeze(1),
                torch.nn.functional.softplus(self.uncertainty_head(latent)).squeeze(1) + 1e-6,
            )


    class _DefaultSharpnessProjector(nn.Module):
        def __init__(self, *, state_dim: int) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.GELU(),
                nn.Linear(128, 1),
            )

        def forward(self, z_t: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.network(z_t)).squeeze(1)


    class _DefaultReplySignalProjector(nn.Module):
        def __init__(self, *, state_dim: int) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.GELU(),
                nn.Linear(256, 1),
            )

        def forward(self, transitioned_latents: torch.Tensor) -> torch.Tensor:
            return self.network(transitioned_latents).squeeze(2)


    class DeliberationLoop(nn.Module):
        """Bounded recurrent refinement loop with rollback and trace emission."""

        def __init__(
            self,
            *,
            state_dim: int = 512,
            memory_dim: int = 256,
            memory_slots: int = 16,
            action_embedding_dim: int = 64,
            top_k_refine: int = 3,
            max_inner_steps: int = 8,
            min_inner_steps: int = 2,
            q_threshold: float = 0.3,
            rollback_threshold: float = 40.0,
            top1_stable_steps: int = 3,
            rollback_buffer_size: int = 4,
            cell: DeliberationCell | None = None,
            selector: CandidateSelector | None = None,
            transition: LatentTransition | None = None,
            value_projector: nn.Module | None = None,
            sharpness_projector: nn.Module | None = None,
            reply_signal_projector: nn.Module | None = None,
        ) -> None:
            super().__init__()
            if state_dim <= 0:
                raise ValueError("state_dim must be positive")
            if memory_dim <= 0:
                raise ValueError("memory_dim must be positive")
            if memory_slots <= 0:
                raise ValueError("memory_slots must be positive")
            if max_inner_steps <= 0:
                raise ValueError("max_inner_steps must be positive")
            if min_inner_steps < 0:
                raise ValueError("min_inner_steps must be non-negative")
            if min_inner_steps > max_inner_steps:
                raise ValueError("min_inner_steps must not exceed max_inner_steps")
            if not 0.0 <= q_threshold <= 1.0:
                raise ValueError("q_threshold must be in [0.0, 1.0]")
            if rollback_threshold < 0.0:
                raise ValueError("rollback_threshold must be non-negative")
            if top1_stable_steps <= 0:
                raise ValueError("top1_stable_steps must be positive")
            if rollback_buffer_size <= 0:
                raise ValueError("rollback_buffer_size must be positive")

            self.state_dim = state_dim
            self.memory_dim = memory_dim
            self.memory_slots = memory_slots
            self.max_inner_steps = max_inner_steps
            self.min_inner_steps = min_inner_steps
            self.q_threshold = q_threshold
            self.rollback_threshold = rollback_threshold
            self.top1_stable_steps = top1_stable_steps
            self.rollback_buffer_size = rollback_buffer_size

            self.cell = cell or DeliberationCell(
                state_dim=state_dim,
                memory_dim=memory_dim,
            )
            self.selector = selector or CandidateSelector(top_k=top_k_refine)
            self.transition = transition or LatentTransition(
                state_dim=state_dim,
                action_embedding_dim=action_embedding_dim,
                hidden_dim=state_dim,
            )
            self.value_projector = value_projector or _DefaultValueProjector(
                state_dim=state_dim,
                memory_dim=memory_dim,
            )
            self.sharpness_projector = sharpness_projector or _DefaultSharpnessProjector(
                state_dim=state_dim,
            )
            self.reply_signal_projector = reply_signal_projector or _DefaultReplySignalProjector(
                state_dim=state_dim,
            )

        def forward(
            self,
            z_root: torch.Tensor,
            candidate_action_indices: torch.Tensor,
            initial_candidate_scores: torch.Tensor,
            candidate_mask: torch.Tensor,
            *,
            single_legal_move: bool = False,
            candidate_uci: list[list[str]] | None = None,
        ) -> dict[str, Any]:
            """Run the bounded latent refinement loop and emit a structured trace."""
            if z_root.ndim != 2 or z_root.shape[1] != self.state_dim:
                raise ValueError(f"z_root must have shape (batch, {self.state_dim})")
            if candidate_action_indices.ndim != 2:
                raise ValueError(
                    "candidate_action_indices must have shape (batch, num_candidates)"
                )
            if initial_candidate_scores.ndim != 2:
                raise ValueError(
                    "initial_candidate_scores must have shape (batch, num_candidates)"
                )
            if candidate_mask.ndim != 2:
                raise ValueError("candidate_mask must have shape (batch, num_candidates)")
            if candidate_action_indices.shape != initial_candidate_scores.shape:
                raise ValueError(
                    "candidate_action_indices and initial_candidate_scores must align"
                )
            if candidate_mask.shape != candidate_action_indices.shape:
                raise ValueError("candidate_mask must align with candidate_action_indices")

            legal_counts = candidate_mask.sum(dim=1)
            if single_legal_move or bool(torch.all(legal_counts <= 1)):
                masked_scores = initial_candidate_scores.masked_fill(
                    ~candidate_mask,
                    float("-inf"),
                )
                top1_indices = torch.argmax(masked_scores, dim=1)
                top1_actions = candidate_action_indices.gather(1, top1_indices.unsqueeze(1)).squeeze(1)
                return {
                    "final_candidate_scores": masked_scores,
                    "refined_top1_action_index": top1_actions,
                    "trace": DeliberationTrace(steps=[]),
                    "step_count": 0,
                    "final_z": z_root,
                    "final_memory": torch.zeros(
                        (z_root.shape[0], self.memory_slots, self.memory_dim),
                        dtype=z_root.dtype,
                        device=z_root.device,
                    ),
                }

            z_t = z_root
            M_t = torch.zeros(
                (z_root.shape[0], self.memory_slots, self.memory_dim),
                dtype=z_root.dtype,
                device=z_root.device,
            )
            C_t = initial_candidate_scores.masked_fill(~candidate_mask, float("-inf"))
            trace_steps: list[DeliberationTraceStep] = []
            top1_history: list[list[int]] = []
            snapshots: deque[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = deque(
                maxlen=self.rollback_buffer_size
            )
            base_value_cp, base_uncertainty = self.value_projector(
                z_t,
                M_t,
                torch.where(torch.isfinite(C_t), C_t, torch.zeros_like(C_t)),
            )
            snapshots.append((z_t.clone(), M_t.clone(), C_t.clone(), base_value_cp.clone()))

            for step_index in range(self.max_inner_steps):
                finite_scores = torch.where(torch.isfinite(C_t), C_t, torch.zeros_like(C_t))
                value_cp, uncertainty = self.value_projector(z_t, M_t, finite_scores)
                sharpness = self.sharpness_projector(z_t)
                top1_indices = torch.argmax(C_t, dim=1)
                top1_actions = candidate_action_indices.gather(1, top1_indices.unsqueeze(1)).squeeze(1)
                top1_history.append(top1_actions.tolist())

                if step_index >= self.min_inner_steps:
                    if bool(torch.all(sharpness < self.q_threshold)):
                        break
                    if len(top1_history) >= self.top1_stable_steps:
                        recent = top1_history[-self.top1_stable_steps :]
                        if all(row == recent[0] for row in recent[1:]):
                            break

                selected_indices = self.selector(
                    z_t,
                    finite_scores,
                    uncertainty.unsqueeze(1),
                    candidate_mask,
                )
                selected_action_indices = candidate_action_indices.gather(1, selected_indices)
                transitioned_latents = self.transition(z_t, selected_action_indices)
                selected_reply_signals = self.reply_signal_projector(transitioned_latents)
                refined_reply_signals = torch.zeros_like(finite_scores)
                refined_reply_signals.scatter_(1, selected_indices, selected_reply_signals)

                z_next, M_next, C_next = self.cell(z_t, M_t, finite_scores, refined_reply_signals)
                C_next = C_next.masked_fill(~candidate_mask, float("-inf"))
                next_value_cp, _next_uncertainty = self.value_projector(
                    z_next,
                    M_next,
                    torch.where(torch.isfinite(C_next), C_next, torch.zeros_like(C_next)),
                )

                rollback_fired = False
                if torch.any((value_cp - next_value_cp) > self.rollback_threshold):
                    rollback_fired = True
                    snapshot_z, snapshot_M, snapshot_C, _snapshot_value = snapshots[-1]
                    z_t = snapshot_z.clone()
                    M_t = snapshot_M.clone()
                    C_t = snapshot_C.clone()
                    penalty = torch.full_like(selected_reply_signals, 1.0)
                    C_t.scatter_add_(1, selected_indices, -penalty)
                    C_t = C_t.masked_fill(~candidate_mask, float("-inf"))
                else:
                    z_t = z_next
                    M_t = M_next
                    C_t = C_next
                    snapshots.append((z_t.clone(), M_t.clone(), C_t.clone(), next_value_cp.clone()))

                pv_scratch = _build_pv_scratch(
                    candidate_action_indices=candidate_action_indices,
                    candidate_scores=C_t,
                    candidate_mask=candidate_mask,
                    candidate_uci=candidate_uci,
                )
                trace_steps.append(
                    DeliberationTraceStep(
                        step=step_index,
                        selected_candidates=selected_indices.tolist(),
                        top1_action_index=top1_actions.tolist(),
                        top1_value_cp=value_cp.tolist(),
                        sharpness=sharpness.tolist(),
                        uncertainty=uncertainty.tolist(),
                        pv_scratch_uci=pv_scratch,
                        rollback_fired=rollback_fired,
                    )
                )

            final_top1_indices = torch.argmax(C_t, dim=1)
            final_top1_actions = candidate_action_indices.gather(
                1,
                final_top1_indices.unsqueeze(1),
            ).squeeze(1)
            return {
                "final_candidate_scores": C_t,
                "refined_top1_action_index": final_top1_actions,
                "trace": DeliberationTrace(steps=trace_steps),
                "step_count": len(trace_steps),
                "final_z": z_t,
                "final_memory": M_t,
            }


def _build_pv_scratch(
    *,
    candidate_action_indices: torch.Tensor,
    candidate_scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    candidate_uci: list[list[str]] | None,
) -> list[list[str]]:
    top_k = min(3, candidate_scores.shape[1])
    masked_scores = candidate_scores.masked_fill(~candidate_mask, float("-inf"))
    topk_indices = torch.topk(masked_scores, k=top_k, dim=1).indices.tolist()
    action_indices = candidate_action_indices.tolist()
    pv_scratch: list[list[str]] = []
    for batch_index, indices in enumerate(topk_indices):
        if candidate_uci is not None:
            pv_scratch.append([candidate_uci[batch_index][index] for index in indices])
        else:
            pv_scratch.append(
                [f"action_{action_indices[batch_index][index]}" for index in indices]
            )
    return pv_scratch
