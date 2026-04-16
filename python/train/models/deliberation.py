"""Model-only bounded deliberation loop for LAPv1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from train.action_space import ACTION_SPACE_SIZE

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised when torch is absent
    torch = None
    nn = None


MASKED_CANDIDATE_SCORE_VALUE = -1e9


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
    frontier_turnover: list[float]
    frontier_revisit: list[float]
    frontier_stable: list[bool]
    pv_scratch_uci: list[list[str]]
    rollback_fired: bool


@dataclass(frozen=True)
class DeliberationTrace:
    """Container for the full bounded refinement trace."""

    steps: list[DeliberationTraceStep]


if torch is not None and nn is not None:

    class DeliberationCell(nn.Module):
        """GRU-like recurrent update over root state, memory, and residual score deltas."""

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
            self.frontier_context_projection = nn.Linear(state_dim * 2, state_dim)
            self.candidate_frontier_state_projection = nn.Linear(state_dim, state_dim)
            self.candidate_frontier_memory_projection = nn.Linear(memory_dim, state_dim)
            self.candidate_delta_network = nn.Sequential(
                nn.Linear(state_dim * 2 + 3, state_dim),
                nn.GELU(),
                nn.Linear(state_dim, 1),
            )
            self.candidate_frontier_delta_network = nn.Sequential(
                nn.Linear(state_dim * 4 + 3, state_dim),
                nn.GELU(),
                nn.Linear(state_dim, 1),
            )

        def forward(
            self,
            z_t: torch.Tensor,
            M_t: torch.Tensor,
            root_scores: torch.Tensor,
            delta_scores: torch.Tensor,
            refined_reply_signals: torch.Tensor,
            candidate_update_mask: torch.Tensor,
            candidate_mask: torch.Tensor,
            candidate_frontier_states: torch.Tensor,
            candidate_frontier_memory: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Update latent root, rolling memory slots, and residual candidate deltas."""
            memory_summary = M_t.mean(dim=1)
            candidate_scores = (root_scores + delta_scores).masked_fill(
                ~candidate_mask,
                0.0,
            )
            valid_candidate_count = candidate_mask.sum(dim=1, keepdim=True).clamp(min=1).to(
                z_t.dtype
            )
            update_count = candidate_update_mask.sum(dim=1, keepdim=True).clamp(min=1).to(
                z_t.dtype
            )
            candidate_summary = candidate_scores.sum(dim=1, keepdim=True) / valid_candidate_count
            reply_summary = refined_reply_signals.sum(dim=1, keepdim=True) / update_count
            masked_candidate_frontier_states = candidate_frontier_states.masked_fill(
                ~candidate_mask.unsqueeze(2),
                0.0,
            )
            masked_candidate_frontier_memory = candidate_frontier_memory.masked_fill(
                ~candidate_mask.unsqueeze(2),
                0.0,
            )
            frontier_state_summary = (
                masked_candidate_frontier_states.sum(dim=1) / valid_candidate_count
            )
            frontier_memory_summary = (
                masked_candidate_frontier_memory.sum(dim=1) / valid_candidate_count
            )
            frontier_context = self.frontier_context_projection(
                torch.cat(
                    [
                        frontier_state_summary,
                        self.candidate_frontier_memory_projection(
                            frontier_memory_summary
                        ),
                    ],
                    dim=1,
                )
            )
            state_input = torch.cat(
                [
                    z_t,
                    self.memory_projection(memory_summary) + frontier_context,
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
            expanded_state = z_next.unsqueeze(1).expand(-1, root_scores.shape[1], -1)
            expanded_memory = self.memory_projection(memory_summary).unsqueeze(1).expand_as(
                expanded_state
            )
            projected_candidate_frontier_state = self.candidate_frontier_state_projection(
                candidate_frontier_states
            )
            projected_candidate_frontier_memory = self.candidate_frontier_memory_projection(
                candidate_frontier_memory
            )
            candidate_inputs = torch.cat(
                [
                    expanded_state,
                    expanded_memory,
                    torch.stack(
                        [
                            candidate_scores,
                            delta_scores.masked_fill(~candidate_mask, 0.0),
                            refined_reply_signals,
                        ],
                        dim=2,
                    ),
                ],
                dim=2,
            )
            delta_update = self.candidate_delta_network(candidate_inputs).squeeze(2)
            delta_update = (
                delta_update
                * candidate_update_mask.to(delta_update.dtype)
                * self.candidate_update_scale
            )
            candidate_frontier_inputs = torch.cat(
                [
                    expanded_state,
                    expanded_memory,
                    projected_candidate_frontier_state,
                    projected_candidate_frontier_memory,
                    torch.stack(
                        [
                            candidate_scores,
                            delta_scores.masked_fill(~candidate_mask, 0.0),
                            refined_reply_signals,
                        ],
                        dim=2,
                    ),
                ],
                dim=2,
            )
            frontier_delta_update = self.candidate_frontier_delta_network(
                candidate_frontier_inputs
            ).squeeze(2)
            frontier_delta_update = (
                frontier_delta_update
                * candidate_update_mask.to(frontier_delta_update.dtype)
                * self.candidate_update_scale
            )
            delta_next = delta_scores + delta_update + frontier_delta_update
            return z_next, M_next, delta_next


    class CandidateSelector(nn.Module):
        """Pick the bounded top-K candidates to refine at one deliberation step."""

        def __init__(
            self,
            *,
            top_k: int = 3,
            revisit_bonus: float = 0.15,
            exploration_bonus: float = 0.25,
            exploration_decay: float = 1.0,
        ) -> None:
            super().__init__()
            if top_k <= 0:
                raise ValueError("top_k must be positive")
            if revisit_bonus < 0.0:
                raise ValueError("revisit_bonus must be non-negative")
            if exploration_bonus < 0.0:
                raise ValueError("exploration_bonus must be non-negative")
            if exploration_decay < 0.0:
                raise ValueError("exploration_decay must be non-negative")
            self.top_k = top_k
            self.revisit_bonus = revisit_bonus
            self.exploration_bonus = exploration_bonus
            self.exploration_decay = exploration_decay

        def forward(
            self,
            z_t: torch.Tensor,
            C_t: torch.Tensor,
            sigma_t: torch.Tensor,
            candidate_mask: torch.Tensor,
            *,
            previous_selected_mask: torch.Tensor | None = None,
            selection_counts: torch.Tensor | None = None,
            candidate_frontier_states: torch.Tensor | None = None,
            candidate_frontier_memory: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Return per-batch candidate indices ordered by current score."""
            if previous_selected_mask is None:
                previous_selected_mask = torch.zeros_like(candidate_mask)
            if selection_counts is None:
                selection_counts = torch.zeros_like(C_t)
            uncertainty = sigma_t.reshape(C_t.shape[0], -1)[:, :1].to(C_t.dtype)
            uncertainty = uncertainty / (1.0 + uncertainty)
            uncertainty = uncertainty.expand_as(C_t)
            revisit_term = (
                previous_selected_mask.to(C_t.dtype)
                * self.revisit_bonus
                * (1.0 - uncertainty)
            )
            novelty_term = self.exploration_bonus * uncertainty / (
                selection_counts.to(C_t.dtype) + 1.0
            ).pow(self.exploration_decay)
            frontier_scores = C_t + revisit_term + novelty_term
            if candidate_frontier_states is not None:
                root_state = z_t.unsqueeze(1).expand_as(candidate_frontier_states)
                state_drift = torch.linalg.vector_norm(
                    candidate_frontier_states - root_state,
                    dim=2,
                ) / (float(candidate_frontier_states.shape[2]) ** 0.5)
                frontier_scores = frontier_scores + (0.05 * uncertainty * state_drift)
            if candidate_frontier_memory is not None:
                memory_norm = torch.linalg.vector_norm(
                    candidate_frontier_memory,
                    dim=2,
                ) / (float(candidate_frontier_memory.shape[2]) ** 0.5)
                frontier_scores = frontier_scores + (0.05 * uncertainty * memory_norm)
            masked_scores = frontier_scores.masked_fill(
                ~candidate_mask,
                MASKED_CANDIDATE_SCORE_VALUE,
            )
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

        def forward(
            self,
            transitioned_latents: torch.Tensor,
            **_kwargs: Any,
        ) -> torch.Tensor:
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
            if max_inner_steps < 0:
                raise ValueError("max_inner_steps must be non-negative")
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
            phase_idx: torch.Tensor | None = None,
            single_legal_move: bool = False,
            candidate_uci: list[list[str]] | None = None,
            candidate_features: torch.Tensor | None = None,
            global_features: torch.Tensor | None = None,
            collect_reply_diagnostics: bool = False,
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
            initial_candidate_frontier_states = z_root.unsqueeze(1).expand(
                -1,
                candidate_action_indices.shape[1],
                -1,
            ).clone()
            initial_candidate_frontier_memory = torch.zeros(
                (
                    z_root.shape[0],
                    candidate_action_indices.shape[1],
                    self.memory_dim,
                ),
                dtype=z_root.dtype,
                device=z_root.device,
            )
            if single_legal_move or bool(torch.all(legal_counts <= 1)) or self.max_inner_steps == 0:
                masked_scores = initial_candidate_scores.masked_fill(
                    ~candidate_mask,
                    MASKED_CANDIDATE_SCORE_VALUE,
                )
                top1_indices = torch.argmax(masked_scores, dim=1)
                top1_actions = candidate_action_indices.gather(1, top1_indices.unsqueeze(1)).squeeze(1)
                return {
                    "final_candidate_scores": masked_scores,
                    "refined_top1_action_index": top1_actions,
                    "trace": DeliberationTrace(steps=[]),
                    "step_count": 0,
                    "step_value_cp_tensors": (),
                    "step_sharpness_tensors": (),
                    "step_candidate_score_tensors": (),
                    "step_selected_candidate_tensors": (),
                    "step_selected_candidate_masks": (),
                    "step_phase_indices": (),
                    "step_active_masks": (),
                    "step_rollback_masks": (),
                    "step_frontier_turnover_tensors": (),
                    "step_frontier_revisit_tensors": (),
                    "step_frontier_stable_masks": (),
                    "step_rollback_flags": (),
                    "step_student_reply_logits_tensors": (),
                    "step_student_pressure_tensors": (),
                    "step_student_uncertainty_tensors": (),
                    "step_teacher_reply_logits_tensors": (),
                    "step_teacher_pressure_tensors": (),
                    "step_teacher_uncertainty_tensors": (),
                    "root_candidate_scores": masked_scores,
                    "final_candidate_deltas": torch.zeros_like(masked_scores),
                    "final_z": z_root,
                    "final_memory": torch.zeros(
                        (z_root.shape[0], self.memory_slots, self.memory_dim),
                        dtype=z_root.dtype,
                        device=z_root.device,
                    ),
                    "candidate_frontier_states": initial_candidate_frontier_states,
                    "candidate_frontier_memory": initial_candidate_frontier_memory,
                    "frontier_state_drift": torch.zeros(
                        (z_root.shape[0],),
                        dtype=z_root.dtype,
                        device=z_root.device,
                    ),
                    "frontier_memory_norm": torch.zeros(
                        (z_root.shape[0],),
                        dtype=z_root.dtype,
                        device=z_root.device,
                    ),
                    "frontier_visit_counts": torch.zeros_like(masked_scores),
                    "frontier_unique_candidate_counts": torch.zeros(
                        (z_root.shape[0],),
                        dtype=torch.long,
                        device=z_root.device,
                    ),
                }

            z_t = z_root
            M_t = torch.zeros(
                (z_root.shape[0], self.memory_slots, self.memory_dim),
                dtype=z_root.dtype,
                device=z_root.device,
            )
            candidate_frontier_states = initial_candidate_frontier_states
            candidate_frontier_memory = initial_candidate_frontier_memory
            root_scores = initial_candidate_scores.masked_fill(
                ~candidate_mask,
                MASKED_CANDIDATE_SCORE_VALUE,
            )
            delta_scores = torch.zeros_like(root_scores)
            trace_steps: list[DeliberationTraceStep] = []
            step_value_cp_tensors: list[torch.Tensor] = []
            step_sharpness_tensors: list[torch.Tensor] = []
            step_candidate_score_tensors: list[torch.Tensor] = []
            step_selected_candidate_tensors: list[torch.Tensor] = []
            step_selected_candidate_masks: list[torch.Tensor] = []
            step_phase_indices: list[torch.Tensor] = []
            step_active_masks: list[torch.Tensor] = []
            step_rollback_masks: list[torch.Tensor] = []
            step_frontier_turnover_tensors: list[torch.Tensor] = []
            step_frontier_revisit_tensors: list[torch.Tensor] = []
            step_frontier_stable_masks: list[torch.Tensor] = []
            step_rollback_flags: list[bool] = []
            step_student_reply_logits_tensors: list[torch.Tensor] = []
            step_student_pressure_tensors: list[torch.Tensor] = []
            step_student_uncertainty_tensors: list[torch.Tensor] = []
            step_teacher_reply_logits_tensors: list[torch.Tensor] = []
            step_teacher_pressure_tensors: list[torch.Tensor] = []
            step_teacher_uncertainty_tensors: list[torch.Tensor] = []
            top1_history: list[list[int]] = []
            active_mask = legal_counts > 1
            previous_selected_mask = torch.zeros_like(candidate_mask)
            frontier_visit_counts = torch.zeros_like(root_scores)
            frontier_visited_mask = torch.zeros_like(candidate_mask)

            for step_index in range(self.max_inner_steps):
                current_scores = (root_scores + delta_scores).masked_fill(
                    ~candidate_mask,
                    MASKED_CANDIDATE_SCORE_VALUE,
                )
                masked_scores = torch.where(
                    candidate_mask,
                    current_scores,
                    torch.zeros_like(current_scores),
                )
                value_cp, uncertainty = self.value_projector(z_t, M_t, masked_scores)
                sharpness = self.sharpness_projector(z_t)
                top1_indices = torch.argmax(current_scores, dim=1)
                top1_actions = candidate_action_indices.gather(1, top1_indices.unsqueeze(1)).squeeze(1)
                top1_history.append(top1_actions.tolist())

                stop_mask = torch.zeros_like(active_mask)
                if step_index >= self.min_inner_steps:
                    stop_mask = active_mask & (sharpness < self.q_threshold)
                    if len(top1_history) >= self.top1_stable_steps:
                        recent = torch.tensor(
                            top1_history[-self.top1_stable_steps :],
                            dtype=top1_actions.dtype,
                            device=top1_actions.device,
                        ).transpose(0, 1)
                        stable_mask = (recent == recent[:, :1]).all(dim=1)
                        stop_mask = stop_mask | (active_mask & stable_mask)

                step_active_mask = active_mask & ~stop_mask
                if not bool(step_active_mask.any().item()):
                    break

                selected_indices = self.selector(
                    z_t,
                    current_scores,
                    uncertainty.unsqueeze(1),
                    candidate_mask,
                    previous_selected_mask=previous_selected_mask,
                    selection_counts=frontier_visit_counts,
                    candidate_frontier_states=candidate_frontier_states,
                    candidate_frontier_memory=candidate_frontier_memory,
                )
                selected_mask = torch.zeros_like(candidate_mask)
                selected_mask.scatter_(
                    1,
                    selected_indices,
                    step_active_mask.unsqueeze(1).expand(-1, selected_indices.shape[1]),
                )
                frontier_overlap_counts = (selected_mask & previous_selected_mask).sum(dim=1)
                selected_counts = selected_mask.sum(dim=1).clamp(min=1)
                has_previous_frontier = step_active_mask & previous_selected_mask.any(dim=1)
                frontier_revisit = torch.where(
                    has_previous_frontier,
                    frontier_overlap_counts.to(value_cp.dtype) / selected_counts.to(value_cp.dtype),
                    torch.zeros_like(value_cp),
                )
                frontier_turnover = torch.where(
                    has_previous_frontier,
                    1.0 - frontier_revisit,
                    torch.zeros_like(frontier_revisit),
                )
                frontier_stable_mask = has_previous_frontier & (
                    selected_mask == previous_selected_mask
                ).all(dim=1)
                frontier_visit_counts = frontier_visit_counts + selected_mask.to(
                    frontier_visit_counts.dtype
                )
                frontier_visited_mask = frontier_visited_mask | selected_mask
                previous_selected_mask = torch.where(
                    step_active_mask.unsqueeze(1),
                    selected_mask,
                    previous_selected_mask,
                )
                selected_action_indices = candidate_action_indices.gather(1, selected_indices)
                selected_frontier_memory = candidate_frontier_memory.gather(
                    1,
                    selected_indices.unsqueeze(2).expand(-1, -1, self.memory_dim),
                )
                transitioned_latents = self.transition(z_t, selected_action_indices)
                reply_diagnostics: dict[str, torch.Tensor | None] | None = None
                if hasattr(self.reply_signal_projector, "predict"):
                    selected_reply_signals, reply_diagnostics = self.reply_signal_projector.predict(
                        transitioned_latents,
                        z_t=z_t,
                        selected_action_indices=selected_action_indices,
                        candidate_action_indices=candidate_action_indices,
                        candidate_features=candidate_features,
                        candidate_mask=candidate_mask,
                        global_features=global_features,
                        collect_distill_targets=collect_reply_diagnostics,
                    )
                else:
                    selected_reply_signals = self.reply_signal_projector(
                        transitioned_latents,
                        z_t=z_t,
                        selected_action_indices=selected_action_indices,
                        candidate_action_indices=candidate_action_indices,
                        candidate_features=candidate_features,
                        candidate_mask=candidate_mask,
                        global_features=global_features,
                    )
                selected_reply_signals = torch.where(
                    step_active_mask.unsqueeze(1),
                    selected_reply_signals,
                    torch.zeros_like(selected_reply_signals),
                )
                proposed_frontier_memory = torch.tanh(
                    self.cell.memory_update(
                        torch.cat(
                            [
                                selected_frontier_memory,
                                transitioned_latents
                                + selected_reply_signals.unsqueeze(2),
                            ],
                            dim=2,
                        )
                    )
                )
                refined_reply_signals = torch.zeros_like(current_scores)
                refined_reply_signals.scatter_(1, selected_indices, selected_reply_signals)
                candidate_update_mask = torch.zeros_like(candidate_mask)
                candidate_update_mask.scatter_(
                    1,
                    selected_indices,
                    step_active_mask.unsqueeze(1).expand(-1, selected_indices.shape[1]),
                )
                candidate_frontier_states_step = _scatter_candidate_updates(
                    base=candidate_frontier_states,
                    indices=selected_indices,
                    updates=transitioned_latents,
                )
                candidate_frontier_memory_step = _scatter_candidate_updates(
                    base=candidate_frontier_memory,
                    indices=selected_indices,
                    updates=proposed_frontier_memory,
                )

                z_next, M_next, delta_next = self.cell(
                    z_t,
                    M_t,
                    root_scores,
                    delta_scores,
                    refined_reply_signals,
                    candidate_update_mask,
                    candidate_mask,
                    candidate_frontier_states_step,
                    candidate_frontier_memory_step,
                )
                C_next = (root_scores + delta_next).masked_fill(
                    ~candidate_mask,
                    MASKED_CANDIDATE_SCORE_VALUE,
                )
                next_value_cp, _next_uncertainty = self.value_projector(
                    z_next,
                    M_next,
                    torch.where(candidate_mask, C_next, torch.zeros_like(C_next)),
                )

                rollback_mask = step_active_mask & (
                    (value_cp - next_value_cp) > self.rollback_threshold
                )
                rollback_fired = bool(rollback_mask.any().item())
                accept_mask = step_active_mask & ~rollback_mask
                rollback_penalties = torch.zeros_like(root_scores)
                rollback_penalties.scatter_(
                    1,
                    selected_indices,
                    torch.ones_like(selected_reply_signals),
                )
                rollback_penalties = rollback_penalties * rollback_mask.unsqueeze(1).to(
                    rollback_penalties.dtype
                )
                rollback_delta_scores = delta_scores - rollback_penalties
                z_t = torch.where(accept_mask.unsqueeze(1), z_next, z_t)
                M_t = torch.where(accept_mask.view(-1, 1, 1), M_next, M_t)
                accepted_transitioned_latents = torch.where(
                    accept_mask.view(-1, 1, 1),
                    transitioned_latents,
                    candidate_frontier_states.gather(
                        1,
                        selected_indices.unsqueeze(2).expand(-1, -1, self.state_dim),
                    ),
                )
                accepted_frontier_memory = torch.where(
                    accept_mask.view(-1, 1, 1),
                    proposed_frontier_memory,
                    selected_frontier_memory,
                )
                candidate_frontier_states = _scatter_candidate_updates(
                    base=candidate_frontier_states,
                    indices=selected_indices,
                    updates=accepted_transitioned_latents,
                )
                candidate_frontier_memory = _scatter_candidate_updates(
                    base=candidate_frontier_memory,
                    indices=selected_indices,
                    updates=accepted_frontier_memory,
                )
                delta_scores = torch.where(
                    accept_mask.unsqueeze(1),
                    delta_next,
                    torch.where(
                        rollback_mask.unsqueeze(1),
                        rollback_delta_scores,
                        delta_scores,
                    ),
                )
                active_mask = step_active_mask
                final_scores = (root_scores + delta_scores).masked_fill(
                    ~candidate_mask,
                    MASKED_CANDIDATE_SCORE_VALUE,
                )

                pv_scratch = _build_pv_scratch(
                    candidate_action_indices=candidate_action_indices,
                    candidate_scores=final_scores,
                    candidate_mask=candidate_mask,
                    candidate_uci=candidate_uci,
                )
                selected_candidates = [
                    selected_indices[batch_index].tolist()
                    if bool(step_active_mask[batch_index].item())
                    else []
                    for batch_index in range(z_root.shape[0])
                ]
                trace_steps.append(
                    DeliberationTraceStep(
                        step=step_index,
                        selected_candidates=selected_candidates,
                        top1_action_index=top1_actions.tolist(),
                        top1_value_cp=value_cp.tolist(),
                        sharpness=sharpness.tolist(),
                        uncertainty=uncertainty.tolist(),
                        frontier_turnover=frontier_turnover.tolist(),
                        frontier_revisit=frontier_revisit.tolist(),
                        frontier_stable=[bool(value) for value in frontier_stable_mask.tolist()],
                        pv_scratch_uci=pv_scratch,
                        rollback_fired=rollback_fired,
                    )
                )
                step_value_cp_tensors.append(value_cp.clone())
                step_sharpness_tensors.append(sharpness.clone())
                step_candidate_score_tensors.append(final_scores.clone())
                step_selected_candidate_tensors.append(selected_indices.clone())
                step_selected_candidate_masks.append(selected_mask.clone())
                if phase_idx is not None:
                    step_phase_indices.append(phase_idx.clone())
                step_active_masks.append(step_active_mask.clone())
                step_rollback_masks.append(rollback_mask.clone())
                step_frontier_turnover_tensors.append(frontier_turnover.clone())
                step_frontier_revisit_tensors.append(frontier_revisit.clone())
                step_frontier_stable_masks.append(frontier_stable_mask.clone())
                step_rollback_flags.append(rollback_fired)
                if reply_diagnostics is not None:
                    assert reply_diagnostics["student_reply_logits"] is not None
                    assert reply_diagnostics["student_pressure"] is not None
                    assert reply_diagnostics["student_uncertainty"] is not None
                    step_student_reply_logits_tensors.append(
                        reply_diagnostics["student_reply_logits"].clone()
                    )
                    step_student_pressure_tensors.append(
                        reply_diagnostics["student_pressure"].clone()
                    )
                    step_student_uncertainty_tensors.append(
                        reply_diagnostics["student_uncertainty"].clone()
                    )
                    if reply_diagnostics["teacher_reply_logits"] is not None:
                        step_teacher_reply_logits_tensors.append(
                            reply_diagnostics["teacher_reply_logits"].clone()
                        )
                        step_teacher_pressure_tensors.append(
                            reply_diagnostics["teacher_pressure"].clone()
                        )
                        step_teacher_uncertainty_tensors.append(
                            reply_diagnostics["teacher_uncertainty"].clone()
                        )

            final_scores = (root_scores + delta_scores).masked_fill(
                ~candidate_mask,
                MASKED_CANDIDATE_SCORE_VALUE,
            )
            final_top1_indices = torch.argmax(final_scores, dim=1)
            final_top1_actions = candidate_action_indices.gather(
                1,
                final_top1_indices.unsqueeze(1),
            ).squeeze(1)
            visited_candidate_mask = frontier_visit_counts > 0
            return {
                "final_candidate_scores": final_scores,
                "refined_top1_action_index": final_top1_actions,
                "trace": DeliberationTrace(steps=trace_steps),
                "step_count": len(trace_steps),
                "step_value_cp_tensors": tuple(step_value_cp_tensors),
                "step_sharpness_tensors": tuple(step_sharpness_tensors),
                "step_candidate_score_tensors": tuple(step_candidate_score_tensors),
                "step_selected_candidate_tensors": tuple(step_selected_candidate_tensors),
                "step_selected_candidate_masks": tuple(step_selected_candidate_masks),
                "step_phase_indices": tuple(step_phase_indices),
                "step_active_masks": tuple(step_active_masks),
                "step_rollback_masks": tuple(step_rollback_masks),
                "step_frontier_turnover_tensors": tuple(step_frontier_turnover_tensors),
                "step_frontier_revisit_tensors": tuple(step_frontier_revisit_tensors),
                "step_frontier_stable_masks": tuple(step_frontier_stable_masks),
                "step_rollback_flags": tuple(step_rollback_flags),
                "step_student_reply_logits_tensors": tuple(
                    step_student_reply_logits_tensors
                ),
                "step_student_pressure_tensors": tuple(step_student_pressure_tensors),
                "step_student_uncertainty_tensors": tuple(
                    step_student_uncertainty_tensors
                ),
                "step_teacher_reply_logits_tensors": tuple(
                    step_teacher_reply_logits_tensors
                ),
                "step_teacher_pressure_tensors": tuple(step_teacher_pressure_tensors),
                "step_teacher_uncertainty_tensors": tuple(
                    step_teacher_uncertainty_tensors
                ),
                "root_candidate_scores": root_scores,
                "final_candidate_deltas": delta_scores,
                "final_z": z_t,
                "final_memory": M_t,
                "candidate_frontier_states": candidate_frontier_states,
                "candidate_frontier_memory": candidate_frontier_memory,
                "frontier_state_drift": _masked_candidate_mean_norm(
                    tensor=candidate_frontier_states - z_root.unsqueeze(1),
                    mask=visited_candidate_mask,
                ),
                "frontier_memory_norm": _masked_candidate_mean_norm(
                    tensor=candidate_frontier_memory,
                    mask=visited_candidate_mask,
                ),
                "frontier_visit_counts": frontier_visit_counts,
                "frontier_unique_candidate_counts": frontier_visited_mask.sum(dim=1),
            }


else:  # pragma: no cover - exercised when torch is absent

    class DeliberationCell:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "PyTorch is required for DeliberationCell. Install the 'train' extra or torch."
            )


    class CandidateSelector:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "PyTorch is required for CandidateSelector. Install the 'train' extra or torch."
            )


    class LatentTransition:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "PyTorch is required for LatentTransition. Install the 'train' extra or torch."
            )


    class DeliberationLoop:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "PyTorch is required for DeliberationLoop. Install the 'train' extra or torch."
            )


def _scatter_candidate_updates(
    *,
    base: torch.Tensor,
    indices: torch.Tensor,
    updates: torch.Tensor,
) -> torch.Tensor:
    updated = base.clone()
    updated.scatter_(
        1,
        indices.unsqueeze(2).expand(-1, -1, base.shape[2]),
        updates,
    )
    return updated


def _masked_candidate_mean_norm(
    *,
    tensor: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    norms = torch.linalg.vector_norm(tensor, dim=2)
    masked_norms = norms * mask.to(norms.dtype)
    counts = mask.sum(dim=1).clamp(min=1).to(norms.dtype)
    return masked_norms.sum(dim=1) / counts


def _build_pv_scratch(
    *,
    candidate_action_indices: torch.Tensor,
    candidate_scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    candidate_uci: list[list[str]] | None,
) -> list[list[str]]:
    top_k = min(3, candidate_scores.shape[1])
    masked_scores = candidate_scores.masked_fill(
        ~candidate_mask,
        MASKED_CANDIDATE_SCORE_VALUE,
    )
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
