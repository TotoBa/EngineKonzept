//! Bounded frontier deliberation over exact legal root candidates.

use std::fmt;

pub const FRONTIER_CANDIDATE_FEATURE_DIM: usize = 18;

const IS_CAPTURE_INDEX: usize = 0;
const IS_PROMOTION_INDEX: usize = 1;
const GIVES_CHECK_INDEX: usize = 4;
const FROM_ATTACKED_BY_OPPONENT_INDEX: usize = 5;
const TO_ATTACKED_BY_OPPONENT_INDEX: usize = 6;
const IS_KING_MOVE_INDEX: usize = 14;
const CAPTURED_PIECE_PRESENT_INDEX: usize = 15;
const CAPTURED_PIECE_MINOR_OR_MAJOR_INDEX: usize = 17;

/// One exact legal candidate available to the frontier planner.
#[derive(Clone, Debug, PartialEq)]
pub struct PlannerCandidate {
    pub move_uci: String,
    pub action_index: u32,
    pub policy_score: f32,
    pub features: [f32; FRONTIER_CANDIDATE_FEATURE_DIM],
}

/// Runtime-configurable bounded budget for root-only deliberation.
#[derive(Clone, Debug, PartialEq)]
pub struct FrontierPlannerConfig {
    pub root_top_k: usize,
    pub beam_width: usize,
    pub min_inner_steps: usize,
    pub max_inner_steps: usize,
    pub stable_margin: f32,
    pub tactical_pressure_scale: f32,
    pub exploration_scale: f32,
    pub revisit_penalty_scale: f32,
    pub stability_hysteresis_steps: usize,
}

impl Default for FrontierPlannerConfig {
    fn default() -> Self {
        Self {
            root_top_k: 6,
            beam_width: 3,
            min_inner_steps: 1,
            max_inner_steps: 4,
            stable_margin: 0.35,
            tactical_pressure_scale: 0.30,
            exploration_scale: 0.25,
            revisit_penalty_scale: 0.10,
            stability_hysteresis_steps: 1,
        }
    }
}

impl FrontierPlannerConfig {
    pub fn validate(&self) -> Result<(), PlannerError> {
        if self.root_top_k == 0 {
            return Err(PlannerError::InvalidConfig(
                "root_top_k must be positive".to_string(),
            ));
        }
        if self.beam_width == 0 {
            return Err(PlannerError::InvalidConfig(
                "beam_width must be positive".to_string(),
            ));
        }
        if self.min_inner_steps > self.max_inner_steps {
            return Err(PlannerError::InvalidConfig(
                "min_inner_steps must not exceed max_inner_steps".to_string(),
            ));
        }
        if self.stable_margin < 0.0 {
            return Err(PlannerError::InvalidConfig(
                "stable_margin must be non-negative".to_string(),
            ));
        }
        if self.tactical_pressure_scale < 0.0 {
            return Err(PlannerError::InvalidConfig(
                "tactical_pressure_scale must be non-negative".to_string(),
            ));
        }
        if self.exploration_scale < 0.0 {
            return Err(PlannerError::InvalidConfig(
                "exploration_scale must be non-negative".to_string(),
            ));
        }
        if self.revisit_penalty_scale < 0.0 {
            return Err(PlannerError::InvalidConfig(
                "revisit_penalty_scale must be non-negative".to_string(),
            ));
        }
        if self.stability_hysteresis_steps == 0 {
            return Err(PlannerError::InvalidConfig(
                "stability_hysteresis_steps must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

/// One deliberation step in the bounded frontier planner.
#[derive(Clone, Debug, PartialEq)]
pub struct FrontierPlannerTraceStep {
    pub step_index: usize,
    pub frontier_moves: Vec<String>,
    pub selected_move: String,
    pub selected_score: f32,
    pub selected_pressure: f32,
    pub leader_margin: f32,
    pub uncertainty: f32,
    pub halted_after_step: bool,
}

/// Final root-only planner decision.
#[derive(Clone, Debug, PartialEq)]
pub struct FrontierPlannerDecision {
    pub bestmove: String,
    pub action_index: u32,
    pub legal_candidate_count: usize,
    pub considered_candidate_count: usize,
    pub applied_inner_steps: usize,
    pub halted_early: bool,
    pub root_margin: f32,
    pub final_uncertainty: f32,
    pub trace: Vec<FrontierPlannerTraceStep>,
}

impl FrontierPlannerDecision {
    #[must_use]
    pub fn info_summary(&self) -> String {
        format!(
            "planner mode=frontier bestmove={} legal_candidates={} considered_candidates={} steps={} halted_early={} root_margin={:.4} uncertainty={:.4}",
            self.bestmove,
            self.legal_candidate_count,
            self.considered_candidate_count,
            self.applied_inner_steps,
            self.halted_early,
            self.root_margin,
            self.final_uncertainty,
        )
    }

    #[must_use]
    pub fn info_trace_strings(&self) -> Vec<String> {
        self.trace
            .iter()
            .map(|step| {
                format!(
                    "planner step={} selected={} frontier={} score={:.4} pressure={:.4} margin={:.4} uncertainty={:.4} halted={}",
                    step.step_index + 1,
                    step.selected_move,
                    step.frontier_moves.join(","),
                    step.selected_score,
                    step.selected_pressure,
                    step.leader_margin,
                    step.uncertainty,
                    step.halted_after_step,
                )
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq)]
struct CandidateState {
    candidate: PlannerCandidate,
    visit_count: usize,
    last_selected_step: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
struct StepCandidateScore {
    index: usize,
    score: f32,
    pressure: f32,
}

/// Planner errors for invalid budgets or empty legal sets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlannerError {
    EmptyCandidateSet,
    InvalidConfig(String),
}

impl fmt::Display for PlannerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCandidateSet => f.write_str("frontier planner received no legal candidates"),
            Self::InvalidConfig(message) => write!(f, "invalid frontier planner config: {message}"),
        }
    }
}

impl std::error::Error for PlannerError {}

/// Choose one legal root move using bounded frontier deliberation.
pub fn choose_move(
    candidates: &[PlannerCandidate],
    config: &FrontierPlannerConfig,
) -> Result<FrontierPlannerDecision, PlannerError> {
    config.validate()?;
    if candidates.is_empty() {
        return Err(PlannerError::EmptyCandidateSet);
    }

    let mut ranked = candidates.to_vec();
    ranked.sort_by(|left, right| {
        right
            .policy_score
            .total_cmp(&left.policy_score)
            .then_with(|| left.action_index.cmp(&right.action_index))
    });

    let legal_candidate_count = ranked.len();
    let considered_candidate_count = ranked.len().min(config.root_top_k);
    let considered = &ranked[..considered_candidate_count];

    let root_margin = margin_from_sorted_scores(
        &considered
            .iter()
            .map(|candidate| candidate.policy_score)
            .collect::<Vec<_>>(),
    );

    if considered_candidate_count == 1 || config.max_inner_steps == 0 {
        let best = &considered[0];
        return Ok(FrontierPlannerDecision {
            bestmove: best.move_uci.clone(),
            action_index: best.action_index,
            legal_candidate_count,
            considered_candidate_count,
            applied_inner_steps: 1,
            halted_early: true,
            root_margin,
            final_uncertainty: uncertainty_from_margin(root_margin),
            trace: vec![FrontierPlannerTraceStep {
                step_index: 0,
                frontier_moves: vec![best.move_uci.clone()],
                selected_move: best.move_uci.clone(),
                selected_score: best.policy_score,
                selected_pressure: tactical_pressure(&best.features),
                leader_margin: root_margin,
                uncertainty: uncertainty_from_margin(root_margin),
                halted_after_step: true,
            }],
        });
    }

    let mut states: Vec<CandidateState> = considered
        .iter()
        .cloned()
        .map(|candidate| CandidateState {
            candidate,
            visit_count: 0,
            last_selected_step: None,
        })
        .collect();

    let mut trace = Vec::new();
    let mut selected_index = 0;
    let mut last_selected_index: Option<usize> = None;
    let mut stable_leader_steps = 0_usize;
    let mut halted_early = false;
    let target_max_steps = adaptive_target_max_steps(root_margin, config);

    for step_index in 0..target_max_steps {
        let step_scores = score_frontier_step(&states, config, step_index);
        let leader_margin =
            margin_from_sorted_scores(&step_scores.iter().map(|row| row.score).collect::<Vec<_>>());
        let uncertainty = uncertainty_from_margin(leader_margin);
        let frontier_moves = step_scores
            .iter()
            .take(config.beam_width.min(step_scores.len()))
            .map(|row| states[row.index].candidate.move_uci.clone())
            .collect::<Vec<_>>();

        let leader = &step_scores[0];
        selected_index = leader.index;
        states[selected_index].visit_count += 1;
        states[selected_index].last_selected_step = Some(step_index);

        if last_selected_index == Some(selected_index) && leader_margin >= config.stable_margin {
            stable_leader_steps += 1;
        } else if leader_margin >= config.stable_margin {
            stable_leader_steps = 1;
        } else {
            stable_leader_steps = 0;
        }

        let haltable = step_index + 1 >= config.min_inner_steps;
        let halted_after_step = haltable
            && stable_leader_steps >= config.stability_hysteresis_steps
            && uncertainty <= uncertainty_from_margin(config.stable_margin);

        trace.push(FrontierPlannerTraceStep {
            step_index,
            frontier_moves,
            selected_move: states[selected_index].candidate.move_uci.clone(),
            selected_score: leader.score,
            selected_pressure: leader.pressure,
            leader_margin,
            uncertainty,
            halted_after_step,
        });

        last_selected_index = Some(selected_index);
        if halted_after_step {
            halted_early = true;
            break;
        }
    }

    let selected = &states[selected_index].candidate;
    let final_uncertainty = trace
        .last()
        .map(|step| step.uncertainty)
        .unwrap_or_else(|| uncertainty_from_margin(root_margin));

    Ok(FrontierPlannerDecision {
        bestmove: selected.move_uci.clone(),
        action_index: selected.action_index,
        legal_candidate_count,
        considered_candidate_count,
        applied_inner_steps: trace.len(),
        halted_early,
        root_margin,
        final_uncertainty,
        trace,
    })
}

fn adaptive_target_max_steps(root_margin: f32, config: &FrontierPlannerConfig) -> usize {
    if config.max_inner_steps <= config.min_inner_steps {
        return config.max_inner_steps;
    }
    if root_margin >= config.stable_margin * 2.0 {
        return config.min_inner_steps.max(1);
    }
    if root_margin >= config.stable_margin {
        return (config.min_inner_steps + 1).min(config.max_inner_steps);
    }
    config.max_inner_steps
}

fn score_frontier_step(
    states: &[CandidateState],
    config: &FrontierPlannerConfig,
    step_index: usize,
) -> Vec<StepCandidateScore> {
    let root_scores = states
        .iter()
        .map(|state| state.candidate.policy_score)
        .collect::<Vec<_>>();
    let root_margin = margin_from_sorted_scores(&root_scores);
    let uncertainty = uncertainty_from_margin(root_margin);

    let mut rows = states
        .iter()
        .enumerate()
        .map(|(index, state)| {
            let pressure = tactical_pressure(&state.candidate.features);
            let exploration_bonus =
                config.exploration_scale * uncertainty / (1.0 + state.visit_count as f32);
            let revisit_penalty = state
                .last_selected_step
                .filter(|last_selected_step| step_index.saturating_sub(*last_selected_step) <= 1)
                .map(|_| config.revisit_penalty_scale * (1.0 - uncertainty))
                .unwrap_or(0.0);
            let score = state.candidate.policy_score
                + (config.tactical_pressure_scale * pressure)
                + exploration_bonus
                - revisit_penalty;
            StepCandidateScore {
                index,
                score,
                pressure,
            }
        })
        .collect::<Vec<_>>();

    rows.sort_by(|left, right| {
        right.score.total_cmp(&left.score).then_with(|| {
            states[left.index]
                .candidate
                .action_index
                .cmp(&states[right.index].candidate.action_index)
        })
    });
    rows
}

fn margin_from_sorted_scores(scores: &[f32]) -> f32 {
    if scores.len() < 2 {
        return f32::INFINITY;
    }
    let mut values = scores.to_vec();
    values.sort_by(|left, right| right.total_cmp(left));
    values[0] - values[1]
}

fn uncertainty_from_margin(margin: f32) -> f32 {
    if !margin.is_finite() {
        return 0.0;
    }
    let clamped = margin.clamp(-8.0, 8.0);
    1.0 / (1.0 + clamped.exp())
}

fn tactical_pressure(features: &[f32; FRONTIER_CANDIDATE_FEATURE_DIM]) -> f32 {
    (0.45 * features[IS_CAPTURE_INDEX])
        + (0.80 * features[IS_PROMOTION_INDEX])
        + (0.90 * features[GIVES_CHECK_INDEX])
        + (0.20 * features[FROM_ATTACKED_BY_OPPONENT_INDEX])
        + (0.20 * features[TO_ATTACKED_BY_OPPONENT_INDEX])
        + (0.10 * features[IS_KING_MOVE_INDEX])
        + (0.15 * features[CAPTURED_PIECE_PRESENT_INDEX])
        + (0.35 * features[CAPTURED_PIECE_MINOR_OR_MAJOR_INDEX])
}

#[cfg(test)]
mod tests {
    use super::{
        choose_move, tactical_pressure, FrontierPlannerConfig, PlannerCandidate,
        FRONTIER_CANDIDATE_FEATURE_DIM,
    };

    fn candidate(
        move_uci: &str,
        action_index: u32,
        policy_score: f32,
        features: [f32; FRONTIER_CANDIDATE_FEATURE_DIM],
    ) -> PlannerCandidate {
        PlannerCandidate {
            move_uci: move_uci.to_string(),
            action_index,
            policy_score,
            features,
        }
    }

    #[test]
    fn clear_leader_halts_after_one_step() {
        let config = FrontierPlannerConfig {
            root_top_k: 3,
            beam_width: 2,
            min_inner_steps: 1,
            max_inner_steps: 4,
            stability_hysteresis_steps: 1,
            ..FrontierPlannerConfig::default()
        };
        let decision = choose_move(
            &[
                candidate("e2e4", 1, 2.0, [0.0; FRONTIER_CANDIDATE_FEATURE_DIM]),
                candidate("d2d4", 2, 0.5, [0.0; FRONTIER_CANDIDATE_FEATURE_DIM]),
                candidate("g1f3", 3, 0.1, [0.0; FRONTIER_CANDIDATE_FEATURE_DIM]),
            ],
            &config,
        )
        .expect("planner succeeds");

        assert_eq!(decision.bestmove, "e2e4");
        assert_eq!(decision.applied_inner_steps, 1);
        assert!(decision.halted_early);
    }

    #[test]
    fn close_race_uses_full_budget() {
        let config = FrontierPlannerConfig {
            root_top_k: 3,
            beam_width: 2,
            min_inner_steps: 1,
            max_inner_steps: 4,
            stable_margin: 1.0,
            ..FrontierPlannerConfig::default()
        };
        let decision = choose_move(
            &[
                candidate("e2e4", 1, 0.50, [0.0; FRONTIER_CANDIDATE_FEATURE_DIM]),
                candidate("d2d4", 2, 0.49, [0.0; FRONTIER_CANDIDATE_FEATURE_DIM]),
                candidate("g1f3", 3, 0.48, [0.0; FRONTIER_CANDIDATE_FEATURE_DIM]),
            ],
            &config,
        )
        .expect("planner succeeds");

        assert_eq!(decision.applied_inner_steps, 4);
        assert!(!decision.trace.is_empty());
    }

    #[test]
    fn tactical_pressure_prefers_forcing_candidate_when_scores_are_close() {
        let config = FrontierPlannerConfig {
            root_top_k: 2,
            beam_width: 2,
            min_inner_steps: 1,
            max_inner_steps: 2,
            stability_hysteresis_steps: 1,
            ..FrontierPlannerConfig::default()
        };
        let mut forcing = [0.0; FRONTIER_CANDIDATE_FEATURE_DIM];
        forcing[0] = 1.0;
        forcing[4] = 1.0;
        forcing[17] = 1.0;
        let decision = choose_move(
            &[
                candidate("quiet", 1, 0.80, [0.0; FRONTIER_CANDIDATE_FEATURE_DIM]),
                candidate("forcing", 2, 0.72, forcing),
            ],
            &config,
        )
        .expect("planner succeeds");

        assert_eq!(decision.bestmove, "forcing");
        assert!(tactical_pressure(&forcing) > 0.0);
    }

    #[test]
    fn invalid_config_is_rejected() {
        let config = FrontierPlannerConfig {
            min_inner_steps: 3,
            max_inner_steps: 2,
            ..FrontierPlannerConfig::default()
        };
        let error = choose_move(
            &[candidate(
                "e2e4",
                1,
                1.0,
                [0.0; FRONTIER_CANDIDATE_FEATURE_DIM],
            )],
            &config,
        )
        .expect_err("invalid config must fail");
        assert!(error.to_string().contains("min_inner_steps"));
    }
}
