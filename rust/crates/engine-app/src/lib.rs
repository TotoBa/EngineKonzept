//! UCI engine loop with exact legality and symbolic proposer scoring.

use std::collections::BTreeSet;
use std::env;
use std::error::Error;
use std::fmt;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use inference::{
    build_symbolic_proposer_inputs, load_symbolic_proposer_runtime, SymbolicProposerRuntime,
};
use planner::{choose_move, FrontierPlannerConfig, PlannerCandidate};
use position::Position;
use rules::{apply_move, legal_moves, MoveError};
use uci_protocol::{parse_command, ParseError, PositionSpec, UciCommand, UciResponse};

pub const ENGINE_NAME: &str = "EngineKonzept";
pub const ENGINE_AUTHOR: &str = "TotoBa";

const GO_OPTION_KEYWORDS: &[&str] = &[
    "searchmoves",
    "ponder",
    "wtime",
    "btime",
    "winc",
    "binc",
    "movestogo",
    "depth",
    "nodes",
    "mate",
    "movetime",
    "infinite",
];

/// Session result for a single input line.
#[derive(Debug, Default)]
pub struct SessionOutcome {
    pub responses: Vec<UciResponse>,
    pub quit: bool,
}

/// Mutable engine state for one UCI session.
#[derive(Debug, Clone)]
pub struct EngineSession {
    current_position: Position,
    debug_enabled: bool,
    proposer_runtime: Option<Arc<SymbolicProposerRuntime>>,
    planner_config: FrontierPlannerConfig,
}

impl Default for EngineSession {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineSession {
    #[must_use]
    pub fn new() -> Self {
        Self {
            current_position: Position::startpos(),
            debug_enabled: false,
            proposer_runtime: default_proposer_runtime().map(Arc::new),
            planner_config: default_frontier_planner_config(),
        }
    }

    #[must_use]
    pub fn with_proposer_runtime(proposer_runtime: Option<Arc<SymbolicProposerRuntime>>) -> Self {
        Self::with_runtime_components(proposer_runtime, default_frontier_planner_config())
    }

    #[must_use]
    pub fn with_runtime_components(
        proposer_runtime: Option<Arc<SymbolicProposerRuntime>>,
        planner_config: FrontierPlannerConfig,
    ) -> Self {
        Self {
            current_position: Position::startpos(),
            debug_enabled: false,
            proposer_runtime,
            planner_config,
        }
    }

    #[must_use]
    pub fn current_position(&self) -> &Position {
        &self.current_position
    }

    #[must_use]
    pub const fn debug_enabled(&self) -> bool {
        self.debug_enabled
    }

    pub fn handle_line(&mut self, line: &str) -> SessionOutcome {
        match parse_command(line) {
            Ok(Some(command)) => self.handle_command(command),
            Ok(None) => SessionOutcome::default(),
            Err(error) => self.handle_error(error),
        }
    }

    fn handle_command(&mut self, command: UciCommand) -> SessionOutcome {
        match command {
            UciCommand::Uci => SessionOutcome {
                responses: vec![
                    UciResponse::IdName(ENGINE_NAME.to_string()),
                    UciResponse::IdAuthor(ENGINE_AUTHOR.to_string()),
                    UciResponse::UciOk,
                ],
                quit: false,
            },
            UciCommand::Debug(enabled) => {
                self.debug_enabled = enabled;
                SessionOutcome::default()
            }
            UciCommand::IsReady => SessionOutcome {
                responses: vec![UciResponse::ReadyOk],
                quit: false,
            },
            UciCommand::UciNewGame => {
                self.current_position = Position::startpos();
                SessionOutcome::default()
            }
            UciCommand::Position { position, moves } => self.handle_position(position, &moves),
            UciCommand::Go { args } => {
                let result = self.select_bestmove(&args);
                let mut responses = result
                    .info_strings
                    .into_iter()
                    .map(UciResponse::InfoString)
                    .collect::<Vec<_>>();
                responses.push(UciResponse::BestMove {
                    bestmove: result.bestmove,
                    ponder: None,
                });
                SessionOutcome {
                    responses,
                    quit: false,
                }
            }
            UciCommand::Stop => SessionOutcome::default(),
            UciCommand::Quit => SessionOutcome {
                responses: Vec::new(),
                quit: true,
            },
        }
    }

    fn handle_position(&mut self, position: PositionSpec, moves: &[String]) -> SessionOutcome {
        match reconstruct_position(position, moves) {
            Ok(position) => {
                self.current_position = position;
                SessionOutcome::default()
            }
            Err(error) => self.handle_error(error),
        }
    }

    fn select_bestmove(&self, go_args: &[String]) -> SelectedMoveResult {
        let searchmoves = searchmoves_from_args(go_args);
        if let Some(runtime) = &self.proposer_runtime {
            if let Ok(scored) = runtime.score_position(&self.current_position) {
                if let Some(result) =
                    self.select_planner_bestmove_from_scored(scored, searchmoves.as_ref(), go_args)
                {
                    return result;
                }
            }
        }
        let bestmove = build_symbolic_proposer_inputs(&self.current_position)
            .map(|inputs| inputs.candidates)
            .unwrap_or_else(|_| {
                legal_moves(&self.current_position)
                    .into_iter()
                    .map(|candidate| inference::SymbolicProposerCandidate {
                        chess_move: candidate,
                        move_uci: candidate.to_uci(),
                        action_index: 0,
                        features: [0.0; inference::SYMBOLIC_CANDIDATE_FEATURE_DIM],
                    })
                    .collect()
            })
            .into_iter()
            .map(|candidate| candidate.move_uci)
            .filter(|candidate| match &searchmoves {
                Some(searchmoves) => searchmoves.contains(candidate),
                None => true,
            })
            .min()
            .unwrap_or_else(|| "0000".to_string());
        SelectedMoveResult {
            bestmove,
            info_strings: Vec::new(),
        }
    }

    fn select_planner_bestmove_from_scored(
        &self,
        scored: Vec<inference::SymbolicScoredCandidate>,
        searchmoves: Option<&BTreeSet<String>>,
        go_args: &[String],
    ) -> Option<SelectedMoveResult> {
        let candidates = scored
            .into_iter()
            .filter(|candidate| match searchmoves {
                Some(searchmoves) => searchmoves.contains(&candidate.candidate.move_uci),
                None => true,
            })
            .map(|candidate| PlannerCandidate {
                move_uci: candidate.candidate.move_uci,
                action_index: candidate.candidate.action_index,
                policy_score: candidate.policy_score,
                features: candidate.candidate.features,
            })
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return None;
        }

        let config = planner_config_for_go_args(&self.planner_config, go_args);
        let decision = choose_move(&candidates, &config).ok()?;
        let mut info_strings = vec![decision.info_summary()];
        if self.debug_enabled {
            info_strings.extend(decision.info_trace_strings());
        }

        Some(SelectedMoveResult {
            bestmove: decision.bestmove,
            info_strings,
        })
    }

    fn handle_error(&self, error: impl fmt::Display) -> SessionOutcome {
        if self.debug_enabled {
            SessionOutcome {
                responses: vec![UciResponse::InfoString(error.to_string())],
                quit: false,
            }
        } else {
            SessionOutcome::default()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SelectedMoveResult {
    bestmove: String,
    info_strings: Vec<String>,
}

/// Runs the UCI loop on arbitrary buffered I/O.
pub fn run_uci_loop<R: BufRead, W: Write>(reader: &mut R, writer: &mut W) -> io::Result<()> {
    let mut session = EngineSession::new();

    for line in reader.lines() {
        let outcome = session.handle_line(&line?);
        for response in outcome.responses {
            writeln!(writer, "{response}")?;
        }
        writer.flush()?;
        if outcome.quit {
            break;
        }
    }

    Ok(())
}

/// Runs the standard-input to standard-output UCI loop.
pub fn run_stdio() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = stdin.lock();
    let mut writer = stdout.lock();
    run_uci_loop(&mut reader, &mut writer)
}

fn reconstruct_position(
    position: PositionSpec,
    moves: &[String],
) -> Result<Position, SessionError> {
    let mut current = match position {
        PositionSpec::Startpos => Position::startpos(),
        PositionSpec::Fen(fen) => {
            Position::from_fen(&normalize_fen(&fen)).map_err(SessionError::InvalidFen)?
        }
    };

    for move_uci in moves {
        let candidate = legal_moves(&current)
            .into_iter()
            .find(|candidate| candidate.to_uci() == *move_uci)
            .ok_or_else(|| SessionError::IllegalMove(move_uci.clone()))?;
        current = apply_move(&current, candidate).map_err(SessionError::ApplyMove)?;
    }

    Ok(current)
}

fn normalize_fen(fen: &str) -> String {
    match fen.split_whitespace().count() {
        4 => format!("{fen} 0 1"),
        _ => fen.to_string(),
    }
}

fn searchmoves_from_args(go_args: &[String]) -> Option<BTreeSet<String>> {
    let searchmoves_start = go_args.iter().position(|token| token == "searchmoves")?;
    let searchmoves: BTreeSet<String> = go_args[searchmoves_start + 1..]
        .iter()
        .take_while(|token| !is_go_option_keyword(token))
        .cloned()
        .collect();

    if searchmoves.is_empty() {
        None
    } else {
        Some(searchmoves)
    }
}

fn is_go_option_keyword(token: &str) -> bool {
    GO_OPTION_KEYWORDS.contains(&token)
}

fn planner_config_for_go_args(
    base: &FrontierPlannerConfig,
    go_args: &[String],
) -> FrontierPlannerConfig {
    let mut config = base.clone();

    if let Some(depth_cap) =
        go_option_value(go_args, "depth").and_then(|value| value.parse::<usize>().ok())
    {
        if depth_cap > 0 {
            config.max_inner_steps = config.max_inner_steps.min(depth_cap);
        }
    }
    if let Some(movetime_ms) =
        go_option_value(go_args, "movetime").and_then(|value| value.parse::<u64>().ok())
    {
        let budget_cap = match movetime_ms {
            0..=25 => 1,
            26..=75 => 2,
            76..=150 => 3,
            _ => config.max_inner_steps,
        };
        config.max_inner_steps = config.max_inner_steps.min(budget_cap);
    }
    config.min_inner_steps = config.min_inner_steps.min(config.max_inner_steps);
    if config.max_inner_steps == 0 {
        config.max_inner_steps = 1;
        config.min_inner_steps = 1;
    }
    config
}

fn go_option_value<'a>(go_args: &'a [String], option_name: &str) -> Option<&'a str> {
    let option_index = go_args.iter().position(|token| token == option_name)?;
    go_args.get(option_index + 1).map(String::as_str)
}

fn default_frontier_planner_config() -> FrontierPlannerConfig {
    let mut config = FrontierPlannerConfig::default();
    if let Some(root_top_k) = env_usize("ENGINEKONZEPT_FRONTIER_ROOT_TOP_K") {
        config.root_top_k = root_top_k.max(1);
    }
    if let Some(beam_width) = env_usize("ENGINEKONZEPT_FRONTIER_BEAM_WIDTH") {
        config.beam_width = beam_width.max(1);
    }
    if let Some(min_inner_steps) = env_usize("ENGINEKONZEPT_FRONTIER_MIN_INNER_STEPS") {
        config.min_inner_steps = min_inner_steps;
    }
    if let Some(max_inner_steps) = env_usize("ENGINEKONZEPT_FRONTIER_MAX_INNER_STEPS") {
        config.max_inner_steps = max_inner_steps.max(1);
    }
    if let Some(stable_margin) = env_f32("ENGINEKONZEPT_FRONTIER_STABLE_MARGIN") {
        config.stable_margin = stable_margin.max(0.0);
    }
    if let Some(scale) = env_f32("ENGINEKONZEPT_FRONTIER_TACTICAL_PRESSURE_SCALE") {
        config.tactical_pressure_scale = scale.max(0.0);
    }
    if let Some(scale) = env_f32("ENGINEKONZEPT_FRONTIER_EXPLORATION_SCALE") {
        config.exploration_scale = scale.max(0.0);
    }
    if let Some(scale) = env_f32("ENGINEKONZEPT_FRONTIER_REVISIT_PENALTY_SCALE") {
        config.revisit_penalty_scale = scale.max(0.0);
    }
    if let Some(steps) = env_usize("ENGINEKONZEPT_FRONTIER_STABILITY_HYSTERESIS") {
        config.stability_hysteresis_steps = steps.max(1);
    }
    if config.validate().is_err() {
        FrontierPlannerConfig::default()
    } else {
        config
    }
}

fn env_usize(name: &str) -> Option<usize> {
    env::var(name).ok()?.parse::<usize>().ok()
}

fn env_f32(name: &str) -> Option<f32> {
    env::var(name).ok()?.parse::<f32>().ok()
}

fn default_proposer_runtime() -> Option<SymbolicProposerRuntime> {
    proposer_bundle_candidates()
        .into_iter()
        .find(|path| path.is_dir())
        .and_then(|path| load_symbolic_proposer_runtime(path).ok())
}

fn proposer_bundle_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(bundle_dir) = env::var("ENGINEKONZEPT_PROPOSER_BUNDLE") {
        candidates.push(PathBuf::from(bundle_dir));
    }
    candidates.push(PathBuf::from(
        "models/proposer/stockfish_pgn_symbolic_v1_v1",
    ));
    candidates.push(repo_root_from_manifest().join("models/proposer/stockfish_pgn_symbolic_v1_v1"));
    candidates
}

fn repo_root_from_manifest() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.."))
}

/// Recoverable session error surfaced only through debug logging.
#[derive(Debug)]
pub enum SessionError {
    Parse(ParseError),
    InvalidFen(position::FenError),
    IllegalMove(String),
    ApplyMove(MoveError),
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(error) => write!(f, "{error}"),
            Self::InvalidFen(error) => write!(f, "invalid position FEN: {error}"),
            Self::IllegalMove(chess_move) => {
                write!(f, "illegal move in position command: {chess_move}")
            }
            Self::ApplyMove(error) => write!(f, "could not apply move: {error}"),
        }
    }
}

impl Error for SessionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Parse(error) => Some(error),
            Self::InvalidFen(error) => Some(error),
            Self::ApplyMove(error) => Some(error),
            Self::IllegalMove(_) => None,
        }
    }
}

impl From<ParseError> for SessionError {
    fn from(error: ParseError) -> Self {
        Self::Parse(error)
    }
}

#[cfg(test)]
mod tests {
    use std::io::BufReader;

    use inference::{SymbolicProposerCandidate, SymbolicScoredCandidate};
    use rules::legal_moves;

    use super::{run_uci_loop, EngineSession, FrontierPlannerConfig};

    #[test]
    fn position_startpos_moves_reconstructs_expected_state() {
        let mut session = EngineSession::new();
        session.handle_line("position startpos moves e2e4 e7e5");
        assert_eq!(
            session.current_position().to_fen(),
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"
        );
    }

    #[test]
    fn position_fen_moves_reconstructs_expected_state() {
        let mut session = EngineSession::new();
        session.handle_line(
            "position fen rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2 moves g1f3",
        );
        assert_eq!(
            session.current_position().to_fen(),
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"
        );
    }

    #[test]
    fn smoke_session_emits_uci_handshake_readyok_and_legal_bestmove() {
        let input = b"uci\nisready\nposition startpos moves e2e4 e7e5\ngo\nquit\n";
        let mut output = Vec::new();
        run_uci_loop(&mut BufReader::new(&input[..]), &mut output).expect("loop succeeds");
        let output = String::from_utf8(output).expect("valid utf-8");

        assert!(output.contains("id name EngineKonzept"));
        assert!(output.contains("id author TotoBa"));
        assert!(output.contains("uciok"));
        assert!(output.contains("readyok"));

        let bestmove_line = output
            .lines()
            .find(|line| line.starts_with("bestmove "))
            .expect("bestmove output");
        let bestmove = bestmove_line.trim_start_matches("bestmove ");

        let mut session = EngineSession::new();
        session.handle_line("position startpos moves e2e4 e7e5");
        let legal: Vec<String> = legal_moves(session.current_position())
            .into_iter()
            .map(|candidate| candidate.to_uci())
            .collect();
        assert!(legal.iter().any(|candidate| candidate == bestmove));
    }

    #[test]
    fn go_searchmoves_restricts_stub_bestmove() {
        let input = b"position startpos\ngo searchmoves h2h3\nquit\n";
        let mut output = Vec::new();
        run_uci_loop(&mut BufReader::new(&input[..]), &mut output).expect("loop succeeds");
        let output = String::from_utf8(output).expect("valid utf-8");

        assert!(output.lines().any(|line| line == "bestmove h2h3"));
    }

    #[test]
    fn go_searchmoves_stops_before_other_go_options() {
        let input = b"position startpos\ngo depth 1 searchmoves h2h3 g1f3 movetime 50\nquit\n";
        let mut output = Vec::new();
        run_uci_loop(&mut BufReader::new(&input[..]), &mut output).expect("loop succeeds");
        let output = String::from_utf8(output).expect("valid utf-8");

        assert!(output.lines().any(|line| line == "bestmove g1f3"));
    }

    #[test]
    fn debug_mode_surfaces_position_errors_as_info_strings() {
        let mut session = EngineSession::new();
        session.handle_line("debug on");
        let outcome = session.handle_line("position startpos moves e2e5");
        assert_eq!(outcome.responses.len(), 1);
        assert!(outcome.responses[0].to_string().starts_with("info string "));
    }

    #[test]
    fn proposer_bundle_candidates_include_repo_models_path() {
        let candidates = super::proposer_bundle_candidates();
        assert!(candidates
            .iter()
            .any(|path| path.ends_with("models/proposer/stockfish_pgn_symbolic_v1_v1")));
    }

    #[test]
    fn planner_helper_respects_searchmoves_filter() {
        let position = position::Position::startpos();
        let legal = legal_moves(&position);
        let scored = legal
            .into_iter()
            .take(3)
            .enumerate()
            .map(|(index, chess_move)| SymbolicScoredCandidate {
                candidate: SymbolicProposerCandidate {
                    move_uci: chess_move.to_uci(),
                    action_index: index as u32,
                    chess_move,
                    features: [0.0; planner::FRONTIER_CANDIDATE_FEATURE_DIM],
                },
                policy_score: 10.0 - index as f32,
            })
            .collect::<Vec<_>>();
        let searchmoves = std::iter::once(scored[2].candidate.move_uci.clone()).collect();
        let session =
            EngineSession::with_runtime_components(None, FrontierPlannerConfig::default());

        let result = session
            .select_planner_bestmove_from_scored(scored.clone(), Some(&searchmoves), &[])
            .expect("planner chooses one filtered move");
        assert_eq!(result.bestmove, scored[2].candidate.move_uci);
    }

    #[test]
    fn planner_helper_emits_summary_and_trace_in_debug_mode() {
        let position = position::Position::startpos();
        let legal = legal_moves(&position);
        let scored = legal
            .into_iter()
            .take(2)
            .enumerate()
            .map(|(index, chess_move)| SymbolicScoredCandidate {
                candidate: SymbolicProposerCandidate {
                    move_uci: chess_move.to_uci(),
                    action_index: index as u32,
                    chess_move,
                    features: [0.0; planner::FRONTIER_CANDIDATE_FEATURE_DIM],
                },
                policy_score: if index == 0 { 0.55 } else { 0.50 },
            })
            .collect::<Vec<_>>();
        let mut session =
            EngineSession::with_runtime_components(None, FrontierPlannerConfig::default());
        session.debug_enabled = true;

        let result = session
            .select_planner_bestmove_from_scored(scored, None, &[])
            .expect("planner returns move");
        assert!(result
            .info_strings
            .iter()
            .any(|line| line.contains("planner mode=frontier")));
        assert!(result
            .info_strings
            .iter()
            .any(|line| line.contains("planner step=")));
    }
}
