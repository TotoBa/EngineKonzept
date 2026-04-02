//! Minimal UCI engine loop for Phase 2.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use std::io::{self, BufRead, Write};

use inference::build_symbolic_proposer_inputs;
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
            UciCommand::Go { args } => SessionOutcome {
                responses: vec![UciResponse::BestMove {
                    bestmove: self.stub_bestmove(&args),
                    ponder: None,
                }],
                quit: false,
            },
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

    fn stub_bestmove(&self, go_args: &[String]) -> String {
        let searchmoves = searchmoves_from_args(go_args);
        build_symbolic_proposer_inputs(&self.current_position)
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
            .unwrap_or_else(|| "0000".to_string())
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

    use rules::legal_moves;

    use super::{run_uci_loop, EngineSession};

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
}
