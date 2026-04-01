//! UCI protocol command and response primitives used by the engine runtime.

use std::error::Error;
use std::fmt;

/// Parsed `position` source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PositionSpec {
    Startpos,
    Fen(String),
}

/// Parsed UCI command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UciCommand {
    Uci,
    Debug(bool),
    IsReady,
    UciNewGame,
    Position {
        position: PositionSpec,
        moves: Vec<String>,
    },
    Go {
        args: Vec<String>,
    },
    Stop,
    Quit,
}

/// UCI response line formatter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UciResponse {
    IdName(String),
    IdAuthor(String),
    UciOk,
    ReadyOk,
    BestMove {
        bestmove: String,
        ponder: Option<String>,
    },
    InfoString(String),
}

impl fmt::Display for UciResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IdName(name) => write!(f, "id name {name}"),
            Self::IdAuthor(author) => write!(f, "id author {author}"),
            Self::UciOk => f.write_str("uciok"),
            Self::ReadyOk => f.write_str("readyok"),
            Self::BestMove { bestmove, ponder } => match ponder {
                Some(ponder) => write!(f, "bestmove {bestmove} ponder {ponder}"),
                None => write!(f, "bestmove {bestmove}"),
            },
            Self::InfoString(message) => write!(f, "info string {message}"),
        }
    }
}

/// UCI parser error for malformed or unsupported commands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParseError {
    UnknownCommand(String),
    MissingArgument(&'static str),
    InvalidArgument {
        command: &'static str,
        detail: String,
    },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownCommand(command) => write!(f, "unknown command '{command}'"),
            Self::MissingArgument(argument) => write!(f, "missing argument: {argument}"),
            Self::InvalidArgument { command, detail } => {
                write!(f, "invalid {command} argument: {detail}")
            }
        }
    }
}

impl Error for ParseError {}

/// Parses a single UCI input line.
pub fn parse_command(input: &str) -> Result<Option<UciCommand>, ParseError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    let tokens: Vec<&str> = trimmed.split_whitespace().collect();
    match tokens[0] {
        "uci" => Ok(Some(UciCommand::Uci)),
        "debug" => parse_debug(&tokens).map(Some),
        "isready" => Ok(Some(UciCommand::IsReady)),
        "ucinewgame" => Ok(Some(UciCommand::UciNewGame)),
        "position" => parse_position(&tokens).map(Some),
        "go" => Ok(Some(UciCommand::Go {
            args: tokens[1..]
                .iter()
                .map(|token| (*token).to_string())
                .collect(),
        })),
        "stop" => Ok(Some(UciCommand::Stop)),
        "quit" => Ok(Some(UciCommand::Quit)),
        unknown => Err(ParseError::UnknownCommand(unknown.to_string())),
    }
}

fn parse_debug(tokens: &[&str]) -> Result<UciCommand, ParseError> {
    let value = tokens
        .get(1)
        .copied()
        .ok_or(ParseError::MissingArgument("debug on|off"))?;
    match value {
        "on" => Ok(UciCommand::Debug(true)),
        "off" => Ok(UciCommand::Debug(false)),
        other => Err(ParseError::InvalidArgument {
            command: "debug",
            detail: other.to_string(),
        }),
    }
}

fn parse_position(tokens: &[&str]) -> Result<UciCommand, ParseError> {
    let selector = tokens
        .get(1)
        .copied()
        .ok_or(ParseError::MissingArgument("position startpos|fen ..."))?;

    let (position, next_index) = match selector {
        "startpos" => (PositionSpec::Startpos, 2),
        "fen" => {
            let moves_index = tokens.iter().position(|token| *token == "moves");
            let fen_end = moves_index.unwrap_or(tokens.len());
            let fen_tokens = &tokens[2..fen_end];
            if fen_tokens.is_empty() {
                return Err(ParseError::MissingArgument("position fen <fen>"));
            }
            (PositionSpec::Fen(fen_tokens.join(" ")), fen_end)
        }
        other => {
            return Err(ParseError::InvalidArgument {
                command: "position",
                detail: other.to_string(),
            });
        }
    };

    let moves = if next_index < tokens.len() {
        if tokens[next_index] != "moves" {
            return Err(ParseError::InvalidArgument {
                command: "position",
                detail: tokens[next_index].to_string(),
            });
        }
        tokens[next_index + 1..]
            .iter()
            .map(|token| (*token).to_string())
            .collect()
    } else {
        Vec::new()
    };

    Ok(UciCommand::Position { position, moves })
}

#[cfg(test)]
mod tests {
    use super::{parse_command, PositionSpec, UciCommand, UciResponse};

    #[test]
    fn parses_position_startpos_with_moves() {
        let command = parse_command("position startpos moves e2e4 e7e5")
            .expect("parse succeeds")
            .expect("command is present");
        assert_eq!(
            command,
            UciCommand::Position {
                position: PositionSpec::Startpos,
                moves: vec!["e2e4".to_string(), "e7e5".to_string()],
            }
        );
    }

    #[test]
    fn parses_position_fen_with_moves() {
        let command = parse_command(
            "position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 moves g1f3",
        )
        .expect("parse succeeds")
        .expect("command is present");
        assert_eq!(
            command,
            UciCommand::Position {
                position: PositionSpec::Fen(
                    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string()
                ),
                moves: vec!["g1f3".to_string()],
            }
        );
    }

    #[test]
    fn parses_go_and_debug_commands() {
        let go = parse_command("go wtime 1000 btime 1000")
            .expect("parse succeeds")
            .expect("command is present");
        assert_eq!(
            go,
            UciCommand::Go {
                args: vec![
                    "wtime".to_string(),
                    "1000".to_string(),
                    "btime".to_string(),
                    "1000".to_string()
                ],
            }
        );

        let debug = parse_command("debug on")
            .expect("parse succeeds")
            .expect("command is present");
        assert_eq!(debug, UciCommand::Debug(true));
    }

    #[test]
    fn formats_bestmove_response() {
        let response = UciResponse::BestMove {
            bestmove: "e2e4".to_string(),
            ponder: None,
        };
        assert_eq!(response.to_string(), "bestmove e2e4");
    }
}
