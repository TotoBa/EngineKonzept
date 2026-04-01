//! Dataset and developer tooling that reuses the exact symbolic rules kernel.

use std::error::Error;
use std::fmt;
use std::io::{self, BufRead, Write};

use action_space::{encode_move, legal_action_encodings, ActionEncodeError};
use core_types::MoveKind;
use encoder::encode_position;
use position::Position;
use rules::{apply_move, is_in_check, legal_moves, MoveError};
use serde::{Deserialize, Serialize};

/// JSON input accepted by the dataset oracle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetOracleInput {
    pub fen: String,
    #[serde(default)]
    pub selected_move_uci: Option<String>,
}

/// JSON output emitted by the dataset oracle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetOracleOutput {
    pub fen: String,
    pub side_to_move: String,
    pub legal_moves: Vec<String>,
    pub legal_action_encodings: Vec<[u32; 3]>,
    pub selected_move_uci: Option<String>,
    pub selected_action_encoding: Option<[u32; 3]>,
    pub next_fen: Option<String>,
    pub position_encoding: PositionEncodingOutput,
    pub annotations: TacticalAnnotationsOutput,
}

/// Model-facing encoded position exported in a JSON-friendly structure.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PositionEncodingOutput {
    pub piece_tokens: Vec<[u32; 3]>,
    pub square_tokens: Vec<[u32; 2]>,
    pub rule_token: [u32; 6],
}

/// Exact and conservative tactical annotations for dataset reporting.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TacticalAnnotationsOutput {
    pub in_check: bool,
    pub is_checkmate: bool,
    pub is_stalemate: bool,
    pub has_legal_en_passant: bool,
    pub has_legal_castle: bool,
    pub has_legal_promotion: bool,
    pub is_low_material_endgame: bool,
    pub legal_move_count: u32,
    pub piece_count: u32,
    pub selected_move_is_capture: Option<bool>,
    pub selected_move_is_promotion: Option<bool>,
    pub selected_move_is_castle: Option<bool>,
    pub selected_move_is_en_passant: Option<bool>,
    pub selected_move_gives_check: Option<bool>,
}

/// Oracle errors surfaced to the Python dataset pipeline.
#[derive(Debug)]
pub enum DatasetOracleError {
    InvalidFen(position::FenError),
    InvalidSelectedMove(String),
    ActionEncoding(ActionEncodeError),
    MoveApplication(MoveError),
    Json(serde_json::Error),
}

impl fmt::Display for DatasetOracleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFen(error) => write!(f, "invalid FEN: {error}"),
            Self::InvalidSelectedMove(chess_move) => {
                write!(
                    f,
                    "selected move is not legal in this position: {chess_move}"
                )
            }
            Self::ActionEncoding(error) => write!(f, "action encoding failed: {error}"),
            Self::MoveApplication(error) => write!(f, "could not apply selected move: {error}"),
            Self::Json(error) => write!(f, "invalid JSON: {error}"),
        }
    }
}

impl Error for DatasetOracleError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidFen(error) => Some(error),
            Self::ActionEncoding(error) => Some(error),
            Self::MoveApplication(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::InvalidSelectedMove(_) => None,
        }
    }
}

/// Labels a dataset input record via the exact rules kernel.
pub fn label_dataset_input(
    input: &DatasetOracleInput,
) -> Result<DatasetOracleOutput, DatasetOracleError> {
    let position = Position::from_fen(&input.fen).map_err(DatasetOracleError::InvalidFen)?;
    let legal = legal_moves(&position);
    let legal_move_strings: Vec<String> =
        legal.iter().map(|candidate| candidate.to_uci()).collect();
    let legal_action_encodings = legal_action_encodings(&position)
        .map_err(DatasetOracleError::ActionEncoding)?
        .into_iter()
        .map(action_encoding_array)
        .collect();

    let selected_move = match &input.selected_move_uci {
        Some(chess_move) => Some(
            legal
                .iter()
                .copied()
                .find(|candidate| candidate.to_uci() == *chess_move)
                .ok_or_else(|| DatasetOracleError::InvalidSelectedMove(chess_move.clone()))?,
        ),
        None => None,
    };

    let next_position = selected_move
        .map(|candidate| {
            apply_move(&position, candidate).map_err(DatasetOracleError::MoveApplication)
        })
        .transpose()?;

    let selected_action_encoding = selected_move
        .map(|candidate| encode_move(candidate).map(action_encoding_array))
        .transpose()
        .map_err(DatasetOracleError::ActionEncoding)?;

    let position_encoding = encode_position(&position);
    let in_check = is_in_check(&position, position.side_to_move());
    let legal_move_count = legal.len() as u32;
    let piece_count = position.iter_pieces().count() as u32;

    let annotations = TacticalAnnotationsOutput {
        in_check,
        is_checkmate: in_check && legal.is_empty(),
        is_stalemate: !in_check && legal.is_empty(),
        has_legal_en_passant: legal
            .iter()
            .any(|candidate| matches!(candidate.kind, MoveKind::EnPassant)),
        has_legal_castle: legal.iter().any(|candidate| {
            matches!(
                candidate.kind,
                MoveKind::CastleKingside | MoveKind::CastleQueenside
            )
        }),
        has_legal_promotion: legal.iter().any(|candidate| {
            matches!(
                candidate.kind,
                MoveKind::Promotion(_) | MoveKind::PromotionCapture(_)
            )
        }),
        is_low_material_endgame: piece_count <= 8,
        legal_move_count,
        piece_count,
        selected_move_is_capture: selected_move.map(|candidate| candidate.kind.is_capture()),
        selected_move_is_promotion: selected_move
            .map(|candidate| candidate.kind.promotion_piece().is_some()),
        selected_move_is_castle: selected_move.map(|candidate| {
            matches!(
                candidate.kind,
                MoveKind::CastleKingside | MoveKind::CastleQueenside
            )
        }),
        selected_move_is_en_passant: selected_move
            .map(|candidate| matches!(candidate.kind, MoveKind::EnPassant)),
        selected_move_gives_check: next_position
            .as_ref()
            .map(|next| is_in_check(next, next.side_to_move())),
    };

    Ok(DatasetOracleOutput {
        fen: input.fen.clone(),
        side_to_move: position.side_to_move().fen_symbol().to_string(),
        legal_moves: legal_move_strings,
        legal_action_encodings,
        selected_move_uci: input.selected_move_uci.clone(),
        selected_action_encoding,
        next_fen: next_position.map(|next| next.to_fen()),
        position_encoding: PositionEncodingOutput {
            piece_tokens: position_encoding.piece_token_matrix(),
            square_tokens: position_encoding.square_token_matrix().to_vec(),
            rule_token: position_encoding.rule_token_vector(),
        },
        annotations,
    })
}

/// Parses a JSON input line and returns the JSON output line for the dataset oracle.
pub fn label_json_line(line: &str) -> Result<String, DatasetOracleError> {
    let input =
        serde_json::from_str::<DatasetOracleInput>(line).map_err(DatasetOracleError::Json)?;
    let output = label_dataset_input(&input)?;
    serde_json::to_string(&output).map_err(DatasetOracleError::Json)
}

/// Process one newline-delimited oracle request stream into newline-delimited responses.
pub fn process_json_lines(
    reader: impl BufRead,
    mut writer: impl Write,
    context: &str,
) -> io::Result<()> {
    for (line_number, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let output = label_json_line(&line).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{context} line {}: {error}", line_number + 1),
            )
        })?;
        writeln!(writer, "{output}")?;
    }

    writer.flush()?;
    Ok(())
}

fn action_encoding_array(encoding: action_space::ActionEncoding) -> [u32; 3] {
    let [from, to, promotion] = encoding.as_indices();
    [from as u32, to as u32, promotion as u32]
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::{label_dataset_input, process_json_lines, DatasetOracleInput};

    #[test]
    fn selected_move_yields_next_state_and_action_label() {
        let output = label_dataset_input(&DatasetOracleInput {
            fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string(),
            selected_move_uci: Some("e2e4".to_string()),
        })
        .expect("oracle labels position");

        assert_eq!(output.legal_moves.len(), 20);
        assert_eq!(
            output.next_fen,
            Some("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1".to_string())
        );
        assert_eq!(output.selected_action_encoding, Some([12, 28, 0]));
        assert_eq!(output.annotations.selected_move_is_capture, Some(false));
    }

    #[test]
    fn terminal_annotations_detect_checkmate() {
        let output = label_dataset_input(&DatasetOracleInput {
            fen: "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1".to_string(),
            selected_move_uci: None,
        })
        .expect("oracle labels position");

        assert!(output.annotations.in_check);
        assert!(output.annotations.is_checkmate);
        assert!(!output.annotations.is_stalemate);
        assert_eq!(output.annotations.legal_move_count, 0);
    }

    #[test]
    fn special_move_coverage_flags_are_exact() {
        let castle = label_dataset_input(&DatasetOracleInput {
            fen: "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1".to_string(),
            selected_move_uci: Some("e1g1".to_string()),
        })
        .expect("oracle labels castling position");
        assert!(castle.annotations.has_legal_castle);
        assert_eq!(castle.annotations.selected_move_is_castle, Some(true));

        let en_passant = label_dataset_input(&DatasetOracleInput {
            fen: "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1".to_string(),
            selected_move_uci: Some("e5d6".to_string()),
        })
        .expect("oracle labels en-passant position");
        assert!(en_passant.annotations.has_legal_en_passant);
        assert_eq!(
            en_passant.annotations.selected_move_is_en_passant,
            Some(true)
        );

        let promotion = label_dataset_input(&DatasetOracleInput {
            fen: "4k3/6P1/8/8/8/8/8/4K3 w - - 0 1".to_string(),
            selected_move_uci: Some("g7g8q".to_string()),
        })
        .expect("oracle labels promotion position");
        assert!(promotion.annotations.has_legal_promotion);
        assert_eq!(promotion.annotations.selected_move_is_promotion, Some(true));
    }

    #[test]
    fn process_json_lines_preserves_record_count() {
        let input = concat!(
            "{\"fen\":\"4k3/8/8/8/8/8/8/4K3 w - - 0 1\",\"selected_move_uci\":null}\n",
            "{\"fen\":\"7k/6Q1/6K1/8/8/8/8/8 b - - 0 1\",\"selected_move_uci\":null}\n"
        );
        let mut output = Vec::new();

        process_json_lines(Cursor::new(input), &mut output, "dataset-oracle")
            .expect("stream processing succeeds");

        let lines: Vec<&str> = std::str::from_utf8(&output)
            .expect("utf8 output")
            .lines()
            .collect();
        assert_eq!(lines.len(), 2);
    }
}
