//! Dataset and developer tooling that reuses the exact symbolic rules kernel.

use std::error::Error;
use std::fmt;
use std::io::{self, BufRead, Write};
use std::time::{Duration, Instant};

use action_space::{encode_move, ActionEncodeError};
use core_types::MoveKind;
use encoder::encode_position;
use position::Position;
use rules::{apply_known_legal_move, is_in_check, legal_moves_profiled, MoveError};
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

/// Aggregated offline timings for one profiled oracle run.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OracleProfileTotals {
    pub records: u64,
    pub json_parse: Duration,
    pub fen_parse: Duration,
    pub legal_generation: Duration,
    pub pseudo_legal_generation: Duration,
    pub self_check_filter: Duration,
    pub attack_check_local: Duration,
    pub attack_check_slider: Duration,
    pub legal_move_uci: Duration,
    pub legal_action_encoding: Duration,
    pub selected_move_resolution: Duration,
    pub selected_move_apply: Duration,
    pub selected_action_encoding: Duration,
    pub position_encoding: Duration,
    pub annotations: Duration,
    pub json_serialize: Duration,
}

impl OracleProfileTotals {
    pub fn total_measured(&self) -> Duration {
        self.json_parse
            + self.fen_parse
            + self.effective_legal_generation()
            + self.legal_move_uci
            + self.legal_action_encoding
            + self.selected_move_resolution
            + self.selected_move_apply
            + self.selected_action_encoding
            + self.position_encoding
            + self.annotations
            + self.json_serialize
    }

    fn effective_legal_generation(&self) -> Duration {
        if self.pseudo_legal_generation > Duration::ZERO || self.self_check_filter > Duration::ZERO
        {
            self.pseudo_legal_generation + self.self_check_filter
        } else {
            self.legal_generation
        }
    }

    fn record_label(&mut self, profile: &OracleRecordProfile) {
        self.records += 1;
        self.fen_parse += profile.fen_parse;
        self.legal_generation += profile.legal_generation;
        self.pseudo_legal_generation += profile.pseudo_legal_generation;
        self.self_check_filter += profile.self_check_filter;
        self.attack_check_local += profile.attack_check_local;
        self.attack_check_slider += profile.attack_check_slider;
        self.legal_move_uci += profile.legal_move_uci;
        self.legal_action_encoding += profile.legal_action_encoding;
        self.selected_move_resolution += profile.selected_move_resolution;
        self.selected_move_apply += profile.selected_move_apply;
        self.selected_action_encoding += profile.selected_action_encoding;
        self.position_encoding += profile.position_encoding;
        self.annotations += profile.annotations;
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct OracleRecordProfile {
    fen_parse: Duration,
    legal_generation: Duration,
    pseudo_legal_generation: Duration,
    self_check_filter: Duration,
    attack_check_local: Duration,
    attack_check_slider: Duration,
    legal_move_uci: Duration,
    legal_action_encoding: Duration,
    selected_move_resolution: Duration,
    selected_move_apply: Duration,
    selected_action_encoding: Duration,
    position_encoding: Duration,
    annotations: Duration,
}

/// Labels a dataset input record via the exact rules kernel.
pub fn label_dataset_input(
    input: &DatasetOracleInput,
) -> Result<DatasetOracleOutput, DatasetOracleError> {
    label_dataset_input_impl(input, None)
}

/// Parses a JSON input line and returns the JSON output line for the dataset oracle.
pub fn label_json_line(line: &str) -> Result<String, DatasetOracleError> {
    let input = parse_oracle_input(line)?;
    let output = label_dataset_input(&input)?;
    serde_json::to_string(&output).map_err(DatasetOracleError::Json)
}

/// Profile one newline-delimited oracle request stream without changing outputs.
pub fn profile_json_lines(reader: impl BufRead, context: &str) -> io::Result<OracleProfileTotals> {
    let mut totals = OracleProfileTotals::default();
    let mut sink = io::sink();
    for (line_number, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let parse_started = Instant::now();
        let input = parse_oracle_input(&line);
        totals.json_parse += parse_started.elapsed();
        let input = input.map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{context} line {}: {error}", line_number + 1),
            )
        })?;

        let mut record_profile = OracleRecordProfile::default();
        let output =
            label_dataset_input_impl(&input, Some(&mut record_profile)).map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("{context} line {}: {error}", line_number + 1),
                )
            })?;
        totals.record_label(&record_profile);

        let serialize_started = Instant::now();
        write_output_json_line(&mut sink, &output).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{context} line {}: {error}", line_number + 1),
            )
        })?;
        totals.json_serialize += serialize_started.elapsed();
    }

    Ok(totals)
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

        let input = parse_oracle_input(&line).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{context} line {}: {error}", line_number + 1),
            )
        })?;
        let output = label_dataset_input(&input).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{context} line {}: {error}", line_number + 1),
            )
        })?;
        write_output_json_line(&mut writer, &output).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{context} line {}: {error}", line_number + 1),
            )
        })?;
    }

    writer.flush()?;
    Ok(())
}

fn action_encoding_array(encoding: action_space::ActionEncoding) -> [u32; 3] {
    let [from, to, promotion] = encoding.as_indices();
    [from as u32, to as u32, promotion as u32]
}

fn parse_oracle_input(line: &str) -> Result<DatasetOracleInput, DatasetOracleError> {
    serde_json::from_str::<DatasetOracleInput>(line).map_err(DatasetOracleError::Json)
}

fn write_output_json_line(
    writer: &mut impl Write,
    output: &DatasetOracleOutput,
) -> Result<(), DatasetOracleError> {
    write_oracle_output_json(&mut *writer, output)?;
    writer.write_all(b"\n").map_err(|error| {
        DatasetOracleError::Json(serde_json::Error::io(io::Error::new(
            error.kind(),
            error.to_string(),
        )))
    })?;
    Ok(())
}

fn write_oracle_output_json(
    writer: &mut impl Write,
    output: &DatasetOracleOutput,
) -> Result<(), DatasetOracleError> {
    writer.write_all(b"{").map_err(json_io_error)?;
    write_json_field_name(writer, "fen")?;
    write_json_string(writer, &output.fen)?;
    writer.write_all(b",").map_err(json_io_error)?;
    write_json_field_name(writer, "side_to_move")?;
    write_json_string(writer, &output.side_to_move)?;
    writer.write_all(b",").map_err(json_io_error)?;
    write_json_field_name(writer, "legal_moves")?;
    write_string_array(writer, &output.legal_moves)?;
    writer.write_all(b",").map_err(json_io_error)?;
    write_json_field_name(writer, "legal_action_encodings")?;
    write_u32_triplet_array(writer, &output.legal_action_encodings)?;
    writer.write_all(b",").map_err(json_io_error)?;
    write_json_field_name(writer, "selected_move_uci")?;
    write_optional_string(writer, output.selected_move_uci.as_deref())?;
    writer.write_all(b",").map_err(json_io_error)?;
    write_json_field_name(writer, "selected_action_encoding")?;
    write_optional_u32_triplet(writer, output.selected_action_encoding)?;
    writer.write_all(b",").map_err(json_io_error)?;
    write_json_field_name(writer, "next_fen")?;
    write_optional_string(writer, output.next_fen.as_deref())?;
    writer.write_all(b",").map_err(json_io_error)?;
    write_json_field_name(writer, "position_encoding")?;
    write_position_encoding(writer, &output.position_encoding)?;
    writer.write_all(b",").map_err(json_io_error)?;
    write_json_field_name(writer, "annotations")?;
    write_annotations(writer, &output.annotations)?;
    writer.write_all(b"}").map_err(json_io_error)?;
    Ok(())
}

fn write_position_encoding(
    writer: &mut impl Write,
    encoding: &PositionEncodingOutput,
) -> Result<(), DatasetOracleError> {
    writer.write_all(b"{").map_err(json_io_error)?;
    write_json_field_name(writer, "piece_tokens")?;
    write_u32_triplet_array(writer, &encoding.piece_tokens)?;
    writer.write_all(b",").map_err(json_io_error)?;
    write_json_field_name(writer, "square_tokens")?;
    write_u32_pair_array(writer, &encoding.square_tokens)?;
    writer.write_all(b",").map_err(json_io_error)?;
    write_json_field_name(writer, "rule_token")?;
    write_u32_six(writer, encoding.rule_token)?;
    writer.write_all(b"}").map_err(json_io_error)?;
    Ok(())
}

fn write_annotations(
    writer: &mut impl Write,
    annotations: &TacticalAnnotationsOutput,
) -> Result<(), DatasetOracleError> {
    writer.write_all(b"{").map_err(json_io_error)?;
    write_json_field_name(writer, "in_check")?;
    write_bool(writer, annotations.in_check)?;
    writer.write_all(b",").map_err(json_io_error)?;
    write_json_field_name(writer, "is_checkmate")?;
    write_bool(writer, annotations.is_checkmate)?;
    writer.write_all(b",").map_err(json_io_error)?;
    write_json_field_name(writer, "is_stalemate")?;
    write_bool(writer, annotations.is_stalemate)?;
    writer.write_all(b",").map_err(json_io_error)?;
    write_json_field_name(writer, "has_legal_en_passant")?;
    write_bool(writer, annotations.has_legal_en_passant)?;
    writer.write_all(b",").map_err(json_io_error)?;
    write_json_field_name(writer, "has_legal_castle")?;
    write_bool(writer, annotations.has_legal_castle)?;
    writer.write_all(b",").map_err(json_io_error)?;
    write_json_field_name(writer, "has_legal_promotion")?;
    write_bool(writer, annotations.has_legal_promotion)?;
    writer.write_all(b",").map_err(json_io_error)?;
    write_json_field_name(writer, "is_low_material_endgame")?;
    write_bool(writer, annotations.is_low_material_endgame)?;
    writer.write_all(b",").map_err(json_io_error)?;
    write_json_field_name(writer, "legal_move_count")?;
    write_u32(writer, annotations.legal_move_count)?;
    writer.write_all(b",").map_err(json_io_error)?;
    write_json_field_name(writer, "piece_count")?;
    write_u32(writer, annotations.piece_count)?;
    writer.write_all(b",").map_err(json_io_error)?;
    write_json_field_name(writer, "selected_move_is_capture")?;
    write_optional_bool(writer, annotations.selected_move_is_capture)?;
    writer.write_all(b",").map_err(json_io_error)?;
    write_json_field_name(writer, "selected_move_is_promotion")?;
    write_optional_bool(writer, annotations.selected_move_is_promotion)?;
    writer.write_all(b",").map_err(json_io_error)?;
    write_json_field_name(writer, "selected_move_is_castle")?;
    write_optional_bool(writer, annotations.selected_move_is_castle)?;
    writer.write_all(b",").map_err(json_io_error)?;
    write_json_field_name(writer, "selected_move_is_en_passant")?;
    write_optional_bool(writer, annotations.selected_move_is_en_passant)?;
    writer.write_all(b",").map_err(json_io_error)?;
    write_json_field_name(writer, "selected_move_gives_check")?;
    write_optional_bool(writer, annotations.selected_move_gives_check)?;
    writer.write_all(b"}").map_err(json_io_error)?;
    Ok(())
}

fn write_json_field_name(writer: &mut impl Write, name: &str) -> Result<(), DatasetOracleError> {
    write_json_string(writer, name)?;
    writer.write_all(b":").map_err(json_io_error)?;
    Ok(())
}

fn write_json_string(writer: &mut impl Write, value: &str) -> Result<(), DatasetOracleError> {
    serde_json::to_writer(&mut *writer, value).map_err(DatasetOracleError::Json)
}

fn write_optional_string(
    writer: &mut impl Write,
    value: Option<&str>,
) -> Result<(), DatasetOracleError> {
    match value {
        Some(value) => write_json_string(writer, value),
        None => writer.write_all(b"null").map_err(json_io_error),
    }
}

fn write_bool(writer: &mut impl Write, value: bool) -> Result<(), DatasetOracleError> {
    if value {
        writer.write_all(b"true").map_err(json_io_error)
    } else {
        writer.write_all(b"false").map_err(json_io_error)
    }
}

fn write_optional_bool(
    writer: &mut impl Write,
    value: Option<bool>,
) -> Result<(), DatasetOracleError> {
    match value {
        Some(value) => write_bool(writer, value),
        None => writer.write_all(b"null").map_err(json_io_error),
    }
}

fn write_u32(writer: &mut impl Write, value: u32) -> Result<(), DatasetOracleError> {
    write!(writer, "{value}").map_err(json_io_error)
}

fn write_u32_triplet_array(
    writer: &mut impl Write,
    values: &[[u32; 3]],
) -> Result<(), DatasetOracleError> {
    writer.write_all(b"[").map_err(json_io_error)?;
    for (index, [a, b, c]) in values.iter().enumerate() {
        if index > 0 {
            writer.write_all(b",").map_err(json_io_error)?;
        }
        write!(writer, "[{a},{b},{c}]").map_err(json_io_error)?;
    }
    writer.write_all(b"]").map_err(json_io_error)?;
    Ok(())
}

fn write_u32_pair_array(
    writer: &mut impl Write,
    values: &[[u32; 2]],
) -> Result<(), DatasetOracleError> {
    writer.write_all(b"[").map_err(json_io_error)?;
    for (index, [a, b]) in values.iter().enumerate() {
        if index > 0 {
            writer.write_all(b",").map_err(json_io_error)?;
        }
        write!(writer, "[{a},{b}]").map_err(json_io_error)?;
    }
    writer.write_all(b"]").map_err(json_io_error)?;
    Ok(())
}

fn write_u32_six(writer: &mut impl Write, values: [u32; 6]) -> Result<(), DatasetOracleError> {
    let [a, b, c, d, e, f] = values;
    write!(writer, "[{a},{b},{c},{d},{e},{f}]").map_err(json_io_error)
}

fn write_string_array(
    writer: &mut impl Write,
    values: &[String],
) -> Result<(), DatasetOracleError> {
    writer.write_all(b"[").map_err(json_io_error)?;
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            writer.write_all(b",").map_err(json_io_error)?;
        }
        write_json_string(writer, value)?;
    }
    writer.write_all(b"]").map_err(json_io_error)?;
    Ok(())
}

fn write_optional_u32_triplet(
    writer: &mut impl Write,
    value: Option<[u32; 3]>,
) -> Result<(), DatasetOracleError> {
    match value {
        Some([a, b, c]) => write!(writer, "[{a},{b},{c}]").map_err(json_io_error),
        None => writer.write_all(b"null").map_err(json_io_error),
    }
}

fn json_io_error(error: io::Error) -> DatasetOracleError {
    DatasetOracleError::Json(serde_json::Error::io(io::Error::new(
        error.kind(),
        error.to_string(),
    )))
}

fn label_dataset_input_impl(
    input: &DatasetOracleInput,
    mut profile: Option<&mut OracleRecordProfile>,
) -> Result<DatasetOracleOutput, DatasetOracleError> {
    let started = Instant::now();
    let position = Position::from_fen(&input.fen).map_err(DatasetOracleError::InvalidFen)?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.fen_parse += started.elapsed();
    }

    let started = Instant::now();
    let (legal, legal_profile) = legal_moves_profiled(&position);
    if let Some(profile) = profile.as_deref_mut() {
        profile.legal_generation += started.elapsed();
        profile.pseudo_legal_generation += legal_profile.pseudo_legal_generation;
        profile.self_check_filter += legal_profile.self_check_filter;
        profile.attack_check_local += legal_profile.attack_check_local;
        profile.attack_check_slider += legal_profile.attack_check_slider;
    }

    let started = Instant::now();
    let legal_move_strings: Vec<String> =
        legal.iter().map(|candidate| candidate.to_uci()).collect();
    if let Some(profile) = profile.as_deref_mut() {
        profile.legal_move_uci += started.elapsed();
    }

    let started = Instant::now();
    let mut legal_action_encodings = legal
        .iter()
        .copied()
        .map(encode_move)
        .collect::<Result<Vec<_>, _>>()
        .map_err(DatasetOracleError::ActionEncoding)?;
    legal_action_encodings.sort_unstable();
    let legal_action_encodings = legal_action_encodings
        .into_iter()
        .map(action_encoding_array)
        .collect();
    if let Some(profile) = profile.as_deref_mut() {
        profile.legal_action_encoding += started.elapsed();
    }

    let started = Instant::now();
    let selected_move = input
        .selected_move_uci
        .as_ref()
        .map(|chess_move| {
            legal
                .iter()
                .copied()
                .find(|candidate| candidate.to_uci() == *chess_move)
                .ok_or_else(|| DatasetOracleError::InvalidSelectedMove(chess_move.clone()))
        })
        .transpose()?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.selected_move_resolution += started.elapsed();
    }

    let started = Instant::now();
    let next_position = selected_move
        .map(|candidate| {
            apply_known_legal_move(&position, candidate)
                .map_err(DatasetOracleError::MoveApplication)
        })
        .transpose()?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.selected_move_apply += started.elapsed();
    }

    let started = Instant::now();
    let selected_action_encoding = selected_move
        .map(|candidate| encode_move(candidate).map(action_encoding_array))
        .transpose()
        .map_err(DatasetOracleError::ActionEncoding)?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.selected_action_encoding += started.elapsed();
    }

    let started = Instant::now();
    let position_encoding = encode_position(&position);
    if let Some(profile) = profile.as_deref_mut() {
        profile.position_encoding += started.elapsed();
    }

    let started = Instant::now();
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
    if let Some(profile) = profile.as_deref_mut() {
        profile.annotations += started.elapsed();
    }

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

#[cfg(test)]
mod tests {
    use serde_json::to_string;
    use std::io::Cursor;
    use std::time::Duration;

    use super::{
        label_dataset_input, process_json_lines, profile_json_lines, write_output_json_line,
        DatasetOracleInput,
    };

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

    #[test]
    fn profile_json_lines_counts_records() {
        let input = concat!(
            "{\"fen\":\"4k3/8/8/8/8/8/8/4K3 w - - 0 1\",\"selected_move_uci\":null}\n",
            "{\"fen\":\"7k/6Q1/6K1/8/8/8/8/8 b - - 0 1\",\"selected_move_uci\":null}\n"
        );

        let profile = profile_json_lines(Cursor::new(input), "dataset-oracle-profile")
            .expect("profiling succeeds");

        assert_eq!(profile.records, 2);
        assert!(profile.total_measured() > Duration::ZERO);
    }

    #[test]
    fn specialized_writer_matches_serde_output() {
        let output = label_dataset_input(&DatasetOracleInput {
            fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string(),
            selected_move_uci: Some("e2e4".to_string()),
        })
        .expect("oracle labels position");

        let expected = format!("{}\n", to_string(&output).expect("serde json"));
        let mut actual = Vec::new();
        write_output_json_line(&mut actual, &output).expect("specialized writer succeeds");

        assert_eq!(String::from_utf8(actual).expect("utf8 output"), expected);
    }
}
