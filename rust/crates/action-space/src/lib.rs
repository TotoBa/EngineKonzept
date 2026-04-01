//! Factorized model-facing move vocabulary.
//!
//! The action space intentionally separates move choice into:
//! - a `from` head over 64 squares
//! - a `to` head over 64 squares
//! - a promotion head over 5 classes (`none`, `knight`, `bishop`, `rook`, `queen`)
//!
//! Exact move semantics such as castling, en passant, capture, or double-pawn-push are
//! recovered against a concrete symbolic position via the exact legal move list.

use std::error::Error;
use std::fmt;

use core_types::{Move, MoveKind, PieceKind, Square};
use position::Position;
use rules::legal_moves;

pub const FROM_HEAD_SIZE: usize = 64;
pub const TO_HEAD_SIZE: usize = 64;
pub const PROMOTION_HEAD_SIZE: usize = 5;

/// Promotion classes used by the model-facing action representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum PromotionTarget {
    None = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
}

impl PromotionTarget {
    #[must_use]
    pub const fn index(self) -> u8 {
        self as u8
    }

    pub fn from_index(index: u8) -> Option<Self> {
        match index {
            0 => Some(Self::None),
            1 => Some(Self::Knight),
            2 => Some(Self::Bishop),
            3 => Some(Self::Rook),
            4 => Some(Self::Queen),
            _ => None,
        }
    }

    pub fn from_piece_kind(piece_kind: PieceKind) -> Result<Self, ActionEncodeError> {
        match piece_kind {
            PieceKind::Knight => Ok(Self::Knight),
            PieceKind::Bishop => Ok(Self::Bishop),
            PieceKind::Rook => Ok(Self::Rook),
            PieceKind::Queen => Ok(Self::Queen),
            unsupported => Err(ActionEncodeError::UnsupportedPromotionPiece(unsupported)),
        }
    }

    #[must_use]
    pub const fn to_piece_kind(self) -> Option<PieceKind> {
        match self {
            Self::None => None,
            Self::Knight => Some(PieceKind::Knight),
            Self::Bishop => Some(PieceKind::Bishop),
            Self::Rook => Some(PieceKind::Rook),
            Self::Queen => Some(PieceKind::Queen),
        }
    }
}

/// Factorized action encoding used for model IO.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ActionEncoding {
    from: u8,
    to: u8,
    promotion: PromotionTarget,
}

impl ActionEncoding {
    #[must_use]
    pub const fn new(from: Square, to: Square, promotion: PromotionTarget) -> Self {
        Self {
            from: from.index(),
            to: to.index(),
            promotion,
        }
    }

    pub fn from_head_indices(
        from: u8,
        to: u8,
        promotion: u8,
    ) -> Result<Self, RawActionEncodingError> {
        if from >= FROM_HEAD_SIZE as u8 {
            return Err(RawActionEncodingError::InvalidFromSquare(from));
        }
        if to >= TO_HEAD_SIZE as u8 {
            return Err(RawActionEncodingError::InvalidToSquare(to));
        }

        let promotion = PromotionTarget::from_index(promotion)
            .ok_or(RawActionEncodingError::InvalidPromotionClass(promotion))?;

        Ok(Self {
            from,
            to,
            promotion,
        })
    }

    #[must_use]
    pub const fn from_index(self) -> u8 {
        self.from
    }

    #[must_use]
    pub const fn to_index(self) -> u8 {
        self.to
    }

    #[must_use]
    pub const fn promotion(self) -> PromotionTarget {
        self.promotion
    }

    #[must_use]
    pub const fn as_indices(self) -> [usize; 3] {
        [
            self.from as usize,
            self.to as usize,
            self.promotion.index() as usize,
        ]
    }

    pub fn from_move(chess_move: Move) -> Result<Self, ActionEncodeError> {
        encode_move(chess_move)
    }

    pub fn decode_in_position(self, position: &Position) -> Result<Move, ActionDecodeError> {
        decode_move(position, self)
    }
}

/// Encodes an exact symbolic move into the factorized action space.
pub fn encode_move(chess_move: Move) -> Result<ActionEncoding, ActionEncodeError> {
    let promotion = match chess_move.kind {
        MoveKind::Promotion(piece_kind) | MoveKind::PromotionCapture(piece_kind) => {
            PromotionTarget::from_piece_kind(piece_kind)?
        }
        _ => PromotionTarget::None,
    };

    Ok(ActionEncoding::new(
        chess_move.from,
        chess_move.to,
        promotion,
    ))
}

/// Returns sorted action encodings for all legal moves in the current position.
pub fn legal_action_encodings(
    position: &Position,
) -> Result<Vec<ActionEncoding>, ActionEncodeError> {
    let mut encodings: Vec<ActionEncoding> = legal_moves(position)
        .into_iter()
        .map(encode_move)
        .collect::<Result<Vec<_>, _>>()?;
    encodings.sort_unstable();
    Ok(encodings)
}

/// Decodes a factorized action back into an exact symbolic move by matching against legal moves.
pub fn decode_move(
    position: &Position,
    encoding: ActionEncoding,
) -> Result<Move, ActionDecodeError> {
    let mut matching_moves = legal_moves(position)
        .into_iter()
        .filter(|candidate| action_matches_move(encoding, *candidate));

    let matching_move = matching_moves
        .next()
        .ok_or(ActionDecodeError::IllegalEncoding(encoding))?;

    if matching_moves.next().is_some() {
        return Err(ActionDecodeError::AmbiguousEncoding(encoding));
    }

    Ok(matching_move)
}

fn action_matches_move(encoding: ActionEncoding, chess_move: Move) -> bool {
    if encoding.from_index() != chess_move.from.index()
        || encoding.to_index() != chess_move.to.index()
    {
        return false;
    }

    match chess_move.kind.promotion_piece() {
        Some(piece_kind) => encoding.promotion().to_piece_kind() == Some(piece_kind),
        None => encoding.promotion() == PromotionTarget::None,
    }
}

/// Error returned when constructing a raw action encoding from model head indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RawActionEncodingError {
    InvalidFromSquare(u8),
    InvalidToSquare(u8),
    InvalidPromotionClass(u8),
}

impl fmt::Display for RawActionEncodingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFromSquare(index) => write!(f, "invalid from-square index: {index}"),
            Self::InvalidToSquare(index) => write!(f, "invalid to-square index: {index}"),
            Self::InvalidPromotionClass(index) => {
                write!(f, "invalid promotion class index: {index}")
            }
        }
    }
}

impl Error for RawActionEncodingError {}

/// Error returned when an exact move cannot be expressed in the current factorization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionEncodeError {
    UnsupportedPromotionPiece(PieceKind),
}

impl fmt::Display for ActionEncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPromotionPiece(piece_kind) => {
                write!(
                    f,
                    "unsupported promotion target in action space: {piece_kind}"
                )
            }
        }
    }
}

impl Error for ActionEncodeError {}

/// Error returned when a factorized action cannot be resolved to an exact legal move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionDecodeError {
    IllegalEncoding(ActionEncoding),
    AmbiguousEncoding(ActionEncoding),
}

impl fmt::Display for ActionDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IllegalEncoding(encoding) => {
                write!(
                    f,
                    "action encoding is not legal in this position: {encoding:?}"
                )
            }
            Self::AmbiguousEncoding(encoding) => {
                write!(
                    f,
                    "action encoding resolves to multiple legal moves: {encoding:?}"
                )
            }
        }
    }
}

impl Error for ActionDecodeError {}

#[cfg(test)]
mod tests {
    use core_types::{MoveKind, PieceKind, Square};

    use super::{
        decode_move, encode_move, legal_action_encodings, ActionEncoding, PromotionTarget,
        PROMOTION_HEAD_SIZE,
    };
    use position::Position;
    use rules::legal_moves;

    #[test]
    fn legal_moves_roundtrip_through_factorized_encoding() {
        let position = Position::startpos();

        for chess_move in legal_moves(&position) {
            let encoding = encode_move(chess_move).expect("move is representable");
            let decoded =
                decode_move(&position, encoding).expect("move decodes in the same position");
            assert_eq!(decoded, chess_move);
        }
    }

    #[test]
    fn castling_and_en_passant_decode_from_same_heads() {
        let castling = Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
            .expect("valid castling position");
        let encoding = ActionEncoding::new(
            Square::from_algebraic("e1").unwrap(),
            Square::from_algebraic("g1").unwrap(),
            PromotionTarget::None,
        );
        let decoded = decode_move(&castling, encoding).expect("castling decodes");
        assert_eq!(decoded.kind, MoveKind::CastleKingside);

        let en_passant = Position::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
            .expect("valid en-passant position");
        let encoding = ActionEncoding::new(
            Square::from_algebraic("e5").unwrap(),
            Square::from_algebraic("d6").unwrap(),
            PromotionTarget::None,
        );
        let decoded = decode_move(&en_passant, encoding).expect("en-passant decodes");
        assert_eq!(decoded.kind, MoveKind::EnPassant);
    }

    #[test]
    fn promotion_mapping_uses_dedicated_promotion_head() {
        let promotion_move = core_types::Move::new(
            Square::from_algebraic("a7").unwrap(),
            Square::from_algebraic("a8").unwrap(),
            MoveKind::Promotion(PieceKind::Knight),
        );
        let encoding = encode_move(promotion_move).expect("promotion is representable");

        assert_eq!(encoding.promotion(), PromotionTarget::Knight);
        assert_eq!(PROMOTION_HEAD_SIZE, 5);
    }

    #[test]
    fn raw_head_indices_validate_shape() {
        let error = ActionEncoding::from_head_indices(64, 0, 0).expect_err("from index is invalid");
        assert_eq!(error, super::RawActionEncodingError::InvalidFromSquare(64));
    }

    #[test]
    fn illegal_action_encoding_is_rejected_in_position() {
        let position = Position::startpos();
        let encoding = ActionEncoding::new(
            Square::from_algebraic("a1").unwrap(),
            Square::from_algebraic("a8").unwrap(),
            PromotionTarget::None,
        );

        assert!(decode_move(&position, encoding).is_err());
    }

    #[test]
    fn legal_action_encodings_are_sorted_and_cover_all_legal_moves() {
        let position =
            Position::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1").expect("valid position");
        let mut expected: Vec<[usize; 3]> = legal_moves(&position)
            .into_iter()
            .map(|chess_move| {
                encode_move(chess_move)
                    .expect("move is representable")
                    .as_indices()
            })
            .collect();
        expected.sort_unstable();

        let actual: Vec<[usize; 3]> = legal_action_encodings(&position)
            .expect("encodings are available")
            .into_iter()
            .map(ActionEncoding::as_indices)
            .collect();

        assert_eq!(actual, expected);
    }
}
