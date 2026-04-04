//! Deterministic exact-position encoding for model-facing consumers.
//!
//! Phase 3 uses an object-centric schema:
//! - piece tokens sorted by square index
//! - square tokens for every board square
//! - one rule token carrying side-to-move, castling, clocks, and repetition state
//!
//! This encoder does not normalize symmetries. Side to move, clocks, castling rights,
//! en-passant state, and exact square locations intentionally affect the encoding.

use std::array;

use core_types::{Color, Piece, PieceKind};
use position::Position;

pub const MAX_PIECE_TOKENS: usize = 32;
pub const SQUARE_TOKEN_COUNT: usize = 64;
pub const PIECE_TOKEN_WIDTH: usize = 3;
pub const SQUARE_TOKEN_WIDTH: usize = 2;
pub const RULE_TOKEN_WIDTH: usize = 6;
pub const PIECE_TOKEN_CAPACITY: usize = MAX_PIECE_TOKENS;
pub const POSITION_FEATURE_SIZE: usize = (PIECE_TOKEN_CAPACITY * PIECE_TOKEN_WIDTH)
    + (SQUARE_TOKEN_COUNT * SQUARE_TOKEN_WIDTH)
    + RULE_TOKEN_WIDTH;

pub const EMPTY_OCCUPANT_CODE: u8 = 0;
pub const NO_EN_PASSANT_SQUARE: u8 = 64;
pub const PIECE_TOKEN_PADDING_VALUE: f32 = -1.0;

/// Object-centric piece token ordered by board square.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PieceToken {
    pub square_index: u8,
    pub color_index: u8,
    pub piece_kind_index: u8,
}

impl PieceToken {
    #[must_use]
    pub const fn as_array(self) -> [u32; PIECE_TOKEN_WIDTH] {
        [
            self.square_index as u32,
            self.color_index as u32,
            self.piece_kind_index as u32,
        ]
    }
}

/// Dense per-square token preserving exact occupancy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SquareToken {
    pub square_index: u8,
    pub occupant_code: u8,
}

impl SquareToken {
    #[must_use]
    pub const fn as_array(self) -> [u32; SQUARE_TOKEN_WIDTH] {
        [self.square_index as u32, self.occupant_code as u32]
    }
}

/// Rule-state token that preserves exact non-board legality state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuleToken {
    pub side_to_move: u8,
    pub castling_bits: u8,
    pub en_passant_square: u8,
    pub halfmove_clock: u32,
    pub fullmove_number: u32,
    pub repetition_count: u32,
}

impl RuleToken {
    #[must_use]
    pub const fn as_array(self) -> [u32; RULE_TOKEN_WIDTH] {
        [
            self.side_to_move as u32,
            self.castling_bits as u32,
            self.en_passant_square as u32,
            self.halfmove_clock,
            self.fullmove_number,
            self.repetition_count,
        ]
    }
}

/// Complete deterministic encoding of a symbolic position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedPosition {
    pub piece_tokens: Vec<PieceToken>,
    pub square_tokens: [SquareToken; SQUARE_TOKEN_COUNT],
    pub rule_token: RuleToken,
}

impl EncodedPosition {
    #[must_use]
    pub fn piece_token_matrix(&self) -> Vec<[u32; PIECE_TOKEN_WIDTH]> {
        self.piece_tokens
            .iter()
            .copied()
            .map(PieceToken::as_array)
            .collect()
    }

    #[must_use]
    pub fn square_token_matrix(&self) -> [[u32; SQUARE_TOKEN_WIDTH]; SQUARE_TOKEN_COUNT] {
        array::from_fn(|index| self.square_tokens[index].as_array())
    }

    #[must_use]
    pub const fn rule_token_vector(&self) -> [u32; RULE_TOKEN_WIDTH] {
        self.rule_token.as_array()
    }
}

/// Encodes a symbolic position into deterministic piece, square, and rule tokens.
#[must_use]
pub fn encode_position(position: &Position) -> EncodedPosition {
    let piece_tokens: Vec<PieceToken> = position
        .iter_pieces()
        .map(|(square, piece)| PieceToken {
            square_index: square.index(),
            color_index: color_index(piece.color),
            piece_kind_index: piece_kind_index(piece.kind),
        })
        .collect();
    debug_assert!(piece_tokens.len() <= MAX_PIECE_TOKENS);

    let square_tokens = array::from_fn(|index| {
        let piece = position.board()[index];
        SquareToken {
            square_index: index as u8,
            occupant_code: occupant_code(piece),
        }
    });

    let castling = position.castling_rights();
    let rule_token = RuleToken {
        side_to_move: color_index(position.side_to_move()),
        castling_bits: u8::from(castling.white_kingside())
            | (u8::from(castling.white_queenside()) << 1)
            | (u8::from(castling.black_kingside()) << 2)
            | (u8::from(castling.black_queenside()) << 3),
        en_passant_square: position
            .en_passant()
            .map_or(NO_EN_PASSANT_SQUARE, |square| square.index()),
        halfmove_clock: position.halfmove_clock(),
        fullmove_number: position.fullmove_number(),
        repetition_count: position.repetition_count() as u32,
    };

    EncodedPosition {
        piece_tokens,
        square_tokens,
        rule_token,
    }
}

/// Flatten the exact token encoder output into the fixed-width 230-dim model vector.
#[must_use]
pub fn pack_position_features(position: &Position) -> [f32; POSITION_FEATURE_SIZE] {
    let encoded = encode_position(position);
    let mut features = [0.0_f32; POSITION_FEATURE_SIZE];
    let mut offset = 0;

    for piece_index in 0..PIECE_TOKEN_CAPACITY {
        if let Some(token) = encoded.piece_tokens.get(piece_index) {
            let token_array = token.as_array();
            features[offset] = token_array[0] as f32;
            features[offset + 1] = token_array[1] as f32;
            features[offset + 2] = token_array[2] as f32;
        } else {
            features[offset] = PIECE_TOKEN_PADDING_VALUE;
            features[offset + 1] = PIECE_TOKEN_PADDING_VALUE;
            features[offset + 2] = PIECE_TOKEN_PADDING_VALUE;
        }
        offset += PIECE_TOKEN_WIDTH;
    }

    for token in encoded.square_tokens {
        let token_array = token.as_array();
        features[offset] = token_array[0] as f32;
        features[offset + 1] = token_array[1] as f32;
        offset += SQUARE_TOKEN_WIDTH;
    }

    for value in encoded.rule_token.as_array() {
        features[offset] = value as f32;
        offset += 1;
    }

    features
}

fn color_index(color: Color) -> u8 {
    match color {
        Color::White => 0,
        Color::Black => 1,
    }
}

fn piece_kind_index(piece_kind: PieceKind) -> u8 {
    match piece_kind {
        PieceKind::Pawn => 0,
        PieceKind::Knight => 1,
        PieceKind::Bishop => 2,
        PieceKind::Rook => 3,
        PieceKind::Queen => 4,
        PieceKind::King => 5,
    }
}

fn occupant_code(piece: Option<Piece>) -> u8 {
    match piece {
        None => EMPTY_OCCUPANT_CODE,
        Some(piece) => {
            let color_offset = match piece.color {
                Color::White => 0,
                Color::Black => 6,
            };
            1 + color_offset + piece_kind_index(piece.kind)
        }
    }
}

#[cfg(test)]
mod tests {
    use core_types::Square;

    use super::{
        encode_position, pack_position_features, EMPTY_OCCUPANT_CODE, NO_EN_PASSANT_SQUARE,
        PIECE_TOKEN_PADDING_VALUE, POSITION_FEATURE_SIZE, RULE_TOKEN_WIDTH, SQUARE_TOKEN_COUNT,
    };
    use position::Position;

    #[test]
    fn encoding_is_deterministic_for_same_position() {
        let position = Position::from_fen("r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 7 22")
            .expect("valid position");

        assert_eq!(encode_position(&position), encode_position(&position));
    }

    #[test]
    fn piece_tokens_are_sorted_by_square_index() {
        let position =
            Position::from_fen("8/8/8/8/3k4/8/4P3/4K3 w - - 0 1").expect("valid position");
        let encoded = encode_position(&position);
        let squares: Vec<u8> = encoded
            .piece_tokens
            .iter()
            .map(|token| token.square_index)
            .collect();

        assert_eq!(squares, vec![4, 12, 27]);
    }

    #[test]
    fn square_and_rule_tokens_preserve_exact_state() {
        let position = Position::from_fen("r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 7 22")
            .expect("valid position");
        let encoded = encode_position(&position);

        assert_eq!(encoded.square_tokens.len(), SQUARE_TOKEN_COUNT);
        assert_eq!(encoded.square_tokens[0].occupant_code, 1 + 3);
        assert_eq!(encoded.square_tokens[35].occupant_code, 1 + 6);
        assert_eq!(
            encoded.rule_token.as_array(),
            [
                0,
                0b1111,
                u32::from(Square::from_algebraic("d6").unwrap().index()),
                7,
                22,
                1,
            ]
        );
        assert_eq!(encoded.rule_token.as_array().len(), RULE_TOKEN_WIDTH);
    }

    #[test]
    fn empty_and_non_empty_square_codes_are_distinct() {
        let position = Position::startpos();
        let encoded = encode_position(&position);

        assert_eq!(encoded.square_tokens[27].occupant_code, EMPTY_OCCUPANT_CODE);
        assert_ne!(encoded.square_tokens[4].occupant_code, EMPTY_OCCUPANT_CODE);
    }

    #[test]
    fn exported_matrices_use_documented_shapes() {
        let position =
            Position::from_fen("8/8/8/8/3k4/8/4P3/4K3 w - - 0 1").expect("valid position");
        let encoded = encode_position(&position);

        assert_eq!(encoded.piece_token_matrix().len(), 3);
        assert_eq!(encoded.square_token_matrix().len(), 64);
        assert_eq!(encoded.rule_token_vector().len(), 6);
    }

    #[test]
    fn no_en_passant_uses_reserved_square_index() {
        let position = Position::startpos();
        let encoded = encode_position(&position);

        assert_eq!(encoded.rule_token.en_passant_square, NO_EN_PASSANT_SQUARE);
    }

    #[test]
    fn packed_features_use_documented_width_and_padding() {
        let position =
            Position::from_fen("8/8/8/8/3k4/8/4P3/4K3 w - - 0 1").expect("valid position");
        let packed = pack_position_features(&position);

        assert_eq!(packed.len(), POSITION_FEATURE_SIZE);
        assert_eq!(packed[9], PIECE_TOKEN_PADDING_VALUE);
    }
}
