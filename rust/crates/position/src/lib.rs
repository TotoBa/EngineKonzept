//! Exact board-state representation, FEN handling, and repetition bookkeeping.

mod castling;

use std::error::Error;
use std::fmt;

use castling::parse_castling_rights;
pub use castling::CastlingRights;
use core_types::{Color, Piece, PieceKind, Rank, Square, SquareParseError};

pub const STARTPOS_FEN: &str = "rn1qkbnr/pppb1ppp/3pp3/8/3P4/2N1PN2/PPP1BPPP/R1BQK2R w KQkq - 0 1";
pub const CLASSICAL_STARTPOS_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// Repetition-relevant snapshot of a position state.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PositionKey {
    board: [Option<Piece>; 64],
    side_to_move: Color,
    castling_rights: CastlingRights,
    en_passant: Option<Square>,
}

impl PositionKey {
    #[must_use]
    pub const fn side_to_move(&self) -> Color {
        self.side_to_move
    }

    #[must_use]
    pub const fn castling_rights(&self) -> CastlingRights {
        self.castling_rights
    }

    #[must_use]
    pub const fn en_passant(&self) -> Option<Square> {
        self.en_passant
    }
}

/// Exact board state plus the rule state required for legal move generation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Position {
    board: [Option<Piece>; 64],
    side_to_move: Color,
    castling_rights: CastlingRights,
    en_passant: Option<Square>,
    halfmove_clock: u32,
    fullmove_number: u32,
    repetition_history: Vec<PositionKey>,
}

impl Position {
    /// Returns the classical chess start position.
    #[must_use]
    pub fn startpos() -> Self {
        Self::from_fen(CLASSICAL_STARTPOS_FEN).expect("start position FEN must remain valid")
    }

    /// Parses a position from a full FEN string.
    pub fn from_fen(fen: &str) -> Result<Self, FenError> {
        let fields: Vec<&str> = fen.split_whitespace().collect();
        if fields.len() != 6 {
            return Err(FenError::WrongFieldCount);
        }

        let board = parse_board(fields[0])?;
        let side_to_move = parse_side_to_move(fields[1])?;
        let castling_rights = parse_castling_rights(fields[2])?;
        let en_passant = parse_en_passant(fields[3])?;
        let halfmove_clock = fields[4]
            .parse::<u32>()
            .map_err(|_| FenError::InvalidHalfmoveClock)?;
        let fullmove_number = fields[5]
            .parse::<u32>()
            .map_err(|_| FenError::InvalidFullmoveNumber)?;

        if fullmove_number == 0 {
            return Err(FenError::InvalidFullmoveNumber);
        }

        let mut position = Self {
            board,
            side_to_move,
            castling_rights,
            en_passant,
            halfmove_clock,
            fullmove_number,
            repetition_history: Vec::new(),
        };
        position.push_current_key();
        Ok(position)
    }

    /// Serializes the current state back to FEN.
    #[must_use]
    pub fn to_fen(&self) -> String {
        let mut board = String::new();

        for rank_index in (0..8).rev() {
            let mut empty_count = 0;
            for file_index in 0..8 {
                let square = Square::from_coords(
                    core_types::File::from_index(file_index).expect("rank iteration is bounded"),
                    Rank::from_index(rank_index).expect("file iteration is bounded"),
                );
                match self.piece_at(square) {
                    Some(piece) => {
                        if empty_count > 0 {
                            board.push_str(&empty_count.to_string());
                            empty_count = 0;
                        }
                        board.push(piece.fen_char());
                    }
                    None => empty_count += 1,
                }
            }
            if empty_count > 0 {
                board.push_str(&empty_count.to_string());
            }
            if rank_index > 0 {
                board.push('/');
            }
        }

        let en_passant = self
            .en_passant
            .map_or_else(|| "-".to_string(), |square| square.to_string());

        format!(
            "{} {} {} {} {} {}",
            board,
            self.side_to_move.fen_symbol(),
            self.castling_rights.to_fen(),
            en_passant,
            self.halfmove_clock,
            self.fullmove_number
        )
    }

    #[must_use]
    pub const fn side_to_move(&self) -> Color {
        self.side_to_move
    }

    #[must_use]
    pub const fn castling_rights(&self) -> CastlingRights {
        self.castling_rights
    }

    #[must_use]
    pub const fn en_passant(&self) -> Option<Square> {
        self.en_passant
    }

    #[must_use]
    pub const fn halfmove_clock(&self) -> u32 {
        self.halfmove_clock
    }

    #[must_use]
    pub const fn fullmove_number(&self) -> u32 {
        self.fullmove_number
    }

    #[must_use]
    pub fn board(&self) -> &[Option<Piece>; 64] {
        &self.board
    }

    #[must_use]
    pub fn repetition_history(&self) -> &[PositionKey] {
        &self.repetition_history
    }

    #[must_use]
    pub fn current_key(&self) -> PositionKey {
        PositionKey {
            board: self.board,
            side_to_move: self.side_to_move,
            castling_rights: self.castling_rights,
            en_passant: self.en_passant,
        }
    }

    pub fn piece_at(&self, square: Square) -> Option<Piece> {
        self.board[usize::from(square.index())]
    }

    pub fn set_piece_at(&mut self, square: Square, piece: Option<Piece>) {
        self.board[usize::from(square.index())] = piece;
    }

    pub fn set_side_to_move(&mut self, side_to_move: Color) {
        self.side_to_move = side_to_move;
    }

    pub fn set_castling_rights(&mut self, castling_rights: CastlingRights) {
        self.castling_rights = castling_rights;
    }

    pub fn set_en_passant(&mut self, en_passant: Option<Square>) {
        self.en_passant = en_passant;
    }

    pub fn set_halfmove_clock(&mut self, halfmove_clock: u32) {
        self.halfmove_clock = halfmove_clock;
    }

    pub fn set_fullmove_number(&mut self, fullmove_number: u32) {
        self.fullmove_number = fullmove_number;
    }

    pub fn set_repetition_history(&mut self, repetition_history: Vec<PositionKey>) {
        self.repetition_history = repetition_history;
    }

    pub fn push_current_key(&mut self) {
        self.repetition_history.push(self.current_key());
    }

    #[must_use]
    pub fn repetition_count(&self) -> usize {
        let current = self.current_key();
        self.repetition_history
            .iter()
            .filter(|entry| **entry == current)
            .count()
    }

    pub fn iter_pieces(&self) -> impl Iterator<Item = (Square, Piece)> + '_ {
        self.board.iter().enumerate().filter_map(|(index, piece)| {
            piece.map(|piece| {
                let square = Square::new(index as u8).expect("board indices are always in range");
                (square, piece)
            })
        })
    }

    #[must_use]
    pub fn king_square(&self, color: Color) -> Option<Square> {
        self.iter_pieces()
            .find(|(_, piece)| piece.color == color && piece.kind == PieceKind::King)
            .map(|(square, _)| square)
    }
}

fn parse_board(board_field: &str) -> Result<[Option<Piece>; 64], FenError> {
    let ranks: Vec<&str> = board_field.split('/').collect();
    if ranks.len() != 8 {
        return Err(FenError::WrongRankCount);
    }

    let mut board = [None; 64];
    for (rank_offset, rank_str) in ranks.iter().enumerate() {
        let rank_index = 7_u8.saturating_sub(rank_offset as u8);
        let mut file_index = 0_u8;

        for symbol in rank_str.chars() {
            if symbol.is_ascii_digit() {
                let count = symbol.to_digit(10).ok_or(FenError::InvalidBoardEncoding)? as u8;
                if count == 0 {
                    return Err(FenError::InvalidBoardEncoding);
                }
                file_index = file_index
                    .checked_add(count)
                    .ok_or(FenError::InvalidBoardEncoding)?;
                continue;
            }

            let piece = Piece::from_fen_char(symbol).ok_or(FenError::InvalidPiece(symbol))?;
            if file_index >= 8 {
                return Err(FenError::InvalidBoardEncoding);
            }
            let square = Square::from_coords(
                core_types::File::from_index(file_index).expect("file is range-checked"),
                Rank::from_index(rank_index).expect("rank is range-checked"),
            );
            board[usize::from(square.index())] = Some(piece);
            file_index += 1;
        }

        if file_index != 8 {
            return Err(FenError::InvalidBoardEncoding);
        }
    }

    Ok(board)
}

fn parse_side_to_move(field: &str) -> Result<Color, FenError> {
    let mut chars = field.chars();
    let symbol = chars.next().ok_or(FenError::InvalidSideToMove)?;
    if chars.next().is_some() {
        return Err(FenError::InvalidSideToMove);
    }
    Color::from_fen_symbol(symbol).ok_or(FenError::InvalidSideToMove)
}

fn parse_en_passant(field: &str) -> Result<Option<Square>, FenError> {
    if field == "-" {
        return Ok(None);
    }

    let square = Square::from_algebraic(field).map_err(FenError::InvalidEnPassant)?;
    let rank = square.rank().index();
    if rank != 2 && rank != 5 {
        return Err(FenError::InvalidEnPassant(SquareParseError));
    }
    Ok(Some(square))
}

/// Errors returned while parsing or serializing FEN.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FenError {
    WrongFieldCount,
    WrongRankCount,
    InvalidBoardEncoding,
    InvalidPiece(char),
    InvalidSideToMove,
    InvalidCastlingRights,
    InvalidEnPassant(SquareParseError),
    InvalidHalfmoveClock,
    InvalidFullmoveNumber,
}

impl fmt::Display for FenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongFieldCount => f.write_str("expected six FEN fields"),
            Self::WrongRankCount => f.write_str("expected eight board ranks"),
            Self::InvalidBoardEncoding => f.write_str("invalid board encoding"),
            Self::InvalidPiece(symbol) => write!(f, "invalid piece symbol '{symbol}'"),
            Self::InvalidSideToMove => f.write_str("invalid side-to-move field"),
            Self::InvalidCastlingRights => f.write_str("invalid castling-rights field"),
            Self::InvalidEnPassant(_) => f.write_str("invalid en-passant field"),
            Self::InvalidHalfmoveClock => f.write_str("invalid halfmove clock"),
            Self::InvalidFullmoveNumber => f.write_str("invalid fullmove number"),
        }
    }
}

impl Error for FenError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidEnPassant(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Position, CLASSICAL_STARTPOS_FEN};
    use core_types::Square;

    #[test]
    fn start_position_roundtrips_to_fen() {
        let position = Position::from_fen(CLASSICAL_STARTPOS_FEN).expect("valid start position");
        assert_eq!(position.to_fen(), CLASSICAL_STARTPOS_FEN);
    }

    #[test]
    fn custom_position_roundtrips_to_fen() {
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPB1PPP/R3K2R w KQkq - 0 1";
        let position = Position::from_fen(fen).expect("valid fen");
        assert_eq!(position.to_fen(), fen);
    }

    #[test]
    fn repetition_history_starts_with_current_position() {
        let position = Position::startpos();
        assert_eq!(position.repetition_history().len(), 1);
        assert_eq!(position.repetition_count(), 1);
    }

    #[test]
    fn piece_lookup_uses_a1_zero_based_mapping() {
        let position = Position::startpos();
        let a1 = Square::from_algebraic("a1").expect("valid square");
        let e8 = Square::from_algebraic("e8").expect("valid square");
        assert_eq!(position.piece_at(a1).expect("piece on a1").fen_char(), 'R');
        assert_eq!(position.piece_at(e8).expect("piece on e8").fen_char(), 'k');
    }
}
