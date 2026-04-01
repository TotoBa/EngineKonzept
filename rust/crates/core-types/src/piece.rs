use std::fmt;

use crate::Color;

/// Chess piece without color.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PieceKind {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

impl PieceKind {
    pub const PROMOTION_PIECES: [Self; 4] = [Self::Queen, Self::Rook, Self::Bishop, Self::Knight];

    /// Parses a single FEN piece letter.
    pub fn from_fen_char(value: char) -> Option<(Color, Self)> {
        let color = if value.is_ascii_uppercase() {
            Color::White
        } else {
            Color::Black
        };

        let kind = match value.to_ascii_lowercase() {
            'p' => Self::Pawn,
            'n' => Self::Knight,
            'b' => Self::Bishop,
            'r' => Self::Rook,
            'q' => Self::Queen,
            'k' => Self::King,
            _ => return None,
        };

        Some((color, kind))
    }

    /// Returns the lower-case piece symbol used in FEN and UCI promotion suffixes.
    #[must_use]
    pub const fn symbol(self) -> char {
        match self {
            Self::Pawn => 'p',
            Self::Knight => 'n',
            Self::Bishop => 'b',
            Self::Rook => 'r',
            Self::Queen => 'q',
            Self::King => 'k',
        }
    }

    /// Returns the FEN letter for a piece of the given color.
    #[must_use]
    pub const fn fen_char(self, color: Color) -> char {
        let symbol = self.symbol();
        match color {
            Color::White => symbol.to_ascii_uppercase(),
            Color::Black => symbol,
        }
    }
}

impl fmt::Display for PieceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Pawn => "pawn",
            Self::Knight => "knight",
            Self::Bishop => "bishop",
            Self::Rook => "rook",
            Self::Queen => "queen",
            Self::King => "king",
        })
    }
}

/// Concrete chess piece including color.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Piece {
    pub color: Color,
    pub kind: PieceKind,
}

impl Piece {
    #[must_use]
    pub const fn new(color: Color, kind: PieceKind) -> Self {
        Self { color, kind }
    }

    #[must_use]
    pub const fn fen_char(self) -> char {
        self.kind.fen_char(self.color)
    }

    pub fn from_fen_char(value: char) -> Option<Self> {
        PieceKind::from_fen_char(value).map(|(color, kind)| Self { color, kind })
    }
}

impl fmt::Display for Piece {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.color, self.kind)
    }
}
