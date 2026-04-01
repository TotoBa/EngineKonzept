use std::fmt;

/// Chess side to move or piece color.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Color {
    White,
    Black,
}

impl Color {
    /// Returns the opposite side.
    #[must_use]
    pub const fn opposite(self) -> Self {
        match self {
            Self::White => Self::Black,
            Self::Black => Self::White,
        }
    }

    /// Returns the canonical FEN symbol for the side to move field.
    #[must_use]
    pub const fn fen_symbol(self) -> char {
        match self {
            Self::White => 'w',
            Self::Black => 'b',
        }
    }

    /// Parses the FEN side-to-move symbol.
    pub fn from_fen_symbol(value: char) -> Option<Self> {
        match value {
            'w' => Some(Self::White),
            'b' => Some(Self::Black),
            _ => None,
        }
    }

    /// Returns the pawn forward direction in ranks.
    #[must_use]
    pub const fn pawn_push_delta(self) -> i8 {
        match self {
            Self::White => 1,
            Self::Black => -1,
        }
    }

    /// Returns the rank index of the pawn home rank.
    #[must_use]
    pub const fn pawn_home_rank(self) -> u8 {
        match self {
            Self::White => 1,
            Self::Black => 6,
        }
    }

    /// Returns the rank index of the promotion rank.
    #[must_use]
    pub const fn promotion_rank(self) -> u8 {
        match self {
            Self::White => 7,
            Self::Black => 0,
        }
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::White => "white",
            Self::Black => "black",
        })
    }
}
