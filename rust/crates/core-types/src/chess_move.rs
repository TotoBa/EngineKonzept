use std::fmt;

use crate::{PieceKind, Square};

/// Exact move classification used by the symbolic rules layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MoveKind {
    Quiet,
    DoublePawnPush,
    Capture,
    EnPassant,
    CastleKingside,
    CastleQueenside,
    Promotion(PieceKind),
    PromotionCapture(PieceKind),
}

impl MoveKind {
    #[must_use]
    pub const fn is_capture(self) -> bool {
        matches!(
            self,
            Self::Capture | Self::EnPassant | Self::PromotionCapture(_)
        )
    }

    #[must_use]
    pub const fn promotion_piece(self) -> Option<PieceKind> {
        match self {
            Self::Promotion(piece) | Self::PromotionCapture(piece) => Some(piece),
            _ => None,
        }
    }
}

/// Exact move representation shared between move generation and state transitions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Move {
    pub from: Square,
    pub to: Square,
    pub kind: MoveKind,
}

impl Move {
    #[must_use]
    pub const fn new(from: Square, to: Square, kind: MoveKind) -> Self {
        Self { from, to, kind }
    }

    /// Returns the move in UCI notation.
    #[must_use]
    pub fn to_uci(self) -> String {
        let mut uci = format!("{}{}", self.from, self.to);
        if let Some(piece) = self.kind.promotion_piece() {
            uci.push(piece.symbol());
        }
        uci
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_uci())
    }
}
