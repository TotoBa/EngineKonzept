use crate::FenError;

/// Castling-rights state carried by the exact position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct CastlingRights {
    white_kingside: bool,
    white_queenside: bool,
    black_kingside: bool,
    black_queenside: bool,
}

impl CastlingRights {
    #[must_use]
    pub const fn new(
        white_kingside: bool,
        white_queenside: bool,
        black_kingside: bool,
        black_queenside: bool,
    ) -> Self {
        Self {
            white_kingside,
            white_queenside,
            black_kingside,
            black_queenside,
        }
    }

    #[must_use]
    pub const fn white_kingside(self) -> bool {
        self.white_kingside
    }

    #[must_use]
    pub const fn white_queenside(self) -> bool {
        self.white_queenside
    }

    #[must_use]
    pub const fn black_kingside(self) -> bool {
        self.black_kingside
    }

    #[must_use]
    pub const fn black_queenside(self) -> bool {
        self.black_queenside
    }

    pub fn clear_white(&mut self) {
        self.white_kingside = false;
        self.white_queenside = false;
    }

    pub fn clear_black(&mut self) {
        self.black_kingside = false;
        self.black_queenside = false;
    }

    pub fn clear_white_kingside(&mut self) {
        self.white_kingside = false;
    }

    pub fn clear_white_queenside(&mut self) {
        self.white_queenside = false;
    }

    pub fn clear_black_kingside(&mut self) {
        self.black_kingside = false;
    }

    pub fn clear_black_queenside(&mut self) {
        self.black_queenside = false;
    }

    #[must_use]
    pub fn to_fen(self) -> String {
        let mut rights = String::new();
        if self.white_kingside {
            rights.push('K');
        }
        if self.white_queenside {
            rights.push('Q');
        }
        if self.black_kingside {
            rights.push('k');
        }
        if self.black_queenside {
            rights.push('q');
        }
        if rights.is_empty() {
            rights.push('-');
        }
        rights
    }
}

pub(crate) fn parse_castling_rights(field: &str) -> Result<CastlingRights, FenError> {
    if field == "-" {
        return Ok(CastlingRights::default());
    }

    let mut rights = CastlingRights::default();
    for symbol in field.chars() {
        match symbol {
            'K' if !rights.white_kingside => rights.white_kingside = true,
            'Q' if !rights.white_queenside => rights.white_queenside = true,
            'k' if !rights.black_kingside => rights.black_kingside = true,
            'q' if !rights.black_queenside => rights.black_queenside = true,
            _ => return Err(FenError::InvalidCastlingRights),
        }
    }
    Ok(rights)
}
