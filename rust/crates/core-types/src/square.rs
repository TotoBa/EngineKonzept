use std::error::Error;
use std::fmt;

/// File on the chessboard, from `a` to `h`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum File {
    A = 0,
    B = 1,
    C = 2,
    D = 3,
    E = 4,
    F = 5,
    G = 6,
    H = 7,
}

impl File {
    #[must_use]
    pub const fn index(self) -> u8 {
        self as u8
    }

    pub fn from_index(index: u8) -> Option<Self> {
        match index {
            0 => Some(Self::A),
            1 => Some(Self::B),
            2 => Some(Self::C),
            3 => Some(Self::D),
            4 => Some(Self::E),
            5 => Some(Self::F),
            6 => Some(Self::G),
            7 => Some(Self::H),
            _ => None,
        }
    }

    pub fn from_char(value: char) -> Option<Self> {
        match value.to_ascii_lowercase() {
            'a' => Some(Self::A),
            'b' => Some(Self::B),
            'c' => Some(Self::C),
            'd' => Some(Self::D),
            'e' => Some(Self::E),
            'f' => Some(Self::F),
            'g' => Some(Self::G),
            'h' => Some(Self::H),
            _ => None,
        }
    }

    #[must_use]
    pub const fn to_char(self) -> char {
        (b'a' + self.index()) as char
    }
}

impl fmt::Display for File {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_char())
    }
}

/// Rank on the chessboard, from `1` to `8`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Rank {
    First = 0,
    Second = 1,
    Third = 2,
    Fourth = 3,
    Fifth = 4,
    Sixth = 5,
    Seventh = 6,
    Eighth = 7,
}

impl Rank {
    #[must_use]
    pub const fn index(self) -> u8 {
        self as u8
    }

    pub fn from_index(index: u8) -> Option<Self> {
        match index {
            0 => Some(Self::First),
            1 => Some(Self::Second),
            2 => Some(Self::Third),
            3 => Some(Self::Fourth),
            4 => Some(Self::Fifth),
            5 => Some(Self::Sixth),
            6 => Some(Self::Seventh),
            7 => Some(Self::Eighth),
            _ => None,
        }
    }

    pub fn from_char(value: char) -> Option<Self> {
        match value {
            '1' => Some(Self::First),
            '2' => Some(Self::Second),
            '3' => Some(Self::Third),
            '4' => Some(Self::Fourth),
            '5' => Some(Self::Fifth),
            '6' => Some(Self::Sixth),
            '7' => Some(Self::Seventh),
            '8' => Some(Self::Eighth),
            _ => None,
        }
    }

    #[must_use]
    pub const fn to_char(self) -> char {
        (b'1' + self.index()) as char
    }
}

impl fmt::Display for Rank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_char())
    }
}

/// Board square encoded as `a1 = 0` through `h8 = 63`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Square(u8);

impl Square {
    pub const fn new(index: u8) -> Option<Self> {
        if index < 64 {
            Some(Self(index))
        } else {
            None
        }
    }

    #[must_use]
    pub const fn index(self) -> u8 {
        self.0
    }

    #[must_use]
    pub const fn file(self) -> File {
        match self.0 % 8 {
            0 => File::A,
            1 => File::B,
            2 => File::C,
            3 => File::D,
            4 => File::E,
            5 => File::F,
            6 => File::G,
            _ => File::H,
        }
    }

    #[must_use]
    pub const fn rank(self) -> Rank {
        match self.0 / 8 {
            0 => Rank::First,
            1 => Rank::Second,
            2 => Rank::Third,
            3 => Rank::Fourth,
            4 => Rank::Fifth,
            5 => Rank::Sixth,
            6 => Rank::Seventh,
            _ => Rank::Eighth,
        }
    }

    #[must_use]
    pub const fn from_coords(file: File, rank: Rank) -> Self {
        Self(rank.index() * 8 + file.index())
    }

    pub fn from_algebraic(value: &str) -> Result<Self, SquareParseError> {
        let mut chars = value.chars();
        let file_char = chars.next().ok_or(SquareParseError)?;
        let rank_char = chars.next().ok_or(SquareParseError)?;
        if chars.next().is_some() {
            return Err(SquareParseError);
        }

        let file = File::from_char(file_char).ok_or(SquareParseError)?;
        let rank = Rank::from_char(rank_char).ok_or(SquareParseError)?;
        Ok(Self::from_coords(file, rank))
    }

    pub fn offset(self, file_delta: i8, rank_delta: i8) -> Option<Self> {
        let file = i16::from(self.file().index()) + i16::from(file_delta);
        let rank = i16::from(self.rank().index()) + i16::from(rank_delta);
        if !(0..=7).contains(&file) || !(0..=7).contains(&rank) {
            return None;
        }

        Some(Self::from_coords(
            File::from_index(file as u8).expect("validated file"),
            Rank::from_index(rank as u8).expect("validated rank"),
        ))
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.file(), self.rank())
    }
}

/// Error returned when parsing a square from algebraic notation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SquareParseError;

impl fmt::Display for SquareParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("invalid square")
    }
}

impl Error for SquareParseError {}
