//! Exact attack queries, move generation, move application, and perft helpers.

use std::error::Error;
use std::fmt;

use core_types::{Color, Move, MoveKind, Piece, PieceKind, Square};
use position::{CastlingRights, Position};

const KNIGHT_DELTAS: [(i8, i8); 8] = [
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
];

const KING_DELTAS: [(i8, i8); 8] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];

const BISHOP_DIRECTIONS: [(i8, i8); 4] = [(-1, -1), (-1, 1), (1, -1), (1, 1)];
const ROOK_DIRECTIONS: [(i8, i8); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

/// Error returned when a move cannot be applied to a position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MoveError {
    NoPieceAtSource(Square),
    WrongSideToMove,
    OccupiedTargetSquare(Square),
    MissingCaptureTarget(Square),
    CannotCaptureOwnPiece,
    InvalidPromotionPiece(PieceKind),
    InvalidMoveKind,
    InvalidEnPassantSquare,
    MissingEnPassantPawn,
    MissingCastlingRook,
    IllegalMove,
}

impl fmt::Display for MoveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoPieceAtSource(square) => write!(f, "no piece at source square {square}"),
            Self::WrongSideToMove => f.write_str("move does not match side to move"),
            Self::OccupiedTargetSquare(square) => {
                write!(f, "target square {square} is unexpectedly occupied")
            }
            Self::MissingCaptureTarget(square) => {
                write!(f, "capture target on {square} is missing")
            }
            Self::CannotCaptureOwnPiece => f.write_str("cannot capture a friendly piece"),
            Self::InvalidPromotionPiece(piece) => {
                write!(f, "invalid promotion piece {piece}")
            }
            Self::InvalidMoveKind => f.write_str("move kind is incompatible with source piece"),
            Self::InvalidEnPassantSquare => f.write_str("invalid en-passant target square"),
            Self::MissingEnPassantPawn => f.write_str("missing en-passant pawn to capture"),
            Self::MissingCastlingRook => f.write_str("missing rook required for castling"),
            Self::IllegalMove => f.write_str("move is not legal in the current position"),
        }
    }
}

impl Error for MoveError {}

/// Returns whether `square` is attacked by `attacker`.
#[must_use]
pub fn is_square_attacked(position: &Position, square: Square, attacker: Color) -> bool {
    let pawn_source_rank_delta = -attacker.pawn_push_delta();
    for file_delta in [-1, 1] {
        if let Some(source) = square.offset(file_delta, pawn_source_rank_delta) {
            if position.piece_at(source) == Some(Piece::new(attacker, PieceKind::Pawn)) {
                return true;
            }
        }
    }

    for (file_delta, rank_delta) in KNIGHT_DELTAS {
        if let Some(source) = square.offset(file_delta, rank_delta) {
            if position.piece_at(source) == Some(Piece::new(attacker, PieceKind::Knight)) {
                return true;
            }
        }
    }

    for (file_delta, rank_delta) in KING_DELTAS {
        if let Some(source) = square.offset(file_delta, rank_delta) {
            if position.piece_at(source) == Some(Piece::new(attacker, PieceKind::King)) {
                return true;
            }
        }
    }

    is_attacked_by_slider(position, square, attacker, &BISHOP_DIRECTIONS, true)
        || is_attacked_by_slider(position, square, attacker, &ROOK_DIRECTIONS, false)
}

/// Returns whether the given side is currently in check.
#[must_use]
pub fn is_in_check(position: &Position, color: Color) -> bool {
    position
        .king_square(color)
        .is_some_and(|square| is_square_attacked(position, square, color.opposite()))
}

/// Generates pseudo-legal moves for the side to move.
#[must_use]
pub fn pseudo_legal_moves(position: &Position) -> Vec<Move> {
    let mut moves = Vec::new();
    let side_to_move = position.side_to_move();

    for (square, piece) in position.iter_pieces() {
        if piece.color != side_to_move {
            continue;
        }

        match piece.kind {
            PieceKind::Pawn => generate_pawn_moves(position, square, piece, &mut moves),
            PieceKind::Knight => generate_knight_moves(position, square, piece, &mut moves),
            PieceKind::Bishop => {
                generate_slider_moves(position, square, piece, &BISHOP_DIRECTIONS, &mut moves)
            }
            PieceKind::Rook => {
                generate_slider_moves(position, square, piece, &ROOK_DIRECTIONS, &mut moves)
            }
            PieceKind::Queen => {
                generate_slider_moves(position, square, piece, &BISHOP_DIRECTIONS, &mut moves);
                generate_slider_moves(position, square, piece, &ROOK_DIRECTIONS, &mut moves);
            }
            PieceKind::King => generate_king_moves(position, square, piece, &mut moves),
        }
    }

    moves
}

/// Generates fully legal moves for the side to move.
#[must_use]
pub fn legal_moves(position: &Position) -> Vec<Move> {
    let moving_side = position.side_to_move();
    pseudo_legal_moves(position)
        .into_iter()
        .filter(|candidate| {
            try_apply_pseudo_move_for_check(position, *candidate)
                .map(|next| !is_in_check(&next, moving_side))
                .unwrap_or(false)
        })
        .collect()
}

/// Applies a legal move and returns the next exact position.
pub fn apply_move(position: &Position, chess_move: Move) -> Result<Position, MoveError> {
    if !legal_moves(position).contains(&chess_move) {
        return Err(MoveError::IllegalMove);
    }
    apply_known_legal_move(position, chess_move)
}

/// Applies a move that is already known to be legal.
pub fn apply_known_legal_move(
    position: &Position,
    chess_move: Move,
) -> Result<Position, MoveError> {
    try_apply_pseudo_move(position, chess_move)
}

/// Counts legal positions reachable within `depth` plies.
#[must_use]
pub fn perft(position: &Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }

    legal_moves(position)
        .into_iter()
        .map(|candidate| {
            let next = try_apply_pseudo_move(position, candidate).expect("legal move must apply");
            perft(&next, depth - 1)
        })
        .sum()
}

/// Returns the perft split for each legal root move.
#[must_use]
pub fn divide(position: &Position, depth: u32) -> Vec<(Move, u64)> {
    let mut result = Vec::new();
    for candidate in legal_moves(position) {
        let next = try_apply_pseudo_move(position, candidate).expect("legal move must apply");
        let count = if depth == 0 {
            1
        } else {
            perft(&next, depth - 1)
        };
        result.push((candidate, count));
    }
    result
}

fn is_attacked_by_slider(
    position: &Position,
    square: Square,
    attacker: Color,
    directions: &[(i8, i8)],
    diagonal: bool,
) -> bool {
    for &(file_delta, rank_delta) in directions {
        let mut current = square;
        while let Some(next) = current.offset(file_delta, rank_delta) {
            current = next;
            if let Some(piece) = position.piece_at(current) {
                if piece.color != attacker {
                    break;
                }

                if diagonal {
                    if matches!(piece.kind, PieceKind::Bishop | PieceKind::Queen) {
                        return true;
                    }
                } else if matches!(piece.kind, PieceKind::Rook | PieceKind::Queen) {
                    return true;
                }
                break;
            }
        }
    }
    false
}

fn generate_pawn_moves(position: &Position, square: Square, piece: Piece, moves: &mut Vec<Move>) {
    let push_delta = piece.color.pawn_push_delta();
    if let Some(one_step) = square.offset(0, push_delta) {
        if position.piece_at(one_step).is_none() {
            if one_step.rank().index() == piece.color.promotion_rank() {
                for promotion in PieceKind::PROMOTION_PIECES {
                    moves.push(Move::new(square, one_step, MoveKind::Promotion(promotion)));
                }
            } else {
                moves.push(Move::new(square, one_step, MoveKind::Quiet));
                if square.rank().index() == piece.color.pawn_home_rank() {
                    if let Some(two_step) = square.offset(0, push_delta * 2) {
                        if position.piece_at(two_step).is_none() {
                            moves.push(Move::new(square, two_step, MoveKind::DoublePawnPush));
                        }
                    }
                }
            }
        }
    }

    for file_delta in [-1, 1] {
        let Some(target) = square.offset(file_delta, push_delta) else {
            continue;
        };

        if let Some(target_piece) = position.piece_at(target) {
            if target_piece.color == piece.color {
                continue;
            }

            if target.rank().index() == piece.color.promotion_rank() {
                for promotion in PieceKind::PROMOTION_PIECES {
                    moves.push(Move::new(
                        square,
                        target,
                        MoveKind::PromotionCapture(promotion),
                    ));
                }
            } else {
                moves.push(Move::new(square, target, MoveKind::Capture));
            }
        } else if position.en_passant() == Some(target) {
            moves.push(Move::new(square, target, MoveKind::EnPassant));
        }
    }
}

fn generate_knight_moves(position: &Position, square: Square, piece: Piece, moves: &mut Vec<Move>) {
    for (file_delta, rank_delta) in KNIGHT_DELTAS {
        let Some(target) = square.offset(file_delta, rank_delta) else {
            continue;
        };

        match position.piece_at(target) {
            None => moves.push(Move::new(square, target, MoveKind::Quiet)),
            Some(target_piece) if target_piece.color != piece.color => {
                moves.push(Move::new(square, target, MoveKind::Capture))
            }
            Some(_) => {}
        }
    }
}

fn generate_slider_moves(
    position: &Position,
    square: Square,
    piece: Piece,
    directions: &[(i8, i8)],
    moves: &mut Vec<Move>,
) {
    for &(file_delta, rank_delta) in directions {
        let mut current = square;
        while let Some(target) = current.offset(file_delta, rank_delta) {
            current = target;
            match position.piece_at(target) {
                None => moves.push(Move::new(square, target, MoveKind::Quiet)),
                Some(target_piece) if target_piece.color != piece.color => {
                    moves.push(Move::new(square, target, MoveKind::Capture));
                    break;
                }
                Some(_) => break,
            }
        }
    }
}

fn generate_king_moves(position: &Position, square: Square, piece: Piece, moves: &mut Vec<Move>) {
    for (file_delta, rank_delta) in KING_DELTAS {
        let Some(target) = square.offset(file_delta, rank_delta) else {
            continue;
        };

        match position.piece_at(target) {
            None => moves.push(Move::new(square, target, MoveKind::Quiet)),
            Some(target_piece) if target_piece.color != piece.color => {
                moves.push(Move::new(square, target, MoveKind::Capture))
            }
            Some(_) => {}
        }
    }

    if is_in_check(position, piece.color) {
        return;
    }

    let rights = position.castling_rights();
    match piece.color {
        Color::White => {
            maybe_add_castle(
                position,
                square,
                piece,
                rights.white_kingside(),
                true,
                moves,
            );
            maybe_add_castle(
                position,
                square,
                piece,
                rights.white_queenside(),
                false,
                moves,
            );
        }
        Color::Black => {
            maybe_add_castle(
                position,
                square,
                piece,
                rights.black_kingside(),
                true,
                moves,
            );
            maybe_add_castle(
                position,
                square,
                piece,
                rights.black_queenside(),
                false,
                moves,
            );
        }
    }
}

fn maybe_add_castle(
    position: &Position,
    king_square: Square,
    king: Piece,
    right_available: bool,
    kingside: bool,
    moves: &mut Vec<Move>,
) {
    if !right_available {
        return;
    }

    let (rook_from, empty_squares, safe_squares, king_target, kind) = match (king.color, kingside) {
        (Color::White, true) => (
            Square::from_algebraic("h1").expect("valid square"),
            [
                Square::from_algebraic("f1").expect("valid square"),
                Square::from_algebraic("g1").expect("valid square"),
                Square::from_algebraic("a1").expect("placeholder"),
            ],
            [
                Square::from_algebraic("f1").expect("valid square"),
                Square::from_algebraic("g1").expect("valid square"),
            ],
            Square::from_algebraic("g1").expect("valid square"),
            MoveKind::CastleKingside,
        ),
        (Color::White, false) => (
            Square::from_algebraic("a1").expect("valid square"),
            [
                Square::from_algebraic("b1").expect("valid square"),
                Square::from_algebraic("c1").expect("valid square"),
                Square::from_algebraic("d1").expect("valid square"),
            ],
            [
                Square::from_algebraic("d1").expect("valid square"),
                Square::from_algebraic("c1").expect("valid square"),
            ],
            Square::from_algebraic("c1").expect("valid square"),
            MoveKind::CastleQueenside,
        ),
        (Color::Black, true) => (
            Square::from_algebraic("h8").expect("valid square"),
            [
                Square::from_algebraic("f8").expect("valid square"),
                Square::from_algebraic("g8").expect("valid square"),
                Square::from_algebraic("a1").expect("placeholder"),
            ],
            [
                Square::from_algebraic("f8").expect("valid square"),
                Square::from_algebraic("g8").expect("valid square"),
            ],
            Square::from_algebraic("g8").expect("valid square"),
            MoveKind::CastleKingside,
        ),
        (Color::Black, false) => (
            Square::from_algebraic("a8").expect("valid square"),
            [
                Square::from_algebraic("b8").expect("valid square"),
                Square::from_algebraic("c8").expect("valid square"),
                Square::from_algebraic("d8").expect("valid square"),
            ],
            [
                Square::from_algebraic("d8").expect("valid square"),
                Square::from_algebraic("c8").expect("valid square"),
            ],
            Square::from_algebraic("c8").expect("valid square"),
            MoveKind::CastleQueenside,
        ),
    };

    if position.piece_at(rook_from) != Some(Piece::new(king.color, PieceKind::Rook)) {
        return;
    }

    let empties = if kingside {
        &empty_squares[..2]
    } else {
        &empty_squares[..3]
    };
    if empties
        .iter()
        .any(|square| position.piece_at(*square).is_some())
    {
        return;
    }

    if safe_squares
        .iter()
        .any(|square| is_square_attacked(position, *square, king.color.opposite()))
    {
        return;
    }

    moves.push(Move::new(king_square, king_target, kind));
}

fn try_apply_pseudo_move(position: &Position, chess_move: Move) -> Result<Position, MoveError> {
    let moving_piece = position
        .piece_at(chess_move.from)
        .ok_or(MoveError::NoPieceAtSource(chess_move.from))?;
    if moving_piece.color != position.side_to_move() {
        return Err(MoveError::WrongSideToMove);
    }

    let mut next = position.clone();
    next.set_en_passant(None);
    next.set_piece_at(chess_move.from, None);

    let mut castling_rights = next.castling_rights();
    let mut captured_piece = None;
    let mut captured_square = None;
    let pawn_move = moving_piece.kind == PieceKind::Pawn;

    match chess_move.kind {
        MoveKind::Quiet => {
            if next.piece_at(chess_move.to).is_some() {
                return Err(MoveError::OccupiedTargetSquare(chess_move.to));
            }
            next.set_piece_at(chess_move.to, Some(moving_piece));
        }
        MoveKind::DoublePawnPush => {
            if moving_piece.kind != PieceKind::Pawn {
                return Err(MoveError::InvalidMoveKind);
            }
            let intermediate = chess_move
                .from
                .offset(0, moving_piece.color.pawn_push_delta())
                .ok_or(MoveError::InvalidMoveKind)?;
            if next.piece_at(intermediate).is_some() || next.piece_at(chess_move.to).is_some() {
                return Err(MoveError::OccupiedTargetSquare(chess_move.to));
            }
            next.set_piece_at(chess_move.to, Some(moving_piece));
            next.set_en_passant(Some(intermediate));
        }
        MoveKind::Capture => {
            let target_piece = next
                .piece_at(chess_move.to)
                .ok_or(MoveError::MissingCaptureTarget(chess_move.to))?;
            if target_piece.color == moving_piece.color {
                return Err(MoveError::CannotCaptureOwnPiece);
            }
            captured_piece = Some(target_piece);
            captured_square = Some(chess_move.to);
            next.set_piece_at(chess_move.to, Some(moving_piece));
        }
        MoveKind::EnPassant => {
            if moving_piece.kind != PieceKind::Pawn {
                return Err(MoveError::InvalidMoveKind);
            }
            if position.en_passant() != Some(chess_move.to) {
                return Err(MoveError::InvalidEnPassantSquare);
            }
            let victim_square = chess_move
                .to
                .offset(0, -moving_piece.color.pawn_push_delta())
                .ok_or(MoveError::InvalidEnPassantSquare)?;
            let victim = next
                .piece_at(victim_square)
                .ok_or(MoveError::MissingEnPassantPawn)?;
            if victim != Piece::new(moving_piece.color.opposite(), PieceKind::Pawn) {
                return Err(MoveError::MissingEnPassantPawn);
            }
            next.set_piece_at(victim_square, None);
            next.set_piece_at(chess_move.to, Some(moving_piece));
            captured_piece = Some(victim);
            captured_square = Some(victim_square);
        }
        MoveKind::CastleKingside | MoveKind::CastleQueenside => {
            if moving_piece.kind != PieceKind::King {
                return Err(MoveError::InvalidMoveKind);
            }
            let (rook_from, rook_to) = rook_squares_for_castle(moving_piece.color, chess_move.kind);
            let rook = next
                .piece_at(rook_from)
                .ok_or(MoveError::MissingCastlingRook)?;
            if rook != Piece::new(moving_piece.color, PieceKind::Rook) {
                return Err(MoveError::MissingCastlingRook);
            }
            if next.piece_at(chess_move.to).is_some() || next.piece_at(rook_to).is_some() {
                return Err(MoveError::OccupiedTargetSquare(chess_move.to));
            }
            next.set_piece_at(chess_move.to, Some(moving_piece));
            next.set_piece_at(rook_from, None);
            next.set_piece_at(rook_to, Some(rook));
        }
        MoveKind::Promotion(promotion) => {
            validate_promotion_piece(promotion)?;
            if moving_piece.kind != PieceKind::Pawn {
                return Err(MoveError::InvalidMoveKind);
            }
            if next.piece_at(chess_move.to).is_some() {
                return Err(MoveError::OccupiedTargetSquare(chess_move.to));
            }
            next.set_piece_at(
                chess_move.to,
                Some(Piece::new(moving_piece.color, promotion)),
            );
        }
        MoveKind::PromotionCapture(promotion) => {
            validate_promotion_piece(promotion)?;
            if moving_piece.kind != PieceKind::Pawn {
                return Err(MoveError::InvalidMoveKind);
            }
            let target_piece = next
                .piece_at(chess_move.to)
                .ok_or(MoveError::MissingCaptureTarget(chess_move.to))?;
            if target_piece.color == moving_piece.color {
                return Err(MoveError::CannotCaptureOwnPiece);
            }
            captured_piece = Some(target_piece);
            captured_square = Some(chess_move.to);
            next.set_piece_at(
                chess_move.to,
                Some(Piece::new(moving_piece.color, promotion)),
            );
        }
    }

    update_castling_rights_for_move(&mut castling_rights, moving_piece, chess_move.from);
    if let Some(square) = captured_square {
        update_castling_rights_for_capture(&mut castling_rights, captured_piece, square);
    }
    next.set_castling_rights(castling_rights);

    let halfmove_clock = if pawn_move || captured_piece.is_some() {
        0
    } else {
        position.halfmove_clock() + 1
    };
    next.set_halfmove_clock(halfmove_clock);
    next.set_fullmove_number(
        position.fullmove_number() + u32::from(position.side_to_move() == Color::Black),
    );
    next.set_side_to_move(position.side_to_move().opposite());
    next.push_current_key();
    Ok(next)
}

fn try_apply_pseudo_move_for_check(
    position: &Position,
    chess_move: Move,
) -> Result<Position, MoveError> {
    let moving_piece = position
        .piece_at(chess_move.from)
        .ok_or(MoveError::NoPieceAtSource(chess_move.from))?;
    if moving_piece.color != position.side_to_move() {
        return Err(MoveError::WrongSideToMove);
    }

    let mut next = position.clone();
    next.set_piece_at(chess_move.from, None);

    match chess_move.kind {
        MoveKind::Quiet => {
            if next.piece_at(chess_move.to).is_some() {
                return Err(MoveError::OccupiedTargetSquare(chess_move.to));
            }
            next.set_piece_at(chess_move.to, Some(moving_piece));
        }
        MoveKind::DoublePawnPush => {
            if moving_piece.kind != PieceKind::Pawn {
                return Err(MoveError::InvalidMoveKind);
            }
            let intermediate = chess_move
                .from
                .offset(0, moving_piece.color.pawn_push_delta())
                .ok_or(MoveError::InvalidMoveKind)?;
            if next.piece_at(intermediate).is_some() || next.piece_at(chess_move.to).is_some() {
                return Err(MoveError::OccupiedTargetSquare(chess_move.to));
            }
            next.set_piece_at(chess_move.to, Some(moving_piece));
        }
        MoveKind::Capture => {
            let target_piece = next
                .piece_at(chess_move.to)
                .ok_or(MoveError::MissingCaptureTarget(chess_move.to))?;
            if target_piece.color == moving_piece.color {
                return Err(MoveError::CannotCaptureOwnPiece);
            }
            next.set_piece_at(chess_move.to, Some(moving_piece));
        }
        MoveKind::EnPassant => {
            if moving_piece.kind != PieceKind::Pawn {
                return Err(MoveError::InvalidMoveKind);
            }
            if position.en_passant() != Some(chess_move.to) {
                return Err(MoveError::InvalidEnPassantSquare);
            }
            let victim_square = chess_move
                .to
                .offset(0, -moving_piece.color.pawn_push_delta())
                .ok_or(MoveError::InvalidEnPassantSquare)?;
            let victim = next
                .piece_at(victim_square)
                .ok_or(MoveError::MissingEnPassantPawn)?;
            if victim != Piece::new(moving_piece.color.opposite(), PieceKind::Pawn) {
                return Err(MoveError::MissingEnPassantPawn);
            }
            next.set_piece_at(victim_square, None);
            next.set_piece_at(chess_move.to, Some(moving_piece));
        }
        MoveKind::CastleKingside | MoveKind::CastleQueenside => {
            if moving_piece.kind != PieceKind::King {
                return Err(MoveError::InvalidMoveKind);
            }
            let (rook_from, rook_to) = rook_squares_for_castle(moving_piece.color, chess_move.kind);
            let rook = next
                .piece_at(rook_from)
                .ok_or(MoveError::MissingCastlingRook)?;
            if rook != Piece::new(moving_piece.color, PieceKind::Rook) {
                return Err(MoveError::MissingCastlingRook);
            }
            if next.piece_at(chess_move.to).is_some() || next.piece_at(rook_to).is_some() {
                return Err(MoveError::OccupiedTargetSquare(chess_move.to));
            }
            next.set_piece_at(chess_move.to, Some(moving_piece));
            next.set_piece_at(rook_from, None);
            next.set_piece_at(rook_to, Some(rook));
        }
        MoveKind::Promotion(promotion) => {
            validate_promotion_piece(promotion)?;
            if moving_piece.kind != PieceKind::Pawn {
                return Err(MoveError::InvalidMoveKind);
            }
            if next.piece_at(chess_move.to).is_some() {
                return Err(MoveError::OccupiedTargetSquare(chess_move.to));
            }
            next.set_piece_at(
                chess_move.to,
                Some(Piece::new(moving_piece.color, promotion)),
            );
        }
        MoveKind::PromotionCapture(promotion) => {
            validate_promotion_piece(promotion)?;
            if moving_piece.kind != PieceKind::Pawn {
                return Err(MoveError::InvalidMoveKind);
            }
            let target_piece = next
                .piece_at(chess_move.to)
                .ok_or(MoveError::MissingCaptureTarget(chess_move.to))?;
            if target_piece.color == moving_piece.color {
                return Err(MoveError::CannotCaptureOwnPiece);
            }
            next.set_piece_at(
                chess_move.to,
                Some(Piece::new(moving_piece.color, promotion)),
            );
        }
    }

    Ok(next)
}

fn update_castling_rights_for_move(
    castling_rights: &mut CastlingRights,
    piece: Piece,
    from: Square,
) {
    match piece.kind {
        PieceKind::King => match piece.color {
            Color::White => castling_rights.clear_white(),
            Color::Black => castling_rights.clear_black(),
        },
        PieceKind::Rook => match (piece.color, from.to_string().as_str()) {
            (Color::White, "a1") => castling_rights.clear_white_queenside(),
            (Color::White, "h1") => castling_rights.clear_white_kingside(),
            (Color::Black, "a8") => castling_rights.clear_black_queenside(),
            (Color::Black, "h8") => castling_rights.clear_black_kingside(),
            _ => {}
        },
        _ => {}
    }
}

fn update_castling_rights_for_capture(
    castling_rights: &mut CastlingRights,
    captured_piece: Option<Piece>,
    captured_square: Square,
) {
    if captured_piece.map(|piece| piece.kind) != Some(PieceKind::Rook) {
        return;
    }

    match captured_square.to_string().as_str() {
        "a1" => castling_rights.clear_white_queenside(),
        "h1" => castling_rights.clear_white_kingside(),
        "a8" => castling_rights.clear_black_queenside(),
        "h8" => castling_rights.clear_black_kingside(),
        _ => {}
    }
}

fn rook_squares_for_castle(color: Color, kind: MoveKind) -> (Square, Square) {
    match (color, kind) {
        (Color::White, MoveKind::CastleKingside) => (
            Square::from_algebraic("h1").expect("valid square"),
            Square::from_algebraic("f1").expect("valid square"),
        ),
        (Color::White, MoveKind::CastleQueenside) => (
            Square::from_algebraic("a1").expect("valid square"),
            Square::from_algebraic("d1").expect("valid square"),
        ),
        (Color::Black, MoveKind::CastleKingside) => (
            Square::from_algebraic("h8").expect("valid square"),
            Square::from_algebraic("f8").expect("valid square"),
        ),
        (Color::Black, MoveKind::CastleQueenside) => (
            Square::from_algebraic("a8").expect("valid square"),
            Square::from_algebraic("d8").expect("valid square"),
        ),
        _ => unreachable!("rook squares are requested only for castling moves"),
    }
}

fn validate_promotion_piece(piece: PieceKind) -> Result<(), MoveError> {
    if matches!(piece, PieceKind::Pawn | PieceKind::King) {
        return Err(MoveError::InvalidPromotionPiece(piece));
    }
    Ok(())
}
