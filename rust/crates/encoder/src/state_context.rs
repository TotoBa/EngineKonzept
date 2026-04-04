use core_types::{Color, MoveKind, Piece, PieceKind, Square};
use position::Position;
use rules::{legal_moves, pseudo_legal_moves};

pub const STATE_CONTEXT_V1_VERSION: u32 = 1;
pub const STATE_CONTEXT_V1_SQUARE_FEATURE_DIM: usize = 7;
pub const STATE_CONTEXT_V1_GLOBAL_FEATURE_DIM: usize = 11;
pub const STATE_CONTEXT_V1_FEATURE_DIM: usize =
    (64 * STATE_CONTEXT_V1_SQUARE_FEATURE_DIM) + STATE_CONTEXT_V1_GLOBAL_FEATURE_DIM;

pub const STATE_CONTEXT_V1_SQUARE_FEATURE_ORDER: [&str; STATE_CONTEXT_V1_SQUARE_FEATURE_DIM] = [
    "own_attackers_count",
    "opp_attackers_count",
    "reaches_own_king",
    "reaches_opp_king",
    "pin_axis_orthogonal",
    "pin_axis_diagonal",
    "xray_attackers_count",
];
pub const STATE_CONTEXT_V1_GLOBAL_FEATURE_ORDER: [&str; STATE_CONTEXT_V1_GLOBAL_FEATURE_DIM] = [
    "in_check",
    "own_king_attackers_count",
    "opp_king_attackers_count",
    "own_king_escape_squares",
    "opp_king_escape_squares",
    "material_phase",
    "single_legal_move",
    "legal_move_count_normalized",
    "has_legal_castle",
    "has_legal_en_passant",
    "has_legal_promotion",
];

const MATERIAL_PHASE_MAX: f64 = 24.0;
const RAY_DIRECTIONS: [(i8, i8); 8] = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReachabilityEdge {
    pub src_square: u8,
    pub dst_square: u8,
    pub piece_type: u8,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StateContextV1 {
    pub feature_values: Vec<f64>,
    pub edge_src_square: Vec<u8>,
    pub edge_dst_square: Vec<u8>,
    pub edge_piece_type: Vec<u8>,
}

impl StateContextV1 {
    #[must_use]
    pub fn from_position(position: &Position) -> Self {
        let own_color = position.side_to_move();
        let opp_color = own_color.opposite();
        let own_king_square = position
            .king_square(own_color)
            .expect("valid position must contain own king");
        let opp_king_square = position
            .king_square(opp_color)
            .expect("valid position must contain opponent king");
        let legal = legal_moves(position);

        let mut feature_values = Vec::with_capacity(STATE_CONTEXT_V1_FEATURE_DIM);
        for square_index in 0_u8..64 {
            let square = Square::new(square_index).expect("square index must be valid");
            let (pin_axis_orthogonal, pin_axis_diagonal) = pin_axis_flags(position, square);
            feature_values.push(attackers_count(position, own_color, square) as f64);
            feature_values.push(attackers_count(position, opp_color, square) as f64);
            feature_values.push(
                if square_reaches_target_king(position, square, own_king_square) {
                    1.0
                } else {
                    0.0
                },
            );
            feature_values.push(
                if square_reaches_target_king(position, square, opp_king_square) {
                    1.0
                } else {
                    0.0
                },
            );
            feature_values.push(pin_axis_orthogonal);
            feature_values.push(pin_axis_diagonal);
            feature_values.push(xray_attackers_count(position, square) as f64);
        }

        feature_values.extend([
            if rules::is_in_check(position, own_color) {
                1.0
            } else {
                0.0
            },
            attackers_count(position, opp_color, own_king_square) as f64,
            attackers_count(position, own_color, opp_king_square) as f64,
            king_escape_square_count(position, own_color) as f64,
            king_escape_square_count(position, opp_color) as f64,
            material_phase(position),
            if legal.len() == 1 { 1.0 } else { 0.0 },
            legal.len() as f64 / 256.0,
            if legal.iter().any(|candidate| {
                matches!(
                    candidate.kind,
                    MoveKind::CastleKingside | MoveKind::CastleQueenside
                )
            }) {
                1.0
            } else {
                0.0
            },
            if legal
                .iter()
                .any(|candidate| matches!(candidate.kind, MoveKind::EnPassant))
            {
                1.0
            } else {
                0.0
            },
            if legal
                .iter()
                .any(|candidate| candidate.kind.promotion_piece().is_some())
            {
                1.0
            } else {
                0.0
            },
        ]);

        let reachability_edges = build_reachability_edges(position);
        let edge_src_square = reachability_edges
            .iter()
            .map(|edge| edge.src_square)
            .collect();
        let edge_dst_square = reachability_edges
            .iter()
            .map(|edge| edge.dst_square)
            .collect();
        let edge_piece_type = reachability_edges
            .iter()
            .map(|edge| edge.piece_type)
            .collect();

        Self {
            feature_values,
            edge_src_square,
            edge_dst_square,
            edge_piece_type,
        }
    }

    #[must_use]
    pub fn reachability_edges(&self) -> Vec<ReachabilityEdge> {
        self.edge_src_square
            .iter()
            .copied()
            .zip(self.edge_dst_square.iter().copied())
            .zip(self.edge_piece_type.iter().copied())
            .map(|((src_square, dst_square), piece_type)| ReachabilityEdge {
                src_square,
                dst_square,
                piece_type,
            })
            .collect()
    }
}

fn attackers_count(position: &Position, color: Color, target: Square) -> usize {
    position
        .iter_pieces()
        .filter(|(square, piece)| {
            piece.color == color && piece_attacks_square(position, *square, *piece, target)
        })
        .count()
}

fn piece_attacks_square(position: &Position, from: Square, piece: Piece, target: Square) -> bool {
    if from == target {
        return false;
    }

    let file_delta = target.file().index() as i8 - from.file().index() as i8;
    let rank_delta = target.rank().index() as i8 - from.rank().index() as i8;
    match piece.kind {
        PieceKind::Pawn => rank_delta == piece.color.pawn_push_delta() && file_delta.abs() == 1,
        PieceKind::Knight => matches!((file_delta.abs(), rank_delta.abs()), (1, 2) | (2, 1)),
        PieceKind::King => file_delta.abs() <= 1 && rank_delta.abs() <= 1,
        PieceKind::Bishop => {
            file_delta.abs() == rank_delta.abs()
                && ray_clear(
                    position,
                    from,
                    target,
                    file_delta.signum(),
                    rank_delta.signum(),
                )
        }
        PieceKind::Rook => {
            (file_delta == 0 || rank_delta == 0)
                && ray_clear(
                    position,
                    from,
                    target,
                    file_delta.signum(),
                    rank_delta.signum(),
                )
        }
        PieceKind::Queen => {
            ((file_delta == 0 || rank_delta == 0) || file_delta.abs() == rank_delta.abs())
                && ray_clear(
                    position,
                    from,
                    target,
                    file_delta.signum(),
                    rank_delta.signum(),
                )
        }
    }
}

fn ray_clear(
    position: &Position,
    from: Square,
    target: Square,
    file_step: i8,
    rank_step: i8,
) -> bool {
    let mut current = from.offset(file_step, rank_step);
    while let Some(square) = current {
        if square == target {
            return true;
        }
        if position.piece_at(square).is_some() {
            return false;
        }
        current = square.offset(file_step, rank_step);
    }
    false
}

fn square_reaches_target_king(
    position: &Position,
    square: Square,
    target_king_square: Square,
) -> bool {
    position
        .piece_at(square)
        .is_some_and(|piece| piece_attacks_square(position, square, piece, target_king_square))
}

fn pin_axis_flags(position: &Position, square: Square) -> (f64, f64) {
    let Some(piece) = position.piece_at(square) else {
        return (0.0, 0.0);
    };
    if piece.kind == PieceKind::King {
        return (0.0, 0.0);
    }
    let king_square = position
        .king_square(piece.color)
        .expect("valid position must contain own king");
    let file_delta = square.file().index() as i8 - king_square.file().index() as i8;
    let rank_delta = square.rank().index() as i8 - king_square.rank().index() as i8;

    let (file_step, rank_step, orthogonal, diagonal) = if file_delta == 0 && rank_delta != 0 {
        (0, rank_delta.signum(), 1.0, 0.0)
    } else if rank_delta == 0 && file_delta != 0 {
        (file_delta.signum(), 0, 1.0, 0.0)
    } else if file_delta.abs() == rank_delta.abs() && file_delta != 0 {
        (file_delta.signum(), rank_delta.signum(), 0.0, 1.0)
    } else {
        return (0.0, 0.0);
    };

    let mut between = king_square.offset(file_step, rank_step);
    while let Some(current) = between {
        if current == square {
            break;
        }
        if position.piece_at(current).is_some() {
            return (0.0, 0.0);
        }
        between = current.offset(file_step, rank_step);
    }

    let mut beyond = square.offset(file_step, rank_step);
    while let Some(current) = beyond {
        if let Some(attacker) = position.piece_at(current) {
            if attacker.color == piece.color.opposite()
                && slider_matches_ray(attacker.kind, orthogonal == 1.0)
            {
                return (orthogonal, diagonal);
            }
            return (0.0, 0.0);
        }
        beyond = current.offset(file_step, rank_step);
    }
    (0.0, 0.0)
}

fn slider_matches_ray(piece_kind: PieceKind, orthogonal: bool) -> bool {
    if orthogonal {
        matches!(piece_kind, PieceKind::Rook | PieceKind::Queen)
    } else {
        matches!(piece_kind, PieceKind::Bishop | PieceKind::Queen)
    }
}

fn xray_attackers_count(position: &Position, square: Square) -> usize {
    let mut count = 0;
    for (file_step, rank_step) in RAY_DIRECTIONS {
        let mut current = square.offset(file_step, rank_step);
        let mut blocker_seen = false;
        while let Some(target) = current {
            if let Some(piece) = position.piece_at(target) {
                if !blocker_seen {
                    blocker_seen = true;
                } else {
                    if slider_matches_ray(piece.kind, file_step == 0 || rank_step == 0) {
                        count += 1;
                    }
                    break;
                }
            }
            current = target.offset(file_step, rank_step);
        }
    }
    count
}

fn material_phase(position: &Position) -> f64 {
    let phase_units = position
        .iter_pieces()
        .map(|(_, piece)| match piece.kind {
            PieceKind::Knight | PieceKind::Bishop => 1.0,
            PieceKind::Rook => 2.0,
            PieceKind::Queen => 4.0,
            PieceKind::Pawn | PieceKind::King => 0.0,
        })
        .sum::<f64>();
    phase_units / MATERIAL_PHASE_MAX
}

fn king_escape_square_count(position: &Position, color: Color) -> usize {
    let king_square = position
        .king_square(color)
        .expect("valid position must contain king");
    let mut for_color = position.clone();
    for_color.set_side_to_move(color);
    legal_moves(&for_color)
        .into_iter()
        .filter(|candidate| candidate.from == king_square)
        .count()
}

fn build_reachability_edges(position: &Position) -> Vec<ReachabilityEdge> {
    let mut edges = Vec::new();
    for (square, piece) in position.iter_pieces() {
        let mut for_color = position.clone();
        for_color.set_side_to_move(piece.color);
        let mut destinations = pseudo_legal_moves(&for_color)
            .into_iter()
            .filter(|candidate| candidate.from == square)
            .map(|candidate| candidate.to.index())
            .collect::<Vec<_>>();
        destinations.sort_unstable();
        destinations.dedup();
        for destination in destinations {
            edges.push(ReachabilityEdge {
                src_square: square.index(),
                dst_square: destination,
                piece_type: piece_type_code(piece.kind),
            });
        }
    }
    edges.sort_unstable_by_key(|edge| (edge.src_square, edge.dst_square, edge.piece_type));
    edges
}

const fn piece_type_code(piece_kind: PieceKind) -> u8 {
    match piece_kind {
        PieceKind::Pawn => 1,
        PieceKind::Knight => 2,
        PieceKind::Bishop => 3,
        PieceKind::Rook => 4,
        PieceKind::Queen => 5,
        PieceKind::King => 6,
    }
}
