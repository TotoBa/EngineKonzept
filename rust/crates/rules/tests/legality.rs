use std::collections::{BTreeMap, BTreeSet};

use core_types::Square;
use position::Position;
use rules::{apply_move, legal_moves, pseudo_legal_moves};

fn edge_case_positions() -> BTreeMap<&'static str, &'static str> {
    include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../tests/positions/edge_cases.txt"
    ))
    .lines()
    .filter_map(|line| {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            return None;
        }

        let (name, fen) = trimmed.split_once('|')?;
        Some((name, fen))
    })
    .collect()
}

fn position(name: &str) -> Position {
    let fen = edge_case_positions()
        .get(name)
        .copied()
        .unwrap_or_else(|| panic!("missing test position '{name}'"));
    Position::from_fen(fen).unwrap_or_else(|error| panic!("invalid fixture {name}: {error}"))
}

fn legal_move_set(position: &Position) -> BTreeSet<String> {
    legal_moves(position)
        .into_iter()
        .map(|candidate| candidate.to_uci())
        .collect()
}

fn pseudo_move_count_from(position: &Position, source: &str) -> usize {
    let source = Square::from_algebraic(source).expect("valid test square");
    pseudo_legal_moves(position)
        .into_iter()
        .filter(|candidate| candidate.from == source)
        .count()
}

fn apply_uci(position: &Position, uci: &str) -> Position {
    let chess_move = legal_moves(position)
        .into_iter()
        .find(|candidate| candidate.to_uci() == uci)
        .unwrap_or_else(|| panic!("missing legal move '{uci}'"));
    apply_move(position, chess_move).expect("legal move should apply")
}

#[test]
fn piece_move_generators_cover_all_core_piece_types() {
    assert_eq!(pseudo_move_count_from(&position("pawn_center"), "d2"), 2);
    assert_eq!(pseudo_move_count_from(&position("knight_center"), "d5"), 8);
    assert_eq!(pseudo_move_count_from(&position("bishop_center"), "d5"), 13);
    assert_eq!(pseudo_move_count_from(&position("rook_center"), "d5"), 14);
    assert_eq!(pseudo_move_count_from(&position("queen_center"), "d5"), 27);
    assert_eq!(pseudo_move_count_from(&position("king_center"), "d5"), 8);
}

#[test]
fn castling_is_generated_only_when_the_path_is_safe() {
    let open = legal_move_set(&position("castle_open"));
    assert!(open.contains("e1g1"));
    assert!(open.contains("e1c1"));

    let attacked = legal_move_set(&position("castle_kingside_attacked"));
    assert!(!attacked.contains("e1g1"));
    assert!(attacked.contains("e1c1"));
}

#[test]
fn illegal_en_passant_is_filtered_when_it_exposes_the_king() {
    let moves = legal_move_set(&position("illegal_en_passant_pin"));
    assert!(!moves.contains("e5d6"));
}

#[test]
fn promotion_moves_include_all_required_underpromotions() {
    let pushes = legal_move_set(&position("promotion_lane"));
    assert!(pushes.contains("g7g8q"));
    assert!(pushes.contains("g7g8r"));
    assert!(pushes.contains("g7g8b"));
    assert!(pushes.contains("g7g8n"));

    let captures = legal_move_set(&position("promotion_capture"));
    assert!(captures.contains("a7b8q"));
    assert!(captures.contains("a7b8r"));
    assert!(captures.contains("a7b8b"));
    assert!(captures.contains("a7b8n"));
}

#[test]
fn check_evasions_are_restricted_correctly() {
    let single = legal_move_set(&position("single_check"));
    assert!(single.contains("e1e2"));
    assert!(single.contains("e1f1"));
    assert!(single.contains("e1d1"));
    assert!(!single.contains("e1f2"));

    let double = legal_moves(&position("double_check"));
    assert!(double
        .iter()
        .all(|candidate| candidate.from.to_string() == "e1"));
}

#[test]
fn repetition_history_tracks_repeated_exact_positions() {
    let mut position = Position::startpos();
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"] {
        position = apply_uci(&position, uci);
    }

    assert_eq!(position.current_key(), Position::startpos().current_key());
    assert_eq!(position.repetition_count(), 2);
    assert_eq!(position.repetition_history().len(), 5);
}
