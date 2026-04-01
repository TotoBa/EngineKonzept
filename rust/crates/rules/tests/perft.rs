use position::Position;
use rules::perft;

#[test]
fn standard_reference_positions_match_known_perft_counts() {
    let fixtures = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../tests/perft/reference_positions.txt"
    ));

    for line in fixtures.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let mut parts = trimmed.split('|');
        let name = parts.next().expect("fixture name");
        let fen = parts.next().expect("fixture fen");
        let position = Position::from_fen(fen)
            .unwrap_or_else(|error| panic!("invalid perft fixture {name}: {error}"));

        for entry in parts {
            let (depth, expected) = entry
                .split_once('=')
                .unwrap_or_else(|| panic!("invalid perft entry in {name}: {entry}"));
            let depth = depth
                .parse::<u32>()
                .unwrap_or_else(|_| panic!("invalid depth in {name}: {depth}"));
            let expected = expected
                .parse::<u64>()
                .unwrap_or_else(|_| panic!("invalid count in {name}: {expected}"));

            assert_eq!(perft(&position, depth), expected, "{name} depth {depth}");
        }
    }
}
