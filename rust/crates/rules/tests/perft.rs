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

#[test]
fn epd_perft_suite_matches_through_depth_three() {
    run_epd_perft_suite(3);
}

#[test]
#[ignore = "full perftsuite depths are intentionally too expensive for default checks"]
fn epd_perft_suite_matches_all_available_depths() {
    run_epd_perft_suite(u32::MAX);
}

fn run_epd_perft_suite(depth_cap: u32) {
    let fixtures = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../perft-data/perftsuite.epd"
    ));

    for (line_index, line) in fixtures.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let mut fields = trimmed.split(';').filter(|field| !field.is_empty());
        let fen = fields
            .next()
            .unwrap_or_else(|| panic!("missing FEN in EPD line {}", line_index + 1));
        let fen = normalize_epd_position(fen).unwrap_or_else(|| {
            panic!(
                "unsupported EPD position shape on line {}: '{}'",
                line_index + 1,
                fen
            )
        });
        let position = Position::from_fen(&fen)
            .unwrap_or_else(|error| panic!("invalid EPD FEN on line {}: {error}", line_index + 1));

        for (offset, expected) in fields.enumerate() {
            let depth = (offset as u32) + 1;
            if depth > depth_cap {
                break;
            }

            let expected = expected.parse::<u64>().unwrap_or_else(|_| {
                panic!(
                    "invalid node count '{}' on EPD line {} depth {}",
                    expected,
                    line_index + 1,
                    depth
                )
            });

            assert_eq!(
                perft(&position, depth),
                expected,
                "epd line {} depth {}",
                line_index + 1,
                depth
            );
        }
    }
}

fn normalize_epd_position(position: &str) -> Option<String> {
    match position.split_whitespace().count() {
        4 => Some(format!("{position} 0 1")),
        6 => Some(position.to_string()),
        _ => None,
    }
}
