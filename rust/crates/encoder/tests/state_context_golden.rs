use std::{fs, path::PathBuf};

use encoder::StateContextV1;
use position::Position;

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../artifacts/golden/state_context_v1_golden.json")
}

#[test]
fn state_context_v1_matches_python_goldens_exactly() {
    let payload = fs::read_to_string(golden_path()).expect("golden file should exist");
    let decoded: serde_json::Value =
        serde_json::from_str(&payload).expect("golden file should be valid JSON");
    let examples = decoded["examples"]
        .as_array()
        .expect("golden file should contain an examples array");

    assert_eq!(examples.len(), 10);
    for example in examples {
        let fen = example["fen"]
            .as_str()
            .expect("golden example must contain a fen");
        let position = Position::from_fen(fen).expect("golden FEN should be valid");
        let actual = StateContextV1::from_position(&position);

        let expected_feature_values: Vec<f64> = example["feature_values"]
            .as_array()
            .expect("golden example must contain feature_values")
            .iter()
            .map(|value| value.as_f64().expect("feature value must be numeric"))
            .collect();
        let expected_edge_src_square: Vec<u8> = example["edge_src_square"]
            .as_array()
            .expect("golden example must contain edge_src_square")
            .iter()
            .map(|value| value.as_u64().expect("edge src must be numeric") as u8)
            .collect();
        let expected_edge_dst_square: Vec<u8> = example["edge_dst_square"]
            .as_array()
            .expect("golden example must contain edge_dst_square")
            .iter()
            .map(|value| value.as_u64().expect("edge dst must be numeric") as u8)
            .collect();
        let expected_edge_piece_type: Vec<u8> = example["edge_piece_type"]
            .as_array()
            .expect("golden example must contain edge_piece_type")
            .iter()
            .map(|value| value.as_u64().expect("edge piece type must be numeric") as u8)
            .collect();

        assert_eq!(
            actual.feature_values, expected_feature_values,
            "feature drift for {fen}"
        );
        assert_eq!(
            actual.edge_src_square, expected_edge_src_square,
            "edge_src drift for {fen}"
        );
        assert_eq!(
            actual.edge_dst_square, expected_edge_dst_square,
            "edge_dst drift for {fen}"
        );
        assert_eq!(
            actual.edge_piece_type, expected_edge_piece_type,
            "edge_piece_type drift for {fen}"
        );
        assert_eq!(
            actual.reachability_edges().len(),
            expected_edge_src_square.len(),
            "reachability edge count drift for {fen}"
        );
    }
}
