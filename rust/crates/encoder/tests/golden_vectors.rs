use std::{fs, path::PathBuf};

use encoder::{pack_position_features, POSITION_FEATURE_SIZE};
use position::Position;

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../artifacts/golden/encoder_golden_v1.json")
}

#[test]
fn packed_features_match_encoder_goldens_exactly() {
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
        let expected_values = example["features"]
            .as_array()
            .expect("golden example must contain features");
        assert_eq!(expected_values.len(), POSITION_FEATURE_SIZE);

        let position = Position::from_fen(fen).expect("golden FEN should be valid");
        let actual = pack_position_features(&position);
        let expected: Vec<f32> = expected_values
            .iter()
            .map(|value| value.as_f64().expect("feature must be numeric") as f32)
            .collect();

        assert_eq!(
            actual.as_slice(),
            expected.as_slice(),
            "golden drift for {fen}"
        );
    }
}
