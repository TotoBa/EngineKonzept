use std::io::{self, BufRead, Write};

use encoder::StateContextV1;
use position::Position;
use serde::Serialize;

#[derive(Serialize)]
struct StateContextDumpOutput {
    fen: String,
    feature_values: Vec<f64>,
    edge_src_square: Vec<u8>,
    edge_dst_square: Vec<u8>,
    edge_piece_type: Vec<u8>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    for raw_line in stdin.lock().lines() {
        let fen = raw_line?;
        let fen = fen.trim();
        if fen.is_empty() {
            continue;
        }
        let position = Position::from_fen(fen)?;
        let context = StateContextV1::from_position(&position);
        let output = StateContextDumpOutput {
            fen: fen.to_string(),
            feature_values: context.feature_values,
            edge_src_square: context.edge_src_square,
            edge_dst_square: context.edge_dst_square,
            edge_piece_type: context.edge_piece_type,
        };
        serde_json::to_writer(&mut stdout, &output)?;
        writeln!(&mut stdout)?;
    }
    stdout.flush()?;
    Ok(())
}
