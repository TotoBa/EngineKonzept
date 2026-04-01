use std::io;

use tools::process_json_lines;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    process_json_lines(stdin.lock(), stdout.lock(), "dataset-oracle")?;
    Ok(())
}
