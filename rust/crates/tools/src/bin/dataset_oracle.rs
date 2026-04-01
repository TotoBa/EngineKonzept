use std::io::{self, BufRead, Write};

use tools::label_json_line;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut writer = stdout.lock();

    for (line_number, line) in stdin.lock().lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let output = label_json_line(&line).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("dataset-oracle line {}: {error}", line_number + 1),
            )
        })?;
        writeln!(writer, "{output}")?;
    }

    writer.flush()?;
    Ok(())
}
