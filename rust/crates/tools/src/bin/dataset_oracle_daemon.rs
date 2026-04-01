use std::env;
use std::fs;
use std::io::{BufReader, BufWriter};
use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};

use tools::process_json_lines;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let socket_path = socket_path_from_args(env::args().skip(1).collect())?;
    if socket_path.exists() {
        fs::remove_file(&socket_path)?;
    }

    let listener = UnixListener::bind(&socket_path)?;
    eprintln!(
        "dataset-oracle-daemon listening on {}",
        socket_path.display()
    );

    for stream in listener.incoming() {
        let stream = match stream {
            Ok(stream) => stream,
            Err(error) => {
                eprintln!("dataset-oracle-daemon accept error: {error}");
                continue;
            }
        };
        let reader = BufReader::new(stream.try_clone()?);
        let writer = BufWriter::new(stream);
        if let Err(error) = process_json_lines(reader, writer, "dataset-oracle-daemon") {
            eprintln!("dataset-oracle-daemon connection error: {error}");
        }
    }

    Ok(())
}

fn socket_path_from_args(args: Vec<String>) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if args.len() != 2 || args[0] != "--socket" {
        return Err("usage: dataset-oracle-daemon --socket /path/to/socket".into());
    }
    let path = Path::new(&args[1]);
    if path.as_os_str().is_empty() {
        return Err("socket path must be non-empty".into());
    }
    Ok(path.to_path_buf())
}
