fn main() {
    if let Err(error) = engine_app::run_stdio() {
        eprintln!("engine-app failed: {error}");
        std::process::exit(1);
    }
}
