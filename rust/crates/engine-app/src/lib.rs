//! Phase-0 placeholder for the future engine binary crate.
//!
//! This crate intentionally contains no chess logic, no UCI handling, and no
//! runtime planner implementation yet.

/// Returns the current bootstrap message for the Phase-0 binary.
pub fn startup_message() -> &'static str {
    "EngineKonzept Phase 0 bootstrap"
}

#[cfg(test)]
mod tests {
    use super::startup_message;

    #[test]
    fn startup_message_is_non_empty() {
        assert!(!startup_message().is_empty());
    }
}
