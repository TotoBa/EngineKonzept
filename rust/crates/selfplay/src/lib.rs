//! Placeholder crate for future selfplay orchestration and data collection.

/// Returns the purpose of this crate during Phase 0.
pub fn crate_purpose() -> &'static str {
    "Selfplay placeholders"
}

#[cfg(test)]
mod tests {
    use super::crate_purpose;

    #[test]
    fn purpose_is_non_empty() {
        assert!(!crate_purpose().is_empty());
    }
}
