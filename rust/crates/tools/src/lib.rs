//! Placeholder crate for future developer tools such as perft runners,
//! converters, and dataset sanity checks.

/// Returns the purpose of this crate during Phase 0.
pub fn crate_purpose() -> &'static str {
    "Developer-tool placeholders"
}

#[cfg(test)]
mod tests {
    use super::crate_purpose;

    #[test]
    fn purpose_is_non_empty() {
        assert!(!crate_purpose().is_empty());
    }
}
