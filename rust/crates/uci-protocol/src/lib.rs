//! Placeholder crate for future UCI protocol parsing and state handling.

/// Returns the purpose of this crate during Phase 0.
pub fn crate_purpose() -> &'static str {
    "UCI parser and protocol state placeholders"
}

#[cfg(test)]
mod tests {
    use super::crate_purpose;

    #[test]
    fn purpose_is_non_empty() {
        assert!(!crate_purpose().is_empty());
    }
}
