//! Placeholder crate for future model-facing move vocabulary and tokenization.

/// Returns the purpose of this crate during Phase 0.
pub fn crate_purpose() -> &'static str {
    "Action-space placeholders"
}

#[cfg(test)]
mod tests {
    use super::crate_purpose;

    #[test]
    fn purpose_is_non_empty() {
        assert!(!crate_purpose().is_empty());
    }
}
