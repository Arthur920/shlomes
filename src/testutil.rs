//! Shared test-only helpers.
//!
//! The eval harnesses across the crate all read a `//`-commented JSONL corpus
//! (one labelled case per line, blank lines and `//` comments ignored). This
//! centralises that convention so every harness skips comments and blanks the
//! same way and only differs in how it deserialises a row.
#![cfg(test)]

/// Iterate the data rows of a `//`-commented JSONL corpus: yields
/// `(line_number, row)` for each non-blank, non-comment line, where
/// `line_number` is 1-based (so a parse panic can point at the source line).
pub fn corpus_rows(corpus: &str) -> impl Iterator<Item = (usize, &str)> {
    corpus.lines().enumerate().filter_map(|(i, raw)| {
        let raw = raw.trim();
        (!raw.is_empty() && !raw.starts_with("//")).then_some((i + 1, raw))
    })
}
