//! Candidate-claim extraction for the Layer 3 judge.
//!
//! Pull behavioural propositions out of doc prose — complete sentences/bullets
//! that reference code and read like assertions — and ground each claim's
//! backtick tokens to the code index. Deliberately heuristic (the NLI judge is
//! the real filter), but it only hands the model *propositions*: soft-wrapped
//! lines are reassembled, and fragments/quoted examples/feature entries dropped.

use std::collections::HashSet;
use std::sync::OnceLock;

use regex::Regex;

use crate::claim::Provenance;
use crate::code::CodeIndex;

use super::ProseClaim;

/// Pull candidate behavioural claims from doc prose: complete sentences/bullets
/// that reference code (an inline backtick span) and read like an assertion. Each
/// claim's backtick tokens are grounded to the code index. Deliberately heuristic
/// — the NLI judge is the filter — but the NLI model is text-trained and brittle,
/// so we only hand it *propositions*: soft-wrapped lines are reassembled into one
/// logical claim (so it isn't judged as a truncated fragment), and sentence
/// fragments and quoted illustrative examples are dropped. Skips fenced code,
/// headings, and table rows.
pub fn candidate_claims(text: &str, doc_path: &str, index: &CodeIndex) -> Vec<ProseClaim> {
    let modules = index.module_set();
    let mut out = Vec::new();
    for (start, block) in logical_lines(text) {
        let cleaned = block
            .trim_start_matches(['-', '*', '>', ' ', '\t'])
            .trim()
            .to_string();
        if cleaned.split_whitespace().count() < 6 || !cleaned.contains('`') {
            continue;
        }
        // The NLI judge can only rule on a complete proposition. Drop the shapes
        // that aren't one: mid-clause fragments (soft-wrap / list continuations),
        // quoted examples of some other rule, `**Bold** — gloss` feature entries,
        // and lowercase-leading list continuations. All skew to false verdicts.
        if is_fragment(&cleaned)
            || is_quoted_example(&cleaned)
            || is_feature_entry(&cleaned)
            || starts_lowercase(&cleaned)
        {
            continue;
        }
        let provenance = ground_claim(&cleaned, index, &modules);
        out.push(ProseClaim {
            text: cleaned,
            doc_ref: format!("{doc_path}:{}", start + 1),
            provenance,
        });
    }
    out
}

/// Collapse markdown prose into logical lines for claim extraction: each list
/// item or paragraph becomes one `(start_line, joined_text)`, with soft-wrapped
/// continuation lines folded in. Fenced code, blank lines, headings, and table
/// rows act as separators (and are never emitted). `start_line` is 0-based.
fn logical_lines(text: &str) -> Vec<(usize, String)> {
    let mut blocks: Vec<(usize, String)> = Vec::new();
    let mut cur: Option<(usize, String)> = None;
    let mut in_fence = false;
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.starts_with("```") || line.starts_with("~~~") {
            in_fence = !in_fence;
            if let Some(b) = cur.take() {
                blocks.push(b);
            }
            continue;
        }
        if in_fence {
            continue;
        }
        // Separators close the current block without starting a new one.
        if line.is_empty() || line.starts_with('#') || line.starts_with('|') {
            if let Some(b) = cur.take() {
                blocks.push(b);
            }
            continue;
        }
        let starts_item = line.starts_with('-')
            || line.starts_with('*')
            || line.starts_with('>')
            || is_numbered_item(line);
        if starts_item {
            if let Some(b) = cur.take() {
                blocks.push(b);
            }
            cur = Some((i, line.to_string()));
        } else if let Some((_, buf)) = cur.as_mut() {
            // Soft-wrapped continuation of the current paragraph/item.
            buf.push(' ');
            buf.push_str(line);
        } else {
            cur = Some((i, line.to_string()));
        }
    }
    if let Some(b) = cur {
        blocks.push(b);
    }
    blocks
}

/// A markdown ordered-list marker: `1.` / `2)` etc. at the start of the line.
fn is_numbered_item(line: &str) -> bool {
    let digits: String = line.chars().take_while(|c| c.is_ascii_digit()).collect();
    !digits.is_empty() && line[digits.len()..].starts_with(['.', ')'])
}

/// True when the line ends mid-clause — a trailing comma/semicolon or a dangling
/// conjunction/preposition/article — i.e. it is a fragment, not a full assertion.
fn is_fragment(s: &str) -> bool {
    let trimmed = s.trim_end_matches(|c: char| c.is_whitespace());
    if trimmed.ends_with(',') || trimmed.ends_with(';') || trimmed.ends_with(':') {
        return true;
    }
    let last = trimmed
        .rsplit(char::is_whitespace)
        .next()
        .unwrap_or("")
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_ascii_lowercase();
    DANGLING.contains(&last.as_str())
}

/// Words that, ending a line, mean the sentence was cut off.
const DANGLING: &[&str] = &[
    "and", "or", "but", "with", "that", "which", "the", "a", "an", "of", "to", "for", "in", "on",
    "at", "by", "from", "as", "into", "than", "then", "if", "when", "where", "while", "via",
];

/// A definition/feature-list entry: a `**bold lead-in**` immediately followed by
/// an em/en-dash gloss (`Local & offline** — the model runs locally`). These
/// describe a feature; they are not checkable propositions about specific code.
/// (The leading `**` is already stripped as a list marker, so we match the close.)
fn is_feature_entry(s: &str) -> bool {
    ["** —", "**—", "** –", "**–"]
        .iter()
        .any(|sep| s.contains(sep))
}

/// A claim that opens with a lowercase letter is a list continuation or sentence
/// fragment — a real assertion opens with a capital or a code span. Leading
/// emphasis markers are unwrapped first; a leading backtick code span is kept.
fn starts_lowercase(s: &str) -> bool {
    let t = s.trim_start_matches(['*', '_', ' ']);
    matches!(t.chars().next(), Some(c) if c.is_ascii_lowercase())
}

/// True when the line's code spans live inside a double-quoted illustrative
/// example (e.g. a rule shown by example: `"`controllers` must not import `db`"`),
/// which asserts nothing about *this* codebase.
fn is_quoted_example(s: &str) -> bool {
    let mut in_quote = false;
    let mut quoted_backtick = false;
    for c in s.chars() {
        match c {
            '"' => {
                if in_quote && quoted_backtick {
                    return true;
                }
                in_quote = !in_quote;
                quoted_backtick = false;
            }
            '`' if in_quote => quoted_backtick = true,
            _ => {}
        }
    }
    false
}

/// Ground a claim's backtick tokens to code: each token that names an indexed
/// symbol becomes a symbol anchor (preferred — survives moves), else a module
/// anchor if it matches a real module path. Tokens that match neither (paths,
/// commands, prose) are ignored.
pub(super) fn ground_claim(line: &str, index: &CodeIndex, modules: &HashSet<String>) -> Provenance {
    let mut prov = Provenance::default();
    for tok in backtick_tokens(line) {
        if let Some(sym) = index.symbols.iter().find(|s| {
            s.qualified_name == tok
                || s.name == tok
                || s.qualified_name.ends_with(&format!("::{tok}"))
        }) {
            if !prov.symbols.contains(&sym.qualified_name) {
                prov.symbols.push(sym.qualified_name.clone());
            }
        } else if let Some(m) = modules.iter().find(|m| crate::rules::matches(m, &tok)) {
            if !prov.modules.contains(m) {
                prov.modules.push(m.clone());
            }
        }
    }
    prov
}

/// All backtick-quoted tokens in a line.
fn backtick_tokens(line: &str) -> Vec<String> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"`([^`]+)`").unwrap());
    re.captures_iter(line).map(|c| c[1].to_string()).collect()
}
