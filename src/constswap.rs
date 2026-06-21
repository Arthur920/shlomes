//! Deterministic constant-swap detector (Layer 1).
//!
//! Catches the one drift class the Layer-3 NLI judge is measurably worst at:
//! a doc that binds a *named symbol* to a *specific literal* that the code no
//! longer backs — "`port` defaults to 8080" while the code says
//! `unwrap_or(5432)`. These minimal-pair, one-token logic flips are exactly the
//! adversarial cases the cross-encoder reads as supported (it leans on lexical
//! overlap), so handling them symbolically here both raises recall and keeps the
//! verdict inside Layer 1's zero-false-positive regime.
//!
//! The whole design is built around *not* false-alarming. We only fire when:
//!   1. the prose ties a value to a backticked identifier via an explicit
//!      default/assignment cue ("defaults to", "is set to", "= N", …);
//!   2. exactly one code symbol's `name` matches that identifier (modulo a
//!      `DEFAULT_` affix), so there is no ambiguity about *which* definition; and
//!   3. that symbol carries exactly one distinct literal of the claimed *type*,
//!      and it differs from the claimed value.
//!
//! Integers, bools, and (delimited) strings are compared. A string claim only
//! counts when the prose value is itself delimited (backticked/quoted) and both
//! sides are "plain" — no interpolation, format placeholders, or escapes — since
//! those have no statically-knowable value to contradict. A code-side string is
//! further required to be a single whitespace-free token (a docstring or message
//! is not a value) and not equal to the symbol's own name (a self-referential
//! accessor key is not a default). Bool spellings are matched case-insensitively
//! so a delimited `True`/`False` is typed as a bool, not a string.
//!
//! If the symbol has several competing literals, or none, we stay silent — an
//! ambiguous body is "unverifiable", never "contradicted".

use std::sync::OnceLock;

use regex::Regex;

use crate::claim::Provenance;
use crate::code::CodeIndex;
use crate::findings::{Finding, Verdict};

/// A doc assertion that a named symbol holds a specific literal value.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ConstClaim {
    /// the backticked identifier the value is attributed to.
    symbol: String,
    /// the claimed value, normalized to the same shape as `Facts.constants`
    /// (bare integer/float digits, `true`/`false`, or `"..."` for strings).
    value: Value,
    /// `path:line` of the prose.
    origin: String,
    /// raw matched phrase, for the human report.
    phrase: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Value {
    /// canonical decimal-integer text (suffix/underscore stripped).
    Int(String),
    Bool(bool),
    /// the *inner* content of a string literal (quotes/prefix stripped). Only
    /// ever produced from a delimited prose value, never a bare word.
    Str(String),
}

impl Value {
    /// Canonical comparison/display key. Strings are re-quoted so the report
    /// reads naturally and a string key can never collide with an int/bool one.
    fn render(&self) -> String {
        match self {
            Value::Int(n) => n.clone(),
            Value::Bool(b) => b.to_string(),
            Value::Str(s) => format!("\"{s}\""),
        }
    }
}

/// Entry point, mirroring the other Layer-1 `check(text, doc_path, index)` passes.
pub fn check(markdown: &str, doc_path: &str, index: &CodeIndex) -> Vec<Finding> {
    let mut findings = Vec::new();
    for claim in extract_claims(markdown, doc_path) {
        if let Some(f) = check_claim(&claim, index) {
            findings.push(f);
        }
    }
    findings
}

/// Pull `(symbol, value)` claims out of prose. Two phrasings, both anchored on a
/// backticked identifier so we never guess at what an English noun refers to:
///   * `` `SYM` <cue> <value> `` — "`port` defaults to 8080"
///   * `` `SYM` = <value> ``     — "`retries` = 3"
fn extract_claims(markdown: &str, doc_path: &str) -> Vec<ConstClaim> {
    let mut out = Vec::new();
    // Lines inside ``` / ~~~ fences are code samples, not prose claims — a
    // literal `secret_key="..."` there is an example, not a statement about the
    // symbol's default. Skipping them keeps us inside the zero-FP regime, exactly
    // as the path and config passes do.
    let fenced = crate::extract::fenced_lines(markdown);
    for (i, line) in markdown.lines().enumerate() {
        if fenced[i] {
            continue;
        }
        for caps in claim_re().captures_iter(line) {
            // A backticked identifier is always an intended code anchor. A bare
            // one is only trusted when its *shape* is unambiguously code
            // (`SCREAMING_SNAKE`, `snake_case`, `camelCase`) — a plain English
            // word like "report" or "Default" is rejected, which is where bare
            // matching would otherwise invent false anchors.
            let symbol = match (caps.name("bsym"), caps.name("psym")) {
                (Some(m), _) => m.as_str().to_string(),
                (None, Some(m)) if is_code_shaped(m.as_str()) => m.as_str().to_string(),
                _ => continue,
            };
            // A delimited value (backticked/quoted) may be a string OR a
            // number; a bare value is only ever trusted as a number/bool.
            let delimited = caps.name("dvalb").or_else(|| caps.name("dvalq"));
            let value = if let Some(m) = delimited {
                let Some(v) = parse_delimited(m.as_str()) else {
                    continue;
                };
                v
            } else {
                let Some(v) = parse_value(caps.name("val").unwrap().as_str()) else {
                    continue;
                };
                v
            };
            out.push(ConstClaim {
                symbol,
                value,
                origin: format!("{doc_path}:{}", i + 1),
                phrase: caps.get(0).unwrap().as_str().trim().to_string(),
            });
        }
    }
    out
}

/// `<identifier> <cue> <value>` where the cue is an explicit default /
/// assignment phrase. The identifier is either backticked (`bsym`) or bare
/// (`psym`, later shape-filtered); the value may itself be backticked. We
/// deliberately keep the cue list closed — "is" alone is too loose and would
/// match prose like "`port` is configurable".
///
/// Note `(?i)` is intentionally *not* set here: the identifier groups must stay
/// case-sensitive so `is_code_shaped` can read casing. The cue keywords are
/// instead spelled with explicit `[Dd]`-style classes.
fn claim_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r#"(?x)
            (?:
                  ` (?P<bsym>[A-Za-z_][A-Za-z0-9_]*) `   # backticked identifier
                | \b (?P<psym>[A-Za-z_][A-Za-z0-9_]*)    # bare identifier
            )
            \s*
            (?i:                                      # an explicit value cue (case-insensitive)
                  defaults?\s+to
                | default(?:s|\sis|\sof|:)?
                | is\s+set\s+to
                | is\s+(?:by\s+default\s+)?
                | set\s+to
                | =
            )
            \s*
            (?:
                  ` (?P<dvalb>[^`]+) `                # backtick-delimited value (string or number)
                | " (?P<dvalq>[^"]+) "                # double-quote-delimited value
                | (?P<val>-?\d[\d_]*(?:\.\d+)?|true|false)  # bare numeric / bool
            )
            "#,
        )
        .unwrap()
    })
}

/// Normalize a *bare* matched literal to `Facts.constants` shape, or `None` if
/// it isn't a type we trust un-delimited (only integers and bools — floats carry
/// too much formatting variance, and a bare word is never read as a string).
/// Delimited values go through [`parse_delimited`], which also handles strings.
fn parse_value(raw: &str) -> Option<Value> {
    // Bool spellings are matched case-insensitively so a delimited Python/Go
    // `True`/`False` is typed as a bool rather than falling through to the string
    // path (where it would spuriously compare against, e.g., a property docstring).
    match raw.to_ascii_lowercase().as_str() {
        "true" => return Some(Value::Bool(true)),
        "false" => return Some(Value::Bool(false)),
        _ => {}
    }
    if raw.contains('.') {
        return None; // float: skip, see doc comment.
    }
    canon_int(raw).map(Value::Int)
}

/// Normalize a *delimited* prose value (one written inside backticks or quotes,
/// so the author clearly meant a literal). A delimited value that parses as an
/// int/bool is treated as such; otherwise it is a string, but only a "plain"
/// one — we refuse interpolated/escaped content (`${x}`, `{0}`, `\n`) since its
/// runtime value isn't statically knowable, which would risk a false contradiction.
fn parse_delimited(raw: &str) -> Option<Value> {
    if let Some(v) = parse_value(raw) {
        return Some(v);
    }
    if is_plain_string(raw) {
        return Some(Value::Str(raw.to_string()));
    }
    None
}

/// A string with no interpolation/format/escape machinery — safe to compare for
/// exact equality. Rejects empty/whitespace-only content too.
fn is_plain_string(s: &str) -> bool {
    !s.trim().is_empty() && !s.chars().any(|c| matches!(c, '\\' | '{' | '}' | '$'))
}

/// Strip a string literal's optional prefix (r, f, b, rb, …) and one matching
/// pair of surrounding quotes (double, single, or backtick), returning the inner
/// content — or `None` if `c` isn't a recognizable, plain string literal.
fn unquote_code_string(c: &str) -> Option<String> {
    let c = c.trim();
    // Skip a short ascii-alpha prefix that precedes the opening quote.
    let body = match c.find(['"', '\'', '`']) {
        Some(0) => c,
        Some(i) if i <= 2 && c[..i].chars().all(|ch| ch.is_ascii_alphabetic()) => &c[i..],
        _ => return None,
    };
    let mut chars = body.chars();
    let quote = chars.next()?;
    if !matches!(quote, '"' | '\'' | '`') {
        return None;
    }
    let inner = chars.as_str().strip_suffix(quote)?;
    if !is_plain_string(inner) {
        return None;
    }
    // A genuine default *value* is a single token (`us-east-1`, `production`,
    // `INFO`); a string literal carrying internal whitespace is overwhelmingly a
    // docstring or message, not a value, so it must not ground a value claim.
    // (This is the pydantic `@property` whose lone literal is its docstring.)
    if inner.chars().any(|c| c.is_whitespace()) {
        return None;
    }
    Some(inner.to_string())
}

/// Canonical decimal text for an integer literal: drop `_` separators and any
/// trailing Rust/C type suffix (`5432u16`, `3_000usize`). Returns `None` if the
/// remainder isn't a plain integer.
fn canon_int(raw: &str) -> Option<String> {
    let neg = raw.starts_with('-');
    let digits: String = raw
        .trim_start_matches('-')
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '_')
        .filter(|c| *c != '_')
        .collect();
    if digits.is_empty() {
        return None;
    }
    // Re-parse to fold leading zeros / overlong values to a canonical form.
    let n: i128 = digits.parse().ok()?;
    Some(if neg { format!("-{n}") } else { n.to_string() })
}

/// Decide a single claim against the index, returning a Contradicted finding
/// only when the zero-FP guards all hold.
fn check_claim(claim: &ConstClaim, index: &CodeIndex) -> Option<Finding> {
    // (1) the symbols this claim could be about.
    let matched: Vec<&crate::code::symbol::Symbol> = index
        .symbols
        .iter()
        .filter(|s| name_matches(&s.name, &claim.symbol))
        .collect();
    if matched.is_empty() {
        return None; // nothing to compare against — not our job to flag.
    }

    // (2) the distinct literals of the claimed *type* across those symbols.
    let mut observed: Vec<String> = Vec::new();
    for s in &matched {
        for c in &s.facts.constants {
            if let Some(v) = constant_of_kind(c, &claim.value) {
                // A string literal identical to the symbol's own name is a
                // self-referential accessor key (`get file -> _get("file")`), not
                // a configured default — skip it. (The vite `file` getter FP.)
                if matches!(claim.value, Value::Str(_))
                    && v.eq_ignore_ascii_case(&format!("\"{}\"", s.name))
                {
                    continue;
                }
                if !observed.contains(&v) {
                    observed.push(v);
                }
            }
        }
    }

    let want = claim.value.render();
    // Already correct — record as supported (ledgered + scored, not reported).
    if observed.contains(&want) {
        return Some(Finding::supported(
            claim.phrase.clone(),
            claim.origin.clone(),
            Provenance::symbol(matched[0].qualified_name.clone()),
        ));
    }
    // (3) fire only on an *unambiguous* single competing literal.
    if observed.len() != 1 {
        return None; // zero (nothing to disagree with) or many (ambiguous).
    }
    let found = &observed[0];

    let refs: Vec<String> = matched
        .iter()
        .map(|s| format!("{}:{}", s.span.path, s.span.start_line))
        .collect();
    Some(
        Finding::problem(
            Verdict::Contradicted,
            claim.phrase.clone(),
            claim.origin.clone(),
            format!(
                "doc says `{}` is {want}, but code has {found} (in `{}`)",
                claim.symbol, matched[0].qualified_name,
            ),
        )
        .anchored(Provenance::symbol(matched[0].qualified_name.clone()))
        .with_refs(refs),
    )
}

/// A `Facts.constants` entry, projected onto the claim's type, or `None` if it
/// is a different kind (a string literal when we're comparing an int, etc.).
fn constant_of_kind(constant: &str, want: &Value) -> Option<String> {
    match want {
        Value::Bool(_) => match constant.to_ascii_lowercase().as_str() {
            "true" => Some("true".to_string()),
            "false" => Some("false".to_string()),
            _ => None,
        },
        Value::Int(_) => {
            // Reject quoted strings outright so `"8080"` never matches int 8080.
            if constant.starts_with('"') {
                return None;
            }
            canon_int(constant)
        }
        // Compare only against genuine string literals, re-quoted to the same
        // key shape as `Value::render` so neither side can collide with a number.
        Value::Str(_) => unquote_code_string(constant).map(|s| format!("\"{s}\"")),
    }
}

/// Whether a *bare* (un-backticked) token is unambiguously code-shaped, and so
/// safe to treat as a symbol anchor. True for `snake_case`, `SCREAMING_SNAKE`,
/// `SCREAMINGCASE` (≥2 letters, all caps), and `camelCase`. False for plain
/// English words (`port`, `report`, `Default`) — those need backticks, since a
/// lone lowercase or Capitalized word is where bare matching invents false
/// anchors.
fn is_code_shaped(s: &str) -> bool {
    if s.contains('_') {
        return true; // snake_case / SCREAMING_SNAKE
    }
    let has_upper = s.chars().any(|c| c.is_ascii_uppercase());
    let has_lower = s.chars().any(|c| c.is_ascii_lowercase());
    // ALLCAPS (PORT, HTTP2) — all-uppercase with at least two letters.
    if has_upper && !has_lower && s.chars().filter(|c| c.is_ascii_alphabetic()).count() >= 2 {
        return true;
    }
    // camelCase: a lowercase letter somewhere followed later by an uppercase one
    // (`maxRetries`). PascalCase (`Port`, `Default`) deliberately does not match.
    let mut seen_lower = false;
    for c in s.chars() {
        if c.is_ascii_lowercase() {
            seen_lower = true;
        } else if c.is_ascii_uppercase() && seen_lower {
            return true;
        }
    }
    false
}

/// Case-insensitive identifier match, tolerating a `DEFAULT_`/`_DEFAULT` affix on
/// the code side (`DEFAULT_PORT` vs the doc's `port`). Exact otherwise — we never
/// substring-match, which would invite false anchors.
fn name_matches(code_name: &str, claimed: &str) -> bool {
    let a = code_name.to_ascii_lowercase();
    let b = claimed.to_ascii_lowercase();
    if a == b {
        return true;
    }
    let stripped = a
        .strip_prefix("default_")
        .or_else(|| a.strip_suffix("_default"));
    stripped == Some(b.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code::symbol::{Facts, Span, Symbol, SymbolKind, Visibility};

    fn sym(name: &str, constants: &[&str]) -> Symbol {
        Symbol {
            qualified_name: format!("src::cfg::{name}"),
            name: name.to_string(),
            kind: SymbolKind::Function,
            visibility: Visibility::Public,
            module: "src/cfg".to_string(),
            span: Span {
                path: "src/cfg.rs".to_string(),
                start_line: 10,
                end_line: 12,
            },
            body_span: Span::zero(),
            signature: None,
            doc: None,
            facts: Facts {
                constants: constants.iter().map(|c| c.to_string()).collect(),
                ..Default::default()
            },
            calls: vec![],
            members: vec![],
        }
    }

    fn index_of(symbols: Vec<Symbol>) -> CodeIndex {
        CodeIndex {
            symbols,
            edges: vec![],
            module_edges: vec![],
            ref_callers: Default::default(),
        }
    }

    fn verdicts(md: &str, index: &CodeIndex) -> Vec<Verdict> {
        check(md, "README.md", index)
            .into_iter()
            .map(|f| f.verdict)
            .collect()
    }

    #[test]
    fn flags_a_clear_swap() {
        let idx = index_of(vec![sym("port", &["5432"])]);
        let f = check("The `port` defaults to 8080.", "README.md", &idx);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].verdict, Verdict::Contradicted);
        assert!(f[0].detail.contains("8080"));
        assert!(f[0].detail.contains("5432"));
        assert_eq!(f[0].code_refs, vec!["src/cfg.rs:10"]);
    }

    #[test]
    fn matches_default_prefixed_const() {
        let idx = index_of(vec![sym("DEFAULT_PORT", &["5432"])]);
        assert_eq!(verdicts("`port` = 8080", &idx), vec![Verdict::Contradicted]);
    }

    #[test]
    fn agreement_is_supported_not_reported() {
        let idx = index_of(vec![sym("port", &["8080"])]);
        let v = verdicts("`port` defaults to 8080", &idx);
        assert_eq!(v, vec![Verdict::Supported]);
        // and Supported is filtered from the human report.
        assert!(!v[0].is_reportable());
    }

    #[test]
    fn ambiguous_body_stays_silent() {
        // two competing int literals → we cannot know which the doc means.
        let idx = index_of(vec![sym("port", &["5432", "8080"])]);
        assert!(verdicts("`port` defaults to 1234", &idx).is_empty());
    }

    #[test]
    fn no_matching_symbol_is_silent() {
        let idx = index_of(vec![sym("timeout", &["30"])]);
        assert!(verdicts("`port` defaults to 8080", &idx).is_empty());
    }

    #[test]
    fn unit_suffix_and_underscores_normalize() {
        let idx = index_of(vec![sym("limit", &["10_000usize"])]);
        // doc agrees once underscores/suffix are folded away.
        assert_eq!(
            verdicts("`limit` defaults to 10000", &idx),
            vec![Verdict::Supported]
        );
        assert_eq!(
            verdicts("`limit` defaults to 9999", &idx),
            vec![Verdict::Contradicted]
        );
    }

    #[test]
    fn bool_swap() {
        let idx = index_of(vec![sym("verbose", &["false"])]);
        assert_eq!(
            verdicts("`verbose` is true", &idx),
            vec![Verdict::Contradicted]
        );
    }

    #[test]
    fn string_swap() {
        let idx = index_of(vec![sym("region", &["\"eu-west-1\""])]);
        let f = check("The `region` defaults to `us-east-1`.", "README.md", &idx);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].verdict, Verdict::Contradicted);
        assert!(f[0].detail.contains("us-east-1"));
        assert!(f[0].detail.contains("eu-west-1"));
    }

    #[test]
    fn string_agreement_is_supported() {
        let idx = index_of(vec![sym("region", &["\"us-east-1\""])]);
        let v = verdicts("`region` defaults to `us-east-1`", &idx);
        assert_eq!(v, vec![Verdict::Supported]);
        assert!(!v[0].is_reportable());
    }

    #[test]
    fn string_quote_styles_and_prefixes_normalize() {
        // single-quoted (Python), prefixed (Rust raw) code literals compare by
        // inner content; a double-quoted prose value works too.
        let idx = index_of(vec![sym("mode", &["'production'"])]);
        assert_eq!(
            verdicts("`mode` is set to `staging`", &idx),
            vec![Verdict::Contradicted]
        );
        let idx = index_of(vec![sym("mode", &["r\"production\""])]);
        assert_eq!(
            verdicts("`mode` defaults to \"production\"", &idx),
            vec![Verdict::Supported]
        );
    }

    #[test]
    fn interpolated_string_value_is_silent() {
        // a format/interpolated literal has no static value — never contradict.
        let idx = index_of(vec![sym("greeting", &["\"hello ${name}\""])]);
        assert!(verdicts("`greeting` defaults to `hello world`", &idx).is_empty());
        // and an interpolated *prose* value is rejected at extraction.
        let idx = index_of(vec![sym("greeting", &["\"hello world\""])]);
        assert!(verdicts("`greeting` defaults to `hi ${name}`", &idx).is_empty());
    }

    #[test]
    fn bare_string_value_is_not_trusted() {
        // an un-delimited word is never read as a string value (too loose).
        let idx = index_of(vec![sym("mode", &["\"production\""])]);
        assert!(verdicts("`mode` defaults to staging", &idx).is_empty());
    }

    #[test]
    fn string_claim_ignores_numeric_constant() {
        // claimed string vs a symbol whose only literal is an int → no string to
        // compare, stay silent.
        let idx = index_of(vec![sym("port", &["8080"])]);
        assert!(verdicts("`port` defaults to `https`", &idx).is_empty());
    }

    #[test]
    fn type_mismatch_does_not_match_string_literal() {
        // code constant is the string "8080", doc claims int 8080 → no int to
        // compare, stay silent rather than spuriously support/контradict.
        let idx = index_of(vec![sym("port", &["\"8080\""])]);
        assert!(verdicts("`port` defaults to 9090", &idx).is_empty());
    }

    #[test]
    fn bare_code_shaped_identifier_fires() {
        // SCREAMING and snake_case bare tokens are trusted anchors.
        let idx = index_of(vec![sym("MAX_RETRIES", &["3"])]);
        assert_eq!(
            verdicts("MAX_RETRIES defaults to 5", &idx),
            vec![Verdict::Contradicted]
        );
        let idx = index_of(vec![sym("maxRetries", &["3"])]);
        assert_eq!(
            verdicts("maxRetries is set to 5", &idx),
            vec![Verdict::Contradicted]
        );
    }

    #[test]
    fn bare_english_word_is_ignored() {
        // "report" is a real symbol name AND an English word; without backticks
        // we must not anchor to it.
        let idx = index_of(vec![sym("report", &["3"])]);
        assert!(verdicts("the report defaults to 5 lines", &idx).is_empty());
        // backticked, the same claim *is* trusted.
        assert_eq!(
            verdicts("the `report` defaults to 5", &idx),
            vec![Verdict::Contradicted]
        );
    }

    #[test]
    fn code_shape_classifier() {
        for yes in ["MAX_RETRIES", "snake_case", "PORT", "HTTP2", "maxRetries"] {
            assert!(is_code_shaped(yes), "{yes} should be code-shaped");
        }
        for no in ["port", "report", "Default", "The", "x"] {
            assert!(!is_code_shaped(no), "{no} should not be code-shaped");
        }
    }

    // Drives `check` over a checked-in labeled corpus and reports precision /
    // recall, gating the Layer-1 zero-false-positive contract: any "none" /
    // "supported" case that yields a Contradicted is a precision leak and fails.
    // Mirrors the `prose_corpus_precision_recall` harness in rules.rs.
    #[test]
    fn constswap_corpus_precision_recall() {
        let corpus = include_str!("../tests/fixtures/constswap_corpus.jsonl");
        // contradiction-detection confusion counts.
        let (mut tp, mut fp, mut fn_) = (0usize, 0usize, 0usize);
        let mut leaks: Vec<String> = Vec::new();
        let mut misses: Vec<String> = Vec::new();

        for (lineno, raw) in crate::testutil::corpus_rows(corpus) {
            let v: serde_json::Value = serde_json::from_str(raw)
                .unwrap_or_else(|e| panic!("corpus line {lineno}: {e}\n{raw}"));
            let text = v["text"].as_str().unwrap();
            let tag = v["tag"].as_str().unwrap_or("?");
            let want = v["expect"].as_str().unwrap();

            let symbols: Vec<Symbol> = v["symbols"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| {
                    let consts: Vec<&str> = s["constants"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|c| c.as_str().unwrap())
                        .collect();
                    sym(s["name"].as_str().unwrap(), &consts)
                })
                .collect();
            let idx = index_of(symbols);

            let got = check(text, "doc.md", &idx);
            let got_contradiction = got.iter().any(|f| f.verdict == Verdict::Contradicted);

            match want {
                "contradicted" => {
                    if got_contradiction {
                        tp += 1;
                    } else {
                        fn_ += 1;
                        misses.push(format!("  MISS line {lineno} [{tag}]"));
                    }
                }
                "supported" | "none" => {
                    if got_contradiction {
                        fp += 1;
                        leaks.push(format!("  line {lineno} [{tag}]: false contradiction"));
                    }
                }
                other => panic!("unknown expect {other:?} on line {lineno}"),
            }
        }

        let precision = if tp + fp == 0 {
            1.0
        } else {
            tp as f64 / (tp + fp) as f64
        };
        let recall = if tp + fn_ == 0 {
            1.0
        } else {
            tp as f64 / (tp + fn_) as f64
        };
        eprintln!(
            "constswap corpus: tp={tp} fp={fp} fn={fn_} precision={precision:.3} recall={recall:.3}"
        );

        // Layer-1 contract: zero false positives, no exceptions.
        assert_eq!(
            fp,
            0,
            "precision regression: {fp} false contradiction(s)\n{}",
            leaks.join("\n")
        );
        // Recall floor — ratchet up as the extractor improves.
        assert!(
            recall >= 0.90,
            "recall regression: {recall:.3} < 0.90 floor (tp={tp} fn={fn_})\n{}",
            misses.join("\n")
        );
    }

    #[test]
    fn fenced_code_sample_is_not_a_claim() {
        // an assignment inside a ``` block is example code, not a prose claim —
        // even though its shape matches, we must not ground it (this is the
        // fastapi `secret_key="supersecret"` wild false positive).
        let idx = index_of(vec![sym("secret_key", &["\"realdefault\""])]);
        let md = "```python\nsecret_key=\"supersecret\"\n```";
        assert!(check(md, "README.md", &idx).is_empty());
    }

    #[test]
    fn docstring_literal_is_not_a_value() {
        // a property whose only literal is its docstring (multi-word, spaces) must
        // not ground a string value claim. This is the pydantic `serialize_as_any`
        // wild false positive: prose "set to `True`" vs a body that is just a
        // docstring. Capitalized `True` is now typed as a bool, and the docstring
        // is rejected as a string value — both guards independently silence it.
        let idx = index_of(vec![sym(
            "serialize_as_any",
            &["\"The serialize_as_any argument set during serialization.\""],
        )]);
        assert!(verdicts("`serialize_as_any` set to `True`", &idx).is_empty());
        assert!(verdicts("`serialize_as_any` set to `False`", &idx).is_empty());
        // a single-token string value still grounds normally.
        let idx = index_of(vec![sym("mode", &["\"production\""])]);
        assert_eq!(
            verdicts("`mode` is set to `staging`", &idx),
            vec![Verdict::Contradicted]
        );
    }

    #[test]
    fn self_referential_key_is_not_a_value() {
        // the vite `file` getter: its lone string literal is its own accessor key
        // (`_get("file")`), equal to the symbol name — not a default. A hypothetical
        // "If `file` is `'foo/bar'`" must not contradict it.
        let idx = index_of(vec![sym("file", &["\"file\""])]);
        assert!(verdicts("If `file` is `foo/bar`", &idx).is_empty());
        // a different literal is still a real value to compare against.
        let idx = index_of(vec![sym("file", &["\"default.txt\""])]);
        assert_eq!(
            verdicts("`file` defaults to `other.txt`", &idx),
            vec![Verdict::Contradicted]
        );
    }

    #[test]
    fn capitalized_bool_grounds_as_bool() {
        // delimited `True`/`False` are bools, matching a code bool literal of
        // either casing (Python `True`, Rust `false`).
        let idx = index_of(vec![sym("strict", &["false"])]);
        assert_eq!(
            verdicts("`strict` defaults to `True`", &idx),
            vec![Verdict::Contradicted]
        );
        let idx = index_of(vec![sym("strict", &["True"])]);
        assert_eq!(
            verdicts("`strict` defaults to `True`", &idx),
            vec![Verdict::Supported]
        );
    }

    #[test]
    fn loose_prose_is_not_a_claim() {
        let idx = index_of(vec![sym("port", &["5432"])]);
        // "is configurable" has no value cue; "is 8080" needs the value to be
        // present — here there's none, so nothing matches.
        assert!(check("`port` is configurable", "README.md", &idx).is_empty());
    }
}
