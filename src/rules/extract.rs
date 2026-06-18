//! Parsing architecture rules out of doc prose.
//!
//! Two extractors: the production [`extract_prose_rules`] (operands always
//! backtick-quoted, narrow phrasings) and the experimental, audit-only
//! [`extract_bare_rules`] (un-backticked operands kept safe by grounding + a
//! stopword denylist). Both compile to the shared [`Rule`] vocabulary.

use std::collections::HashSet;
use std::sync::OnceLock;

use regex::Regex;

use super::{Rule, SourcedRule};

/// Parse architectural rules out of one markdown doc's prose. Operands are
/// always backtick-quoted; phrasings are deliberately narrow to avoid matching
/// ordinary prose.
pub fn extract_prose_rules(markdown: &str, doc_path: &str) -> Vec<SourcedRule> {
    let mut rules = Vec::new();
    for (i, line) in markdown.lines().enumerate() {
        // Drop double-quoted spans: an author quoting an *example* rule
        // ("no `eval`") is describing the feature, not stating an enforced rule.
        let line = quoted_re().replace_all(line, "");
        let line = line.as_ref();
        let origin = format!("{doc_path}:{}", i + 1);
        let mut push = |rule| {
            rules.push(SourcedRule {
                rule,
                origin: origin.clone(),
            })
        };

        for c in forbid_edge_re().captures_iter(line) {
            push(Rule::ForbidEdge {
                from: c[1].to_string(),
                to: c[2].to_string(),
            });
        }
        for c in never_edge_re().captures_iter(line) {
            push(Rule::ForbidEdge {
                from: c[1].to_string(),
                to: c[2].to_string(),
            });
        }
        // "`domain` must not transitively/indirectly reach `infra`" — a path,
        // not just a direct edge. Checked before the direct verbs so the
        // transitive marker is consumed here rather than left dangling.
        for c in forbid_reach_re().captures_iter(line) {
            push(Rule::ForbidReach {
                from: c[1].to_string(),
                to: c[2].to_string(),
            });
        }
        // "`db` must not be imported by `api`" — reverse direction (api -> db).
        for c in forbid_by_re().captures_iter(line) {
            push(Rule::ForbidEdge {
                from: c[2].to_string(),
                to: c[1].to_string(),
            });
        }
        // "`domain` is independent of `infra`" — symmetric: forbid both edges.
        for c in independent_re().captures_iter(line) {
            push(Rule::ForbidEdge {
                from: c[1].to_string(),
                to: c[2].to_string(),
            });
            push(Rule::ForbidEdge {
                from: c[2].to_string(),
                to: c[1].to_string(),
            });
        }
        for c in depends_nothing_re().captures_iter(line) {
            push(Rule::Layer {
                module: c[1].to_string(),
                allowed: Vec::new(),
            });
        }
        for c in only_depends_re().captures_iter(line) {
            let allowed = backtick_tokens(&c[2]);
            if !allowed.is_empty() {
                push(Rule::Layer {
                    module: c[1].to_string(),
                    allowed,
                });
            }
        }
        for c in forbid_symbol_re().captures_iter(line) {
            push(Rule::ForbidSymbol {
                symbol: c[1].to_string(),
                except: except_modules(line),
            });
        }
    }
    rules
}

/// EXPERIMENTAL (audit-only): extract dependency rules whose module operands are
/// *not* backtick-quoted — the dominant real-world phrasing ("the EVSE module
/// must not depend on the Station module", "**Repository** layer cannot
/// reference **Service**"). Backticks are normally required precisely because
/// they keep precision at 100%; here we instead lean entirely on **grounding**:
/// a bare operand is only accepted if it resolves to a real module in the graph,
/// and generic prose nouns (`modules`, `details`, `low-level`, …) are denylisted
/// so SOLID/RFC boilerplate ("high-level modules should not depend on …") cannot
/// fire even when a same-named directory happens to exist.
///
/// This is wired into `staleguard rules` (the dry-run audit) ONLY, so we can
/// measure recall/precision on real repos before letting it affect `check`.
/// Emitted operands are canonicalised to the real module segment they matched,
/// and each origin is tagged `[bare]` so the report can flag them.
pub fn extract_bare_rules(
    markdown: &str,
    doc_path: &str,
    modules: &HashSet<String>,
) -> Vec<SourcedRule> {
    let mut rules = Vec::new();
    for (i, line) in markdown.lines().enumerate() {
        let line = quoted_re().replace_all(line, "");
        let line = line.as_ref();
        let origin = format!("{doc_path}:{} [bare]", i + 1);

        let mut push = |from: &str, to: &str, transitive: bool| {
            let (Some(from), Some(to)) = (
                canonical_operand(from, modules),
                canonical_operand(to, modules),
            ) else {
                return;
            };
            if from == to {
                return; // a module depending on itself is not a real rule
            }
            let rule = if transitive {
                Rule::ForbidReach { from, to }
            } else {
                Rule::ForbidEdge { from, to }
            };
            rules.push(SourcedRule {
                rule,
                origin: origin.clone(),
            });
        };

        for c in bare_reach_re().captures_iter(line) {
            push(&c[1], &c[2], true);
        }
        for c in bare_edge_re().captures_iter(line) {
            push(&c[1], &c[2], false);
        }
    }
    rules
}

/// Generic prose nouns that are never module names — the denylist that, together
/// with grounding, keeps SOLID/RFC/security boilerplate from being read as a
/// rule. Compared case-insensitively against the bare operand token.
const BARE_STOPWORDS: &[&str] = &[
    "module",
    "modules",
    "layer",
    "layers",
    "package",
    "packages",
    "crate",
    "crates",
    "component",
    "components",
    "code",
    "library",
    "libraries",
    "class",
    "classes",
    "interface",
    "interfaces",
    "detail",
    "details",
    "abstraction",
    "abstractions",
    "concretion",
    "concretions",
    "anything",
    "nothing",
    "something",
    "them",
    "it",
    "this",
    "that",
    "these",
    "those",
    "other",
    "others",
    "any",
    "all",
    "each",
    "both",
    "one",
    "low-level",
    "high-level",
    "client",
    "clients",
    "user",
    "users",
    "entity",
    "entities",
    "file",
    "files",
    "system",
    "systems",
    "function",
    "functions",
    "method",
    "methods",
    "data",
    "type",
    "types",
    "thing",
    "things",
    "implementation",
    "implementations",
    "framework",
    "frameworks",
    "dependency",
    "dependencies",
    "runtime",
    "run-time",
    "server",
    "scope",
    "state",
    "everything",
    "concretions",
    "abstractions",
];

/// Resolve a bare prose operand to the real module segment it names, or `None`
/// if it grounds to no module (case-insensitive) or is a denylisted noun. The
/// returned string is a real path segment, so [`super::matches`] accepts it verbatim.
fn canonical_operand(op: &str, modules: &HashSet<String>) -> Option<String> {
    let op = op.trim_matches(|c: char| !c.is_alphanumeric());
    if op.is_empty() || BARE_STOPWORDS.iter().any(|w| w.eq_ignore_ascii_case(op)) {
        return None;
    }
    // Single-segment operand: match a real path segment, preserving its case.
    if !op.contains('/') {
        for m in modules {
            for seg in m.split('/') {
                if seg.eq_ignore_ascii_case(op) {
                    return Some(seg.to_string());
                }
            }
        }
        return None;
    }
    // Path-style operand (`dafny/specs`): ground by case-insensitive subtree /
    // leaf / interior match, storing the lowercased form.
    let lop = op.to_lowercase();
    for m in modules {
        let lm = m.to_lowercase();
        if lm == lop
            || lm.starts_with(&format!("{lop}/"))
            || lm.ends_with(&format!("/{lop}"))
            || lm.contains(&format!("/{lop}/"))
        {
            return Some(lop);
        }
    }
    None
}

/// All backtick-quoted tokens in a string.
fn backtick_tokens(s: &str) -> Vec<String> {
    backtick_re()
        .captures_iter(s)
        .map(|c| c[1].to_string())
        .collect()
}

/// Backtick tokens following an `outside`/`except` keyword on a forbid-symbol
/// line, forming the rule's exception list.
fn except_modules(line: &str) -> Vec<String> {
    let lower = line.to_lowercase();
    let Some(pos) = ["outside", "except"].iter().find_map(|kw| lower.find(kw)) else {
        return Vec::new();
    };
    backtick_tokens(&line[pos..])
}

// ---- prose patterns -------------------------------------------------------

fn forbid_edge_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?:(?:modules?|code|files?|classes|components?|anything)\s+(?:in|under|within)\s+)?`([^`]+)`(?:\s+(?:layer|module|package|crate|component))?\s+(?:(?:must|should|may|can|does|do)\s+not|cannot|can'?t)\s+(?:imports?\s+(?:anything\s+)?from|import|imports|depend\s+on|depends\s+on|use|uses|reference|references|access|accesses|touch|touches|calls?\s+into)\s+(?:the\s+)?`([^`]+)`",
        )
        .unwrap()
    })
}

fn never_edge_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"`([^`]+)`(?:\s+(?:layer|module|package|crate|component))?\s+(?:(?:must|should|may|can)\s+)?never\s+(?:imports?|depends?\s+on|uses?|references?)\s+(?:the\s+)?`([^`]+)`",
        )
        .unwrap()
    })
}

/// "`X` must not be imported/used/referenced by `Y`" — captures the forbidden
/// target (1) and the dependent (2); the edge runs Y -> X.
fn forbid_reach_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"`([^`]+)`(?:\s+(?:layer|module|package|crate|component))?\s+(?:must|should|may|can|cannot)\s+not\s+(?:(?:even\s+)?(?:transitively|indirectly)\s+(?:import|imports|depend\s+on|depends\s+on|use|uses|reference|references|reach|reaches)|reach|reaches)\s+(?:the\s+)?`([^`]+)`",
        )
        .unwrap()
    })
}
fn forbid_by_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"`([^`]+)`(?:\s+(?:layer|module|package|crate|component))?\s+(?:must|should|may|can)\s+not\s+be\s+(?:imported|used|referenced|accessed|depended\s+on)\s+(?:by|from|in)\s+(?:the\s+)?`([^`]+)`",
        )
        .unwrap()
    })
}

/// "`X` is independent of `Y`" / "`X` has no dependency on `Y`" — a symmetric
/// no-edge rule (both directions forbidden).
fn independent_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"`([^`]+)`(?:\s+(?:layer|module|package|crate|component))?\s+(?:is|are|stays?|remains?)\s+independent\s+(?:of|from)\s+(?:the\s+)?`([^`]+)`",
        )
        .unwrap()
    })
}

fn depends_nothing_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"`([^`]+)`(?:\s+(?:layer|module|package|crate|component))?\s+(?:depends?\s+on|imports?|has)\s+(?:nothing|no\s+(?:dependencies|deps|imports))",
        )
        .unwrap()
    })
}

fn only_depends_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"`([^`]+)`(?:\s+(?:layer|module|package|crate|component))?\s+(?:(?:must|may|can|should)\s+)?only\s+(?:depends?\s+on|imports?)\s+(.*)").unwrap()
    })
}

/// Forbid-symbol phrasings, all requiring a use/call signal so a bare "no
/// `config`" in prose is not mistaken for a rule.
fn forbid_symbol_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)(?:(?:must|should|may)\s+not\s+(?:use|call|invoke|reference)|don'?t\s+(?:use|call)|never\s+(?:use|call)|no\s+(?:direct|raw)(?:\s+(?:use|usage|calls?|reference)\s+(?:of|to))?|no\s+(?:use|usage|calls?)\s+(?:of|to))\s+`([^`]+)`",
        )
        .unwrap()
    })
}

/// EXPERIMENTAL bare-operand direct-edge pattern (no backticks). Operands are
/// captured as bare tokens (optionally **bold**, optionally `the …`, optionally
/// trailed by a noun like `layer`/`module`/`code`); grounding + the stopword
/// denylist downstream are what keep this safe. Case-insensitive.
fn bare_edge_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)(?:the\s+)?\*{0,2}([a-z][\w.-]*(?:/[\w.-]+)*)\*{0,2}(?:\s+(?:layer|module|package|crate|component|code|library|internals?|implementations?|classes))?\s+(?:(?:must|should|may|can)\s+not|cannot|can'?t)\s+(?:import|imports|depend\s+on|depends\s+on|reference|references|access|accesses|use|uses)\s+(?:the\s+)?\*{0,2}([a-z][\w.-]*(?:/[\w.-]+)*)\*{0,2}",
        )
        .unwrap()
    })
}

/// EXPERIMENTAL bare-operand transitive pattern (no backticks). Mirrors
/// [`forbid_reach_re`] but with bare tokens.
fn bare_reach_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)(?:the\s+)?\*{0,2}([a-z][\w.-]*(?:/[\w.-]+)*)\*{0,2}(?:\s+(?:layer|module|package|crate|component|code|library|internals?|implementations?|classes))?\s+(?:(?:must|should|may|can)\s+not|cannot|can'?t)\s+(?:(?:even\s+)?(?:transitively|indirectly)\s+(?:import|imports|depend\s+on|depends\s+on|use|uses|reference|references|reach|reaches)|reach|reaches)\s+(?:the\s+)?\*{0,2}([a-z][\w.-]*(?:/[\w.-]+)*)\*{0,2}",
        )
        .unwrap()
    })
}

fn backtick_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"`([^`]+)`").unwrap())
}

/// A double-quoted span (straight or curly quotes).
fn quoted_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r#""[^"]*"|“[^”]*”"#).unwrap())
}
