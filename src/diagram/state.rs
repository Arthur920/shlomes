//! State-diagram grounding (Layer 1, deterministic). Parses Mermaid
//! `stateDiagram` / `stateDiagram-v2` and grounds the drawn **states** against
//! the variants of a real enum (`Symbol.members`).
//!
//! Conservative, per the zero-FP stance: a diagram is grounded only against the
//! single enum whose variants its states best match (clearing [`MIN_GROUND`] and
//! at least half the states); a diagram that grounds to no enum emits nothing.
//! **Transitions** are not deterministically groundable (a transition table
//! isn't extracted) and are deferred to the Layer-3 judge — only the state set is
//! checked here. PlantUML state diagrams are not yet parsed.

use std::collections::HashSet;
use std::sync::OnceLock;

use regex::Regex;

use super::Format;
use crate::claim::Provenance;
use crate::code::symbol::{Symbol, SymbolKind};
use crate::code::CodeIndex;
use crate::findings::{Finding, Verdict};

/// Minimum matched states before a diagram is considered grounded to an enum.
const MIN_GROUND: usize = 2;

/// State-diagram findings for one embedded diagram, or empty if it isn't a state
/// diagram in `format` or grounds to no enum.
pub(super) fn check(format: Format, body: &str, origin: &str, index: &CodeIndex) -> Vec<Finding> {
    let Some(states) = parse(format, body) else {
        return Vec::new();
    };
    let Some(en) = ground(&states, index) else {
        return Vec::new();
    };
    diff(&states, en, origin)
}

/// Distinct drawn state names (excluding the `[*]` start/end pseudo-state), in
/// first-seen order. `None` if `body` isn't a Mermaid state diagram.
fn parse(format: Format, body: &str) -> Option<Vec<String>> {
    if format != Format::Mermaid {
        return None;
    }
    let header = body
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with("%%"))?;
    let first = header.split_whitespace().next().unwrap_or("");
    if first != "stateDiagram" && first != "stateDiagram-v2" {
        return None;
    }

    let mut states: Vec<String> = Vec::new();
    let push = |s: &str, states: &mut Vec<String>| {
        if s != "[*]" && !s.is_empty() && !states.iter().any(|x| x == s) {
            states.push(s.to_string());
        }
    };
    for raw in body.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with("%%") || line == header {
            continue;
        }
        if let Some(c) = transition_re().captures(line) {
            push(&c[1], &mut states);
            push(&c[2], &mut states);
        } else if let Some(c) = state_decl_re().captures(line) {
            push(&c[1], &mut states);
        }
    }
    (!states.is_empty()).then_some(states)
}

/// Pick the enum whose variants best overlap the drawn states. `None` unless the
/// best overlap clears `MIN_GROUND` and covers ≥ half the states.
fn ground<'a>(states: &[String], index: &'a CodeIndex) -> Option<&'a Symbol> {
    let mut best: Option<(usize, &Symbol)> = None;
    for s in &index.symbols {
        if s.kind != SymbolKind::Enum || s.members.is_empty() {
            continue;
        }
        let matched = states
            .iter()
            .filter(|st| s.members.iter().any(|v| v == *st))
            .count();
        if matched == 0 {
            continue;
        }
        match best {
            Some((m, _)) if matched < m => {}
            Some((m, prev)) if matched == m && s.qualified_name >= prev.qualified_name => {}
            _ => best = Some((matched, s)),
        }
    }
    let (matched, en) = best?;
    (matched >= MIN_GROUND && matched >= states.len().div_ceil(2)).then_some(en)
}

fn diff(states: &[String], en: &Symbol, origin: &str) -> Vec<Finding> {
    let variants: HashSet<&str> = en.members.iter().map(String::as_str).collect();
    let drawn: HashSet<&str> = states.iter().map(String::as_str).collect();
    let prov = Provenance::symbol(en.qualified_name.clone());
    let mut out = Vec::new();

    // Drawn states checked against the enum's variants.
    for st in states {
        if variants.contains(st.as_str()) {
            out.push(Finding::supported(
                format!("state `{st}` of `{}`", en.name),
                origin.to_string(),
                prov.clone(),
            ));
        } else {
            out.push(
                Finding::problem(
                    Verdict::Stale,
                    format!("state `{st}`"),
                    origin.to_string(),
                    format!(
                        "Stale state: the diagram draws state `{st}`, but enum `{}` has no such variant.",
                        en.name
                    ),
                )
                .anchored(prov.clone()),
            );
        }
    }
    // Variants the diagram never draws.
    for v in &en.members {
        if !drawn.contains(v.as_str()) {
            out.push(
                Finding::problem(
                    Verdict::Undocumented,
                    format!("variant `{v}` of `{}`", en.name),
                    origin.to_string(),
                    format!(
                        "Undrawn state: enum `{}` has variant `{v}`, but the state diagram omits it.",
                        en.name
                    ),
                )
                .anchored(prov.clone()),
            );
        }
    }
    out
}

fn transition_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // `A --> B` / `A --> B : label` / `[*] --> A`.
    RE.get_or_init(|| {
        Regex::new(r"^(\[\*\]|[A-Za-z_]\w*)\s*-->\s*(\[\*\]|[A-Za-z_]\w*)\s*(?::.*)?$").unwrap()
    })
}

fn state_decl_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // `state Foo` / `state "desc" as Foo`.
    RE.get_or_init(|| {
        Regex::new(r#"^state\s+(?:"?[^"\n]+"?\s+as\s+)?([A-Za-z_]\w*)\s*\{?\s*$"#).unwrap()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code::symbol::{Facts, Span, Visibility};

    fn enum_sym(name: &str, variants: &[&str]) -> Symbol {
        Symbol {
            qualified_name: format!("m::{name}"),
            name: name.to_string(),
            kind: SymbolKind::Enum,
            visibility: Visibility::Public,
            module: "m".to_string(),
            span: Span { path: "m.rs".into(), start_line: 1, end_line: 1 },
            body_span: Span::zero(),
            signature: None,
            doc: None,
            facts: Facts::default(),
            calls: Vec::new(),
            members: variants.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn index(syms: Vec<Symbol>) -> CodeIndex {
        CodeIndex { symbols: syms, ..Default::default() }
    }

    #[test]
    fn stale_state_not_a_variant_is_flagged() {
        let idx = index(vec![enum_sym("State", &["Idle", "Running", "Done"])]);
        let body = "stateDiagram-v2\n  [*] --> Idle\n  Idle --> Running : start\n  Running --> Paused\n";
        let out = check(Format::Mermaid, body, "d.md:1", &idx);
        assert!(out
            .iter()
            .any(|f| f.verdict == Verdict::Stale && f.detail.contains("Paused")));
        // Idle + Running match; Done is undrawn.
        assert!(out
            .iter()
            .any(|f| f.verdict == Verdict::Undocumented && f.detail.contains("Done")));
        assert!(out.iter().any(|f| f.verdict == Verdict::Supported && f.claim.contains("Idle")));
    }

    #[test]
    fn diagram_grounding_to_no_enum_emits_nothing() {
        let idx = index(vec![enum_sym("State", &["Idle", "Running"])]);
        // States overlap no enum variant -> ungrounded -> silent.
        let body = "stateDiagram-v2\n  [*] --> Alpha\n  Alpha --> Beta\n";
        assert!(check(Format::Mermaid, body, "d.md:1", &idx).is_empty());
    }

    #[test]
    fn non_state_diagram_is_ignored() {
        let idx = index(vec![enum_sym("State", &["Idle", "Running"])]);
        assert!(check(Format::Mermaid, "graph TD\n A-->B\n", "d.md:1", &idx).is_empty());
    }
}
