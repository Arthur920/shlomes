//! Layer 1: qualified code references in docs that the code index can't
//! resolve. A backtick token like `verify::check_paths` or `CodeIndex::build`
//! names a member of a module or type; if that module/type is local but the
//! final member exists nowhere as a symbol, the reference is `stale`.
//!
//! Scoped to *local* references to keep false positives at zero. A reference is
//! only checked when its head segment anchors to this repo — a `crate`/`self`/
//! `super` prefix, or a head that matches a known module segment or symbol name.
//! `std::collections::HashMap` and `serde::Serialize` (external heads) are left
//! alone. Grounding is over-approximate, mirroring `resolve_refs`: a member is
//! considered present if its name matches *any* symbol or module segment, so a
//! real member is never falsely flagged.

use std::collections::HashSet;
use std::sync::OnceLock;

use regex::Regex;

use crate::claim::Provenance;
use crate::code::CodeIndex;
use crate::findings::{Finding, Verdict};

/// Name sets distilled from the code index for resolving doc references.
pub struct Grounding {
    /// Every symbol name, qualified-name tail, and module-path segment — the
    /// pool a referenced member must hit to count as present.
    names: HashSet<String>,
    /// Heads that mark a reference as local: module segments + symbol names,
    /// plus the Rust path keywords.
    anchors: HashSet<String>,
    /// Names of type containers (struct/class/enum/trait/interface). Their
    /// members — enum variants, methods, associated items — are not indexed as
    /// standalone symbols, so a `Type::member` reference is unverifiable rather
    /// than drift. Module qualifiers are deliberately excluded: a module's
    /// functions *are* indexed, so `module::missing_fn` is real drift.
    type_names: HashSet<String>,
}

impl Grounding {
    pub fn from_index(index: &CodeIndex) -> Grounding {
        use crate::code::symbol::SymbolKind;
        let mut names = HashSet::new();
        let mut type_names = HashSet::new();
        let mut anchors: HashSet<String> = ["crate", "self", "super"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        for s in &index.symbols {
            names.insert(s.name.clone());
            anchors.insert(s.name.clone());
            if matches!(
                s.kind,
                SymbolKind::Struct
                    | SymbolKind::Class
                    | SymbolKind::Enum
                    | SymbolKind::Trait
                    | SymbolKind::Interface
            ) {
                type_names.insert(s.name.clone());
            }
            if let Some(tail) = s.qualified_name.rsplit("::").next() {
                names.insert(tail.to_string());
            }
            for seg in s.module.split('/') {
                if !seg.is_empty() {
                    names.insert(seg.to_string());
                    anchors.insert(seg.to_string());
                }
            }
        }
        Grounding {
            names,
            anchors,
            type_names,
        }
    }
}

/// Check every `::`-qualified reference in `markdown` against the index.
pub fn check(markdown: &str, doc_path: &str, g: &Grounding) -> Vec<Finding> {
    let mut findings = Vec::new();
    let mut seen = HashSet::new();
    for (line, reference) in qualified_refs(markdown) {
        let segs: Vec<&str> = reference.split("::").collect();
        let Some(effective) = local_path(&segs, &g.anchors) else {
            continue; // external or non-local — not our claim.
        };
        let Some(member) = effective.last() else {
            continue;
        };
        if !seen.insert((line, reference.clone())) {
            continue;
        }
        let doc_ref = format!("{doc_path}:{line}");
        let claim = format!("references `{reference}`");
        // The qualifier directly before the member (`MappingTarget` in
        // `MappingTarget::MapToUnknown`). Enum variants, methods and associated
        // items are not indexed as their own symbols, so a member access on a
        // *known* type is unverifiable, not drift — ground it on the type.
        let qualifier = (effective.len() >= 2).then(|| effective[effective.len() - 2]);
        if g.names.contains(*member) {
            // Anchor to the member name; the drift changed-set includes symbol
            // short names, so this carries forward until that symbol changes.
            findings.push(Finding::supported(
                claim,
                doc_ref,
                Provenance::symbol((*member).to_string()),
            ));
        } else if let Some(qualifier) = qualifier.filter(|q| g.type_names.contains(*q)) {
            // Member of a real type we can't introspect (variant/method/assoc).
            findings.push(Finding::supported(
                claim,
                doc_ref,
                Provenance::symbol(qualifier.to_string()),
            ));
        } else {
            findings.push(Finding::problem(
                Verdict::Stale,
                claim,
                doc_ref,
                format!(
                    "`{reference}` is named in docs but `{member}` resolves to no symbol or module in the code."
                ),
            ));
        }
    }
    findings
}

/// Decide whether a `::` path is a local reference and, if so, return the
/// segments that carry meaning (a leading `crate`/`self`/`super` run is
/// stripped). `None` means the head is external — skip it.
fn local_path<'a>(segs: &'a [&'a str], anchors: &HashSet<String>) -> Option<&'a [&'a str]> {
    const KEYWORDS: &[&str] = &["crate", "self", "super"];
    let mut rest = segs;
    let mut had_keyword = false;
    while let [head, tail @ ..] = rest {
        if KEYWORDS.contains(head) {
            had_keyword = true;
            rest = tail;
        } else {
            break;
        }
    }
    if rest.is_empty() {
        return None;
    }
    // `crate::…` is explicitly local; otherwise the head must anchor locally.
    if had_keyword || anchors.contains(rest[0]) {
        Some(rest)
    } else {
        None
    }
}

/// `::`-joined identifier paths (≥ 2 segments) appearing in inline `backtick`
/// spans. Generic args and trailing `()` are excluded by the identifier shape.
fn qualified_refs(markdown: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    for (i, line) in markdown.lines().enumerate() {
        for code in inline_code_re().captures_iter(line) {
            for cap in qualified_re().captures_iter(&code[1]) {
                out.push((i + 1, cap[1].to_string()));
            }
        }
    }
    out
}

fn inline_code_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"`([^`]+)`").unwrap())
}

fn qualified_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"([A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+)").unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code::symbol::{Facts, Span, Symbol, SymbolKind, Visibility};

    fn sym(name: &str, qualified: &str, module: &str) -> Symbol {
        Symbol {
            qualified_name: qualified.to_string(),
            name: name.to_string(),
            kind: SymbolKind::Function,
            visibility: Visibility::Public,
            module: module.to_string(),
            span: Span {
                path: format!("{module}.rs"),
                start_line: 1,
                end_line: 1,
            },
            body_span: Span::zero(),
            signature: None,
            doc: None,
            facts: Facts::default(),
            calls: Vec::new(),
            members: Vec::new(),
        }
    }

    fn typ(name: &str, qualified: &str, module: &str, kind: SymbolKind) -> Symbol {
        Symbol {
            kind,
            ..sym(name, qualified, module)
        }
    }

    fn grounding() -> Grounding {
        let index = CodeIndex {
            symbols: vec![
                sym("check_paths", "src::verify::check_paths", "src/verify"),
                sym("CodeIndex", "src::code::CodeIndex", "src/code"),
                sym("build", "src::code::CodeIndex::build", "src/code"),
                typ(
                    "MappingTarget",
                    "src::map::MappingTarget",
                    "src/map",
                    SymbolKind::Enum,
                ),
            ],
            edges: vec![],
            module_edges: vec![],
            ref_callers: Default::default(),
        };
        Grounding::from_index(&index)
    }

    #[test]
    fn local_resolvable_ref_is_not_flagged() {
        let g = grounding();
        let f = check("See `verify::check_paths`.", "README.md", &g);
        assert!(f.iter().all(|x| !x.verdict.is_reportable()), "{f:?}");
    }

    #[test]
    fn local_unresolvable_member_is_flagged() {
        let g = grounding();
        let flagged = check("Call `verify::deleted_fn`.", "README.md", &g);
        assert_eq!(flagged.len(), 1);
        assert!(flagged[0].detail.contains("deleted_fn"));
    }

    #[test]
    fn external_paths_are_skipped() {
        let g = grounding();
        // std/serde heads are not local anchors → never flagged.
        let md = "Uses `std::collections::HashMap` and `serde::Serialize`.";
        assert!(check(md, "README.md", &g).is_empty());
    }

    #[test]
    fn crate_prefix_forces_local_check() {
        let g = grounding();
        // `crate::` is explicitly local, so a bogus member is caught.
        let flagged = check("`crate::nowhere::gone`", "README.md", &g);
        assert_eq!(flagged.len(), 1);
        assert!(flagged[0].detail.contains("gone"));
    }

    #[test]
    fn enum_variant_of_known_type_is_not_flagged() {
        let g = grounding();
        // `MapToUnknown` is an enum variant — not indexed as its own symbol —
        // but `MappingTarget` is a real enum, so the reference is unverifiable,
        // not drift.
        let f = check("Use `MappingTarget::MapToUnknown`.", "README.md", &g);
        assert!(f.iter().all(|x| !x.verdict.is_reportable()), "{f:?}");
    }

    #[test]
    fn missing_fn_in_known_module_is_still_flagged() {
        let g = grounding();
        // `verify` is a module (not a type); its functions are indexed, so a
        // missing one is genuine drift — the type fallback must not mask it.
        let flagged = check("Call `verify::deleted_fn`.", "README.md", &g);
        assert_eq!(flagged.len(), 1, "{flagged:?}");
    }

    #[test]
    fn type_qualified_method_resolves() {
        let g = grounding();
        // Head is a known symbol name (CodeIndex), member build exists.
        let f = check("`CodeIndex::build` indexes the repo.", "README.md", &g);
        assert!(f.iter().all(|x| !x.verdict.is_reportable()), "{f:?}");
    }
}
