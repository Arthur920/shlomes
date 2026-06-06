//! Class-diagram grounding (Layer 1, deterministic). Parses Mermaid
//! `classDiagram` and PlantUML class diagrams and grounds drawn classes +
//! methods against the AST symbols in [`CodeIndex`].
//!
//! Scope is deliberately tight to stay zero-FP. The index carries methods as
//! flat `module::method` symbols with no type-level link back to their class,
//! and class boxes are bare type names indistinguishable from external types
//! (`String`, `HashMap`). So:
//!
//! - a class that grounds to a real type → `Supported`;
//! - a **method** drawn under a *grounded* class that exists nowhere in that
//!   class's module → `Stale` (a reliable "drawn method that's gone" defect);
//! - a bare class name that grounds to nothing is **not** flagged (could be an
//!   external type — deferred, like the graph diff's bare-box case);
//! - relations (inheritance / association) need type-level edges we don't
//!   extract — parsed-over, not diffed (Layer 2/3).

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use regex::Regex;

use super::Format;
use crate::claim::Provenance;
use crate::code::symbol::{SymbolKind, Symbol};
use crate::code::CodeIndex;
use crate::findings::{Finding, Verdict};

/// A declared class and the method names drawn on it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ClassDecl {
    name: String,
    methods: Vec<String>,
}

struct ClassDiagram {
    classes: Vec<ClassDecl>,
    origin: String,
}

/// Class-diagram findings for one embedded diagram, or empty if `body` isn't a
/// class diagram in `format`.
pub(super) fn check(format: Format, body: &str, origin: &str, index: &CodeIndex) -> Vec<Finding> {
    let Some(d) = parse(format, body, origin) else {
        return Vec::new();
    };
    diff(&d, index)
}

fn parse(format: Format, body: &str, origin: &str) -> Option<ClassDiagram> {
    match format {
        Format::Mermaid => {
            let header = body
                .lines()
                .map(str::trim)
                .find(|l| !l.is_empty() && !l.starts_with("%%"))?;
            if header.split_whitespace().next() != Some("classDiagram") {
                return None;
            }
        }
        Format::PlantUml => {
            // A class diagram declares `class X` (sequence/component don't).
            if !body.lines().any(|l| class_open_re().is_match(l.trim())) {
                return None;
            }
        }
        Format::Dot => return None,
    }
    parse_body(body, origin)
}

fn parse_body(body: &str, origin: &str) -> Option<ClassDiagram> {
    let mut classes: Vec<ClassDecl> = Vec::new();
    let mut current: Option<usize> = None; // index into `classes` inside a `{ }`

    let ensure = |classes: &mut Vec<ClassDecl>, name: &str| -> usize {
        match classes.iter().position(|c| c.name == name) {
            Some(i) => i,
            None => {
                classes.push(ClassDecl { name: name.to_string(), methods: Vec::new() });
                classes.len() - 1
            }
        }
    };

    for raw in body.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with("%%") || line.starts_with('\'') {
            continue;
        }
        // Close a member block.
        if line.starts_with('}') {
            current = None;
            continue;
        }
        // `class Foo {` / `class Foo` / `class "Foo" as F`.
        if let Some(c) = class_open_re().captures(line) {
            let name = c[1].trim_matches('"').trim();
            let idx = ensure(&mut classes, name);
            current = line.ends_with('{').then_some(idx);
            continue;
        }
        // Inside a `{ }` block: a member line. Only methods (with `()`) count.
        if let Some(idx) = current {
            if let Some(m) = method_name(line) {
                classes[idx].methods.push(m);
            }
            continue;
        }
        // Inline member: `Foo : +bar()`.
        if let Some(c) = inline_re().captures(line) {
            let idx = ensure(&mut classes, c[1].trim());
            if let Some(m) = method_name(&c[2]) {
                classes[idx].methods.push(m);
            }
        }
        // Anything else (relations, notes) is ignored.
    }

    (!classes.is_empty()).then_some(ClassDiagram { classes, origin: origin.to_string() })
}

/// The method name of a member line (`+isMammal()`, `-save() : bool`), or `None`
/// for a field (no parens) — fields are ambiguous to name and skipped.
fn method_name(member: &str) -> Option<String> {
    method_re().captures(member).map(|c| c[1].to_string())
}

fn diff(d: &ClassDiagram, index: &CodeIndex) -> Vec<Finding> {
    // Real types by name, and the set of method names per module.
    let types: HashMap<&str, &Symbol> = index
        .symbols
        .iter()
        .filter(|s| is_type(&s.kind))
        .map(|s| (s.name.as_str(), s))
        .collect();
    let mut methods_by_module: HashMap<&str, HashSet<&str>> = HashMap::new();
    for s in &index.symbols {
        if matches!(s.kind, SymbolKind::Method | SymbolKind::Function) {
            methods_by_module
                .entry(s.module.as_str())
                .or_default()
                .insert(s.name.as_str());
        }
    }

    let mut out = Vec::new();
    for c in &d.classes {
        let Some(sym) = types.get(c.name.as_str()) else {
            continue; // ungrounded bare name — could be external; not flagged
        };
        let prov = Provenance::symbol(sym.qualified_name.clone());
        out.push(Finding::supported(
            format!("class diagram declares `{}`", c.name),
            d.origin.clone(),
            prov.clone(),
        ));
        let module_methods = methods_by_module.get(sym.module.as_str());
        for m in &c.methods {
            let exists = module_methods.map(|set| set.contains(m.as_str())).unwrap_or(false);
            if exists {
                out.push(Finding::supported(
                    format!("class `{}` method `{m}`", c.name),
                    d.origin.clone(),
                    prov.clone(),
                ));
            } else {
                out.push(
                    Finding::problem(
                        Verdict::Stale,
                        format!("class `{}` method `{m}`", c.name),
                        d.origin.clone(),
                        format!(
                            "Stale member: the class diagram draws `{}.{m}()`, but `{}` defines no such method.",
                            c.name, sym.module
                        ),
                    )
                    .anchored(prov.clone()),
                );
            }
        }
    }
    out
}

fn is_type(kind: &SymbolKind) -> bool {
    matches!(
        kind,
        SymbolKind::Class
            | SymbolKind::Struct
            | SymbolKind::Trait
            | SymbolKind::Interface
            | SymbolKind::Enum
    )
}

fn class_open_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // `class Foo`, `class Foo {`, `class "Foo" as F`.
    RE.get_or_init(|| Regex::new(r#"^class\s+("?[A-Za-z_][\w]*"?)(?:\s+as\s+\w+)?\s*\{?\s*$"#).unwrap())
}

fn inline_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // `Foo : +bar()` (mermaid inline member).
    RE.get_or_init(|| Regex::new(r"^([A-Za-z_]\w*)\s*:\s*(.+)$").unwrap())
}

fn method_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // The identifier immediately before `(` — visibility markers/types ignored.
    RE.get_or_init(|| Regex::new(r"([A-Za-z_]\w*)\s*\(").unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code::symbol::{Facts, Span, Visibility};

    fn sym(name: &str, kind: SymbolKind, module: &str) -> Symbol {
        Symbol {
            qualified_name: format!("{module}::{name}"),
            name: name.to_string(),
            kind,
            visibility: Visibility::Public,
            module: module.to_string(),
            span: Span { path: format!("{module}.rs"), start_line: 1, end_line: 1 },
            body_span: Span::zero(),
            signature: None,
            doc: None,
            facts: Facts::default(),
            calls: Vec::new(),
            members: Vec::new(),
        }
    }

    fn index() -> CodeIndex {
        CodeIndex {
            symbols: vec![
                sym("Account", SymbolKind::Struct, "src/account"),
                sym("balance", SymbolKind::Method, "src/account"),
                sym("deposit", SymbolKind::Method, "src/account"),
            ],
            ..Default::default()
        }
    }

    #[test]
    fn grounded_class_and_method_are_supported() {
        let body = "classDiagram\n  class Account {\n    +balance()\n    +deposit()\n  }\n";
        let out = check(Format::Mermaid, body, "d.md:1", &index());
        assert!(out.iter().all(|f| f.verdict == Verdict::Supported));
        assert_eq!(out.len(), 3); // class + 2 methods
    }

    #[test]
    fn drawn_method_absent_from_module_is_stale() {
        let body = "classDiagram\n  class Account {\n    +balance()\n    +withdraw()\n  }\n";
        let out = check(Format::Mermaid, body, "d.md:1", &index());
        assert!(out
            .iter()
            .any(|f| f.verdict == Verdict::Stale && f.detail.contains("withdraw")));
        assert!(out.iter().any(|f| f.verdict == Verdict::Supported && f.claim.contains("balance")));
    }

    #[test]
    fn external_bare_class_is_not_flagged() {
        let body = "classDiagram\n  class HashMap\n  class Account\n  HashMap <|-- Account\n";
        let out = check(Format::Mermaid, body, "d.md:1", &index());
        // HashMap grounds to nothing -> skipped (no Stale); Account grounds -> one Supported.
        assert!(out.iter().all(|f| f.verdict == Verdict::Supported));
        assert_eq!(out.len(), 1);
        assert!(out[0].claim.contains("Account"));
    }

    #[test]
    fn inline_member_syntax_is_parsed() {
        let body = "classDiagram\n  Account : +deposit()\n";
        let out = check(Format::Mermaid, body, "d.md:1", &index());
        assert!(out.iter().any(|f| f.claim.contains("deposit")));
    }
}
