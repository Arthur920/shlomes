//! Language-aware code extractor: turns source into symbols + dependency edges.
//!
//! The shared substrate for coverage-gaps, diagram edge-diff, architecture
//! rules, and drift provenance/fingerprints. Default build (no `ml` feature).

mod extract;
pub mod facts;
pub mod lang;
mod resolve;
pub mod schema;
pub mod symbol;

use std::collections::{HashMap, HashSet};
use std::path::Path;

use serde::Serialize;

use extract::RawRef;
use lang::Language;
use symbol::{DepEdge, RefEdge, Symbol};

/// All symbols, module dependency edges, and symbol-level reference edges
/// extracted from a repo.
#[derive(Debug, Default, Serialize)]
pub struct CodeIndex {
    pub symbols: Vec<Symbol>,
    /// Raw dependency edges: `from_module` is a repo module path, `to_module` is
    /// the import *as written* (`crate::x`, `./mod`, `os`).
    pub edges: Vec<DepEdge>,
    /// Resolved internal module graph: both endpoints are repo module paths, and
    /// only edges whose target resolves to a real module are kept. The clean
    /// substrate architecture-rule checks run against.
    pub module_edges: Vec<DepEdge>,
    pub ref_edges: Vec<RefEdge>,
}

impl CodeIndex {
    /// Walk every code file under `repo_root` and extract symbols + edges, then
    /// resolve raw references into symbol-level reference edges across files.
    pub fn build(repo_root: &Path) -> CodeIndex {
        let mut symbols = Vec::new();
        let mut edges = Vec::new();
        let mut raw_refs = Vec::new();
        for file in lang::code_files(repo_root) {
            let (s, e, r) = extract::extract_file(&file, repo_root);
            symbols.extend(s);
            edges.extend(e);
            raw_refs.extend(r);
        }
        let ref_edges = resolve_refs(&symbols, raw_refs);
        let module_edges = resolve_module_edges(&symbols, &edges);
        CodeIndex {
            symbols,
            edges,
            module_edges,
            ref_edges,
        }
    }

    /// The set of real internal module paths (symbol modules + edge sources).
    /// Used to ground architecture-rule operands.
    pub fn module_set(&self) -> HashSet<String> {
        self.symbols
            .iter()
            .map(|s| s.module.clone())
            .chain(self.edges.iter().map(|e| e.from_module.clone()))
            .collect()
    }

    /// Symbols defined in a given module path. (Consumed by coverage-gaps.)
    #[allow(dead_code)]
    pub fn symbols_in<'a>(&'a self, module: &'a str) -> impl Iterator<Item = &'a Symbol> {
        self.symbols.iter().filter(move |s| s.module == module)
    }

    /// Number of distinct symbols that reference `qualified_name` — the
    /// per-symbol risk signal for coverage-gaps, and the basis for the
    /// dead-code-vs-undocumented distinction.
    pub fn symbol_fan_in(&self, qualified_name: &str) -> usize {
        self.ref_edges
            .iter()
            .filter(|e| e.to_symbol == qualified_name)
            .map(|e| e.from_symbol.as_str())
            .collect::<HashSet<_>>()
            .len()
    }
}

/// Turn raw import edges into a resolved internal module graph: map each
/// `to_module` (the import as written) to a real repo module via the source
/// language's rules, dropping externals, self-edges, and duplicates.
fn resolve_module_edges(symbols: &[Symbol], edges: &[DepEdge]) -> Vec<DepEdge> {
    let module_set: HashSet<String> = symbols
        .iter()
        .map(|s| s.module.clone())
        .chain(edges.iter().map(|e| e.from_module.clone()))
        .collect();

    // First-seen language per module, from each symbol's source file.
    let mut module_lang: HashMap<&str, Language> = HashMap::new();
    for s in symbols {
        if let Some(l) = Language::from_path(Path::new(&s.span.path)) {
            module_lang.entry(s.module.as_str()).or_insert(l);
        }
    }

    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut out = Vec::new();
    for e in edges {
        let Some(&lang) = module_lang.get(e.from_module.as_str()) else {
            continue;
        };
        let Some(to) = resolve::resolve_import(&e.to_module, &e.from_module, lang, &module_set)
        else {
            continue;
        };
        if to == e.from_module {
            continue;
        }
        if seen.insert((e.from_module.clone(), to.clone())) {
            out.push(DepEdge {
                from_module: e.from_module.clone(),
                to_module: to,
            });
        }
    }
    out
}

/// Resolve raw references (name + enclosing symbol) into symbol-level edges.
/// A reference name is matched to *every* same-named definition
/// (over-approximate — never under-counts callers); self-edges and duplicate
/// `(from, to)` pairs are dropped.
fn resolve_refs(symbols: &[Symbol], raw_refs: Vec<RawRef>) -> Vec<RefEdge> {
    let mut by_name: HashMap<&str, Vec<&str>> = HashMap::new();
    for s in symbols {
        by_name
            .entry(s.name.as_str())
            .or_default()
            .push(s.qualified_name.as_str());
    }

    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut edges = Vec::new();
    for r in &raw_refs {
        let Some(targets) = by_name.get(r.name.as_str()) else {
            continue;
        };
        for &to in targets {
            if to == r.from {
                continue;
            }
            if seen.insert((r.from.clone(), to.to_string())) {
                edges.push(RefEdge {
                    from_symbol: r.from.clone(),
                    to_symbol: to.to_string(),
                });
            }
        }
    }
    edges
}

#[cfg(test)]
mod tests {
    use super::*;
    use symbol::{Facts, Span, SymbolKind, Visibility};

    fn sym(name: &str, qualified: &str) -> Symbol {
        Symbol {
            qualified_name: qualified.to_string(),
            name: name.to_string(),
            kind: SymbolKind::Function,
            visibility: Visibility::Public,
            module: "m".to_string(),
            span: Span {
                path: "m.rs".to_string(),
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

    #[test]
    fn fan_in_counts_distinct_callers() {
        let symbols = vec![sym("target", "m::target"), sym("a", "m::a"), sym("b", "m::b")];
        // a and b each call target; a calls it twice -> still one distinct caller.
        let raw = vec![
            RawRef { from: "m::a".into(), name: "target".into() },
            RawRef { from: "m::a".into(), name: "target".into() },
            RawRef { from: "m::b".into(), name: "target".into() },
        ];
        let ref_edges = resolve_refs(&symbols, raw);
        let index = CodeIndex { symbols, edges: vec![], module_edges: vec![], ref_edges };
        assert_eq!(index.symbol_fan_in("m::target"), 2);
    }

    #[test]
    fn over_approximates_on_name_collision() {
        let symbols = vec![sym("run", "a::run"), sym("run", "b::run"), sym("caller", "c::caller")];
        let raw = vec![RawRef { from: "c::caller".into(), name: "run".into() }];
        let edges = resolve_refs(&symbols, raw);
        assert!(edges.iter().any(|e| e.to_symbol == "a::run"));
        assert!(edges.iter().any(|e| e.to_symbol == "b::run"));
    }

    #[test]
    fn drops_self_edges() {
        let symbols = vec![sym("foo", "m::foo")];
        let raw = vec![RawRef { from: "m::foo".into(), name: "foo".into() }];
        assert!(resolve_refs(&symbols, raw).is_empty());
    }
}
