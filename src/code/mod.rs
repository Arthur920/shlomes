//! Language-aware code extractor: turns source into symbols + dependency edges.
//!
//! The shared substrate for coverage-gaps, diagram edge-diff, architecture
//! rules, and drift provenance/fingerprints. Default build (no `ml` feature).

mod extract;
pub mod lang;
pub mod symbol;

use std::collections::HashSet;
use std::path::Path;

use serde::Serialize;

use symbol::{DepEdge, Symbol};

/// All symbols and module dependency edges extracted from a repo.
#[derive(Debug, Default, Serialize)]
pub struct CodeIndex {
    pub symbols: Vec<Symbol>,
    pub edges: Vec<DepEdge>,
}

impl CodeIndex {
    /// Walk every code file under `repo_root` and extract symbols + edges.
    pub fn build(repo_root: &Path) -> CodeIndex {
        let mut symbols = Vec::new();
        let mut edges = Vec::new();
        for file in lang::code_files(repo_root) {
            let (s, e) = extract::extract_file(&file, repo_root);
            symbols.extend(s);
            edges.extend(e);
        }
        CodeIndex { symbols, edges }
    }

    /// Symbols defined in a given module path. (Consumed by coverage-gaps.)
    #[allow(dead_code)]
    pub fn symbols_in<'a>(&'a self, module: &'a str) -> impl Iterator<Item = &'a Symbol> {
        self.symbols.iter().filter(move |s| s.module == module)
    }

    /// Number of distinct modules that import `module` — a risk-weighting
    /// signal for coverage-gaps. Matches on suffix since import targets aren't
    /// resolved to internal module paths yet.
    #[allow(dead_code)]
    pub fn fan_in(&self, module: &str) -> usize {
        self.edges
            .iter()
            .filter(|e| e.to_module == module || e.to_module.ends_with(module))
            .map(|e| e.from_module.as_str())
            .collect::<HashSet<_>>()
            .len()
    }
}
