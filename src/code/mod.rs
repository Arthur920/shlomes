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
use std::sync::Arc;

use rayon::prelude::*;
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
    /// Symbol reference graph keyed by target: `ref_callers[to]` is the distinct
    /// set of `from` symbols that reference `to`. This replaces a flat
    /// `Vec<RefEdge>` so [`Self::symbol_fan_in`] is an O(1) lookup rather than an
    /// O(edges) scan — coverage calls it once per symbol, so the old flat shape
    /// was O(symbols × edges), the quadratic that hung on large repos (litellm).
    /// Each edge also stores one interned `Arc<str>` caller (under its shared
    /// target key) instead of two cloned names. Serializes back to the original
    /// flat `[{from_symbol, to_symbol}]` array under the key `ref_edges`, so the
    /// `index` dump shape is unchanged.
    #[serde(rename = "ref_edges", serialize_with = "serialize_ref_callers")]
    pub ref_callers: HashMap<Arc<str>, Vec<Arc<str>>>,
}

/// Flatten the target-keyed caller map back into the historical
/// `[{from_symbol, to_symbol}]` array, sorted for a deterministic dump.
fn serialize_ref_callers<S: serde::Serializer>(
    callers: &HashMap<Arc<str>, Vec<Arc<str>>>,
    s: S,
) -> Result<S::Ok, S::Error> {
    use serde::ser::SerializeSeq;
    let mut flat: Vec<(&str, &str)> = callers
        .iter()
        .flat_map(|(to, froms)| froms.iter().map(move |from| (from.as_ref(), to.as_ref())))
        .collect();
    flat.sort_unstable();
    let mut seq = s.serialize_seq(Some(flat.len()))?;
    for (from, to) in flat {
        seq.serialize_element(&RefEdge {
            from_symbol: Arc::from(from),
            to_symbol: Arc::from(to),
        })?;
    }
    seq.end()
}

/// Build the target-keyed caller map from flat edges, deduping callers per
/// target. The inverse of the flat serialization above; used by tests that
/// construct a [`CodeIndex`] from explicit edges.
#[cfg(test)]
pub fn ref_callers_from(edges: impl IntoIterator<Item = RefEdge>) -> HashMap<Arc<str>, Vec<Arc<str>>> {
    let mut map: HashMap<Arc<str>, Vec<Arc<str>>> = HashMap::new();
    for e in edges {
        let froms = map.entry(e.to_symbol).or_default();
        if !froms.contains(&e.from_symbol) {
            froms.push(e.from_symbol);
        }
    }
    map
}

/// Prebuilt symbol lookups shared across the claim-grounding and evidence passes,
/// so neither re-scans the whole symbol table per claim (the old shape was
/// O(claims × symbols)). Borrows the index, so rebuild it if the index changes.
/// Only the `ml` build's Layer 2/3 uses it.
#[cfg(feature = "ml")]
pub struct SymbolLookup<'a> {
    symbols: &'a [Symbol],
    /// `qualified_name` -> all symbol indices with that name (ascending).
    by_qname: HashMap<&'a str, Vec<usize>>,
    /// `name` -> earliest symbol index with that leaf name.
    by_name: HashMap<&'a str, usize>,
    /// leaf of `qualified_name` (segment after the last `::`) -> earliest index.
    by_leaf: HashMap<&'a str, usize>,
    /// module path -> symbol indices defined in it (ascending).
    by_module: HashMap<&'a str, Vec<usize>>,
    /// the index's `module_set` (symbol modules + edge sources), borrowed.
    modules: HashSet<&'a str>,
}

#[cfg(feature = "ml")]
impl<'a> SymbolLookup<'a> {
    pub fn build(index: &'a CodeIndex) -> SymbolLookup<'a> {
        let symbols = index.symbols.as_slice();
        let mut by_qname: HashMap<&str, Vec<usize>> = HashMap::new();
        let mut by_name: HashMap<&str, usize> = HashMap::new();
        let mut by_leaf: HashMap<&str, usize> = HashMap::new();
        let mut by_module: HashMap<&str, Vec<usize>> = HashMap::new();
        let mut modules: HashSet<&str> = HashSet::new();
        for (i, s) in symbols.iter().enumerate() {
            by_qname.entry(&s.qualified_name).or_default().push(i);
            by_name.entry(&s.name).or_insert(i);
            let leaf = s
                .qualified_name
                .rsplit("::")
                .next()
                .unwrap_or(&s.qualified_name);
            by_leaf.entry(leaf).or_insert(i);
            by_module.entry(&s.module).or_default().push(i);
            modules.insert(&s.module);
        }
        for e in &index.edges {
            modules.insert(&e.from_module);
        }
        SymbolLookup {
            symbols,
            by_qname,
            by_name,
            by_leaf,
            by_module,
            modules,
        }
    }

    /// The earliest symbol whose `qualified_name == tok`, `name == tok`, or
    /// `qualified_name` ends with `::tok` — faithfully matching the original
    /// linear `find` (first symbol satisfying any of the three). Tokens that
    /// contain `::` fall back to a scan, since a general suffix match can't be
    /// served by the hashed indexes; such tokens are rare for backtick names.
    pub fn resolve_token(&self, tok: &str) -> Option<&'a Symbol> {
        if tok.contains("::") {
            return self.symbols.iter().find(|s| {
                s.qualified_name == tok
                    || s.name == tok
                    || s.qualified_name.ends_with(&format!("::{tok}"))
            });
        }
        // Each map already holds the earliest index for its key, so the min over
        // the three is the earliest symbol matching any condition.
        [
            self.by_qname.get(tok).map(|v| v[0]),
            self.by_name.get(tok).copied(),
            self.by_leaf.get(tok).copied(),
        ]
        .into_iter()
        .flatten()
        .min()
        .map(|i| &self.symbols[i])
    }

    /// All symbols whose `qualified_name` equals `qn`.
    pub fn by_qualified(&self, qn: &str) -> impl Iterator<Item = &'a Symbol> + '_ {
        self.by_qname
            .get(qn)
            .into_iter()
            .flatten()
            .map(move |&i| &self.symbols[i])
    }

    /// Public symbols defined in module `m`.
    pub fn public_in_module(&self, m: &str) -> impl Iterator<Item = &'a Symbol> + '_ {
        self.by_module
            .get(m)
            .into_iter()
            .flatten()
            .map(move |&i| &self.symbols[i])
            .filter(|s| s.visibility == symbol::Visibility::Public)
    }

    /// The first module whose path fuzzily matches `tok` (the module fallback for
    /// a backtick token that grounded to no symbol).
    pub fn module_matches(&self, tok: &str) -> Option<&'a str> {
        self.modules
            .iter()
            .find(|m| crate::rules::matches(m, tok))
            .copied()
    }
}

impl CodeIndex {
    /// Walk every code file under `repo_root` and extract symbols + edges, then
    /// resolve raw references into symbol-level reference edges across files.
    pub fn build(repo_root: &Path) -> CodeIndex {
        // Parse files in parallel — each `extract_file` owns its tree-sitter
        // parser, so there's no shared state. `collect` into an ordered Vec keeps
        // the merge deterministic (stable symbol order = stable output).
        let files = lang::code_files(repo_root);
        let per_file: Vec<_> = files
            .par_iter()
            .map(|file| extract::extract_file(file, repo_root))
            .collect();

        // Merge symbols/edges into flat Vecs, but keep each file's raw refs in
        // their own already-allocated Vec rather than concatenating them — the raw
        // (pre-resolution) ref set is the largest collection on large repos, and a
        // single flattened copy would double its peak footprint. `resolve_refs`
        // only streams them once, so we hand it a lazy `flatten()` instead.
        let mut symbols = Vec::new();
        let mut edges = Vec::new();
        let mut raw_per_file: Vec<Vec<RawRef>> = Vec::with_capacity(per_file.len());
        for (s, e, r) in per_file {
            symbols.extend(s);
            edges.extend(e);
            raw_per_file.push(r);
        }
        let ref_callers = resolve_refs(&symbols, raw_per_file.into_iter().flatten());
        let module_edges = resolve_module_edges(&symbols, &edges);
        CodeIndex {
            symbols,
            edges,
            module_edges,
            ref_callers,
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
        // Callers are deduped per target at construction, so the stored length is
        // already the distinct-caller count — an O(1) lookup.
        self.ref_callers.get(qualified_name).map_or(0, Vec::len)
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

/// A name defined more than this many times is a common identifier (`new`,
/// `get`, `build`, `next`): name-based resolution would fan every reference out
/// to all of them, an `O(refs × defs)` blow-up that produced hundreds of
/// millions of edges (and OOM) on large repos. Such names carry no signal, so
/// we skip resolving them rather than explode.
const MAX_DEFS_PER_NAME: usize = 32;

/// Resolve raw references (name + enclosing symbol) into symbol-level edges.
/// A reference name is matched to every same-named definition (over-approximate
/// — never under-counts callers), except names with more than
/// [`MAX_DEFS_PER_NAME`] definitions, which are dropped as noise. Self-edges and
/// duplicate `(from, to)` pairs are dropped.
///
/// Dedup is done over interned `u32` ids rather than cloned `String` pairs, so
/// the `seen` set costs 8 bytes per pair instead of two heap allocations, and the
/// target-keyed caller map is built in a single pass (no intermediate edge list).
fn resolve_refs(
    symbols: &[Symbol],
    raw_refs: impl IntoIterator<Item = RawRef>,
) -> HashMap<Arc<str>, Vec<Arc<str>>> {
    let mut by_name: HashMap<&str, Vec<&str>> = HashMap::new();
    for s in symbols {
        by_name
            .entry(s.name.as_str())
            .or_default()
            .push(s.qualified_name.as_str());
    }

    // Intern qualified names to small ids *and* a shared `Arc<str>` per distinct
    // endpoint (linear in symbols), so each emitted edge reuses one allocation
    // rather than cloning two long names. `pool[id]` is the interned name for id.
    // `intern` is a free fn (not a closure capturing `pool`), so its borrow of
    // `pool` is released at each return — letting us read `pool[id]` in the same
    // loop and build the caller map in one pass, with no intermediate edge list.
    fn intern(ids: &mut HashMap<Arc<str>, u32>, pool: &mut Vec<Arc<str>>, s: &str) -> u32 {
        if let Some(&i) = ids.get(s) {
            return i;
        }
        let i = pool.len() as u32;
        let arc: Arc<str> = Arc::from(s);
        pool.push(arc.clone());
        ids.insert(arc, i);
        i
    }

    let mut ids: HashMap<Arc<str>, u32> = HashMap::new();
    let mut pool: Vec<Arc<str>> = Vec::new();
    // `seen` dedups `(from, to)` over 8-byte id pairs (no string allocs); callers
    // are grouped by target as we go. A hot callee's many callers each cost one
    // `Arc` clone (a refcount bump), not a fresh copy of its long name.
    let mut seen: HashSet<(u32, u32)> = HashSet::new();
    let mut callers: HashMap<Arc<str>, Vec<Arc<str>>> = HashMap::new();
    for r in raw_refs {
        let Some(targets) = by_name.get(r.name.as_str()) else {
            continue;
        };
        if targets.len() > MAX_DEFS_PER_NAME {
            continue;
        }
        let from_id = intern(&mut ids, &mut pool, &r.from);
        for &to in targets {
            if to == r.from {
                continue;
            }
            let to_id = intern(&mut ids, &mut pool, to);
            if seen.insert((from_id, to_id)) {
                let to_arc = pool[to_id as usize].clone();
                callers
                    .entry(to_arc)
                    .or_default()
                    .push(pool[from_id as usize].clone());
            }
        }
    }
    callers
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
        let symbols = vec![
            sym("target", "m::target"),
            sym("a", "m::a"),
            sym("b", "m::b"),
        ];
        // a and b each call target; a calls it twice -> still one distinct caller.
        let raw = vec![
            RawRef {
                from: "m::a".into(),
                name: "target".into(),
            },
            RawRef {
                from: "m::a".into(),
                name: "target".into(),
            },
            RawRef {
                from: "m::b".into(),
                name: "target".into(),
            },
        ];
        let ref_callers = resolve_refs(&symbols, raw);
        let index = CodeIndex {
            symbols,
            edges: vec![],
            module_edges: vec![],
            ref_callers,
        };
        assert_eq!(index.symbol_fan_in("m::target"), 2);
    }

    #[test]
    fn over_approximates_on_name_collision() {
        let symbols = vec![
            sym("run", "a::run"),
            sym("run", "b::run"),
            sym("caller", "c::caller"),
        ];
        let raw = vec![RawRef {
            from: "c::caller".into(),
            name: "run".into(),
        }];
        let callers = resolve_refs(&symbols, raw);
        assert!(callers.contains_key("a::run"));
        assert!(callers.contains_key("b::run"));
    }

    #[test]
    fn drops_self_edges() {
        let symbols = vec![sym("foo", "m::foo")];
        let raw = vec![RawRef {
            from: "m::foo".into(),
            name: "foo".into(),
        }];
        assert!(resolve_refs(&symbols, raw).is_empty());
    }
}
