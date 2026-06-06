//! Coverage gaps: the code → doc traversal (Layer 1, deterministic).
//!
//! The inverse of `verify` — instead of checking a doc claim against the code,
//! it starts from the code's public surface and asks whether any doc describes
//! it. A public symbol whose name appears in no doc is an `undocumented` gap.
//!
//! Scope (see `docs/coverage-gaps.md`): public surface only; "documented" means
//! the name appears as a token anywhere in any doc (loose presence, fewest false
//! positives). Gaps are never suppressed — they are **risk-ranked** by a
//! composite of fan-in, churn, branch-count, and a net-new (no-co-changed-doc)
//! signal, so the riskiest undocumented surface surfaces first. A public symbol
//! with no callers and no doc still flags (it may be a true public entry point),
//! just ranked last. `term-drift` and `under-documented` remain Layer 2/3.
//!
//! The `coverage-gaps.md` §2 fourth quadrant — a doc names a symbol that *exists
//! but nothing calls* ("removed feature") — is deliberately **not** emitted here:
//! at Layer 1 it is indistinguishable from a legitimate public entry point
//! (binaries' true API has fan-in 0 too), so flagging it would break the zero-FP
//! stance. That semantic call is deferred to the Layer-3 judge. A doc naming a
//! symbol that does *not* exist is already caught by `entrypoints`/`verify`.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::OnceLock;

use regex::Regex;

use crate::code::symbol::{Symbol, SymbolKind, Visibility};
use crate::code::CodeIndex;
use crate::drift::coupling;
use crate::findings::{Finding, Verdict};
use crate::git;

/// Commits of history to mine for churn + co-change (newest first).
const MAX_COMMITS: usize = 1000;

/// Identifier-like tokens (len ≥ 2). Underscores are part of a token, so a name
/// like `check_paths` matches as a whole; runs over the whole doc, so prose and
/// `backtick` spans are both covered.
fn ident_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[A-Za-z_][A-Za-z0-9_]+").unwrap())
}

/// Extract code → doc coverage gaps for a repo (the standalone `coverage`
/// subcommand path — builds its own index).
pub fn run(repo_root: &Path) -> Vec<Finding> {
    let index = CodeIndex::build(repo_root);
    gaps(&index, repo_root)
}

/// Coverage gaps over an already-built index. The `check` pipeline calls this so
/// gaps flow into `drift::run` and become a dimension of the alignment score
/// (`coverage-gaps.md` "Score integration"): each `Undocumented` gap is a claim
/// anchored to its symbol, so it lowers that module's score and trips the
/// regression gate, symmetric with a broken doc→code claim.
pub fn gaps(index: &CodeIndex, repo_root: &Path) -> Vec<Finding> {
    let terms = build_doc_terms(repo_root);
    let risk = RiskSignals::from_history(&git::file_change_history(repo_root, MAX_COMMITS));
    find_gaps(index, &terms, &risk)
}

/// Git-derived risk signals that *rank* (never suppress) coverage gaps.
#[derive(Debug, Default)]
struct RiskSignals {
    /// repo-relative file path -> number of commits it changed in (churn).
    churn: HashMap<String, usize>,
    /// code files no doc has ever co-changed with — the net-new gap signal.
    no_codoc: HashSet<String>,
}

impl RiskSignals {
    fn from_history(history: &[Vec<String>]) -> RiskSignals {
        let mut churn: HashMap<String, usize> = HashMap::new();
        for files in history {
            for f in files {
                *churn.entry(f.clone()).or_default() += 1;
            }
        }
        RiskSignals {
            churn,
            no_codoc: coupling::code_without_codoc(history),
        }
    }

    fn churn_of(&self, path: &str) -> usize {
        self.churn.get(path).copied().unwrap_or(0)
    }

    /// A symbol is "net-new" when its file is tracked in history but no doc ever
    /// co-changed with it (a fresh public surface no doc follows).
    fn is_net_new(&self, path: &str) -> bool {
        self.no_codoc.contains(path)
    }
}

/// Every identifier-like token mentioned across all markdown docs in the repo.
fn build_doc_terms(repo_root: &Path) -> HashSet<String> {
    let mut terms = HashSet::new();
    for doc in crate::collect_docs(repo_root) {
        if let Ok(text) = std::fs::read_to_string(&doc) {
            for m in ident_re().find_iter(&text) {
                terms.insert(m.as_str().to_string());
            }
        }
    }
    terms
}

/// Per-gap risk inputs, computed once so the rank key and the finding detail
/// stay in sync.
struct Gap<'a> {
    sym: &'a Symbol,
    fan_in: usize,
    churn: usize,
    /// Branch count — a free cyclomatic-complexity proxy from the facts walk.
    complexity: usize,
    net_new: bool,
}

impl Gap<'_> {
    /// Composite risk score (higher = riskier). Net-new surface dominates, then
    /// the additive fan-in + churn + complexity signals.
    fn composite(&self) -> usize {
        self.fan_in + self.churn + self.complexity
    }
}

/// Public symbols whose name is mentioned in no doc, risk-ranked. Net-new gaps
/// first, then by composite risk (fan-in + churn + branch-count), then by
/// location for stable output. Nothing is suppressed.
fn find_gaps(index: &CodeIndex, terms: &HashSet<String>, risk: &RiskSignals) -> Vec<Finding> {
    let mut gaps: Vec<Gap> = index
        .symbols
        .iter()
        .filter(|s| s.visibility == Visibility::Public)
        .filter(|s| !terms.contains(&s.name))
        .map(|s| Gap {
            sym: s,
            fan_in: index.symbol_fan_in(&s.qualified_name),
            churn: risk.churn_of(&s.span.path),
            complexity: s.facts.predicates.len(),
            net_new: risk.is_net_new(&s.span.path),
        })
        .collect();

    gaps.sort_by(|a, b| {
        b.net_new
            .cmp(&a.net_new)
            .then_with(|| b.composite().cmp(&a.composite()))
            .then_with(|| a.sym.span.path.cmp(&b.sym.span.path))
            .then_with(|| a.sym.span.start_line.cmp(&b.sym.span.start_line))
    });

    gaps.into_iter().map(finding_for).collect()
}

fn finding_for(g: Gap) -> Finding {
    let s = g.sym;
    let kind = kind_label(&s.kind);
    // Soft reachability hint: a zero-caller public symbol is either dead code or
    // a true entry point. We don't suppress it — just flag the signal and let it
    // rank last.
    let reach = if g.fan_in == 0 {
        "no internal callers".to_string()
    } else {
        format!("fan-in {}", g.fan_in)
    };
    // Net-new is the strongest signal — a fresh public surface no doc tracks.
    let lead = if g.net_new {
        "new public surface, no doc has tracked it; "
    } else {
        ""
    };
    Finding::problem(
        Verdict::Undocumented,
        format!("public {} `{}` has no doc reference", kind.to_lowercase(), s.name),
        format!("{}:{}", s.span.path, s.span.start_line),
        format!(
            "{} `{}` ({}) is documented nowhere; {}{}.",
            kind, s.name, s.module, lead, reach
        ),
    )
    .anchored(crate::claim::Provenance::symbol(s.qualified_name.clone()))
}

fn kind_label(kind: &SymbolKind) -> String {
    match kind {
        SymbolKind::Other(s) => s.clone(),
        other => format!("{other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code::symbol::{Facts, RefEdge, Span};

    fn sym(name: &str, vis: Visibility, module: &str) -> Symbol {
        Symbol {
            qualified_name: format!("{module}::{name}"),
            name: name.to_string(),
            kind: SymbolKind::Function,
            visibility: vis,
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

    fn terms(words: &[&str]) -> HashSet<String> {
        words.iter().map(|w| w.to_string()).collect()
    }

    #[test]
    fn public_symbol_absent_from_docs_is_flagged() {
        let index = CodeIndex {
            symbols: vec![sym("frobnicate", Visibility::Public, "m")],
            edges: vec![],
            module_edges: vec![],
            ref_edges: vec![],
        };
        let gaps = find_gaps(&index, &terms(&["something", "else"]), &RiskSignals::default());
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].verdict, Verdict::Undocumented);
        assert!(gaps[0].detail.contains("frobnicate"));
        // zero callers -> the soft reachability hint.
        assert!(gaps[0].detail.contains("no internal callers"));
    }

    #[test]
    fn documented_symbol_not_flagged() {
        let index = CodeIndex {
            symbols: vec![sym("frobnicate", Visibility::Public, "m")],
            edges: vec![],
            module_edges: vec![],
            ref_edges: vec![],
        };
        assert!(find_gaps(&index, &terms(&["frobnicate"]), &RiskSignals::default()).is_empty());
    }

    #[test]
    fn private_and_internal_symbols_not_flagged() {
        let index = CodeIndex {
            symbols: vec![
                sym("helper", Visibility::Private, "m"),
                sym("internal", Visibility::Internal, "m"),
            ],
            edges: vec![],
            module_edges: vec![],
            ref_edges: vec![],
        };
        assert!(find_gaps(&index, &HashSet::new(), &RiskSignals::default()).is_empty());
    }

    #[test]
    fn higher_fan_in_ranks_first() {
        // hot_fn has two distinct callers; cold_fn has none.
        let index = CodeIndex {
            symbols: vec![
                sym("cold_fn", Visibility::Public, "cold"),
                sym("hot_fn", Visibility::Public, "hot"),
            ],
            edges: vec![],
            module_edges: vec![],
            ref_edges: vec![
                RefEdge {
                    from_symbol: "cold::cold_fn".into(),
                    to_symbol: "hot::hot_fn".into(),
                },
                RefEdge {
                    from_symbol: "other::caller".into(),
                    to_symbol: "hot::hot_fn".into(),
                },
            ],
        };
        let gaps = find_gaps(&index, &HashSet::new(), &RiskSignals::default());
        assert_eq!(gaps.len(), 2);
        assert!(gaps[0].detail.contains("hot_fn"));
        assert!(gaps[0].detail.contains("fan-in 2"));
        assert!(gaps[1].detail.contains("cold_fn"));
        assert!(gaps[1].detail.contains("no internal callers"));
    }

    #[test]
    fn net_new_gap_outranks_a_busier_but_tracked_gap() {
        // `cold` has high fan-in but a doc has tracked its file; `fresh` is a
        // brand-new file no doc co-changed with. Net-new must win the rank.
        let index = CodeIndex {
            symbols: vec![
                sym("cold_fn", Visibility::Public, "cold"),
                sym("fresh_fn", Visibility::Public, "fresh"),
            ],
            edges: vec![],
            module_edges: vec![],
            ref_edges: vec![
                RefEdge { from_symbol: "a::x".into(), to_symbol: "cold::cold_fn".into() },
                RefEdge { from_symbol: "b::y".into(), to_symbol: "cold::cold_fn".into() },
                RefEdge { from_symbol: "c::z".into(), to_symbol: "cold::cold_fn".into() },
            ],
        };
        let mut no_codoc = HashSet::new();
        no_codoc.insert("fresh.rs".to_string()); // matches sym()'s `{module}.rs` path
        let risk = RiskSignals { churn: HashMap::new(), no_codoc };

        let gaps = find_gaps(&index, &HashSet::new(), &risk);
        assert_eq!(gaps.len(), 2);
        assert!(gaps[0].detail.contains("fresh_fn"), "{}", gaps[0].detail);
        assert!(gaps[0].detail.contains("new public surface"));
        assert!(gaps[1].detail.contains("cold_fn"));
    }

    #[test]
    fn churn_breaks_ties_above_a_quiet_gap() {
        // Two equally-uncalled, doc-tracked gaps; the churned file ranks first.
        let index = CodeIndex {
            symbols: vec![
                sym("quiet_fn", Visibility::Public, "quiet"),
                sym("busy_fn", Visibility::Public, "busy"),
            ],
            edges: vec![],
            module_edges: vec![],
            ref_edges: vec![],
        };
        let mut churn = HashMap::new();
        churn.insert("busy.rs".to_string(), 9);
        let risk = RiskSignals { churn, no_codoc: HashSet::new() };

        let gaps = find_gaps(&index, &HashSet::new(), &risk);
        assert!(gaps[0].detail.contains("busy_fn"), "{}", gaps[0].detail);
        assert!(gaps[1].detail.contains("quiet_fn"));
    }

    #[test]
    fn run_flags_only_undocumented_public_symbol() {
        use std::fs;
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("shlomes-cov-{nanos}"));
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(
            dir.join("src/lib.rs"),
            "pub fn documented_fn() {}\npub fn hidden_fn() {}\n",
        )
        .unwrap();
        fs::write(dir.join("README.md"), "We expose `documented_fn` for callers.\n").unwrap();

        let findings = run(&dir);
        assert!(findings.iter().any(|f| f.detail.contains("hidden_fn")));
        assert!(!findings.iter().any(|f| f.detail.contains("documented_fn")));
    }
}
