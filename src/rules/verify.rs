//! Verifying compiled rules against the resolved module graph.
//!
//! Each [`Rule`] compiles to a dependency-graph or source query checked against
//! the index; a violation is a hard `contradicted` [`Finding`], a grounded rule
//! that holds emits a `supported` claim, and an ungrounded operand is skipped
//! rather than guessed. The per-rule graph checks here are shared with the
//! dry-run [`super::audit`].

use std::collections::{HashMap, HashSet};
use std::path::Path;

use rayon::prelude::*;

use crate::claim::Provenance;
use crate::code::lang;
use crate::code::CodeIndex;
use crate::findings::{Finding, Verdict};

use super::{grounded, matches, quote_list, Rule, SourcedRule};

/// Verify every rule against the index, returning `contradicted` findings for
/// violations.
pub fn check(rules: &[SourcedRule], index: &CodeIndex, repo_root: &Path) -> Vec<Finding> {
    let modules = index.module_set();
    // Read every source file once, up front — but only if some forbid-symbol
    // rule actually needs the textual scan. Otherwise N such rules would each
    // re-walk and re-read the whole repo.
    let sources = if rules
        .iter()
        .any(|r| matches!(r.rule, Rule::ForbidSymbol { .. }))
    {
        read_sources(repo_root)
    } else {
        Vec::new()
    };
    let mut findings = Vec::new();
    for sr in rules {
        let before = findings.len();
        match &sr.rule {
            Rule::ForbidEdge { from, to } => {
                check_forbid_edge(sr, from, to, index, &modules, &mut findings);
                add_supported(sr, &modules, before, &mut findings);
            }
            Rule::ForbidReach { from, to } => {
                check_forbid_reach(sr, from, to, index, &modules, &mut findings);
                add_supported(sr, &modules, before, &mut findings);
            }
            Rule::Layer { module, allowed } => {
                check_layer(sr, module, allowed, index, &modules, &mut findings);
                add_supported(sr, &modules, before, &mut findings);
            }
            Rule::ForbidSymbol { symbol, except } => {
                // The symbol check returns the modules it actually scanned; a
                // clean rule is anchored to them so adding the symbol in any of
                // them re-opens the claim (precise lineage, fixing the old empty
                // provenance that could never carry forward).
                let scanned =
                    check_forbid_symbol(sr, symbol, except, index, &sources, &mut findings);
                if findings.len() == before && !scanned.is_empty() {
                    findings.push(Finding::supported(
                        format!("forbids `{symbol}`"),
                        sr.origin.clone(),
                        Provenance::modules(scanned),
                    ));
                }
            }
        }
    }
    findings
}

/// Record the `Supported` claim for a grounded module rule that held (no new
/// findings since `before`). Ungrounded module rules produce nothing.
fn add_supported(
    sr: &SourcedRule,
    modules: &HashSet<String>,
    before: usize,
    findings: &mut Vec<Finding>,
) {
    if findings.len() == before {
        if let Some(claim) = supported_rule(sr, modules) {
            findings.push(claim);
        }
    }
}

/// A `Supported` claim for a grounded module rule that held, anchored to the
/// modules it constrains. Returns `None` for an ungrounded rule (unverifiable,
/// not supported). `ForbidSymbol` is anchored to its scanned modules inline in
/// [`check`], so it is not handled here.
fn supported_rule(sr: &SourcedRule, modules: &HashSet<String>) -> Option<Finding> {
    let (claim, prov) = match &sr.rule {
        Rule::ForbidEdge { from, to } => {
            if !grounded(from, modules) || !grounded(to, modules) {
                return None;
            }
            (
                format!("`{from}` must not import `{to}`"),
                Provenance::modules([from.clone(), to.clone()]),
            )
        }
        Rule::ForbidReach { from, to } => {
            if !grounded(from, modules) || !grounded(to, modules) {
                return None;
            }
            (
                format!("`{from}` must not reach `{to}`"),
                Provenance::modules([from.clone(), to.clone()]),
            )
        }
        Rule::Layer { module, allowed } => {
            if !grounded(module, modules) {
                return None;
            }
            let claim = if allowed.is_empty() {
                format!("`{module}` depends on nothing")
            } else {
                format!("`{module}` may depend only on {}", quote_list(allowed))
            };
            (claim, Provenance::modules([module.clone()]))
        }
        Rule::ForbidSymbol { .. } => return None,
    };
    Some(Finding::supported(claim, sr.origin.clone(), prov))
}

pub(super) fn check_forbid_edge(
    sr: &SourcedRule,
    from: &str,
    to: &str,
    index: &CodeIndex,
    modules: &HashSet<String>,
    out: &mut Vec<Finding>,
) {
    if !grounded(from, modules) || !grounded(to, modules) {
        return; // operand names no real module — unverifiable, don't guess.
    }
    for e in &index.module_edges {
        if matches(&e.from_module, from) && matches(&e.to_module, to) {
            out.push(violation(
                sr,
                format!("`{from}` must not import `{to}`"),
                format!("`{}` imports `{}`.", e.from_module, e.to_module),
                &e.from_module,
                &e.to_module,
            ));
        }
    }
}

/// Check a transitive reachability rule: is any module matching `to` reachable
/// from any module matching `from` through a chain of import edges? Reports the
/// shortest offending path per distinct source module (one finding each), so the
/// output stays bounded and each violation names the exact chain to break.
pub(super) fn check_forbid_reach(
    sr: &SourcedRule,
    from: &str,
    to: &str,
    index: &CodeIndex,
    modules: &HashSet<String>,
    out: &mut Vec<Finding>,
) {
    if !grounded(from, modules) || !grounded(to, modules) {
        return; // operand names no real module — unverifiable, don't guess.
    }
    // Adjacency over concrete module paths.
    let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
    for e in &index.module_edges {
        adj.entry(&e.from_module).or_default().push(&e.to_module);
    }
    // Sorted source modules for deterministic output.
    let mut sources: Vec<&str> = modules
        .iter()
        .map(String::as_str)
        .filter(|m| matches(m, from))
        .collect();
    sources.sort_unstable();

    for src in sources {
        // A source that itself matches `to` (same subtree) isn't a violation.
        if matches(src, to) {
            continue;
        }
        if let Some(path) = shortest_path(src, to, from, &adj) {
            // A direct edge is already reported by `ForbidEdge`-style detail; we
            // still flag it here but make the path explicit so length-1 and
            // length-N read the same way.
            let chain = path.join(" → ");
            let target = *path.last().unwrap();
            out.push(
                violation(
                    sr,
                    format!("`{from}` must not reach `{to}`"),
                    format!("`{src}` transitively reaches `{target}` via {chain}."),
                    src,
                    target,
                )
                .with_refs(vec![format!("{src} -> {target}")]),
            );
        }
    }
}

/// BFS for the shortest path from `src` to any module matching `to`, never
/// passing through a node that matches `from`'s own subtree as the target.
/// Returns the path as concrete module names (including both endpoints), or
/// `None` if `to` is unreachable.
fn shortest_path<'a>(
    src: &'a str,
    to: &str,
    from: &str,
    adj: &HashMap<&'a str, Vec<&'a str>>,
) -> Option<Vec<&'a str>> {
    let mut queue: std::collections::VecDeque<&str> = std::collections::VecDeque::new();
    let mut prev: HashMap<&str, &str> = HashMap::new();
    let mut seen: HashSet<&str> = HashSet::new();
    queue.push_back(src);
    seen.insert(src);
    while let Some(cur) = queue.pop_front() {
        for &next in adj.get(cur).map(Vec::as_slice).unwrap_or(&[]) {
            if !seen.insert(next) {
                continue;
            }
            prev.insert(next, cur);
            // A hit: `next` matches `to` but is not itself in `from`'s subtree.
            if matches(next, to) && !matches(next, from) {
                let mut path = vec![next];
                let mut node = next;
                while let Some(&p) = prev.get(node) {
                    path.push(p);
                    node = p;
                }
                path.reverse();
                return Some(path);
            }
            queue.push_back(next);
        }
    }
    None
}

pub(super) fn check_layer(
    sr: &SourcedRule,
    module: &str,
    allowed: &[String],
    index: &CodeIndex,
    modules: &HashSet<String>,
    out: &mut Vec<Finding>,
) {
    if !grounded(module, modules) {
        return;
    }
    for e in &index.module_edges {
        if !matches(&e.from_module, module) {
            continue;
        }
        // Edges within the module's own subtree are not external dependencies.
        if matches(&e.to_module, module) {
            continue;
        }
        if allowed.iter().any(|a| matches(&e.to_module, a)) {
            continue;
        }
        let claim = if allowed.is_empty() {
            format!("`{module}` depends on nothing")
        } else {
            format!("`{module}` may depend only on {}", quote_list(allowed))
        };
        out.push(violation(
            sr,
            claim,
            format!("`{}` imports `{}`.", e.from_module, e.to_module),
            &e.from_module,
            &e.to_module,
        ));
    }
}

/// Read every in-budget source file once: its module path and full text. Shared
/// across all forbid-symbol rules so the repo is walked + read a single time per
/// `check` run rather than once per rule. Files that fail to read (non-UTF-8,
/// permissions) are dropped — they can't be textually scanned anyway.
pub(super) fn read_sources(repo_root: &Path) -> Vec<(String, String)> {
    lang::code_files(repo_root)
        .par_iter()
        .filter_map(|file| {
            let module = lang::module_path(file, repo_root);
            let text = std::fs::read_to_string(file).ok()?;
            Some((module, text))
        })
        .collect()
}

/// Check a forbid-symbol rule. Returns the (deduped) modules it scanned so a
/// clean rule can be anchored to them. Two passes: a textual scan of source
/// lines, and — when the symbol resolves to exactly one indexed definition — a
/// scan of the resolved `ref_edges` for indirect/re-exported references the text
/// scan can't see. The ref pass is skipped on ambiguous (multi-target) symbols
/// to keep zero false positives, and skips modules the text pass already flagged.
pub(super) fn check_forbid_symbol(
    sr: &SourcedRule,
    symbol: &str,
    except: &[String],
    index: &CodeIndex,
    sources: &[(String, String)],
    out: &mut Vec<Finding>,
) -> Vec<String> {
    let is_excepted = |module: &str| except.iter().any(|e| matches(module, e));
    let matcher = symbol_matcher(symbol);
    let mut scanned = Vec::new();
    let mut text_flagged: HashSet<String> = HashSet::new();

    for (module, text) in sources {
        if is_excepted(module) {
            continue;
        }
        scanned.push(module.clone());
        for (i, line) in text.lines().enumerate() {
            if matcher.is_match(line) {
                let at = format!("{module}:{}", i + 1);
                text_flagged.insert(module.clone());
                out.push(
                    Finding::problem(
                        Verdict::Contradicted,
                        format!("forbids `{symbol}`"),
                        sr.origin.clone(),
                        format!("Rule forbids `{symbol}`, but it appears in `{at}`."),
                    )
                    .anchored(Provenance::modules([module.clone()]))
                    .with_refs(vec![at]),
                );
            }
        }
    }

    // Indirect references: only trusted when the symbol names exactly one
    // indexed definition (name-based ref resolution is otherwise ambiguous).
    let targets: Vec<&str> = index
        .symbols
        .iter()
        .filter(|s| symbol_identifies(symbol, s))
        .map(|s| s.qualified_name.as_str())
        .collect();
    if let [target] = targets[..] {
        for edge in &index.ref_edges {
            if edge.to_symbol != target {
                continue;
            }
            let from_module = module_of(&edge.from_symbol);
            if is_excepted(from_module) || text_flagged.contains(from_module) {
                continue;
            }
            out.push(
                Finding::problem(
                    Verdict::Contradicted,
                    format!("forbids `{symbol}`"),
                    sr.origin.clone(),
                    format!(
                        "Rule forbids `{symbol}`, but `{}` references it.",
                        edge.from_symbol
                    ),
                )
                .anchored(Provenance::symbol(edge.from_symbol.clone()))
                .with_refs(vec![edge.from_symbol.clone()]),
            );
        }
    }

    scanned.sort();
    scanned.dedup();
    scanned
}

/// Whether a forbid-symbol operand names this indexed symbol — by full
/// `qualified_name`, by a `::`-qualified suffix, or by leaf name.
fn symbol_identifies(operand: &str, s: &crate::code::symbol::Symbol) -> bool {
    let leaf = operand.rsplit([':', '.']).next().unwrap_or(operand);
    s.qualified_name == operand
        || s.qualified_name.ends_with(&format!("::{operand}"))
        || s.name == leaf
}

/// Module path of a `module::name` qualified symbol.
fn module_of(qualified: &str) -> &str {
    qualified
        .rsplit_once("::")
        .map(|(m, _)| m)
        .unwrap_or(qualified)
}

/// Identifier symbols match on word boundaries; anything else (e.g.
/// `os.environ`) matches as a literal substring.
fn symbol_matcher(symbol: &str) -> regex::Regex {
    if symbol.chars().all(|c| c.is_alphanumeric() || c == '_') {
        regex::Regex::new(&format!(r"\b{}\b", regex::escape(symbol))).unwrap()
    } else {
        regex::Regex::new(&regex::escape(symbol)).unwrap()
    }
}

/// A finding for a violated module-graph rule.
fn violation(sr: &SourcedRule, claim: String, detail: String, from: &str, to: &str) -> Finding {
    Finding::problem(
        Verdict::Contradicted,
        claim,
        sr.origin.clone(),
        format!("Rule violated: {detail}"),
    )
    .anchored(Provenance::modules([from.to_string(), to.to_string()]))
    .with_refs(vec![format!("{from} -> {to}")])
}
