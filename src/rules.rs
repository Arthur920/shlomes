//! Layer 1: architecture-rule fitness functions.
//!
//! Docs constantly state architectural invariants — "`controllers` must not
//! import `db`", "`domain` depends on nothing", "no direct use of `eval`".
//! These are negative/absence claims the other checks can't see. Here we
//! extract such rules from doc prose, compile each to a dependency-graph or
//! source query, and verify it against the resolved module graph. A violation
//! is a hard `contradicted` verdict — no ML.
//!
//! Zero false positives: a rule whose module operands don't resolve to any real
//! module is skipped rather than guessed, and module matching is grounded
//! against the index's `module_set`.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::OnceLock;

use rayon::prelude::*;
use regex::Regex;

use crate::claim::Provenance;
use crate::code::lang;
use crate::code::CodeIndex;
use crate::findings::{Finding, Verdict};

/// A compiled architectural invariant.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Rule {
    /// `from` must not depend on `to` (a direct import edge).
    ForbidEdge { from: String, to: String },
    /// `from` must not *transitively* reach `to` through any chain of imports
    /// (a path of length ≥ 1 in the module graph). Subsumes `ForbidEdge` but is
    /// opt-in via explicit "transitively"/"indirectly"/"reach" phrasing, so a
    /// plain "must not import" stays a precise direct-edge check.
    ForbidReach { from: String, to: String },
    /// `module` may depend only on `allowed` (empty ⇒ "depends on nothing").
    Layer {
        module: String,
        allowed: Vec<String>,
    },
    /// `symbol` must not appear outside the `except` modules.
    ForbidSymbol { symbol: String, except: Vec<String> },
}

impl Rule {
    /// A compact one-line summary of the invariant, for the `rules` audit.
    pub fn describe(&self) -> String {
        match self {
            Rule::ForbidEdge { from, to } => format!("`{from}` ✗→ `{to}`"),
            Rule::ForbidReach { from, to } => format!("`{from}` ✗⇢ `{to}` (transitive)"),
            Rule::Layer { module, allowed } if allowed.is_empty() => {
                format!("`{module}` depends on nothing")
            }
            Rule::Layer { module, allowed } => {
                format!("`{module}` → only {}", quote_list(allowed))
            }
            Rule::ForbidSymbol { symbol, except } if except.is_empty() => {
                format!("forbid symbol `{symbol}`")
            }
            Rule::ForbidSymbol { symbol, except } => {
                format!("forbid symbol `{symbol}` (except {})", quote_list(except))
            }
        }
    }
}

/// A rule plus where it came from (a doc `path:line`, or the rules file).
#[derive(Debug, Clone)]
pub struct SourcedRule {
    pub rule: Rule,
    pub origin: String,
}

// ---- rule sources ---------------------------------------------------------

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
/// returned string is a real path segment, so [`matches`] accepts it verbatim.
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

// ---- checking -------------------------------------------------------------

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

// ---- audit (dry-run visibility) -------------------------------------------

/// Outcome of auditing one extracted rule against the index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuleStatus {
    /// Grounded and verified: the invariant holds in the current code.
    Holds,
    /// Grounded and verified, but violated `count` time(s) in the import graph.
    Violated(usize),
    /// Skipped, not guessed: this operand matched no real module, so the rule
    /// is unverifiable. The string is the offending operand.
    Ungrounded(String),
}

/// One audited rule: what was extracted, where from, and how it fared. This is
/// the data behind `staleguard rules` — it turns the otherwise-silent prose
/// extraction into something a user can see and debug.
#[derive(Debug, Clone)]
pub struct AuditRow {
    pub rule: Rule,
    pub origin: String,
    pub status: RuleStatus,
}

/// Audit every extracted rule against the index without emitting findings —
/// reusing the exact same grounding and graph checks as [`check`], so the
/// report can never disagree with a real run.
pub fn audit(rules: &[SourcedRule], index: &CodeIndex, repo_root: &Path) -> Vec<AuditRow> {
    let modules = index.module_set();
    let sources = if rules
        .iter()
        .any(|r| matches!(r.rule, Rule::ForbidSymbol { .. }))
    {
        read_sources(repo_root)
    } else {
        Vec::new()
    };

    rules
        .iter()
        .map(|sr| {
            let status = match &sr.rule {
                Rule::ForbidEdge { from, to } => {
                    if !grounded(from, &modules) {
                        RuleStatus::Ungrounded(from.clone())
                    } else if !grounded(to, &modules) {
                        RuleStatus::Ungrounded(to.clone())
                    } else {
                        let mut out = Vec::new();
                        check_forbid_edge(sr, from, to, index, &modules, &mut out);
                        violated_or_holds(out.len())
                    }
                }
                Rule::ForbidReach { from, to } => {
                    if !grounded(from, &modules) {
                        RuleStatus::Ungrounded(from.clone())
                    } else if !grounded(to, &modules) {
                        RuleStatus::Ungrounded(to.clone())
                    } else {
                        let mut out = Vec::new();
                        check_forbid_reach(sr, from, to, index, &modules, &mut out);
                        violated_or_holds(out.len())
                    }
                }
                Rule::Layer { module, allowed } => {
                    if !grounded(module, &modules) {
                        RuleStatus::Ungrounded(module.clone())
                    } else {
                        let mut out = Vec::new();
                        check_layer(sr, module, allowed, index, &modules, &mut out);
                        violated_or_holds(out.len())
                    }
                }
                Rule::ForbidSymbol { symbol, except } => {
                    // Symbol rules ground against source text, not the module
                    // set, so they are always checkable.
                    let mut out = Vec::new();
                    check_forbid_symbol(sr, symbol, except, index, &sources, &mut out);
                    violated_or_holds(out.len())
                }
            };
            AuditRow {
                rule: sr.rule.clone(),
                origin: sr.origin.clone(),
                status,
            }
        })
        .collect()
}

fn violated_or_holds(count: usize) -> RuleStatus {
    if count == 0 {
        RuleStatus::Holds
    } else {
        RuleStatus::Violated(count)
    }
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

fn check_forbid_edge(
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
fn check_forbid_reach(
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

fn check_layer(
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

/// Check a forbid-symbol rule. Returns the (deduped) modules it scanned so a
/// clean rule can be anchored to them. Two passes: a textual scan of source
/// lines, and — when the symbol resolves to exactly one indexed definition — a
/// scan of the resolved `ref_edges` for indirect/re-exported references the text
/// scan can't see. The ref pass is skipped on ambiguous (multi-target) symbols
/// to keep zero false positives, and skips modules the text pass already flagged.
/// Read every in-budget source file once: its module path and full text. Shared
/// across all forbid-symbol rules so the repo is walked + read a single time per
/// `check` run rather than once per rule. Files that fail to read (non-UTF-8,
/// permissions) are dropped — they can't be textually scanned anyway.
fn read_sources(repo_root: &Path) -> Vec<(String, String)> {
    lang::code_files(repo_root)
        .par_iter()
        .filter_map(|file| {
            let module = lang::module_path(file, repo_root);
            let text = std::fs::read_to_string(file).ok()?;
            Some((module, text))
        })
        .collect()
}

fn check_forbid_symbol(
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

// ---- matching helpers -----------------------------------------------------

/// A module path matches an operand by exact equality, subtree prefix
/// (`op/…`), leaf suffix (`…/op`), or interior segment (`…/op/…`) — so a
/// conceptual name (`controllers`) matches a real path (`src/controllers`).
pub(crate) fn matches(module: &str, operand: &str) -> bool {
    let op = operand.trim_matches('/');
    module == op
        || module.starts_with(&format!("{op}/"))
        || module.ends_with(&format!("/{op}"))
        || module.contains(&format!("/{op}/"))
}

/// True if an operand matches at least one real module.
pub(crate) fn grounded(operand: &str, modules: &HashSet<String>) -> bool {
    modules.iter().any(|m| matches(m, operand))
}

/// Identifier symbols match on word boundaries; anything else (e.g.
/// `os.environ`) matches as a literal substring.
fn symbol_matcher(symbol: &str) -> Regex {
    if symbol.chars().all(|c| c.is_alphanumeric() || c == '_') {
        Regex::new(&format!(r"\b{}\b", regex::escape(symbol))).unwrap()
    } else {
        Regex::new(&regex::escape(symbol)).unwrap()
    }
}

fn quote_list(items: &[String]) -> String {
    items
        .iter()
        .map(|s| format!("`{s}`"))
        .collect::<Vec<_>>()
        .join(", ")
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code::symbol::{DepEdge, Facts, RefEdge, Span, Symbol, SymbolKind, Visibility};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn edge(from: &str, to: &str) -> DepEdge {
        DepEdge {
            from_module: from.to_string(),
            to_module: to.to_string(),
        }
    }

    fn symbol(name: &str, qualified: &str, module: &str) -> Symbol {
        Symbol {
            qualified_name: qualified.to_string(),
            name: name.to_string(),
            kind: SymbolKind::Class,
            visibility: Visibility::Public,
            module: module.to_string(),
            span: Span::zero(),
            body_span: Span::zero(),
            signature: None,
            doc: None,
            facts: Facts::default(),
            calls: Vec::new(),
            members: Vec::new(),
        }
    }

    fn scratch_dir(tag: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("staleguard-rules-{tag}-{nanos}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn index(edges: Vec<DepEdge>) -> CodeIndex {
        // module_set comes from from_module + symbol modules; mirror endpoints
        // into edges so both ends ground.
        CodeIndex {
            symbols: vec![],
            edges: edges
                .iter()
                .flat_map(|e| [edge(&e.from_module, "x"), edge(&e.to_module, "x")])
                .collect(),
            module_edges: edges,
            ref_edges: vec![],
        }
    }

    fn rule(r: Rule) -> Vec<SourcedRule> {
        vec![SourcedRule {
            rule: r,
            origin: "rules".into(),
        }]
    }

    #[test]
    fn forbid_edge_violation_is_contradicted() {
        let idx = index(vec![edge("src/api", "src/db")]);
        let rules = rule(Rule::ForbidEdge {
            from: "src/api".into(),
            to: "src/db".into(),
        });
        let f = check(&rules, &idx, Path::new("."));
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].verdict, Verdict::Contradicted);
        assert_eq!(f[0].code_refs, vec!["src/api -> src/db"]);
    }

    #[test]
    fn forbid_edge_clean_repo_passes() {
        let idx = index(vec![edge("src/api", "src/domain")]);
        let rules = rule(Rule::ForbidEdge {
            from: "src/api".into(),
            to: "src/db".into(),
        });
        assert!(check(&rules, &idx, Path::new(".")).is_empty());
    }

    #[test]
    fn ungrounded_operand_is_skipped() {
        let idx = index(vec![edge("src/api", "src/db")]);
        // `ghost` matches no real module → rule unverifiable, not flagged.
        let rules = rule(Rule::ForbidEdge {
            from: "ghost".into(),
            to: "src/db".into(),
        });
        assert!(check(&rules, &idx, Path::new(".")).is_empty());
    }

    #[test]
    fn conceptual_name_matches_real_path() {
        let idx = index(vec![edge("src/api", "src/db")]);
        let rules = rule(Rule::ForbidEdge {
            from: "api".into(),
            to: "db".into(),
        });
        assert_eq!(check(&rules, &idx, Path::new(".")).len(), 1);
    }

    #[test]
    fn audit_reports_holds_violated_and_ungrounded() {
        let idx = index(vec![edge("src/api", "src/db")]);
        let rules = vec![
            SourcedRule {
                rule: Rule::ForbidEdge {
                    from: "api".into(),
                    to: "db".into(),
                },
                origin: "ARCH.md:1".into(),
            },
            SourcedRule {
                rule: Rule::ForbidEdge {
                    from: "api".into(),
                    to: "domain".into(),
                },
                origin: "ARCH.md:2".into(),
            },
            SourcedRule {
                rule: Rule::ForbidEdge {
                    from: "api".into(),
                    to: "ghost".into(),
                },
                origin: "ARCH.md:3".into(),
            },
        ];
        let rows = audit(&rules, &idx, Path::new("."));
        assert_eq!(rows[0].status, RuleStatus::Violated(1));
        // `domain` isn't a real module here, so this edge can't exist → holds
        // only if grounded; domain is ungrounded, so it's skipped, not "holds".
        assert_eq!(rows[1].status, RuleStatus::Ungrounded("domain".into()));
        assert_eq!(rows[2].status, RuleStatus::Ungrounded("ghost".into()));
    }

    #[test]
    fn prose_forbid_reach_extracted() {
        for md in [
            "`handlers` must not transitively import `store`.",
            "`a` must not indirectly depend on `b`.",
            "`a` must not even indirectly use `b`.",
            "`a` must not reach `b`.",
        ] {
            let rules = extract_prose_rules(md, "ARCH.md");
            assert!(
                matches!(
                    rules.first().map(|r| &r.rule),
                    Some(Rule::ForbidReach { .. })
                ),
                "expected a reach rule from: {md:?}"
            );
        }
        // A plain direct import must stay a direct ForbidEdge, not a reach rule.
        let direct = extract_prose_rules("`a` must not import `b`.", "ARCH.md");
        assert!(matches!(direct[0].rule, Rule::ForbidEdge { .. }));
    }

    #[test]
    fn forbid_reach_flags_transitive_path() {
        // a -> b -> c; "a must not reach c" is violated through b.
        let idx = index(vec![edge("src/a", "src/b"), edge("src/b", "src/c")]);
        let rules = rule(Rule::ForbidReach {
            from: "a".into(),
            to: "c".into(),
        });
        let f = check(&rules, &idx, Path::new("."));
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].verdict, Verdict::Contradicted);
        assert!(f[0].detail.contains("src/a"));
        assert!(f[0].detail.contains("src/b"));
        assert!(f[0].detail.contains("src/c"));
    }

    #[test]
    fn forbid_reach_holds_when_unreachable() {
        // a -> b, and an isolated c. "a must not reach c" holds.
        let idx = index(vec![edge("src/a", "src/b"), edge("src/c", "src/b")]);
        let rules = rule(Rule::ForbidReach {
            from: "a".into(),
            to: "c".into(),
        });
        // Grounded + holds emits a Supported claim, so assert no contradiction
        // rather than an empty result.
        let f = check(&rules, &idx, Path::new("."));
        assert!(f.iter().all(|x| x.verdict != Verdict::Contradicted));
    }

    #[test]
    fn forbid_reach_ungrounded_is_skipped() {
        let idx = index(vec![edge("src/a", "src/b")]);
        let rows = audit(
            &rule(Rule::ForbidReach {
                from: "a".into(),
                to: "ghost".into(),
            }),
            &idx,
            Path::new("."),
        );
        assert_eq!(rows[0].status, RuleStatus::Ungrounded("ghost".into()));
    }

    fn mods(items: &[&str]) -> HashSet<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn bare_extracts_grounded_dependency_rules() {
        let m = mods(&["src/handlers/h", "src/store/db", "src/util/u"]);
        // un-backticked operands, the dominant real-world phrasing.
        let r = extract_bare_rules(
            "The handlers module must not depend on the store module.",
            "ARCH.md",
            &m,
        );
        assert_eq!(
            r[0].rule,
            Rule::ForbidEdge {
                from: "handlers".into(),
                to: "store".into()
            }
        );
        assert!(r[0].origin.ends_with("[bare]"));
    }

    #[test]
    fn bare_handles_bold_and_cannot() {
        let m = mods(&["src/service/s", "src/util/u"]);
        // **bold** operands + "cannot <verb>" (no following "not").
        let r = extract_bare_rules(
            "The **service** module cannot import **util**.",
            "ARCH.md",
            &m,
        );
        assert_eq!(
            r[0].rule,
            Rule::ForbidEdge {
                from: "service".into(),
                to: "util".into()
            }
        );
    }

    #[test]
    fn bare_extracts_transitive_reach() {
        let m = mods(&["src/handlers/h", "src/store/db"]);
        let r = extract_bare_rules(
            "The handlers layer must not transitively reach store.",
            "ARCH.md",
            &m,
        );
        assert_eq!(
            r[0].rule,
            Rule::ForbidReach {
                from: "handlers".into(),
                to: "store".into()
            }
        );
    }

    #[test]
    fn bare_suppresses_solid_and_ungrounded_noise() {
        let m = mods(&["src/handlers/h", "src/store/db"]);
        // SOLID boilerplate (stopwords), and operands matching no module.
        for noise in [
            "High-level modules should not depend on low-level modules.",
            "Clients should not be forced to depend on interfaces they do not use.",
            "The frobnicator must not depend on the wizbang.",
            "Abstractions should not depend on details.",
        ] {
            assert!(
                extract_bare_rules(noise, "ARCH.md", &m).is_empty(),
                "should not fire on: {noise:?}"
            );
        }
    }

    #[test]
    fn bare_preserves_real_module_case() {
        // Java-style CamelCase module segment: operand canonicalises to the real
        // segment so `matches` (case-sensitive) still finds it downstream.
        let m = mods(&["api/Handler", "store/Db"]);
        let r = extract_bare_rules("The Handler module must not depend on Db.", "ARCH.md", &m);
        assert_eq!(
            r[0].rule,
            Rule::ForbidEdge {
                from: "Handler".into(),
                to: "Db".into()
            }
        );
    }

    #[test]
    fn audit_holds_when_grounded_and_clean() {
        let idx = index(vec![edge("src/api", "src/domain")]);
        let rules = vec![SourcedRule {
            rule: Rule::ForbidEdge {
                from: "api".into(),
                to: "domain".into(),
            },
            origin: "ARCH.md:1".into(),
        }];
        // both operands ground to real modules; the forbidden edge does exist.
        assert_eq!(
            audit(&rules, &idx, Path::new("."))[0].status,
            RuleStatus::Violated(1)
        );

        let idx = index(vec![
            edge("src/api", "src/domain"),
            edge("src/api", "src/util"),
        ]);
        let rules = vec![SourcedRule {
            rule: Rule::ForbidEdge {
                from: "util".into(),
                to: "domain".into(),
            },
            origin: "ARCH.md:1".into(),
        }];
        // util and domain both real; util→domain edge absent → holds.
        assert_eq!(
            audit(&rules, &idx, Path::new("."))[0].status,
            RuleStatus::Holds
        );
    }

    #[test]
    fn layer_depends_on_nothing() {
        let idx = index(vec![edge("src/domain", "src/infra")]);
        let rules = rule(Rule::Layer {
            module: "src/domain".into(),
            allowed: vec![],
        });
        assert_eq!(check(&rules, &idx, Path::new(".")).len(), 1);
    }

    #[test]
    fn layer_allows_listed_and_subtree() {
        let idx = index(vec![
            edge("src/api", "src/domain"),
            edge("src/api", "src/api/util"),
            edge("src/api", "src/db"),
        ]);
        let rules = rule(Rule::Layer {
            module: "src/api".into(),
            allowed: vec!["src/domain".into()],
        });
        let f = check(&rules, &idx, Path::new("."));
        // domain (allowed) and api/util (own subtree) pass; db is flagged.
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].code_refs, vec!["src/api -> src/db"]);
    }

    #[test]
    fn prose_forbid_edge_extracted() {
        let md = "The `controllers` layer must not import `db` directly.";
        let rules = extract_prose_rules(md, "ARCH.md");
        assert_eq!(
            rules[0].rule,
            Rule::ForbidEdge {
                from: "controllers".into(),
                to: "db".into()
            }
        );
        assert_eq!(rules[0].origin, "ARCH.md:1");
    }

    #[test]
    fn prose_depends_on_nothing_extracted() {
        let md = "`domain` depends on nothing.";
        let rules = extract_prose_rules(md, "ARCH.md");
        assert_eq!(
            rules[0].rule,
            Rule::Layer {
                module: "domain".into(),
                allowed: vec![]
            }
        );
    }

    #[test]
    fn prose_only_depends_extracted() {
        let md = "`api` must only depend on `domain` and `util`.";
        let rules = extract_prose_rules(md, "ARCH.md");
        assert_eq!(
            rules[0].rule,
            Rule::Layer {
                module: "api".into(),
                allowed: vec!["domain".into(), "util".into()]
            }
        );
    }

    #[test]
    fn prose_forbid_symbol_with_except() {
        let md = "There must be no direct `os.environ` outside `config`.";
        let rules = extract_prose_rules(md, "ARCH.md");
        assert_eq!(
            rules[0].rule,
            Rule::ForbidSymbol {
                symbol: "os.environ".into(),
                except: vec!["config".into()]
            }
        );
    }

    #[test]
    fn prose_forbid_symbol_no_direct_use_of() {
        // "no direct use of `X`" — a very common phrasing that previously fell
        // between the `no direct` and `no use of` branches and silently dropped.
        for md in [
            "There must be no direct use of `process.env` outside `config`.",
            "no raw usage of `os.environ`",
            "no direct calls to `eval`",
        ] {
            let rules = extract_prose_rules(md, "ARCH.md");
            assert!(
                matches!(
                    rules.first().map(|r| &r.rule),
                    Some(Rule::ForbidSymbol { .. })
                ),
                "expected a forbid-symbol rule from: {md:?}"
            );
        }
        // The bare-backtick form still works (optional group absent).
        let md = "no direct `process.env`";
        assert!(matches!(
            extract_prose_rules(md, "ARCH.md").first().map(|r| &r.rule),
            Some(Rule::ForbidSymbol { .. })
        ));
    }

    #[test]
    fn quoted_example_rule_is_ignored() {
        // An author illustrating the feature in quotes is not stating a rule.
        let md = r#"- forbidden call/symbol: "no direct `os.environ` outside config""#;
        assert!(extract_prose_rules(md, "ARCH.md").is_empty());
        let md2 = r#"For example, "`api` must not import `db`" is a forbidden edge."#;
        assert!(extract_prose_rules(md2, "ARCH.md").is_empty());
    }

    #[test]
    fn bare_no_x_is_not_a_rule() {
        // "no `foo`" without a use/call signal must not become a rule.
        let md = "There is no `config` file in this layout.";
        assert!(extract_prose_rules(md, "ARCH.md").is_empty());
    }

    #[test]
    fn prose_forbid_by_reverses_direction() {
        let md = "`db` must not be imported by `controllers`.";
        let rules = extract_prose_rules(md, "ARCH.md");
        assert_eq!(
            rules[0].rule,
            Rule::ForbidEdge {
                from: "controllers".into(),
                to: "db".into()
            }
        );
    }

    #[test]
    fn prose_independent_is_symmetric() {
        let md = "`domain` is independent of `infra`.";
        let kinds: Vec<Rule> = extract_prose_rules(md, "ARCH.md")
            .into_iter()
            .map(|s| s.rule)
            .collect();
        assert!(kinds.contains(&Rule::ForbidEdge {
            from: "domain".into(),
            to: "infra".into()
        }));
        assert!(kinds.contains(&Rule::ForbidEdge {
            from: "infra".into(),
            to: "domain".into()
        }));
    }

    #[test]
    fn clean_forbid_symbol_is_anchored_to_scanned_modules() {
        let dir = scratch_dir("clean-symbol");
        fs::write(dir.join("safe.rs"), "fn ok() {}\n").unwrap();
        let rules = rule(Rule::ForbidSymbol {
            symbol: "eval".into(),
            except: vec![],
        });
        let f = check(&rules, &index(vec![]), &dir);
        let supported: Vec<&Finding> = f
            .iter()
            .filter(|x| x.verdict == Verdict::Supported)
            .collect();
        assert_eq!(supported.len(), 1);
        assert!(
            !supported[0].provenance.modules.is_empty(),
            "must anchor to scanned modules"
        );
    }

    #[test]
    fn forbid_symbol_catches_indirect_ref() {
        let mut idx = index(vec![]);
        idx.symbols = vec![
            symbol("Client", "src/legacy::Client", "src/legacy"),
            symbol("run", "src/app::run", "src/app"),
        ];
        idx.ref_edges = vec![RefEdge {
            from_symbol: "src/app::run".into(),
            to_symbol: "src/legacy::Client".into(),
        }];
        let rules = rule(Rule::ForbidSymbol {
            symbol: "legacy::Client".into(),
            except: vec![],
        });
        // Empty repo dir → text scan finds nothing; only the ref edge fires.
        let f = check(&rules, &idx, &scratch_dir("indirect"));
        assert!(f
            .iter()
            .any(|x| x.verdict == Verdict::Contradicted && x.code_refs == vec!["src/app::run"]));
    }

    #[test]
    fn forbid_symbol_skips_ambiguous_indirect_ref() {
        let mut idx = index(vec![]);
        // Two symbols share the leaf `Client` → ambiguous → no ref-edge findings.
        idx.symbols = vec![
            symbol("Client", "src/legacy::Client", "src/legacy"),
            symbol("Client", "src/modern::Client", "src/modern"),
            symbol("run", "src/app::run", "src/app"),
        ];
        idx.ref_edges = vec![RefEdge {
            from_symbol: "src/app::run".into(),
            to_symbol: "src/legacy::Client".into(),
        }];
        let rules = rule(Rule::ForbidSymbol {
            symbol: "Client".into(),
            except: vec![],
        });
        let f = check(&rules, &idx, &scratch_dir("ambiguous"));
        assert!(f.iter().all(|x| x.verdict != Verdict::Contradicted));
    }

    // ---- prose-eval harness ------------------------------------------------
    //
    // Drives both extractors over a checked-in labeled corpus and reports
    // precision/recall, gating on the Layer-1 zero-false-positive contract.
    // This is the measurement substrate for improving prose recall: any change
    // to the extractors moves the printed numbers, and the asserts ratchet.

    /// Stable comparable key for a compiled rule.
    fn rule_key(r: &Rule) -> String {
        match r {
            Rule::ForbidEdge { from, to } => format!("edge:{from}->{to}"),
            Rule::ForbidReach { from, to } => format!("reach:{from}->{to}"),
            Rule::Layer { module, allowed } => {
                let mut a = allowed.clone();
                a.sort();
                format!("layer:{module}:{}", a.join(","))
            }
            Rule::ForbidSymbol { symbol, except } => {
                let mut e = except.clone();
                e.sort();
                format!("symbol:{symbol}:{}", e.join(","))
            }
        }
    }

    /// Same key shape, derived from a gold JSON entry in the corpus.
    fn gold_key(v: &serde_json::Value) -> String {
        let strs = |k: &str| -> Vec<String> {
            v[k].as_array()
                .map(|a| a.iter().map(|s| s.as_str().unwrap().to_string()).collect())
                .unwrap_or_default()
        };
        let s = |k: &str| v[k].as_str().unwrap().to_string();
        match v["kind"].as_str().unwrap() {
            "forbid_edge" => format!("edge:{}->{}", s("from"), s("to")),
            "forbid_reach" => format!("reach:{}->{}", s("from"), s("to")),
            "layer" => {
                let mut a = strs("allowed");
                a.sort();
                format!("layer:{}:{}", s("module"), a.join(","))
            }
            "forbid_symbol" => {
                let mut e = strs("except");
                e.sort();
                format!("symbol:{}:{}", s("symbol"), e.join(","))
            }
            other => panic!("unknown gold kind {other:?}"),
        }
    }

    #[test]
    fn prose_corpus_precision_recall() {
        let corpus = include_str!("../tests/fixtures/prose_corpus.jsonl");
        let (mut tp, mut fp, mut fn_, mut gold_total) = (0usize, 0usize, 0usize, 0usize);
        let mut leaks: Vec<String> = Vec::new();

        for (i, raw) in corpus.lines().enumerate() {
            let raw = raw.trim();
            if raw.is_empty() || raw.starts_with("//") {
                continue;
            }
            let v: serde_json::Value = serde_json::from_str(raw)
                .unwrap_or_else(|e| panic!("corpus line {}: {e}\n{raw}", i + 1));
            let text = v["text"].as_str().unwrap();
            let modules: HashSet<String> = v["modules"]
                .as_array()
                .unwrap()
                .iter()
                .map(|m| m.as_str().unwrap().to_string())
                .collect();
            let mut gold: HashSet<String> =
                v["gold"].as_array().unwrap().iter().map(gold_key).collect();
            gold_total += gold.len();

            // Mirror main.rs: prose rules first, then bare deduped against them.
            let mut got: Vec<Rule> = extract_prose_rules(text, "doc.md")
                .into_iter()
                .map(|s| s.rule)
                .collect();
            let known: HashSet<String> = got.iter().map(rule_key).collect();
            for s in extract_bare_rules(text, "doc.md", &modules) {
                if !known.contains(&rule_key(&s.rule)) {
                    got.push(s.rule);
                }
            }

            let got_keys: HashSet<String> = got.iter().map(rule_key).collect();
            for k in &got_keys {
                if gold.remove(k) {
                    tp += 1;
                } else {
                    fp += 1;
                    leaks.push(format!("  line {} [{}]: extracted {k}", i + 1, v["tag"]));
                }
            }
            fn_ += gold.len(); // gold rules left unmatched
            for k in &gold {
                eprintln!("  MISS line {} [{}]: {k}", i + 1, v["tag"]);
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
            "prose eval: tp={tp} fp={fp} fn={fn_} gold={gold_total} \
             precision={precision:.3} recall={recall:.3}"
        );

        // Zero-FP is the Layer-1 contract — any extracted rule not in gold is a
        // hard failure, with the offending sentences named.
        assert!(
            fp == 0,
            "precision regression: {fp} false positive(s)\n{}",
            leaks.join("\n")
        );
        // Recall ratchet: never drop below the current measured floor.
        assert!(
            recall >= 0.90,
            "recall regression: {recall:.3} < 0.90 floor (tp={tp} fn={fn_})"
        );
    }
}
