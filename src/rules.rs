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

use std::collections::HashSet;
use std::path::Path;
use std::sync::OnceLock;

use rayon::prelude::*;
use regex::Regex;

use crate::claim::Provenance;
use crate::code::lang;
use crate::code::CodeIndex;
use crate::findings::{Finding, Verdict};

/// A compiled architectural invariant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rule {
    /// `from` must not depend on `to`.
    ForbidEdge { from: String, to: String },
    /// `module` may depend only on `allowed` (empty ⇒ "depends on nothing").
    Layer {
        module: String,
        allowed: Vec<String>,
    },
    /// `symbol` must not appear outside the `except` modules.
    ForbidSymbol { symbol: String, except: Vec<String> },
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
            r"`([^`]+)`(?:\s+(?:layer|module|package|crate|component))?\s+(?:must|should|may|can|cannot|does|do)\s+not\s+(?:import|imports|depend\s+on|depends\s+on|use|uses|reference|references|access|accesses|touch|touches)\s+(?:the\s+)?`([^`]+)`",
        )
        .unwrap()
    })
}

fn never_edge_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"`([^`]+)`(?:\s+(?:layer|module|package|crate|component))?\s+never\s+(?:imports?|depends?\s+on|uses?|references?)\s+(?:the\s+)?`([^`]+)`",
        )
        .unwrap()
    })
}

/// "`X` must not be imported/used/referenced by `Y`" — captures the forbidden
/// target (1) and the dependent (2); the edge runs Y -> X.
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
        Regex::new(r"`([^`]+)`(?:\s+(?:layer|module|package|crate|component))?\s+(?:must\s+)?only\s+(?:depends?\s+on|imports?)\s+(.*)").unwrap()
    })
}

/// Forbid-symbol phrasings, all requiring a use/call signal so a bare "no
/// `config`" in prose is not mistaken for a rule.
fn forbid_symbol_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?:(?:must|should|may)\s+not\s+(?:use|call|invoke|reference)|don'?t\s+(?:use|call)|never\s+(?:use|call)|no\s+(?:direct|raw)|no\s+(?:use|usage|calls?)\s+(?:of|to))\s+`([^`]+)`",
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
}
