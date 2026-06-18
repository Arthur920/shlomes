//! Layer 1: architecture-rule fitness functions.
//!
//! Docs constantly state architectural invariants — "`controllers` must not
//! import `db`", "`domain` depends on nothing", "no direct use of `eval`".
//! These are negative/absence claims the other checks can't see. We
//! [`extract`](self::extract) such rules from doc prose, compile each to a
//! dependency-graph or source query, and [`verify`](self::verify) it against the
//! resolved module graph. A violation is a hard `contradicted` verdict — no ML.
//! [`audit`](self::audit) reuses the same checks for the dry-run `rules` report.
//!
//! Zero false positives: a rule whose module operands don't resolve to any real
//! module is skipped rather than guessed, and module matching is grounded
//! against the index's `module_set`.

mod audit;
mod extract;
mod verify;

use std::collections::HashSet;
#[allow(unused_imports)]
use std::path::Path;

pub use audit::{audit, AuditRow, RuleStatus};
pub use extract::{extract_bare_rules, extract_prose_rules};
pub use verify::check;

#[allow(unused_imports)]
use crate::code::CodeIndex;
#[allow(unused_imports)]
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

// ---- shared matching helpers ----------------------------------------------

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

/// Render a list of operands as a comma-separated backtick-quoted string.
pub(super) fn quote_list(items: &[String]) -> String {
    items
        .iter()
        .map(|s| format!("`{s}`"))
        .collect::<Vec<_>>()
        .join(", ")
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
        let corpus = include_str!("../../tests/fixtures/prose_corpus.jsonl");
        let (mut tp, mut fp, mut fn_, mut gold_total) = (0usize, 0usize, 0usize, 0usize);
        let mut leaks: Vec<String> = Vec::new();

        for (lineno, raw) in crate::testutil::corpus_rows(corpus) {
            let v: serde_json::Value = serde_json::from_str(raw)
                .unwrap_or_else(|e| panic!("corpus line {lineno}: {e}\n{raw}"));
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
                    leaks.push(format!("  line {lineno} [{}]: extracted {k}", v["tag"]));
                }
            }
            fn_ += gold.len(); // gold rules left unmatched
            for k in &gold {
                eprintln!("  MISS line {lineno} [{}]: {k}", v["tag"]);
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
