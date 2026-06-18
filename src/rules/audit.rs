//! The dry-run rule audit behind `staleguard rules`.
//!
//! [`audit`] runs the exact same grounding and graph checks as [`super::check`]
//! but reports each rule's status instead of emitting findings, so the
//! otherwise-silent prose extraction becomes something a user can see and debug.

use std::path::Path;

use crate::code::CodeIndex;

use super::verify::{
    check_forbid_edge, check_forbid_reach, check_forbid_symbol, check_layer, read_sources,
};
use super::{grounded, Rule, SourcedRule};

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
/// reusing the exact same grounding and graph checks as [`super::check`], so the
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
