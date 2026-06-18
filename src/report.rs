//! Output rendering for every `staleguard` subcommand.
//!
//! One place that knows how to turn each command's result — findings, a drift
//! [`Outcome`](crate::drift::Outcome), the rule audit, the code index — into
//! either human `text` or machine `json`, so the command dispatch in `main.rs`
//! stays argument-parsing plus a render call.

use crate::code::CodeIndex;
use crate::drift;
use crate::findings::Finding;
use crate::rules;

use clap::ValueEnum;

/// How a command renders its result: `text` (human) or `json` (machine-readable).
#[derive(Clone, Copy, ValueEnum)]
pub enum Format {
    Text,
    Json,
}

pub(crate) fn report(findings: &[Finding], format: Format) {
    match format {
        Format::Json => {
            println!("{}", serde_json::to_string_pretty(findings).unwrap());
        }
        Format::Text => {
            if findings.is_empty() {
                println!("\u{2713} no coherence issues found");
                return;
            }
            for f in findings {
                println!("[{}] {}: {}", f.verdict.as_str(), f.doc_path, f.detail);
            }
            println!("\n{} finding(s)", findings.len());
        }
    }
}

/// Report a completed drift run: the findings plus the alignment score and the
/// lineage/regression summary.
pub(crate) fn report_check(out: &drift::Outcome, format: Format) {
    match format {
        Format::Json => {
            let payload = serde_json::json!({
                "findings": out.findings,
                "score": out.score,
                "carried_forward": out.carried_forward,
                "total_claims": out.total_claims,
                "regression": out.regression.map(|(b, h)| serde_json::json!({ "base": b, "head": h })),
            });
            println!("{}", serde_json::to_string_pretty(&payload).unwrap());
        }
        Format::Text => {
            report(&out.findings, format);
            println!(
                "\nalignment {:.3} | {} claim(s), {} carried forward",
                out.score.repo, out.total_claims, out.carried_forward
            );
            if let Some((base, head)) = out.regression {
                println!("\u{2717} score regressed: {base:.3} (base) -> {head:.3} (head)");
            }
        }
    }
}

/// Print the architecture-rule audit: every rule extracted from doc prose, where
/// it came from, and its status (holds / violated / skipped-ungrounded). This is
/// the visibility layer over the otherwise-silent prose extractor.
pub(crate) fn report_rules(rows: &[rules::AuditRow], format: Format) {
    match format {
        Format::Json => {
            let payload: Vec<_> = rows
                .iter()
                .map(|r| {
                    let (status, detail) = match &r.status {
                        rules::RuleStatus::Holds => ("holds", serde_json::Value::Null),
                        rules::RuleStatus::Violated(n) => {
                            ("violated", serde_json::json!({ "violations": n }))
                        }
                        rules::RuleStatus::Ungrounded(op) => {
                            ("ungrounded", serde_json::json!({ "operand": op }))
                        }
                    };
                    serde_json::json!({
                        "rule": r.rule.describe(),
                        "origin": r.origin,
                        "status": status,
                        "detail": detail,
                        "experimental": r.origin.ends_with("[bare]"),
                    })
                })
                .collect();
            println!("{}", serde_json::to_string_pretty(&payload).unwrap());
        }
        Format::Text => {
            if rows.is_empty() {
                println!("no architecture rules extracted from doc prose.");
                println!(
                    "(rules must be written with both operands in backticks, e.g. \
                     \"`api` must not import `db`\".)"
                );
                return;
            }
            let (mut holds, mut violated, mut ungrounded) = (0, 0, 0);
            let mut bare = 0;
            for r in rows {
                let (mark, note) = match &r.status {
                    rules::RuleStatus::Holds => {
                        holds += 1;
                        ("\u{2713} holds    ", String::new())
                    }
                    rules::RuleStatus::Violated(n) => {
                        violated += 1;
                        ("\u{2717} VIOLATED ", format!("  ({n} violation(s))"))
                    }
                    rules::RuleStatus::Ungrounded(op) => {
                        ungrounded += 1;
                        (
                            "\u{26a0} skipped  ",
                            format!("  (`{op}` matches no real module)"),
                        )
                    }
                };
                // Experimental bare-operand rules are flagged so they are never
                // confused with the enforced (backticked) ones.
                let tag = if r.origin.ends_with("[bare]") {
                    bare += 1;
                    " \u{2248}bare"
                } else {
                    ""
                };
                println!(
                    "{}{}  {:<30}{}  [{}]",
                    mark,
                    tag,
                    r.rule.describe(),
                    note,
                    r.origin
                );
            }
            println!(
                "\n{} rule(s): {holds} hold, {violated} violated, {ungrounded} skipped (ungrounded)",
                rows.len()
            );
            if ungrounded > 0 {
                println!(
                    "note: skipped rules are not enforced — fix the operand name so it \
                     matches a real module, or the rule is silently ignored."
                );
            }
            if bare > 0 {
                println!(
                    "note: {bare} \u{2248}bare rule(s) were parsed from un-backticked prose \
                     (experimental, grounded against the module graph). These are shown for \
                     evaluation and are NOT enforced by `staleguard check`."
                );
            }
        }
    }
}

pub(crate) fn report_index(index: &CodeIndex, format: Format) {
    match format {
        Format::Json => {
            println!("{}", serde_json::to_string_pretty(index).unwrap());
        }
        Format::Text => {
            for s in &index.symbols {
                println!(
                    "[{:?}/{:?}] {} ({}:{})",
                    s.kind, s.visibility, s.qualified_name, s.span.path, s.span.start_line
                );
            }
            for e in &index.edges {
                println!("edge  {} -> {}", e.from_module, e.to_module);
            }
            for e in &index.module_edges {
                println!("mod-edge  {} -> {}", e.from_module, e.to_module);
            }
            for r in &index.ref_edges {
                println!("ref-edge  {} -> {}", r.from_symbol, r.to_symbol);
            }
            println!(
                "\n{} symbol(s), {} edge(s), {} mod-edge(s), {} ref-edge(s)",
                index.symbols.len(),
                index.edges.len(),
                index.module_edges.len(),
                index.ref_edges.len()
            );
        }
    }
}
