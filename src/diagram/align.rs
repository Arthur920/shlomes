//! Sequence-diagram alignment (Layer 1, deterministic). Aligns a sequence
//! diagram's ordered message steps against the code's ordered call sequence
//! (`Symbol.calls`) with a Needleman–Wunsch global alignment, exact-name
//! substitution — no embeddings (semantic substitution is Layer 2). The trace
//! *is* the explanation: matched steps are coherent, gaps and substitutions are
//! missing / undrawn / out-of-order steps.
//!
//! Grounding is conservative: a diagram is aligned only against the single code
//! symbol whose call sequence its steps best match, and only when that match
//! clears [`MIN_GROUND`] and covers at least half the drawn steps. A diagram
//! that grounds to nothing emits nothing (zero-FP), exactly like the graph diff.

use crate::claim::Provenance;
use crate::code::symbol::Symbol;
use crate::code::CodeIndex;
use crate::findings::{Finding, Verdict};

use super::sequence::Sequence;

/// Minimum matched steps before a diagram is considered grounded to a symbol.
const MIN_GROUND: usize = 2;

const MATCH: i32 = 2;
const MISMATCH: i32 = -1;
const GAP: i32 = -2;

/// Alignment findings for one sequence diagram. Empty when no code symbol grounds
/// the diagram's flow.
pub(super) fn check(seq: &Sequence, index: &CodeIndex) -> Vec<Finding> {
    let steps: Vec<String> = seq.messages.iter().map(|m| m.call_token()).collect();
    let Some(driver) = ground(&steps, index) else {
        return Vec::new();
    };
    emit(seq, &steps, driver)
}

/// Pick the code symbol whose ordered calls best match the diagram steps. Returns
/// `None` unless the best match clears `MIN_GROUND` and covers ≥ half the steps.
fn ground<'a>(steps: &[String], index: &'a CodeIndex) -> Option<&'a Symbol> {
    let mut best: Option<(usize, &Symbol)> = None;
    for s in &index.symbols {
        if s.calls.is_empty() {
            continue;
        }
        let matched = overlap(steps, &s.calls);
        if matched == 0 {
            continue;
        }
        match best {
            Some((m, _)) if matched < m => {}
            Some((m, prev)) if matched == m && s.qualified_name >= prev.qualified_name => {}
            _ => best = Some((matched, s)),
        }
    }
    let (matched, sym) = best?;
    let half = steps.len().div_ceil(2);
    (matched >= MIN_GROUND && matched >= half).then_some(sym)
}

/// Count of diagram steps that appear somewhere in the code calls (set overlap),
/// the grounding heuristic — order is judged later by the alignment itself.
fn overlap(steps: &[String], calls: &[String]) -> usize {
    steps
        .iter()
        .filter(|t| calls.iter().any(|c| c == *t))
        .count()
}

/// Traceback ops, oriented diagram-vs-code.
enum Op {
    /// Both consumed: a substitution (match when equal).
    Sub,
    /// Diagram step consumed, code gap — a step the code never makes.
    DiagramOnly,
    /// Code call consumed, diagram gap — a call the diagram omits.
    CodeOnly,
}

fn emit(seq: &Sequence, steps: &[String], driver: &Symbol) -> Vec<Finding> {
    let calls = &driver.calls;
    let trace = align(steps, calls);
    let prov = Provenance::symbol(driver.qualified_name.clone());
    let mut out = Vec::new();
    let (mut i, mut j) = (0usize, 0usize); // indices into steps / calls

    for op in trace {
        match op {
            Op::Sub => {
                let d = &steps[i];
                let c = &calls[j];
                if d == c {
                    out.push(Finding::supported(
                        format!("sequence step `{d}` matches a call in `{}`", driver.name),
                        seq.origin.clone(),
                        prov.clone(),
                    ));
                } else {
                    out.push(
                        Finding::problem(
                            Verdict::Contradicted,
                            format!("sequence step `{d}`"),
                            seq.origin.clone(),
                            format!(
                                "Out-of-order step: the diagram's step `{d}` aligns to code call `{c}` in `{}` — wrong call or wrong order.",
                                driver.name
                            ),
                        )
                        .anchored(prov.clone()),
                    );
                }
                i += 1;
                j += 1;
            }
            Op::DiagramOnly => {
                let d = &steps[i];
                out.push(
                    Finding::problem(
                        Verdict::Contradicted,
                        format!("sequence step `{d}`"),
                        seq.origin.clone(),
                        format!(
                            "Missing step: the diagram shows step `{d}`, but `{}` makes no matching call.",
                            driver.name
                        ),
                    )
                    .anchored(prov.clone()),
                );
                i += 1;
            }
            Op::CodeOnly => {
                let c = &calls[j];
                out.push(
                    Finding::problem(
                        Verdict::Undocumented,
                        format!("call `{c}` in `{}`", driver.name),
                        seq.origin.clone(),
                        format!(
                            "Undrawn step: `{}` calls `{c}`, but the sequence diagram omits it.",
                            driver.name
                        ),
                    )
                    .anchored(prov.clone()),
                );
                j += 1;
            }
        }
    }
    out
}

/// Needleman–Wunsch global alignment of `steps` (rows) vs `calls` (cols); returns
/// the traceback as an ordered op list from start to end.
fn align(steps: &[String], calls: &[String]) -> Vec<Op> {
    let (m, n) = (steps.len(), calls.len());
    // Score matrix (m+1) x (n+1).
    let mut score = vec![vec![0i32; n + 1]; m + 1];
    for (i, row) in score.iter_mut().enumerate() {
        row[0] = i as i32 * GAP;
    }
    for (j, cell) in score[0].iter_mut().enumerate() {
        *cell = j as i32 * GAP;
    }
    for i in 1..=m {
        for j in 1..=n {
            let sub = score[i - 1][j - 1]
                + if steps[i - 1] == calls[j - 1] {
                    MATCH
                } else {
                    MISMATCH
                };
            let del = score[i - 1][j] + GAP; // diagram-only
            let ins = score[i][j - 1] + GAP; // code-only
            score[i][j] = sub.max(del).max(ins);
        }
    }

    // Traceback, preferring diagonal on ties for stable substitution alignment.
    let mut ops = Vec::new();
    let (mut i, mut j) = (m, n);
    while i > 0 || j > 0 {
        if i > 0 && j > 0 {
            let diag = score[i - 1][j - 1]
                + if steps[i - 1] == calls[j - 1] {
                    MATCH
                } else {
                    MISMATCH
                };
            if score[i][j] == diag {
                ops.push(Op::Sub);
                i -= 1;
                j -= 1;
                continue;
            }
        }
        if i > 0 && score[i][j] == score[i - 1][j] + GAP {
            ops.push(Op::DiagramOnly);
            i -= 1;
        } else {
            ops.push(Op::CodeOnly);
            j -= 1;
        }
    }
    ops.reverse();
    ops
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code::symbol::{Facts, Span, SymbolKind, Visibility};

    fn driver(name: &str, calls: &[&str]) -> Symbol {
        Symbol {
            qualified_name: format!("m::{name}"),
            name: name.to_string(),
            kind: SymbolKind::Function,
            visibility: Visibility::Public,
            module: "m".to_string(),
            span: Span {
                path: "m.rs".into(),
                start_line: 1,
                end_line: 1,
            },
            body_span: Span::zero(),
            signature: None,
            doc: None,
            facts: Facts::default(),
            calls: calls.iter().map(|s| s.to_string()).collect(),
            members: Vec::new(),
        }
    }

    fn seq(steps: &[&str]) -> Sequence {
        use crate::diagram::sequence::Message;
        Sequence {
            participants: vec![],
            messages: steps
                .iter()
                .map(|s| Message {
                    from: "A".into(),
                    to: "B".into(),
                    label: format!("{s}()"),
                })
                .collect(),
            origin: "d.md:1".into(),
        }
    }

    fn index(syms: Vec<Symbol>) -> CodeIndex {
        CodeIndex {
            symbols: syms,
            ..Default::default()
        }
    }

    #[test]
    fn coherent_sequence_only_supported() {
        let idx = index(vec![driver("handler", &["validate", "save", "notify"])]);
        let s = seq(&["validate", "save", "notify"]);
        let out = check(&s, &idx);
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|f| f.verdict == Verdict::Supported));
    }

    #[test]
    fn undrawn_call_is_flagged() {
        // Code inserts `rateLimit` between validate and save; diagram omits it.
        let idx = index(vec![driver("handler", &["validate", "rateLimit", "save"])]);
        let s = seq(&["validate", "save"]);
        let out = check(&s, &idx);
        assert!(out
            .iter()
            .any(|f| f.verdict == Verdict::Undocumented && f.detail.contains("rateLimit")));
        // validate + save still match.
        assert_eq!(
            out.iter()
                .filter(|f| f.verdict == Verdict::Supported)
                .count(),
            2
        );
    }

    #[test]
    fn missing_step_is_flagged() {
        // Diagram shows `audit` the code never calls.
        let idx = index(vec![driver("handler", &["validate", "save"])]);
        let s = seq(&["validate", "audit", "save"]);
        let out = check(&s, &idx);
        assert!(out
            .iter()
            .any(|f| f.verdict == Verdict::Contradicted && f.detail.contains("audit")));
    }

    #[test]
    fn ungrounded_diagram_emits_nothing() {
        // No symbol's calls overlap the diagram steps.
        let idx = index(vec![driver("other", &["foo", "bar"])]);
        let s = seq(&["alpha", "beta", "gamma"]);
        assert!(check(&s, &idx).is_empty());
    }
}
