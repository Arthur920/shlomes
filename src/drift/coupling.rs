//! Evolutionary-coupling staleness prior (no model, no embeddings).
//!
//! Mines git change-coupling: a doc and the code it describes that *used to
//! co-change but no longer do* is a strong staleness signal. If `auth.rs` has
//! churned repeatedly since `docs/auth.md` was last touched — and the two have a
//! history of changing together — the doc is likely stale. Pure git history.
//!
//! Conservative by design (zero-FP stance): a pair must have co-changed at least
//! [`COCHANGE_MIN`] times *and* the code must have changed at least
//! [`DRIFT_MIN`] times since the doc's last edit before we say anything.

use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use crate::claim::Provenance;
use crate::findings::{Finding, Verdict};
use crate::git;

/// Commits to mine from history (newest first).
const MAX_COMMITS: usize = 1000;
/// Minimum historical co-changes before a pair is considered coupled.
const COCHANGE_MIN: usize = 2;
/// Minimum code churn since the doc's last edit to call the doc stale.
const DRIFT_MIN: usize = 3;

/// One coupled doc/code pair where the code has drifted ahead of the doc.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaleDoc {
    pub doc: String,
    pub code: String,
    pub cochange: usize,
    /// Code changes since the doc's last edit.
    pub churn_after: usize,
}

/// Staleness-prior findings for the repo, mined from git history. Empty when git
/// is unavailable or history is too thin to be meaningful.
pub fn check(root: &Path) -> Vec<Finding> {
    let history = git::file_change_history(root, MAX_COMMITS);
    analyze(&history)
        .into_iter()
        .map(|s| {
            Finding::problem(
                Verdict::Stale,
                format!("`{}` documents `{}`", s.doc, s.code),
                s.doc.clone(),
                format!(
                    "Staleness prior: `{}` changed {} times since `{}` was last updated (they co-changed {} times before). The doc likely lags the code.",
                    s.code, s.churn_after, s.doc, s.cochange
                ),
            )
            .anchored(Provenance::path(s.code.clone()))
            .with_refs(vec![s.code])
        })
        .collect()
}

/// Pure analysis over a newest-first commit history (each entry is one commit's
/// changed-file list). Separated from git for testability.
fn analyze(history: &[Vec<String>]) -> Vec<StaleDoc> {
    // file -> commit indices it changed in (ascending; 0 = newest commit).
    let mut appears: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, files) in history.iter().enumerate() {
        for f in files {
            appears.entry(f.as_str()).or_default().push(i);
        }
    }

    let docs: Vec<(&str, &Vec<usize>)> = appears
        .iter()
        .filter(|(f, _)| is_doc(f))
        .map(|(f, v)| (*f, v))
        .collect();
    let code: Vec<(&str, &Vec<usize>)> = appears
        .iter()
        .filter(|(f, _)| !is_doc(f))
        .map(|(f, v)| (*f, v))
        .collect();

    let mut out = Vec::new();
    for (doc, doc_idxs) in &docs {
        // The doc's most recent change = the smallest index it appears at.
        let Some(&doc_last) = doc_idxs.iter().min() else {
            continue;
        };
        for (code_path, code_idxs) in &code {
            let cochange = intersect_count(doc_idxs, code_idxs);
            if cochange < COCHANGE_MIN {
                continue;
            }
            // Code commits strictly newer than the doc's last edit.
            let churn_after = code_idxs.iter().filter(|&&i| i < doc_last).count();
            if churn_after >= DRIFT_MIN {
                out.push(StaleDoc {
                    doc: doc.to_string(),
                    code: code_path.to_string(),
                    cochange,
                    churn_after,
                });
            }
        }
    }
    // Stable, strongest-first ordering.
    out.sort_by(|a, b| {
        b.churn_after
            .cmp(&a.churn_after)
            .then_with(|| a.doc.cmp(&b.doc))
            .then_with(|| a.code.cmp(&b.code))
    });
    out
}

/// Code files that have **never** co-changed with any doc across `history` — the
/// "no doc has ever tracked this code" signal for net-new coverage gaps
/// (`coverage-gaps.md` §4). Conservative: any commit touching both a doc and a
/// code file marks *all* its code files tracked, so we under-flag rather than
/// over-flag. Empty when git history is unavailable.
pub fn code_without_codoc(history: &[Vec<String>]) -> HashSet<String> {
    let mut all_code: HashSet<&str> = HashSet::new();
    let mut tracked: HashSet<&str> = HashSet::new();
    for files in history {
        let has_doc = files.iter().any(|f| is_doc(f));
        for f in files {
            if is_doc(f) {
                continue;
            }
            all_code.insert(f.as_str());
            if has_doc {
                tracked.insert(f.as_str());
            }
        }
    }
    all_code
        .into_iter()
        .filter(|f| !tracked.contains(f))
        .map(str::to_string)
        .collect()
}

fn is_doc(path: &str) -> bool {
    path.ends_with(".md") || path.ends_with(".markdown")
}

/// Count shared elements of two ascending index lists.
fn intersect_count(a: &[usize], b: &[usize]) -> usize {
    let (mut i, mut j, mut n) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                n += 1;
                i += 1;
                j += 1;
            }
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(files: &[&str]) -> Vec<String> {
        files.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn flags_code_that_drifted_ahead_of_its_doc() {
        // Newest first: auth.rs changed in 3 recent commits; the doc + auth.rs
        // co-changed twice further back.
        let history = vec![
            c(&["src/auth.rs"]),                 // 0 newest
            c(&["src/auth.rs"]),                 // 1
            c(&["src/auth.rs"]),                 // 2
            c(&["docs/auth.md", "src/auth.rs"]), // 3  (doc's last edit)
            c(&["docs/auth.md", "src/auth.rs"]), // 4
        ];
        let got = analyze(&history);
        assert_eq!(got.len(), 1, "{got:?}");
        assert_eq!(got[0].doc, "docs/auth.md");
        assert_eq!(got[0].code, "src/auth.rs");
        assert_eq!(got[0].cochange, 2);
        assert_eq!(got[0].churn_after, 3);
    }

    #[test]
    fn no_flag_when_doc_keeps_pace() {
        // Doc changed most recently → no code churn after it.
        let history = vec![
            c(&["docs/auth.md", "src/auth.rs"]),
            c(&["docs/auth.md", "src/auth.rs"]),
            c(&["docs/auth.md", "src/auth.rs"]),
        ];
        assert!(analyze(&history).is_empty());
    }

    #[test]
    fn no_flag_below_cochange_threshold() {
        // They only ever co-changed once → not considered coupled.
        let history = vec![
            c(&["src/auth.rs"]),
            c(&["src/auth.rs"]),
            c(&["src/auth.rs"]),
            c(&["docs/auth.md", "src/auth.rs"]),
        ];
        assert!(analyze(&history).is_empty());
    }
}
