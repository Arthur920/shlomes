//! Verification layers.
//!
//! Layer 1 (deterministic) lives here. Layer 2 (retrieval) is in [`retrieve`]
//! and Layer 3 (the NLI judge) in [`judge`]; both are gated behind the `ml`
//! feature.
//!
//! [`retrieve`]: crate::retrieve
//! [`judge`]: crate::judge

use std::collections::{HashMap, HashSet};
use std::path::Path;

use walkdir::WalkDir;

use crate::claim::Provenance;
use crate::extract::PathClaim;
use crate::findings::{Finding, Verdict};

/// Every path string under `root` (files and dirs), one walk, for the in-memory
/// existence check below. Skips the vendored/build dirs every walker ignores.
/// Built once per run and shared, so path claims don't each re-walk the repo.
pub fn repo_paths(root: &Path) -> Vec<String> {
    WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| !crate::code::lang::is_skip_dir(&e.file_name().to_string_lossy()))
        .filter_map(|e| e.ok())
        .map(|e| e.path().to_string_lossy().into_owned())
        .collect()
}

/// Layer 1: every path a doc names by backtick should exist in the repo. Emits
/// a `Supported` claim for paths that exist and a `Stale` one for those that do
/// not; both are anchored (provenance) to the named path so drift lineage can
/// invalidate them when that file changes.
///
/// A claim exists if `repo_root/c.raw` resolves on disk (a single stat, which
/// handles directories, `./`-prefixed and absolute paths exactly as before) or
/// any pre-walked repo path in `repo_files` ends with it (the suffix case, e.g.
/// a doc naming `index.ts` for `src/index.ts`). Both are memoized so repeated
/// claims cost nothing; the only thing removed versus the old code is the
/// full-tree re-walk that ran once per claim.
pub fn check_paths(claims: &[PathClaim], repo_root: &Path, repo_files: &[String]) -> Vec<Finding> {
    // Existence per distinct token, memoized (one stat / suffix scan each).
    let mut exists: HashMap<&str, bool> = HashMap::new();
    for c in claims {
        exists.entry(c.raw.as_str()).or_insert_with(|| {
            repo_root.join(&c.raw).exists() || repo_files.iter().any(|p| p.ends_with(&c.raw))
        });
    }

    // Migration rows: when one doc line names several paths and at least one
    // resolves, the unresolved siblings are the "before" side of an old → new
    // mapping (e.g. a `| old | new |` table row); their absence is intentional.
    let mut by_line: HashMap<(&str, usize), Vec<&PathClaim>> = HashMap::new();
    for c in claims {
        by_line
            .entry((c.doc_path.as_str(), c.line))
            .or_default()
            .push(c);
    }
    let mut migrated: HashSet<(&str, usize, &str)> = HashSet::new();
    for group in by_line.values() {
        if group.len() >= 2 && group.iter().any(|c| exists[c.raw.as_str()]) {
            for c in group {
                if !exists[c.raw.as_str()] {
                    migrated.insert((c.doc_path.as_str(), c.line, c.raw.as_str()));
                }
            }
        }
    }

    let mut findings = Vec::new();
    for c in claims {
        let present = exists[c.raw.as_str()];
        let doc_ref = format!("{}:{}", c.doc_path, c.line);
        let prov = Provenance::path(c.raw.clone());
        if present {
            findings.push(Finding::supported(
                format!("references `{}`", c.raw),
                doc_ref,
                prov,
            ));
        } else if c.historical || migrated.contains(&(c.doc_path.as_str(), c.line, c.raw.as_str()))
        {
            // Named as deleted / renamed / replaced — its absence confirms the
            // doc rather than contradicting it, so emit nothing (zero-FP).
            continue;
        } else {
            findings.push(
                Finding::problem(
                    Verdict::Stale,
                    format!("references `{}`", c.raw),
                    doc_ref,
                    format!(
                        "Path `{}` is named in docs but does not exist in the repo.",
                        c.raw
                    ),
                )
                .anchored(prov),
            );
        }
    }
    findings
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::extract_path_claims;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn scratch_dir(tag: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("staleguard-{tag}-{nanos}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn missing_path_is_flagged() {
        let dir = scratch_dir("missing");
        fs::write(dir.join("real.py"), "x = 1\n").unwrap();
        let md = "Entry point is `real.py`, config in `does/not/exist.toml`.";
        let claims = extract_path_claims(md, "README.md");
        let findings = check_paths(&claims, &dir, &repo_paths(&dir));

        let flagged: Vec<&str> = findings
            .iter()
            .filter(|f| f.verdict.is_reportable())
            .map(|f| f.claim.as_str())
            .collect();
        assert!(flagged.contains(&"references `does/not/exist.toml`"));
        assert!(!flagged.contains(&"references `real.py`"));
    }

    #[test]
    fn clean_repo_has_no_findings() {
        let dir = scratch_dir("clean");
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(dir.join("src/main.py"), "print('hi')\n").unwrap();
        let claims = extract_path_claims("See `src/main.py`.", "README.md");
        let findings = check_paths(&claims, &dir, &repo_paths(&dir));
        assert!(findings.iter().all(|f| !f.verdict.is_reportable()));
    }

    #[test]
    fn deleted_path_in_deletion_context_is_not_flagged() {
        // A plan documenting a removal names a file that (correctly) does not
        // exist — its absence confirms the doc, so no stale finding.
        let dir = scratch_dir("deleted");
        let md = "**Delete**\n\n- `src/old/handler.ts`\n\n`src/old/handler.ts` no longer exists.";
        let claims = extract_path_claims(md, "PLAN.md");
        let findings = check_paths(&claims, &dir, &repo_paths(&dir));
        assert!(
            findings.iter().all(|f| !f.verdict.is_reportable()),
            "deletion-context path was flagged: {findings:?}"
        );
    }

    #[test]
    fn migration_row_old_path_is_not_flagged() {
        // `| old | new |`: new exists, old doesn't — the old side is the
        // migration source and must not be flagged stale.
        let dir = scratch_dir("migration");
        fs::create_dir_all(dir.join("src/common/query")).unwrap();
        fs::write(dir.join("src/common/query/query-client.ts"), "//\n").unwrap();
        let md = "| `src/lib/query-client.ts` | `src/common/query/query-client.ts` |";
        let claims = extract_path_claims(md, "PLAN.md");
        let findings = check_paths(&claims, &dir, &repo_paths(&dir));
        assert!(
            findings.iter().all(|f| !f.verdict.is_reportable()),
            "migration old-path was flagged: {findings:?}"
        );
    }
}
