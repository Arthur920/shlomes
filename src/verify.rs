//! Verification layers.
//!
//! Layer 1 (deterministic) is implemented. Layers 2 (retrieval) and 3 (LLM
//! judge) are declared here as the integration points.

use std::path::Path;

use walkdir::WalkDir;

use crate::extract::PathClaim;
use crate::findings::{Finding, Verdict};

/// Layer 1: every path a doc names by backtick should exist in the repo.
pub fn check_paths(claims: &[PathClaim], repo_root: &Path) -> Vec<Finding> {
    let mut findings = Vec::new();
    for c in claims {
        let direct = repo_root.join(&c.raw);
        let exists = direct.exists() || tree_contains(repo_root, &c.raw);
        if !exists {
            findings.push(Finding {
                verdict: Verdict::Stale,
                claim: format!("references `{}`", c.raw),
                doc_path: format!("{}:{}", c.doc_path, c.line),
                detail: format!(
                    "Path `{}` is named in docs but does not exist in the repo.",
                    c.raw
                ),
                layer: 1,
                code_refs: Vec::new(),
            });
        }
    }
    findings
}

/// True if any file in the tree has a path ending with `suffix`.
fn tree_contains(root: &Path, suffix: &str) -> bool {
    WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| e.file_name() != ".git")
        .filter_map(|e| e.ok())
        .any(|e| e.path().to_string_lossy().ends_with(suffix))
}

/// Layer 3: LLM-as-judge over (claim, evidence) -> Verdict.
///
/// Not yet implemented. Will live behind a `ml` cargo feature (HTTP to the
/// Anthropic API).
#[allow(dead_code)]
pub fn judge_claim(_claim: &str, _evidence: &[String]) -> Verdict {
    unimplemented!("Layer 3 (LLM verification) is not implemented yet.")
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
        let dir = std::env::temp_dir().join(format!("shlomes-{tag}-{nanos}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn missing_path_is_flagged() {
        let dir = scratch_dir("missing");
        fs::write(dir.join("real.py"), "x = 1\n").unwrap();
        let md = "Entry point is `real.py`, config in `does/not/exist.toml`.";
        let claims = extract_path_claims(md, "README.md");
        let findings = check_paths(&claims, &dir);

        let flagged: Vec<&str> = findings.iter().map(|f| f.claim.as_str()).collect();
        assert!(flagged.contains(&"references `does/not/exist.toml`"));
        assert!(!flagged.contains(&"references `real.py`"));
    }

    #[test]
    fn clean_repo_has_no_findings() {
        let dir = scratch_dir("clean");
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(dir.join("src/main.py"), "print('hi')\n").unwrap();
        let claims = extract_path_claims("See `src/main.py`.", "README.md");
        assert!(check_paths(&claims, &dir).is_empty());
    }
}
