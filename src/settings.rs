//! Optional per-repo configuration read from `.staleguard.toml` at the repo
//! root. Everything here is opt-in: with no config file the defaults reproduce
//! the previous behaviour exactly (check every doc, report every verdict).
//!
//! ```toml
//! # .staleguard.toml — all keys optional
//!
//! # Doc paths to skip entirely, as glob patterns matched against the path
//! # relative to the repo root (`*` matches within a path segment, `**` across
//! # segments). A bare name with no slash also matches by suffix, so
//! # `LEGACY.md` skips it in any directory.
//! exclude = ["docs/legacy/**", "vendor/**", "NOTES.md"]
//!
//! # Verdict categories to drop from the report (and from the failing set), for
//! # teams that opt out of a whole class of finding. One or more of:
//! # "contradicted", "stale", "unverifiable", "undocumented".
//! suppress = ["undocumented"]
//!
//! # Drop findings below this severity from the report, SARIF, and failing set.
//! # "note" (default, keep all) < "warning" < "error". The `--min-severity`
//! # flag overrides this. `warning` hides the high-volume `undocumented` notes.
//! min_severity = "warning"
//! ```

use std::path::Path;

use regex::Regex;
use serde::Deserialize;

use crate::findings::{Finding, Severity, Verdict};

/// Parsed `.staleguard.toml`. Absent file ⇒ [`Settings::default`] (no-op).
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Settings {
    /// Glob patterns of doc paths (relative to root) to skip.
    pub exclude: Vec<String>,
    /// Verdict names to suppress from findings.
    pub suppress: Vec<String>,
    /// Drop reportable findings below this severity (`note` < `warning` < `error`).
    /// Overridden by the `--min-severity` flag when that is given.
    pub min_severity: Option<Severity>,
}

impl Settings {
    /// The config file name looked up at the repo root.
    pub const FILE: &'static str = ".staleguard.toml";

    /// Load `.staleguard.toml` from `root`. A missing file yields the default
    /// (no-op) settings; a present-but-malformed file is a hard error so a typo
    /// never silently disables a check.
    pub fn load(root: &Path) -> Result<Settings, String> {
        let path = root.join(Self::FILE);
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Settings::default()),
            Err(e) => return Err(format!("reading {}: {e}", path.display())),
        };
        toml::from_str(&text).map_err(|e| format!("parsing {}: {e}", path.display()))
    }

    /// Whether the doc at relative path `rel` is excluded by any pattern.
    pub fn is_doc_excluded(&self, rel: &str) -> bool {
        self.exclude.iter().any(|pat| glob_match(pat, rel))
    }

    /// Verdicts to drop, parsed from the `suppress` list (unknown names ignored).
    fn suppressed(&self) -> Vec<Verdict> {
        self.suppress
            .iter()
            .filter_map(|s| match s.to_ascii_lowercase().as_str() {
                "contradicted" => Some(Verdict::Contradicted),
                "stale" => Some(Verdict::Stale),
                "unverifiable" => Some(Verdict::Unverifiable),
                "undocumented" => Some(Verdict::Undocumented),
                _ => None,
            })
            .collect()
    }

    /// Drop findings whose verdict is suppressed. `Supported` claims are never
    /// suppressible (they feed the alignment score, not the report).
    pub fn apply_suppression(&self, findings: &mut Vec<Finding>) {
        let drop = self.suppressed();
        if drop.is_empty() {
            return;
        }
        findings.retain(|f| f.verdict == Verdict::Supported || !drop.contains(&f.verdict));
    }

    /// Drop reportable findings whose severity is below `threshold`. `Supported`
    /// claims are kept regardless (they feed the score, not the report). A `None`
    /// threshold is a no-op.
    pub fn apply_severity_threshold(findings: &mut Vec<Finding>, threshold: Option<Severity>) {
        let Some(min) = threshold else { return };
        findings.retain(|f| f.verdict == Verdict::Supported || f.verdict.severity() >= min);
    }
}

/// Match a relative path against a glob pattern. `*` matches any run of
/// non-`/` characters, `**` matches across `/`, and everything else is literal.
/// A pattern with no `/` also matches by trailing path segment, so `NOTES.md`
/// matches `docs/NOTES.md`.
fn glob_match(pattern: &str, rel: &str) -> bool {
    if !pattern.contains('/') && !pattern.contains('*') && rel.ends_with(pattern) {
        // Bare name: match the whole final segment, not a substring.
        return rel == pattern || rel.ends_with(&format!("/{pattern}"));
    }
    let re = glob_to_regex(pattern);
    re.is_match(rel)
}

fn glob_to_regex(pattern: &str) -> Regex {
    let mut re = String::from("^");
    let bytes = pattern.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'*' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'*' {
                    re.push_str(".*");
                    i += 2;
                    // Swallow a trailing slash after `**/` so `a/**/b` matches `a/b`.
                    if i < bytes.len() && bytes[i] == b'/' {
                        i += 1;
                    }
                    continue;
                }
                re.push_str("[^/]*");
            }
            c => re.push_str(&regex::escape(std::str::from_utf8(&[c]).unwrap())),
        }
        i += 1;
    }
    re.push('$');
    Regex::new(&re).unwrap_or_else(|_| Regex::new("$^").unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::claim::Provenance;

    #[test]
    fn default_excludes_nothing() {
        let s = Settings::default();
        assert!(!s.is_doc_excluded("README.md"));
    }

    #[test]
    fn bare_name_matches_in_any_dir() {
        let s = Settings {
            exclude: vec!["NOTES.md".into()],
            suppress: vec![],
            min_severity: None,
        };
        assert!(s.is_doc_excluded("NOTES.md"));
        assert!(s.is_doc_excluded("docs/NOTES.md"));
        assert!(!s.is_doc_excluded("docs/RELEASE_NOTES.md"));
    }

    #[test]
    fn single_star_stays_within_segment() {
        let s = Settings {
            exclude: vec!["docs/*.md".into()],
            suppress: vec![],
            min_severity: None,
        };
        assert!(s.is_doc_excluded("docs/usage.md"));
        assert!(!s.is_doc_excluded("docs/sub/usage.md"));
    }

    #[test]
    fn double_star_crosses_segments() {
        let s = Settings {
            exclude: vec!["docs/legacy/**".into()],
            suppress: vec![],
            min_severity: None,
        };
        assert!(s.is_doc_excluded("docs/legacy/v1/api.md"));
        assert!(!s.is_doc_excluded("docs/current/api.md"));
    }

    #[test]
    fn suppression_drops_named_verdicts_but_keeps_supported() {
        let s = Settings {
            exclude: vec![],
            suppress: vec!["undocumented".into()],
            min_severity: None,
        };
        let mut findings = vec![
            Finding::problem(Verdict::Undocumented, "c", "a.rs", "d"),
            Finding::problem(Verdict::Stale, "c", "README.md:1", "d"),
            Finding::supported("c", "README.md:2", Provenance::default()),
        ];
        s.apply_suppression(&mut findings);
        assert_eq!(findings.len(), 2);
        assert!(findings.iter().all(|f| f.verdict != Verdict::Undocumented));
        assert!(findings.iter().any(|f| f.verdict == Verdict::Supported));
    }

    #[test]
    fn severity_threshold_drops_below_but_keeps_supported() {
        let mut findings = vec![
            Finding::problem(Verdict::Undocumented, "c", "a.rs", "d"), // note
            Finding::problem(Verdict::Unverifiable, "c", "b.md:1", "d"), // warning
            Finding::problem(Verdict::Stale, "c", "c.md:1", "d"),      // error
            Finding::supported("c", "d.md:2", Provenance::default()),
        ];
        Settings::apply_severity_threshold(&mut findings, Some(Severity::Warning));
        // Note-level undocumented dropped; warning + error kept; supported kept.
        assert!(findings.iter().all(|f| f.verdict != Verdict::Undocumented));
        assert!(findings.iter().any(|f| f.verdict == Verdict::Unverifiable));
        assert!(findings.iter().any(|f| f.verdict == Verdict::Stale));
        assert!(findings.iter().any(|f| f.verdict == Verdict::Supported));
    }

    #[test]
    fn no_threshold_is_a_noop() {
        let mut findings = vec![Finding::problem(Verdict::Undocumented, "c", "a.rs", "d")];
        Settings::apply_severity_threshold(&mut findings, None);
        assert_eq!(findings.len(), 1);
    }

    #[test]
    fn malformed_config_is_an_error() {
        let dir = std::env::temp_dir().join(format!("sg-cfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(Settings::FILE), "exclude = \"not a list\"").unwrap();
        assert!(Settings::load(&dir).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
