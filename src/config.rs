//! Layer 1: env vars and CLI flags named in docs that the code never reads.
//!
//! Grounding is loose presence in source — the same fewest-false-positives
//! rule coverage-gaps uses. An env var is "real" if its name appears as a token
//! anywhere in the source tree (`env::var("NAME")`, `process.env.NAME`, etc.
//! all carry the literal); a `--flag` is "real" if its snake/concat form
//! appears as a source identifier. A documented name grounded nowhere is
//! `stale`.
//!
//! Flags are scoped to *this project's* CLI: only `--flags` attached to a
//! command whose first token is one of the project's own binaries are checked,
//! so a documented third-party invocation (`npm install --save-dev`) is never
//! mistaken for our drift. System env vars (`$HOME`, `$PATH`, …) are excluded.

use std::collections::HashSet;
use std::path::Path;
use std::sync::OnceLock;

use rayon::prelude::*;
use regex::Regex;

use crate::claim::Provenance;
use crate::code::lang;
use crate::commands::command_lines;
use crate::findings::{Finding, Verdict};

/// Every identifier-like token across all source files — the grounding set for
/// env vars and flags. TOML/YAML are in `CODE_EXTS`, so `Cargo.toml` keys (e.g.
/// `features`, `release`) ground the corresponding cargo flags.
pub fn code_tokens(repo_root: &Path) -> HashSet<String> {
    lang::code_files(repo_root)
        .par_iter()
        .map(|file| {
            let mut local = HashSet::new();
            if let Ok(text) = std::fs::read_to_string(file) {
                for m in ident_re().find_iter(&text) {
                    local.insert(m.as_str().to_string());
                }
            }
            local
        })
        .reduce(HashSet::new, |mut a, b| {
            a.extend(b);
            a
        })
}

/// Check env-var and flag claims in `markdown` against the source grounding set.
pub fn check(
    markdown: &str,
    doc_path: &str,
    code_tokens: &HashSet<String>,
    project_bins: &HashSet<String>,
) -> Vec<Finding> {
    let mut findings = Vec::new();
    check_env_vars(markdown, doc_path, code_tokens, &mut findings);
    check_flags(markdown, doc_path, code_tokens, project_bins, &mut findings);
    findings
}

fn check_env_vars(
    markdown: &str,
    doc_path: &str,
    code_tokens: &HashSet<String>,
    findings: &mut Vec<Finding>,
) {
    // Names assigned in the doc itself (`MDBOOK_VERS="…"`) are local shell
    // variables of an example snippet, not env vars the project reads.
    let assigned = assigned_names(markdown);
    let mut seen = HashSet::new();
    for (line, name) in env_var_claims(markdown) {
        if SYSTEM_ENV.contains(&name.as_str())
            || is_placeholder(&name)
            || assigned.contains(&name)
            || !seen.insert((line, name.clone()))
        {
            continue;
        }
        let doc_ref = format!("{doc_path}:{line}");
        let claim = format!("references env var `{name}`");
        if code_tokens.contains(&name) {
            // Token-grounded (no single owning symbol) → empty provenance, so it
            // is always re-checked rather than carried forward. Safe by default.
            findings.push(Finding::supported(claim, doc_ref, Provenance::default()));
        } else {
            findings.push(Finding::problem(
                Verdict::Stale,
                claim,
                doc_ref,
                format!("Env var `{name}` is named in docs but read nowhere in the code."),
            ));
        }
    }
}

fn check_flags(
    markdown: &str,
    doc_path: &str,
    code_tokens: &HashSet<String>,
    project_bins: &HashSet<String>,
    findings: &mut Vec<Finding>,
) {
    if project_bins.is_empty() {
        return; // No known CLI ⇒ can't attribute any flag to us.
    }
    let mut seen = HashSet::new();
    for (line, cmd) in command_lines(markdown) {
        let mut toks = cmd.split_whitespace();
        let Some(bin) = toks.next() else { continue };
        if !project_bins.contains(bin) {
            continue;
        }
        for tok in toks {
            let Some(flag) = long_flag_re().captures(tok).map(|c| c[1].to_string()) else {
                continue;
            };
            if AUTO_FLAGS.contains(&flag.as_str()) || !seen.insert((line, flag.clone())) {
                continue;
            }
            let doc_ref = format!("{doc_path}:{line}");
            let claim = format!("documents flag `--{flag}`");
            if flag_grounded(&flag, code_tokens) {
                findings.push(Finding::supported(claim, doc_ref, Provenance::default()));
            } else {
                findings.push(Finding::problem(
                    Verdict::Stale,
                    claim,
                    doc_ref,
                    format!("Flag `--{flag}` for `{bin}` is documented but absent from the code."),
                ));
            }
        }
    }
}

/// A flag is grounded if any of its conventional identifier spellings is a source
/// token. A `--kebab-flag` rarely appears verbatim in code (the literal is split
/// on `-` by the tokenizer); instead it surfaces as a field or type whose name
/// drops the dashes in one of a few casings (all for `--type-not`):
///
/// - snake_case `type_not` — clap-derive fields, Python/Rust fields
/// - concat `typenot` — joined-lowercase
/// - PascalCase `TypeNot` — per-flag structs (ripgrep) / enum variants
/// - camelCase `typeNot` — JS/TS option fields
///
/// All four are exact `HashSet` lookups, so grounding stays case-exact and adds
/// no corpus-wide lowercasing (which would risk masking real drift).
fn flag_grounded(flag: &str, code_tokens: &HashSet<String>) -> bool {
    // Negation flags (`--no-color`, `--no-encoding`) are commonly auto-generated
    // from the positive flag (clap's `no_` negations, ripgrep's `name_negated`),
    // so they carry no identifier of their own. Ground `--no-foo` whenever `foo`
    // grounds — far cheaper to miss a never-supported negation than to cry drift
    // on a real one.
    if let Some(base) = flag.strip_prefix("no-") {
        if flag_grounded(base, code_tokens) {
            return true;
        }
    }
    let segments: Vec<&str> = flag.split('-').filter(|s| !s.is_empty()).collect();
    let snake = segments.join("_");
    let concat = segments.concat();
    let pascal = capitalize_join(&segments, false);
    let camel = capitalize_join(&segments, true);
    [snake, concat, pascal, camel]
        .iter()
        .any(|form| code_tokens.contains(form))
}

/// Join `segments` with each capitalized (`TypeNot`); when `lower_first`, the
/// leading segment stays lowercase (`typeNot`).
fn capitalize_join(segments: &[&str], lower_first: bool) -> String {
    segments
        .iter()
        .enumerate()
        .map(|(i, seg)| {
            if i == 0 && lower_first {
                seg.to_string()
            } else {
                let mut chars = seg.chars();
                match chars.next() {
                    Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
                    None => String::new(),
                }
            }
        })
        .collect()
}

/// Env-var names from docs: `$NAME` / `${NAME}` whose name is all-uppercase,
/// plus inline `backtick` spans whose whole content is an `UPPER_SNAKE` env
/// identifier (requiring an underscore avoids matching prose acronyms like
/// `API`). Env vars are UPPER_SNAKE by convention; the casing filter on the
/// dollar form skips shell variables that follow other conventions — zsh
/// `$fpath` (lowercase), PowerShell `$OutputEncoding` (PascalCase) — which
/// appear in example snippets but are not the project's own env vars.
fn env_var_claims(markdown: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    let mut in_fence = false;
    for (i, line) in markdown.lines().enumerate() {
        let lineno = i + 1;
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
            continue;
        }
        for cap in dollar_env_re().captures_iter(line) {
            let name = &cap[1];
            if upper_env_re().is_match(name) {
                out.push((lineno, name.to_string()));
            }
        }
        if !in_fence {
            for cap in inline_code_re().captures_iter(line) {
                let inner = cap[1].trim();
                if upper_snake_env_re().is_match(inner) {
                    out.push((lineno, inner.to_string()));
                }
            }
        }
    }
    out
}

/// Names assigned somewhere in the doc as a shell variable (`NAME=...`, with an
/// optional leading `export`). Such names are local to an example snippet and
/// must not be checked as project env vars.
fn assigned_names(markdown: &str) -> HashSet<String> {
    assign_re()
        .captures_iter(markdown)
        .map(|c| c[1].to_string())
        .collect()
}

/// Documentation placeholders standing in for a real name the reader supplies
/// (`${YOUR_ENV}`, `$MY_VAR`). Never the project's own env var.
fn is_placeholder(name: &str) -> bool {
    const PLACEHOLDER_PREFIXES: &[&str] = &["YOUR_", "MY_"];
    const PLACEHOLDER_EXACT: &[&str] = &["FOO", "BAR", "BAZ", "PLACEHOLDER", "CHANGEME"];
    PLACEHOLDER_PREFIXES.iter().any(|p| name.starts_with(p))
        || PLACEHOLDER_EXACT.contains(&name)
        || name.contains("YOUR")
        || name.contains("PLACEHOLDER")
}

/// clap auto-generates these; they never appear as source identifiers.
const AUTO_FLAGS: &[&str] = &["help", "version"];

/// System/shell env vars docs may mention that the project doesn't itself read.
const SYSTEM_ENV: &[&str] = &[
    "HOME",
    "PATH",
    "PWD",
    "OLDPWD",
    "UID",
    "GID",
    "EUID",
    "USER",
    "LOGNAME",
    "SHELL",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "TMPDIR",
    "TMP",
    "TEMP",
    "EDITOR",
    "VISUAL",
    "PAGER",
    "HOSTNAME",
    "DISPLAY",
    "SHLVL",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "FTP_PROXY",
];

fn ident_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[A-Za-z_][A-Za-z0-9_]+").unwrap())
}

fn inline_code_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"`([^`]+)`").unwrap())
}

/// `$NAME` or `${NAME}` (name starts with a letter/underscore, length ≥ 2).
fn dollar_env_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\$\{?([A-Za-z_][A-Za-z0-9_]+)\}?").unwrap())
}

/// A shell assignment `NAME=` (optionally `export NAME=`), name all-uppercase.
fn assign_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)(?:^|[;&|]|\bexport\s+)\s*([A-Z][A-Z0-9_]*)=").unwrap())
}

/// An all-uppercase env-var name (`PORT`, `DATABASE_URL`). Single-word is fine
/// here because the `$`/`${}` sigil already marks it as a variable, not prose.
fn upper_env_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^[A-Z][A-Z0-9_]*$").unwrap())
}

/// `UPPER_SNAKE` with at least one underscore.
fn upper_snake_env_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$").unwrap())
}

/// A long flag token `--kebab-name` (≥ 2 chars after the dashes).
fn long_flag_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^--([a-z][a-z0-9-]+)$").unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(words: &[&str]) -> HashSet<String> {
        words.iter().map(|w| w.to_string()).collect()
    }

    fn bins(words: &[&str]) -> HashSet<String> {
        words.iter().map(|w| w.to_string()).collect()
    }

    #[test]
    fn ungrounded_env_var_is_flagged() {
        let code = tokens(&["DATABASE_URL", "main"]);
        let md = "Set `DATABASE_URL` and `REDIS_URL` before running.";
        let flagged: Vec<String> = check(md, "README.md", &code, &HashSet::new())
            .iter()
            .map(|f| f.detail.clone())
            .collect();
        assert!(flagged.iter().any(|d| d.contains("REDIS_URL")));
        assert!(!flagged.iter().any(|d| d.contains("DATABASE_URL")));
    }

    #[test]
    fn dollar_form_and_system_vars() {
        let code = tokens(&["PORT"]);
        // $HOME is a system var (skipped); $LISTEN_ADDR is ours and ungrounded.
        let md = "Reads `$HOME`, `$PORT`, and `${LISTEN_ADDR}`.";
        let flagged: Vec<String> = check(md, "README.md", &code, &HashSet::new())
            .iter()
            .map(|f| f.detail.clone())
            .collect();
        assert!(flagged.iter().any(|d| d.contains("LISTEN_ADDR")));
        assert!(!flagged.iter().any(|d| d.contains("HOME")));
        assert!(!flagged.iter().any(|d| d.contains("PORT")));
    }

    #[test]
    fn locally_assigned_shell_var_is_not_flagged() {
        let code = HashSet::new();
        // `MDBOOK_VERS` is assigned in the snippet then used — a local shell var.
        let md = "```bash\nMDBOOK_VERS=\"1.0\"\ngh release create v$MDBOOK_VERS\n```";
        let flagged: Vec<String> = check(md, "README.md", &code, &HashSet::new())
            .iter()
            .filter(|f| f.verdict.is_reportable())
            .map(|f| f.detail.clone())
            .collect();
        assert!(flagged.is_empty(), "local shell var leaked: {flagged:?}");
    }

    #[test]
    fn placeholder_env_var_is_not_flagged() {
        let code = HashSet::new();
        let md = "Replace `${YOUR_ENV}` and `$MY_TOKEN` with real values.";
        let flagged: Vec<String> = check(md, "README.md", &code, &HashSet::new())
            .iter()
            .filter(|f| f.verdict.is_reportable())
            .map(|f| f.detail.clone())
            .collect();
        assert!(flagged.is_empty(), "placeholder leaked: {flagged:?}");
    }

    #[test]
    fn flag_grounded_via_snake_form() {
        let code = tokens(&["max_layer", "format"]);
        // clap derive: --max-layer ↔ field max_layer; --format ↔ field format.
        let md = "`app --max-layer 2 --format json`";
        let findings = check(md, "README.md", &code, &bins(&["app"]));
        assert!(findings.iter().all(|f| !f.verdict.is_reportable()));
    }

    #[test]
    fn flag_grounded_via_pascal_and_camel_forms() {
        // ripgrep-style: `--type-not` surfaces only as a PascalCase struct
        // `TypeNot` (the kebab name lives in a split-apart string literal).
        // `--no-color` surfaces as a camelCase JS field `noColor`.
        let code = tokens(&["TypeNot", "noColor"]);
        let md = "`rg --type-not rust --no-color`";
        let findings = check(md, "README.md", &code, &bins(&["rg"]));
        assert!(
            findings.iter().all(|f| !f.verdict.is_reportable()),
            "PascalCase/camelCase fields should ground the flags: {findings:?}"
        );
    }

    #[test]
    fn negation_flag_grounded_via_positive() {
        // `--no-encoding` is a generated negation of `--encoding`; only the
        // positive flag (`Encoding`) exists as an identifier.
        let code = tokens(&["Encoding"]);
        let md = "`rg --no-encoding`";
        let findings = check(md, "README.md", &code, &bins(&["rg"]));
        assert!(
            findings.iter().all(|f| !f.verdict.is_reportable()),
            "PascalCase/camelCase fields should ground the flags: {findings:?}"
        );
    }

    #[test]
    fn ungrounded_project_flag_is_flagged() {
        let code = tokens(&["format"]);
        let md = "`app check --diff main`";
        let flagged = check(md, "README.md", &code, &bins(&["app"]));
        assert_eq!(flagged.len(), 1);
        assert!(flagged[0].detail.contains("--diff"));
    }

    #[test]
    fn third_party_flags_are_not_checked() {
        let code = HashSet::new();
        // npm is not a project bin → its --save-dev flag is none of our business.
        let md = "`npm install --save-dev typescript`";
        assert!(check(md, "README.md", &code, &bins(&["app"])).is_empty());
    }

    #[test]
    fn no_project_bins_means_no_flag_findings() {
        let code = HashSet::new();
        let md = "`app --whatever`";
        assert!(check(md, "README.md", &code, &HashSet::new()).is_empty());
    }
}
