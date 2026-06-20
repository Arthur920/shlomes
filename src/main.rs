//! staleguard command-line entry point.

mod check;
mod claim;
mod code;
mod commands;
mod config;
mod constswap;
mod coverage;
mod diagram;
mod drift;
mod entrypoints;
#[cfg(feature = "ml")]
mod evidence;
mod extract;
mod findings;
mod git;
#[cfg(feature = "ml")]
mod judge;
mod report;
#[cfg(feature = "ml")]
mod rerank;
#[cfg(feature = "ml")]
mod retrieve;
mod rules;
mod sarif;
mod settings;
#[cfg(test)]
mod testutil;
mod verify;

use code::CodeIndex;
use report::Format;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use walkdir::WalkDir;

#[derive(Parser)]
#[command(
    name = "staleguard",
    version,
    about = "Check CLAUDE.md, project docs, and code against each other for coherence drift.",
    after_help = EXAMPLES
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// Worked examples appended to `staleguard --help`.
const EXAMPLES: &str = "\
Examples:
  staleguard check                  full repo, deterministic (layer 1)
  staleguard check --diff main      only re-check what changed vs main
  staleguard check --format json    machine-readable findings (exits non-zero on drift)
  staleguard check --format sarif   SARIF for GitHub code scanning / PR annotations
  staleguard check --write-ledger   set the CI alignment baseline on the base branch
  staleguard index                  print code symbols + module/reference edges
  staleguard rules                  audit architecture rules parsed from doc prose
  staleguard coverage               public code surface no doc describes

Layers 2-3 (retrieval + NLI judge) need the `ml` build; see `staleguard check --help`.
Run `staleguard <command> --help` for per-command options.";

#[derive(Subcommand)]
enum Commands {
    /// Check docs against code for coherence drift.
    Check {
        /// Repo root (default: cwd).
        #[arg(default_value = ".")]
        path: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value_t = Format::Text)]
        format: Format,
        /// Max layer: 1 deterministic (recommended), 2 +retrieval, 3 +NLI judge.
        /// Layers 2-3 require the `ml` feature build. Layer 3 runs a code-aware NLI
        /// cross-encoder (`staleguard`, a UniXcoder fine-tune) over
        /// the retrieved code evidence to render supported/contradicted verdicts.
        #[arg(long, default_value_t = 1)]
        layer: u8,
        /// Drift base: only re-derive claims whose code changed since this git
        /// ref (default: the committed ledger's last commit).
        #[arg(long)]
        diff: Option<String>,
        /// Persist the drift ledger + alignment score under `.staleguard/` (run this
        /// on the base branch to set the CI baseline).
        #[arg(long)]
        write_ledger: bool,
        /// Fail if the alignment score regressed below the committed baseline.
        #[arg(long)]
        fail_on_regression: bool,
        /// Drop findings below this severity (`note` < `warning` < `error`) from
        /// the report, the SARIF, and the failing set. Default `note` keeps
        /// everything; `warning` hides the high-volume `undocumented` notes;
        /// `error` keeps only provable drift (broken refs, contradictions).
        /// Overrides `min_severity` in `.staleguard.toml`.
        #[arg(long, value_enum)]
        min_severity: Option<findings::Severity>,
        /// Restrict doc-vs-code checks to these doc paths (repeatable; matched by
        /// exact relative path or path suffix). Skips the repo-wide coverage and
        /// history passes, so it is far cheaper — useful for checking a single
        /// changed doc (and for keeping the Layer-3 judge to that doc's claims).
        #[arg(long = "doc")]
        docs: Vec<String>,
    },
    /// Extract and print the code index (symbols + dependency edges).
    Index {
        /// Repo root (default: cwd).
        #[arg(default_value = ".")]
        path: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value_t = Format::Text)]
        format: Format,
    },
    /// Audit which architecture rules are extracted from doc prose, and how each
    /// fares against the code — so silent misses (a rule that didn't parse, or an
    /// operand that grounds to no real module) become visible.
    Rules {
        /// Repo root (default: cwd).
        #[arg(default_value = ".")]
        path: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value_t = Format::Text)]
        format: Format,
    },
    /// Report public code surface that no doc describes (code -> doc gaps).
    Coverage {
        /// Repo root (default: cwd).
        #[arg(default_value = ".")]
        path: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value_t = Format::Text)]
        format: Format,
    },
    /// Prepare all layers: fetch the Layer 2/3 models so later runs are offline.
    /// Layer 1 needs nothing; this download-and-load step is only meaningful in
    /// the `ml` build, where it pulls the embedding model and the NLI judge.
    Setup,
    /// Semantic code search using local jina embeddings (requires `ml` feature).
    #[cfg(feature = "ml")]
    Retrieve {
        /// Natural-language or code query.
        query: String,
        /// Repo root (default: cwd).
        #[arg(default_value = ".")]
        path: PathBuf,
        /// Number of chunks to return.
        #[arg(long, default_value_t = 5)]
        k: usize,
    },
}

pub(crate) fn collect_docs(root: &Path) -> Vec<PathBuf> {
    collect_docs_filtered(root, &[])
}

/// `collect_docs`, optionally restricted to the docs named in `filter` (matched
/// by exact relative path or path suffix, e.g. `README.md` or `docs/usage.md`).
/// An empty filter means "every doc" (the changelog exclusion still applies).
pub(crate) fn collect_docs_filtered(root: &Path, filter: &[String]) -> Vec<PathBuf> {
    WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| !crate::code::lang::is_skip_dir(&e.file_name().to_string_lossy()))
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter(crate::code::lang::within_size_limit)
        .map(|e| e.into_path())
        .filter(|p| {
            matches!(
                p.extension().and_then(|s| s.to_str()),
                Some("md") | Some("markdown")
            )
        })
        .filter(|p| !is_changelog_doc(p))
        .filter(|p| {
            if filter.is_empty() {
                return true;
            }
            let rel = p.strip_prefix(root).unwrap_or(p).to_string_lossy();
            filter
                .iter()
                .any(|f| rel == f.as_str() || rel.ends_with(f.as_str()))
        })
        .collect()
}

/// Changelogs and release-note fragments document *past* states, so they
/// legitimately name removed files, old symbols, and external versions — verbatim
/// history, not claims about the current code. Checking them only manufactures
/// false drift, so they are excluded from the doc set.
pub(crate) fn is_changelog_doc(path: &Path) -> bool {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    const NAMES: &[&str] = &[
        "history.md",
        "changelog.md",
        "changelog.markdown",
        "changes.md",
        "news.md",
        "releases.md",
        "release-notes.md",
        "release_notes.md",
        "whatsnew.md",
    ];
    if NAMES.contains(&name.as_str()) {
        return true;
    }
    // Towncrier-style fragment directories: `changes/`, `changelog.d/`, `news.d/`.
    path.components().any(|c| {
        matches!(
            c.as_os_str()
                .to_str()
                .map(str::to_ascii_lowercase)
                .as_deref(),
            Some("changes")
                | Some("changelog.d")
                | Some("changelog")
                | Some("news.d")
                | Some("newsfragments")
        )
    })
}

fn run_check(
    root: &Path,
    opts: &drift::Options,
    layer: u8,
    doc_filter: &[String],
    min_severity: Option<findings::Severity>,
) -> drift::Outcome {
    let _ = layer; // consulted only in `ml` builds for the Layer 3 judge.
                   // `--doc` scoping: restrict every doc-derived pass to the named docs and skip
                   // the repo-wide coverage/history passes (which answer "what code is
                   // undocumented", a whole-repo question that a single-doc check doesn't ask).
                   // This is what makes a scoped run cheap: no 1000-commit history parse, no
                   // coverage ranking, and the Layer-3 judge only sees the target doc's claims.
    let scoped = !doc_filter.is_empty();
    // Optional `.staleguard.toml`: doc-exclude globs + verdict suppression. A
    // malformed file aborts the run rather than silently dropping a check.
    let settings = settings::Settings::load(root).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(2);
    });
    // Repo-wide grounding, built once and shared across every doc.
    let index = CodeIndex::build(root);
    // Manifests are resolved per doc from its nearest ancestor manifest (cached
    // by directory), so a sub-project's docs check against that sub-project's
    // Makefile/Cargo.toml/package.json rather than only the repo-root one.
    let mut manifest_cache: HashMap<PathBuf, commands::Manifests> = HashMap::new();
    let code_tokens = config::code_tokens(root);
    let grounding = entrypoints::Grounding::from_index(&index);
    // One git-history fetch shared by every history-mining pass (coverage risk
    // ranking + the coupling staleness prior). Skipped entirely when scoped.
    let history = if scoped {
        Vec::new()
    } else {
        git::file_change_history(root, drift::coupling::MAX_COMMITS)
    };
    // The repo's path list, walked once, so each doc's path claims match in
    // memory instead of re-walking the whole tree per claim.
    let repo_files = verify::repo_paths(root);

    let ctx = check::CheckContext {
        root,
        index: &index,
        grounding: &grounding,
        code_tokens: &code_tokens,
        repo_files: &repo_files,
    };
    let doc_checks = check::doc_checks();

    // Architecture rules: prose-sourced, accumulated per doc, then verified once
    // (the symbol scan walks the whole repo).
    let mut arch_rules: Vec<rules::SourcedRule> = Vec::new();

    let mut findings = Vec::new();
    for doc_path in collect_docs_filtered(root, doc_filter) {
        let text = match std::fs::read_to_string(&doc_path) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let rel = doc_path
            .strip_prefix(root)
            .unwrap_or(&doc_path)
            .to_string_lossy()
            .to_string();
        if settings.is_doc_excluded(&rel) {
            continue;
        }
        let doc_dir = doc_path.parent().unwrap_or(root).to_path_buf();
        let manifests = manifest_cache
            .entry(doc_dir.clone())
            .or_insert_with(|| commands::Manifests::load_nearest(&doc_dir, root))
            .clone();
        let doc = check::Doc {
            rel,
            text,
            manifests,
        };
        for c in doc_checks {
            findings.extend(c.check(&doc, &ctx));
        }
        arch_rules.extend(rules::extract_prose_rules(&doc.text, &doc.rel));
    }
    findings.extend(rules::check(&arch_rules, &index, root));

    // Standalone Graphviz files (`*.dot`/`*.gv`) live outside the markdown set.
    // They are a whole-repo pass, so a `--doc`-scoped run skips them.
    if !scoped {
        for dot in diagram::collect_dot_files(root) {
            if let Ok(text) = std::fs::read_to_string(&dot) {
                let rel = dot
                    .strip_prefix(root)
                    .unwrap_or(&dot)
                    .to_string_lossy()
                    .to_string();
                findings.extend(diagram::check_dot_file(&text, &rel, &index));
            }
        }
    }

    // Code -> doc coverage gaps: undocumented public surface, anchored to its
    // symbol so it scores as its own dimension of the alignment score. This is a
    // whole-repo question, so a `--doc`-scoped run skips it.
    if !scoped {
        findings.extend(coverage::gaps(&index, root, &history));
    }

    // Layer 3: behavioural prose claims the deterministic layers can't reach.
    // Layer 2 retrieves the evidence; the NLI judge renders the verdict. Gated
    // behind the `ml` feature and `--layer 3`; a model/load failure degrades to
    // the deterministic findings rather than aborting the run.
    #[cfg(feature = "ml")]
    if layer >= 3 {
        eprintln!(
            "note: layer 3 runs the code-aware NLI judge (staleguard); \
             verdicts are advisory — review contradictions before acting."
        );
        let mut claims = Vec::new();
        for doc in collect_docs_filtered(root, doc_filter) {
            if let Ok(text) = std::fs::read_to_string(&doc) {
                let rel = doc
                    .strip_prefix(root)
                    .unwrap_or(&doc)
                    .to_string_lossy()
                    .to_string();
                if settings.is_doc_excluded(&rel) {
                    continue;
                }
                claims.extend(judge::candidate_claims(&text, &rel, &index));
            }
        }
        let cap = judge::max_claims();
        if cap > 0 {
            claims.truncate(cap);
        }
        match judge::check(root, &index, &claims, judge::EVIDENCE_K) {
            Ok(mut judged) => findings.append(&mut judged),
            Err(e) => eprintln!("note: layer 3 judge skipped ({e})"),
        }
    }

    // Layer 0: git-history staleness prior, then the drift pipeline (lineage,
    // carry-forward, fact-hash drift flag, alignment score). The staleness prior
    // needs the (skipped) history, so it only runs on a full, unscoped check.
    if !scoped {
        findings.extend(drift::coupling::check(&history));
    }
    // Verdict suppression from `.staleguard.toml` (e.g. opt out of `undocumented`).
    // Applied before the drift pipeline so suppressed findings neither report nor
    // gate. `Supported` claims are untouched, so the alignment score is unaffected.
    settings.apply_suppression(&mut findings);
    // Severity threshold: the `--min-severity` flag wins, else the config value.
    // Same pre-pipeline placement, so dropped findings neither report nor gate.
    let threshold = min_severity.or(settings.min_severity);
    settings::Settings::apply_severity_threshold(&mut findings, threshold);
    drift::run(findings, &index, root, opts)
}

/// `staleguard setup`: ensure every layer is ready to run. Layer 1 is always
/// available; in the `ml` build this fetches and loads the Layer 2 embedding
/// model and the Layer 3 NLI judge so the first real `check --layer 3` is fully
/// offline (and any model auth/network error surfaces here, not mid-run).
fn run_setup() -> ExitCode {
    println!("Layer 1 (deterministic): ready — no model needed.");

    #[cfg(not(feature = "ml"))]
    {
        println!(
            "Layers 2-3: this binary was built without the `ml` feature, so there \
             are no models to fetch.\n\
             Build with models enabled (the prebuilt/Homebrew binaries omit the \
             heavy ONNX deps):\n  \
             cargo install --git https://github.com/Arthur920/Staleguard --features ml"
        );
        ExitCode::SUCCESS
    }

    #[cfg(feature = "ml")]
    {
        print!("Layer 2 (embeddings): fetching model ... ");
        use std::io::Write;
        let _ = std::io::stdout().flush();
        if let Err(e) = retrieve::prefetch_model() {
            println!("failed");
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
        println!("ready.");

        print!("Layer 3 (NLI judge): fetching model ... ");
        let _ = std::io::stdout().flush();
        if let Err(e) = judge::Judge::load() {
            println!("failed");
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
        println!("ready.");
        println!("All layers ready. Run `staleguard check --layer 3`.");
        ExitCode::SUCCESS
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Commands::Check {
            path,
            format,
            layer,
            diff,
            write_ledger,
            fail_on_regression,
            min_severity,
            docs,
        } => {
            let root = std::fs::canonicalize(&path).unwrap_or(path);
            #[cfg(not(feature = "ml"))]
            if layer > 1 {
                eprintln!("note: layers 2-3 need the `ml` feature; running layer 1 only.");
            }
            #[cfg(feature = "ml")]
            if layer == 2 {
                eprintln!("note: layer 2 is retrieval-only (no verdicts); use --layer 3 for the NLI judge.");
            }
            let opts = drift::Options {
                diff_ref: diff,
                write_ledger,
                fail_on_regression,
            };
            let out = run_check(&root, &opts, layer, &docs, min_severity);
            report::report_check(&out, format);
            // Fail on any reportable finding, or on a score regression in CI.
            if out.findings.is_empty() && out.regression.is_none() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Commands::Index { path, format } => {
            let root = std::fs::canonicalize(&path).unwrap_or(path);
            let index = CodeIndex::build(&root);
            report::report_index(&index, format);
            ExitCode::SUCCESS
        }
        Commands::Rules { path, format } => {
            let root = std::fs::canonicalize(&path).unwrap_or(path);
            let index = CodeIndex::build(&root);
            let modules = index.module_set();
            let mut sourced = Vec::new();
            let mut bare = Vec::new();
            for doc in collect_docs(&root) {
                if let Ok(text) = std::fs::read_to_string(&doc) {
                    let rel = doc
                        .strip_prefix(&root)
                        .unwrap_or(&doc)
                        .to_string_lossy()
                        .to_string();
                    sourced.extend(rules::extract_prose_rules(&text, &rel));
                    bare.extend(rules::extract_bare_rules(&text, &rel, &modules));
                }
            }
            // Experimental bare-operand rules (audit-only): keep only those not
            // already captured by the backticked path, so the report shows the
            // *additional* recall the prototype would buy.
            let known: std::collections::HashSet<_> =
                sourced.iter().map(|s| s.rule.clone()).collect();
            bare.retain(|s| !known.contains(&s.rule));
            sourced.extend(bare);
            let rows = rules::audit(&sourced, &index, &root);
            report::report_rules(&rows, format);
            // A violated rule is real drift; exit non-zero so CI/agents notice.
            let violated = rows
                .iter()
                .any(|r| matches!(r.status, rules::RuleStatus::Violated(_)));
            if violated {
                ExitCode::FAILURE
            } else {
                ExitCode::SUCCESS
            }
        }
        Commands::Coverage { path, format } => {
            let root = std::fs::canonicalize(&path).unwrap_or(path);
            let findings = coverage::run(&root);
            report::report(&findings, format);
            if findings.is_empty() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Commands::Setup => run_setup(),
        #[cfg(feature = "ml")]
        Commands::Retrieve { query, path, k } => {
            let root = std::fs::canonicalize(&path).unwrap_or(path);
            let index = CodeIndex::build(&root);
            match retrieve::retrieve(&root, &index, std::slice::from_ref(&query), k) {
                Ok(per_query) => {
                    for hit in &per_query[0] {
                        println!("{:.3}  {}:{}", hit.score, hit.path, hit.start_line);
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::FAILURE
                }
            }
        }
    }
}
