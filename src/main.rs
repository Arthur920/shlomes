//! shlomes command-line entry point.

mod claim;
mod code;
mod commands;
mod config;
mod coverage;
mod diagram;
mod entrypoints;
mod drift;
mod extract;
mod findings;
mod git;
#[cfg(feature = "ml")]
mod judge;
#[cfg(feature = "ml")]
mod rerank;
#[cfg(feature = "ml")]
mod retrieve;
mod rules;
mod verify;

use code::CodeIndex;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use walkdir::WalkDir;

use crate::findings::Finding;

#[derive(Parser)]
#[command(
    name = "shlomes",
    version,
    about = "Check CLAUDE.md, project docs, and code against each other for coherence drift."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

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
        /// Max layer: 1 deterministic, 2 +retrieval, 3 +NLI judge (2-3 require
        /// the `ml` feature build).
        #[arg(long, default_value_t = 1)]
        layer: u8,
        /// Drift base: only re-derive claims whose code changed since this git
        /// ref (default: the committed ledger's last commit).
        #[arg(long)]
        diff: Option<String>,
        /// Persist the drift ledger + alignment score under `.shlomes/` (run this
        /// on the base branch to set the CI baseline).
        #[arg(long)]
        write_ledger: bool,
        /// Fail if the alignment score regressed below the committed baseline.
        #[arg(long)]
        fail_on_regression: bool,
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
    /// Report public code surface that no doc describes (code -> doc gaps).
    Coverage {
        /// Repo root (default: cwd).
        #[arg(default_value = ".")]
        path: PathBuf,
        /// Output format.
        #[arg(long, value_enum, default_value_t = Format::Text)]
        format: Format,
    },
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

#[derive(Clone, Copy, ValueEnum)]
enum Format {
    Text,
    Json,
}

pub(crate) fn collect_docs(root: &Path) -> Vec<PathBuf> {
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
        .collect()
}

fn run_check(root: &Path, opts: &drift::Options, layer: u8) -> drift::Outcome {
    let _ = layer; // consulted only in `ml` builds for the Layer 3 judge.
    // Repo-wide grounding, built once and shared across every doc.
    let index = CodeIndex::build(root);
    let manifests = commands::Manifests::load(root);
    let code_tokens = config::code_tokens(root);
    let grounding = entrypoints::Grounding::from_index(&index);
    // One git-history fetch shared by every history-mining pass (coverage risk
    // ranking + the coupling staleness prior), rather than fetching + parsing
    // the same 1000 commits twice.
    let history = git::file_change_history(root, drift::coupling::MAX_COMMITS);
    // The repo's path list, walked once, so each doc's path claims match in
    // memory instead of re-walking the whole tree per claim.
    let repo_files = verify::repo_paths(root);

    // Architecture rules: prose-sourced, accumulated per doc, then verified once
    // (the symbol scan walks the whole repo).
    let mut arch_rules: Vec<rules::SourcedRule> = Vec::new();

    let mut findings = Vec::new();
    for doc in collect_docs(root) {
        let text = match std::fs::read_to_string(&doc) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let rel = doc
            .strip_prefix(root)
            .unwrap_or(&doc)
            .to_string_lossy()
            .to_string();
        let claims = extract::extract_path_claims(&text, &rel);
        findings.extend(verify::check_paths(&claims, root, &repo_files));
        findings.extend(commands::check(&text, &rel, &manifests));
        findings.extend(config::check(
            &text,
            &rel,
            &code_tokens,
            manifests.project_bins(),
        ));
        findings.extend(entrypoints::check(&text, &rel, &grounding));
        findings.extend(diagram::check(&text, &rel, &index, root));
        arch_rules.extend(rules::extract_prose_rules(&text, &rel));
    }
    findings.extend(rules::check(&arch_rules, &index, root));

    // Standalone Graphviz files (`*.dot`/`*.gv`) live outside the markdown set.
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

    // Code -> doc coverage gaps: undocumented public surface, anchored to its
    // symbol so it scores as its own dimension of the alignment score.
    findings.extend(coverage::gaps(&index, root, &history));

    // Layer 3: behavioural prose claims the deterministic layers can't reach.
    // Layer 2 retrieves the evidence; the NLI judge renders the verdict. Gated
    // behind the `ml` feature and `--layer 3`; a model/load failure degrades to
    // the deterministic findings rather than aborting the run.
    #[cfg(feature = "ml")]
    if layer >= 3 {
        let mut claims = Vec::new();
        for doc in collect_docs(root) {
            if let Ok(text) = std::fs::read_to_string(&doc) {
                let rel = doc
                    .strip_prefix(root)
                    .unwrap_or(&doc)
                    .to_string_lossy()
                    .to_string();
                claims.extend(judge::candidate_claims(&text, &rel, &index));
            }
        }
        claims.truncate(judge::MAX_CLAIMS);
        match judge::check(root, &claims, judge::EVIDENCE_K) {
            Ok(mut judged) => findings.append(&mut judged),
            Err(e) => eprintln!("note: layer 3 judge skipped ({e})"),
        }
    }

    // Layer 0: git-history staleness prior, then the drift pipeline (lineage,
    // carry-forward, fact-hash drift flag, alignment score).
    findings.extend(drift::coupling::check(&history));
    drift::run(findings, &index, root, opts)
}

fn report(findings: &[Finding], format: Format) {
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
fn report_check(out: &drift::Outcome, format: Format) {
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
                println!(
                    "\u{2717} score regressed: {base:.3} (base) -> {head:.3} (head)"
                );
            }
        }
    }
}

fn report_index(index: &CodeIndex, format: Format) {
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
            let out = run_check(&root, &opts, layer);
            report_check(&out, format);
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
            report_index(&index, format);
            ExitCode::SUCCESS
        }
        Commands::Coverage { path, format } => {
            let root = std::fs::canonicalize(&path).unwrap_or(path);
            let findings = coverage::run(&root);
            report(&findings, format);
            if findings.is_empty() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        #[cfg(feature = "ml")]
        Commands::Retrieve { query, path, k } => {
            let root = std::fs::canonicalize(&path).unwrap_or(path);
            match retrieve::retrieve(&root, std::slice::from_ref(&query), k) {
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
