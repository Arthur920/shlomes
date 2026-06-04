//! doc-aligner command-line entry point.

mod extract;
mod findings;
#[cfg(feature = "ml")]
mod retrieve;
mod verify;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use walkdir::WalkDir;

use crate::findings::Finding;

#[derive(Parser)]
#[command(
    name = "doc-aligner",
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
        /// Max layer: 1 deterministic, 2 +retrieval, 3 +LLM (1 only for now).
        #[arg(long, default_value_t = 1)]
        layer: u8,
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

fn collect_docs(root: &Path) -> Vec<PathBuf> {
    WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            name != ".git" && name != ".doc-aligner"
        })
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .filter(|p| {
            matches!(
                p.extension().and_then(|s| s.to_str()),
                Some("md") | Some("markdown")
            )
        })
        .collect()
}

fn run_check(root: &Path) -> Vec<Finding> {
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
        findings.extend(verify::check_paths(&claims, root));
    }
    findings
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

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Commands::Check {
            path,
            format,
            layer,
        } => {
            let root = std::fs::canonicalize(&path).unwrap_or(path);
            if layer > 1 {
                eprintln!("note: layers 2-3 are not implemented yet; running layer 1.");
            }
            let findings = run_check(&root);
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
