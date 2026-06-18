//! The per-doc detector pipeline.
//!
//! Every deterministic Layer-1 detector that runs *per documentation file* shares
//! one shape — `(doc text, doc-relative path, some repo-wide grounding) -> findings`
//! — so they are expressed here as a uniform [`DocCheck`] trait over a shared
//! [`CheckContext`]. The orchestrator in `main.rs` builds the context once, then
//! runs every registered check against every doc, instead of hand-wiring each
//! detector into the loop body. Adding a detector is now "implement `DocCheck` and
//! append it to [`doc_checks`]", not "edit the orchestrator".
//!
//! Whole-repo passes (coverage, standalone diagrams, the architecture-rule graph
//! check, the history/coupling prior) are *not* per-doc and stay in `main.rs`.

use std::path::Path;

use crate::code::CodeIndex;
use crate::commands::Manifests;
use crate::entrypoints::Grounding;
use crate::findings::Finding;
use crate::{commands, config, constswap, diagram, entrypoints, extract, verify};

/// Repo-wide grounding built once and shared across every doc and every detector.
pub(crate) struct CheckContext<'a> {
    pub root: &'a Path,
    pub index: &'a CodeIndex,
    pub grounding: &'a Grounding,
    /// Identifiers that appear anywhere in the source tree (env vars / flags).
    pub code_tokens: &'a std::collections::HashSet<String>,
    /// The repo's path list, walked once, so path claims match in memory.
    pub repo_files: &'a [String],
}

/// One documentation file in flight: its text, its repo-relative path (used as the
/// finding origin), and the manifests resolved from its nearest ancestor.
pub(crate) struct Doc {
    pub rel: String,
    pub text: String,
    pub manifests: Manifests,
}

/// A deterministic detector that runs once per documentation file.
pub(crate) trait DocCheck {
    fn check(&self, doc: &Doc, ctx: &CheckContext) -> Vec<Finding>;
}

/// The registered per-doc detectors, run in order against every doc.
pub(crate) fn doc_checks() -> [&'static dyn DocCheck; 6] {
    [
        &PathCheck,
        &CommandCheck,
        &ConfigCheck,
        &EntrypointCheck,
        &ConstSwapCheck,
        &DiagramCheck,
    ]
}

/// Documented file/dir paths that don't exist in the repo.
struct PathCheck;
impl DocCheck for PathCheck {
    fn check(&self, doc: &Doc, ctx: &CheckContext) -> Vec<Finding> {
        let claims = extract::extract_path_claims(&doc.text, &doc.rel);
        verify::check_paths(&claims, ctx.root, ctx.repo_files)
    }
}

/// Documented commands with no matching npm script / make target / cargo bin.
struct CommandCheck;
impl DocCheck for CommandCheck {
    fn check(&self, doc: &Doc, _ctx: &CheckContext) -> Vec<Finding> {
        commands::check(&doc.text, &doc.rel, &doc.manifests)
    }
}

/// Documented env vars and CLI flags never read in the code.
struct ConfigCheck;
impl DocCheck for ConfigCheck {
    fn check(&self, doc: &Doc, ctx: &CheckContext) -> Vec<Finding> {
        config::check(
            &doc.text,
            &doc.rel,
            ctx.code_tokens,
            doc.manifests.project_bins(),
        )
    }
}

/// Documented entry points that resolve to no real symbol.
struct EntrypointCheck;
impl DocCheck for EntrypointCheck {
    fn check(&self, doc: &Doc, ctx: &CheckContext) -> Vec<Finding> {
        entrypoints::check(&doc.text, &doc.rel, ctx.grounding)
    }
}

/// Documented default constants that disagree with the lone code literal.
struct ConstSwapCheck;
impl DocCheck for ConstSwapCheck {
    fn check(&self, doc: &Doc, ctx: &CheckContext) -> Vec<Finding> {
        constswap::check(&doc.text, &doc.rel, ctx.index)
    }
}

/// Embedded diagrams diffed against the real dependency graph.
struct DiagramCheck;
impl DocCheck for DiagramCheck {
    fn check(&self, doc: &Doc, ctx: &CheckContext) -> Vec<Finding> {
        diagram::check(&doc.text, &doc.rel, ctx.index, ctx.root)
    }
}
