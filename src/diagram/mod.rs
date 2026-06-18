//! Layer 1 diagram coherence: parse text-based architecture diagrams and
//! set-diff their nodes/edges against the real module dependency graph.
//!
//! Scope (first cut): graph-shaped diagrams only — Mermaid `graph`/`flowchart`,
//! PlantUML component diagrams, and Graphviz DOT. Sequence/class/ER/state
//! diagrams are recognized and skipped (they need symbol/call-graph alignment).
//! No ML: every endpoint is grounded against
//! real modules with [`crate::rules::matches`] / [`crate::rules::grounded`] so we
//! under-report rather than emit false positives.

mod align;
mod class;
mod dot;
mod er;
mod ground;
mod mermaid;
mod plantuml;
mod sequence;
mod state;

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use walkdir::WalkDir;

use crate::claim::Provenance;
use crate::code::CodeIndex;
use crate::findings::{Finding, Verdict};
use crate::rules::matches;

use ground::{ground_label, module_token_index, resolve, Resolution};

/// The shape of a parsed diagram. Only the graph-shaped kinds are diffable at
/// Layer 1; the rest are parsed to `None` by their format parser and skipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagramKind {
    Flowchart,
    Component,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Mermaid,
    PlantUml,
    Dot,
}

impl Format {
    fn label(self) -> &'static str {
        match self {
            Format::Mermaid => "mermaid",
            Format::PlantUml => "plantuml",
            Format::Dot => "dot",
        }
    }
}

/// A diagram box. `id` is the node key used by edges; `label` is the display
/// text (falls back to `id` when a node is only referenced, never declared).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    pub id: String,
    pub label: String,
}

/// A drawn connection between two node ids. `directed` distinguishes `A --> B`
/// (an import direction) from `A --- B` (undirected).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Edge {
    pub from: String,
    pub to: String,
    pub directed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagram {
    pub kind: DiagramKind,
    pub format: Format,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    /// `"rel/path.md:<block-start-line>"`, or a `.dot`/`.gv` file path.
    pub origin: String,
}

impl Diagram {
    /// Display text for a node id — its declared label, else the id itself.
    fn text(&self, id: &str) -> String {
        self.nodes
            .iter()
            .find(|n| n.id == id)
            .map(|n| n.label.clone())
            .unwrap_or_else(|| id.to_string())
    }
}

/// A fenced/embedded diagram source pulled from a markdown doc.
struct Source {
    format: Format,
    body: String,
    /// 1-based line where the block starts.
    line: usize,
}

/// Extract every diagram source embedded in a markdown document: fenced
/// ```` ```mermaid|plantuml|dot ```` blocks plus bare `@startuml…@enduml`
/// regions that sit outside any fence.
fn sources(markdown: &str) -> Vec<Source> {
    let mut out = Vec::new();
    // (format, start-line, body). A non-diagram fence is tracked with `None`
    // format so a `@startuml` inside it isn't mistaken for a bare UML region;
    // its body is discarded on close.
    let mut fence: Option<(Option<Format>, usize, Vec<&str>)> = None;
    let mut uml: Option<(usize, Vec<&str>)> = None;

    for (i, raw) in markdown.lines().enumerate() {
        let trimmed = raw.trim_start();

        // Inside a fenced block: accumulate until the closing fence.
        if let Some((fmt, start, body)) = fence.as_mut() {
            if trimmed.starts_with("```") {
                if let Some(fmt) = fmt {
                    out.push(Source {
                        format: *fmt,
                        body: body.join("\n"),
                        line: *start,
                    });
                }
                fence = None;
            } else {
                body.push(raw);
            }
            continue;
        }

        // Opening fence?
        if let Some(rest) = trimmed.strip_prefix("```") {
            let fmt = match rest.trim().to_ascii_lowercase().as_str() {
                "mermaid" => Some(Format::Mermaid),
                "plantuml" | "puml" | "uml" => Some(Format::PlantUml),
                "dot" | "graphviz" => Some(Format::Dot),
                _ => None,
            };
            fence = Some((fmt, i + 1, Vec::new()));
            continue;
        }

        // Bare PlantUML outside any fence.
        if let Some((start, body)) = uml.as_mut() {
            body.push(raw);
            if trimmed.starts_with("@enduml") {
                out.push(Source {
                    format: Format::PlantUml,
                    body: body.join("\n"),
                    line: *start,
                });
                uml = None;
            }
            continue;
        }
        if trimmed.starts_with("@startuml") {
            uml = Some((i + 1, vec![raw]));
        }
    }
    out
}

/// Diagram-coherence findings for one markdown document. `root` is the repo root
/// (needed only to extract the SQL schema for ER diagrams).
pub fn check(markdown: &str, doc_path: &str, index: &CodeIndex, root: &Path) -> Vec<Finding> {
    let modules = index.module_set();
    let mut out = Vec::new();
    for src in sources(markdown) {
        let origin = format!("{doc_path}:{}", src.line);
        if let Some(d) = parse(src.format, &src.body, &origin) {
            out.extend(diff(&d, index, &modules));
        } else if let Some(seq) = sequence::parse(src.format, &src.body, &origin) {
            // Ordered diagrams are aligned, not set-diffed.
            out.extend(align::check(&seq, index));
        } else {
            // Symbol/schema-grounded kinds. Each returns empty unless the body is
            // its kind, so this stays a clean fall-through.
            out.extend(class::check(src.format, &src.body, &origin, index));
            out.extend(state::check(src.format, &src.body, &origin, index));
            out.extend(er::check(src.format, &src.body, &origin, root));
        }
    }
    out
}

/// Standalone Graphviz files (`*.dot`, `*.gv`) anywhere under `root`.
pub fn collect_dot_files(root: &Path) -> Vec<PathBuf> {
    WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| !crate::code::lang::is_skip_dir(&e.file_name().to_string_lossy()))
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .filter(|p| {
            matches!(
                p.extension().and_then(|s| s.to_str()),
                Some("dot") | Some("gv")
            )
        })
        .collect()
}

/// Diagram-coherence findings for a standalone DOT file.
pub fn check_dot_file(body: &str, origin: &str, index: &CodeIndex) -> Vec<Finding> {
    let modules = index.module_set();
    match parse(Format::Dot, body, origin) {
        Some(d) => diff(&d, index, &modules),
        None => Vec::new(),
    }
}

fn parse(format: Format, body: &str, origin: &str) -> Option<Diagram> {
    match format {
        Format::Mermaid => mermaid::parse(body, origin),
        Format::PlantUml => plantuml::parse(body, origin),
        Format::Dot => dot::parse(body, origin),
    }
}

// ---- the set-diff ---------------------------------------------------------

/// Does a real module edge connect a module matching `from` to one matching
/// `to`? `from`/`to` are diagram labels; `module_edges` endpoints are real paths.
fn real_edge(index: &CodeIndex, from: &str, to: &str) -> bool {
    index
        .module_edges
        .iter()
        .any(|e| matches(&e.from_module, from) && matches(&e.to_module, to))
}

/// Set-diff a parsed diagram against the real import graph. Every comparison is
/// gated by grounding, so external boxes (`User`, `DB`, `Browser`) are ignored.
fn diff(d: &Diagram, index: &CodeIndex, modules: &HashSet<String>) -> Vec<Finding> {
    let mut out = Vec::new();
    let module_tokens = module_token_index(modules);
    let res = |label: &str| resolve(label, modules, &module_tokens);

    // Resolve every node label once (exact or fuzzy).
    let node_res: Vec<(&str, Resolution)> = d
        .nodes
        .iter()
        .map(|n| (n.label.as_str(), res(&n.label)))
        .collect();

    // 1. Phantom edges — drawn, both endpoints name exactly one real module, but
    //    no real import connects them. Exact-unique only: a fuzzily- or ambiguously-
    //    grounded endpoint can't carry an assertion about a specific edge without
    //    risking false positives (the wild audit's conceptual/segment-match arrows).
    for e in &d.edges {
        let from = d.text(&e.from);
        let to = d.text(&e.to);
        let (rfrom, rto) = (res(&from), res(&to));
        let (Some(gfrom), Some(gto)) = (rfrom.exact_module(), rto.exact_module()) else {
            continue; // an endpoint is external/undocumented/ambiguous → skip
        };
        let exists = real_edge(index, gfrom, gto) || (!e.directed && real_edge(index, gto, gfrom));
        let prov = Provenance::modules([from.clone(), to.clone()]);
        if exists {
            out.push(Finding::supported(
                format!("diagram draws `{from}` -> `{to}`"),
                d.origin.clone(),
                prov,
            ));
        } else {
            out.push(
                Finding::problem(
                    Verdict::Contradicted,
                    format!("diagram draws `{from}` -> `{to}`"),
                    d.origin.clone(),
                    format!(
                        "Phantom dependency: the {} diagram draws an edge `{from}` -> `{to}`, but no import connects those modules.",
                        d.format.label()
                    ),
                )
                .anchored(prov)
                .with_refs(vec![format!("{from} -> {to}")]),
            );
        }
    }

    // 2. Stale boxes — a box that clearly names a code module that is gone. Fuzzy
    //    resolution only *reduces* this set (more boxes ground), so it stays safe.
    for (text, resolution) in &node_res {
        if resolution.grounds() {
            continue; // names real code (exact, fuzzy, or ambiguous) → not stale
        }
        let g = ground_label(text);
        if module_intent(g) {
            out.push(Finding::problem(
                Verdict::Stale,
                format!("diagram box `{text}`"),
                d.origin.clone(),
                format!(
                    "Stale box: the {} diagram contains a box `{text}` that resolves to no module in the repo.",
                    d.format.label()
                ),
            ));
        }
    }

    // 3. Missing arrows — a real import between two boxes that are *both* already
    //    drawn, yet no edge connects them. Bounded to depicted modules, so it
    //    never fires for components the author chose to omit.
    //    Exact-unique only: fuzzy/ambiguous boxes must not invent "you forgot an
    //    edge", since abstract diagrams omit edges intentionally (highest-FP class).
    let mut seen = HashSet::new();
    for me in &index.module_edges {
        let from_drawn = node_res.iter().any(|(_, r)| {
            r.exact_module()
                .is_some_and(|g| matches(&me.from_module, g))
        });
        let to_drawn = node_res
            .iter()
            .any(|(_, r)| r.exact_module().is_some_and(|g| matches(&me.to_module, g)));
        if !from_drawn || !to_drawn {
            continue;
        }
        let drawn = d.edges.iter().any(|e| {
            let ft = ground_label(&d.text(&e.from)).to_string();
            let tt = ground_label(&d.text(&e.to)).to_string();
            (matches(&me.from_module, &ft) && matches(&me.to_module, &tt))
                || (!e.directed && matches(&me.from_module, &tt) && matches(&me.to_module, &ft))
        });
        if !drawn && seen.insert((me.from_module.clone(), me.to_module.clone())) {
            out.push(
                Finding::problem(
                    Verdict::Undocumented,
                    format!("import `{}` -> `{}`", me.from_module, me.to_module),
                    d.origin.clone(),
                    format!(
                        "Missing arrow: `{}` imports `{}` and both are drawn in the {} diagram, but no edge connects them.",
                        me.from_module,
                        me.to_module,
                        d.format.label()
                    ),
                )
                .anchored(Provenance::modules([
                    me.from_module.clone(),
                    me.to_module.clone(),
                ]))
                .with_refs(vec![format!("{} -> {}", me.from_module, me.to_module)]),
            );
        }
    }

    out
}

/// True if a box label unambiguously denotes a code module *path* (so an
/// ungrounded one is a stale reference, not a conceptual box). Deliberately
/// conservative — only a clean path/namespace token with a separator counts.
/// A bare word (`User`, `DB`), a URL route (`/items/public/`), a decision-node
/// label (`needed=False<br/>ok`), or call syntax is left alone to keep Layer 1
/// zero-FP. Pass the [`ground_label`]-normalized text so a trailing `.ts`/`.py`
/// doesn't read as a path dot.
fn module_intent(text: &str) -> bool {
    if text.is_empty() || text.chars().any(char::is_whitespace) {
        return false;
    }
    // URL routes / fragments are not code module paths.
    if text.starts_with('/') || text.contains("://") {
        return false;
    }
    // Markup, decision-node labels, call/query syntax — none belong in a path.
    if text.contains([
        '=', '<', '>', '{', '}', '(', ')', '\\', '?', '#', '&', '"', '\'',
    ]) {
        return false;
    }
    // What remains must carry a path or namespace separator.
    text.contains('/') || text.contains("::")
}

#[cfg(test)]
mod tests;
