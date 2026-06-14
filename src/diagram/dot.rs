//! Graphviz DOT parser. `digraph` edges (`->`) are directed; `graph` edges
//! (`--`) are undirected.

use std::sync::OnceLock;

use regex::Regex;

use super::{Diagram, DiagramKind, Edge, Format, Node};

pub(super) fn parse(body: &str, origin: &str) -> Option<Diagram> {
    let stripped = strip_comments(body);
    let directed = graph_kind(&stripped)?;

    let mut nodes: Vec<Node> = Vec::new();
    let mut edges: Vec<Edge> = Vec::new();

    let mut register = |id: &str, label: Option<&str>| {
        let id = id.to_string();
        match nodes.iter_mut().find(|n| n.id == id) {
            Some(existing) => {
                if let Some(label) = label {
                    existing.label = label.to_string();
                }
            }
            None => nodes.push(Node {
                label: label.unwrap_or(&id).to_string(),
                id,
            }),
        }
    };

    for stmt in stripped.split([';', '\n', '{', '}']) {
        let stmt = stmt.trim();
        if stmt.is_empty() {
            continue;
        }
        // Edge: `a -> b` / `"a" -- "b"`. Checked before node decls so an
        // edge statement isn't swallowed as a node attribute.
        if let Some(c) = edge_re().captures(stmt) {
            let from = unquote(&c[1]);
            let to = unquote(&c[2]);
            register(&from, None);
            register(&to, None);
            edges.push(Edge { from, to, directed });
            continue;
        }
        // Node with explicit label: `a [label="A"]`.
        if let Some(c) = node_label_re().captures(stmt) {
            register(&unquote(&c[1]), Some(c[2].trim()));
            continue;
        }
        // Bare node: `a;` / `"a"`.
        if let Some(c) = bare_node_re().captures(stmt) {
            let id = unquote(&c[1]);
            if !is_keyword(&id) {
                register(&id, None);
            }
        }
    }
    let _ = register; // drop closure, releasing the &mut borrow of `nodes`

    if edges.is_empty() && nodes.is_empty() {
        return None;
    }
    Some(Diagram {
        kind: DiagramKind::Component,
        format: Format::Dot,
        nodes,
        edges,
        origin: origin.to_string(),
    })
}

/// `Some(true)` for `digraph`, `Some(false)` for `graph`, `None` otherwise.
fn graph_kind(src: &str) -> Option<bool> {
    let head = src.split('{').next().unwrap_or("");
    if head.split_whitespace().any(|w| w == "digraph") {
        Some(true)
    } else if head.split_whitespace().any(|w| w == "graph") {
        Some(false)
    } else {
        None
    }
}

fn strip_comments(src: &str) -> String {
    let no_block = block_comment_re().replace_all(src, " ");
    no_block
        .lines()
        .map(|l| {
            let l = l.split("//").next().unwrap_or(l);
            l.split('#').next().unwrap_or(l)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn unquote(s: &str) -> String {
    s.trim().trim_matches('"').to_string()
}

fn is_keyword(s: &str) -> bool {
    matches!(
        s.to_ascii_lowercase().as_str(),
        "graph" | "digraph" | "node" | "edge" | "subgraph" | "{" | "}"
    )
}

fn edge_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r#"("[^"]+"|[A-Za-z_]\w*)\s*(?:->|--)\s*("[^"]+"|[A-Za-z_]\w*)"#).unwrap()
    })
}

fn node_label_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r#"^("[^"]+"|[A-Za-z_]\w*)\s*\[[^\]]*label\s*=\s*"([^"]+)"[^\]]*\]"#).unwrap()
    })
}

fn bare_node_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r#"^("[^"]+"|[A-Za-z_][\w/.:-]*)\s*(?:\[[^\]]*\])?$"#).unwrap())
}

fn block_comment_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?s)/\*.*?\*/").unwrap())
}
