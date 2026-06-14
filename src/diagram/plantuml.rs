//! PlantUML component-diagram parser. Sequence diagrams (`participant`, `A -> B
//! : msg`) are recognized and skipped — message-order alignment is Layer 2/3.

use std::sync::OnceLock;

use regex::Regex;

use super::{Diagram, DiagramKind, Edge, Format, Node};

pub(super) fn parse(body: &str, origin: &str) -> Option<Diagram> {
    // Skip sequence diagrams: they declare participants or use `: message`.
    let looks_sequence = body.lines().any(|l| {
        let l = l.trim();
        l.starts_with("participant")
            || l.starts_with("actor")
            || (l.contains("->") && l.contains(':'))
    });
    if looks_sequence {
        return None;
    }

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

    for line in body.lines() {
        let line = line.trim();
        if line.is_empty()
            || line.starts_with('\'')
            || line.starts_with("@")
            || line.starts_with("skinparam")
            || line.starts_with("title")
        {
            continue;
        }

        // Aliases: `component "Auth Service" as auth` / `[Auth] as auth`.
        if let Some(c) = alias_re().captures(line) {
            let label = c[1].trim_matches('"').trim();
            register(&c[2], Some(label));
            continue;
        }
        // Bare component declaration: `[Auth Service]` / `component "Auth"`.
        if let Some(c) = decl_re().captures(line) {
            let label = c.get(1).or_else(|| c.get(2)).unwrap().as_str();
            let label = label.trim_matches('"').trim();
            register(label, Some(label));
            continue;
        }
        // Edges: `A --> B`, `[A] ..> [B]`, `A -- B`.
        if let Some(c) = edge_re().captures(line) {
            let from = endpoint(&c[1]);
            let to = endpoint(&c[3]);
            register(&from, None);
            register(&to, None);
            edges.push(Edge {
                from,
                to,
                directed: c[2].contains('>'),
            });
        }
    }
    let _ = register; // drop closure, releasing the &mut borrow of `nodes`

    if edges.is_empty() && nodes.is_empty() {
        return None;
    }
    Some(Diagram {
        kind: DiagramKind::Component,
        format: Format::PlantUml,
        nodes,
        edges,
        origin: origin.to_string(),
    })
}

/// An edge endpoint is either `[Bracketed Name]` or a bare identifier/alias.
fn endpoint(raw: &str) -> String {
    raw.trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .trim_matches('"')
        .trim()
        .to_string()
}

fn alias_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r#"(?:component|interface|node|\[)\s*("?[^"\]]+"?)\]?\s+as\s+([A-Za-z_]\w*)"#)
            .unwrap()
    })
}

fn decl_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r#"^(?:component|interface|node)\s+("?[^"\n]+?"?)$|^\[([^\]]+)\]$"#).unwrap()
    })
}

fn edge_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // left  arrow  right, where each side is `[Name]` or an identifier.
    RE.get_or_init(|| {
        Regex::new(r"(\[[^\]]+\]|[A-Za-z_]\w*)\s*([-.]{2,}>?|[-.]+>)\s*(\[[^\]]+\]|[A-Za-z_]\w*)")
            .unwrap()
    })
}
