//! Ordered (sequence) diagram parsing: Mermaid `sequenceDiagram` and PlantUML
//! sequence diagrams. Unlike the graph-shaped formats these are an **ordered
//! list of messages**, so they are aligned (not set-diffed) against the code's
//! ordered call sequence — see [`super::align`] and `docs/diagram-coherence.md`.
//!
//! Parsing is deliberately tolerant: a line that doesn't look like a
//! participant declaration or an `A -> B : msg` message is skipped, so control
//! blocks (`alt`/`loop`/`note`) never produce phantom steps.

use std::sync::OnceLock;

use regex::Regex;

use super::Format;

/// One ordered message `from -> to : label`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub from: String,
    pub to: String,
    pub label: String,
}

impl Message {
    /// The leading identifier of the message label — its candidate call name
    /// (`validate()` -> `validate`, `login(user)` -> `login`). Prose labels with
    /// no leading identifier yield the trimmed label, which simply won't match a
    /// real call (keeping the alignment zero-FP).
    pub fn call_token(&self) -> String {
        call_re()
            .captures(&self.label)
            .map(|c| c[1].to_string())
            .unwrap_or_else(|| self.label.trim().to_string())
    }
}

/// A parsed sequence diagram: participant ids/labels and ordered messages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sequence {
    pub participants: Vec<String>,
    pub messages: Vec<Message>,
    /// `"rel/path.md:<block-start-line>"`.
    pub origin: String,
}

/// Parse a sequence diagram, or `None` if `body` isn't one in `format`.
pub(super) fn parse(format: Format, body: &str, origin: &str) -> Option<Sequence> {
    match format {
        Format::Mermaid => {
            let header = body
                .lines()
                .map(str::trim)
                .find(|l| !l.is_empty() && !l.starts_with("%%"))?;
            if header.split_whitespace().next() != Some("sequenceDiagram") {
                return None;
            }
            parse_body(body, origin)
        }
        Format::PlantUml => {
            // Same heuristic plantuml.rs uses to *reject* sequence diagrams.
            let looks_sequence = body.lines().any(|l| {
                let l = l.trim();
                l.starts_with("participant")
                    || l.starts_with("actor")
                    || (l.contains("->") && l.contains(':'))
            });
            looks_sequence.then(|| parse_body(body, origin)).flatten()
        }
        Format::Dot => None,
    }
}

fn parse_body(body: &str, origin: &str) -> Option<Sequence> {
    let mut participants: Vec<String> = Vec::new();
    let mut messages: Vec<Message> = Vec::new();

    let register = |id: &str, participants: &mut Vec<String>| {
        if !participants.iter().any(|p| p == id) {
            participants.push(id.to_string());
        }
    };

    for raw in body.lines() {
        let line = raw.trim();
        if line.is_empty() || is_noise(line) {
            continue;
        }
        if let Some(c) = participant_re().captures(line) {
            // `participant X` / `participant "X" as y` / `actor X` — the alias
            // (last group) is the id edges use, else the bare name.
            let id = c.get(2).map(|m| m.as_str()).unwrap_or(&c[1]);
            register(id.trim_matches('"').trim(), &mut participants);
            continue;
        }
        if let Some(c) = message_re().captures(line) {
            let from = c[1].to_string();
            let to = c[2].to_string();
            register(&from, &mut participants);
            register(&to, &mut participants);
            messages.push(Message {
                from,
                to,
                label: c[3].trim().to_string(),
            });
        }
    }

    if messages.is_empty() {
        return None;
    }
    Some(Sequence {
        participants,
        messages,
        origin: origin.to_string(),
    })
}

/// Control-flow / styling lines that are not participants or messages.
fn is_noise(line: &str) -> bool {
    const KW: &[&str] = &[
        "note", "alt", "else", "opt", "loop", "par", "and", "end", "rect",
        "activate", "deactivate", "autonumber", "title", "@start", "@end",
        "skinparam", "box",
    ];
    line.starts_with('\'')
        || line.starts_with("%%")
        || KW.iter().any(|k| {
            line.len() >= k.len()
                && line[..k.len()].eq_ignore_ascii_case(k)
                && line[k.len()..]
                    .chars()
                    .next()
                    .map(|c| !c.is_alphanumeric())
                    .unwrap_or(true)
        })
}

fn participant_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        // `participant <name|"name"> [as <alias>]` / `actor <name>`.
        Regex::new(r#"^(?:participant|actor)\s+("?[^"\n]+?"?)(?:\s+as\s+([A-Za-z_]\w*))?$"#).unwrap()
    })
}

fn message_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        // from  arrow  to  :  label. Arrows cover Mermaid (`->>`, `-->>`, `-)`,
        // `-x`) and PlantUML (`->`, `-->`, `..>`).
        Regex::new(
            r"^([A-Za-z_]\w*)\s*(?:-{1,2}>>?|-{1,2}\)|-{1,2}x|\.\.>)\s*([A-Za-z_]\w*)\s*:\s*(.+)$",
        )
        .unwrap()
    })
}

fn call_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\s*([A-Za-z_]\w*)").unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_mermaid_sequence() {
        let body = "sequenceDiagram\n  participant A\n  A->>B: validate()\n  B-->>A: ok\n";
        let seq = parse(Format::Mermaid, body, "d.md:1").unwrap();
        assert_eq!(seq.messages.len(), 2);
        assert_eq!(seq.messages[0].from, "A");
        assert_eq!(seq.messages[0].to, "B");
        assert_eq!(seq.messages[0].call_token(), "validate");
        assert!(seq.participants.contains(&"A".to_string()));
        assert!(seq.participants.contains(&"B".to_string()));
    }

    #[test]
    fn parses_plantuml_sequence_and_skips_control_blocks() {
        let body = "@startuml\nparticipant Auth\nAuth -> DB : lookup(id)\nalt found\nDB --> Auth : row\nend\n@enduml\n";
        let seq = parse(Format::PlantUml, body, "d.md:1").unwrap();
        assert_eq!(seq.messages.len(), 2);
        assert_eq!(seq.messages[0].call_token(), "lookup");
        assert_eq!(seq.messages[1].from, "DB");
    }

    #[test]
    fn flowchart_is_not_a_sequence() {
        assert!(parse(Format::Mermaid, "graph TD\n A-->B\n", "d.md:1").is_none());
    }
}
