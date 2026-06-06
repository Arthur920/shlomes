//! ER-diagram grounding (Layer 1, deterministic). Parses Mermaid `erDiagram`
//! and grounds entities + attributes against the repo's SQL [`Schema`].
//!
//! The weakest-grounded diagram kind, so it is the most conservative: it only
//! runs when a `.sql` schema exists, and — mirroring the class check — flags an
//! **attribute** only on an entity that *grounds* to a real table (an unmatched
//! entity may simply be a model our SQL-only extractor didn't see, so it is left
//! alone). Relationships/cardinality are not checked (Layer 2/3).

use std::collections::HashSet;
use std::path::Path;
use std::sync::OnceLock;

use regex::Regex;

use super::Format;
use crate::claim::Provenance;
use crate::code::schema::Schema;
use crate::findings::{Finding, Verdict};

/// One declared entity and its attribute names.
#[derive(Debug, Clone, PartialEq, Eq)]
struct EntityDecl {
    name: String,
    attrs: Vec<String>,
}

/// ER-diagram findings for one embedded diagram. Empty unless `body` is an
/// `erDiagram` *and* the repo has an extractable SQL schema.
pub(super) fn check(format: Format, body: &str, origin: &str, root: &Path) -> Vec<Finding> {
    if format != Format::Mermaid {
        return Vec::new();
    }
    let Some(entities) = parse(body) else {
        return Vec::new();
    };
    // Build the schema only once an ER diagram is actually present.
    let schema = Schema::extract(root);
    if schema.is_empty() {
        return Vec::new();
    }
    diff(&entities, &schema, origin)
}

/// Parse `erDiagram` entities + attributes, or `None` if it isn't one.
fn parse(body: &str) -> Option<Vec<EntityDecl>> {
    let header = body
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with("%%"))?;
    if header.split_whitespace().next() != Some("erDiagram") {
        return None;
    }

    let mut entities: Vec<EntityDecl> = Vec::new();
    let ensure = |entities: &mut Vec<EntityDecl>, name: &str| -> usize {
        match entities.iter().position(|e| e.name == name) {
            Some(i) => i,
            None => {
                entities.push(EntityDecl { name: name.to_string(), attrs: Vec::new() });
                entities.len() - 1
            }
        }
    };

    let mut current: Option<usize> = None;
    for raw in body.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with("%%") || line == header {
            continue;
        }
        if line.starts_with('}') {
            current = None;
            continue;
        }
        // `ENTITY {` opens an attribute block.
        if let Some(c) = block_open_re().captures(line) {
            let idx = ensure(&mut entities, &c[1]);
            current = Some(idx);
            continue;
        }
        // Relationship: `A ||--o{ B : label` — registers both entities.
        if let Some(c) = rel_re().captures(line) {
            ensure(&mut entities, &c[1]);
            ensure(&mut entities, &c[2]);
            continue;
        }
        // Attribute line inside a block: `type name [PK] [comment]` -> name.
        if let Some(idx) = current {
            if let Some(attr) = attr_name(line) {
                entities[idx].attrs.push(attr);
            }
        }
    }

    (!entities.is_empty()).then_some(entities)
}

/// The attribute name of an ER member line — the second whitespace token
/// (`string firstName` -> `firstName`). `None` if the line has fewer than two.
fn attr_name(line: &str) -> Option<String> {
    let mut it = line.split_whitespace();
    let _ty = it.next()?;
    it.next().map(|s| s.to_string())
}

fn diff(entities: &[EntityDecl], schema: &Schema, origin: &str) -> Vec<Finding> {
    let mut out = Vec::new();
    for e in entities {
        let Some(table) = schema.entity(&e.name) else {
            continue; // unmatched entity — may be an unextracted model; skip
        };
        let prov = Provenance::path(format!("schema:{}", table.name));
        let cols: HashSet<&str> = table.fields.iter().map(String::as_str).collect();
        out.push(Finding::supported(
            format!("ER entity `{}`", e.name),
            origin.to_string(),
            prov.clone(),
        ));
        for a in &e.attrs {
            let lower = a.to_ascii_lowercase();
            if cols.contains(lower.as_str()) {
                out.push(Finding::supported(
                    format!("entity `{}` attribute `{a}`", e.name),
                    origin.to_string(),
                    prov.clone(),
                ));
            } else {
                out.push(
                    Finding::problem(
                        Verdict::Stale,
                        format!("entity `{}` attribute `{a}`", e.name),
                        origin.to_string(),
                        format!(
                            "Stale attribute: the ER diagram draws `{}.{a}`, but table `{}` has no such column.",
                            e.name, table.name
                        ),
                    )
                    .anchored(prov.clone()),
                );
            }
        }
    }
    out
}

fn block_open_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^([A-Za-z_]\w*)\s*\{\s*$").unwrap())
}

fn rel_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // `A <card>--<card> B : label`, e.g. `CUSTOMER ||--o{ ORDER : places`.
    RE.get_or_init(|| {
        Regex::new(r"^([A-Za-z_]\w*)\s*[|}o{][|}o{.-]*--[|}o{.-]*[|}o{]?\s*([A-Za-z_]\w*)\s*:")
            .unwrap()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code::schema::Entity;

    fn schema() -> Schema {
        Schema {
            entities: vec![Entity {
                name: "customers".into(),
                fields: vec!["id".into(), "name".into()],
            }],
        }
    }

    fn check_with(body: &str, schema: &Schema) -> Vec<Finding> {
        // The parse+diff path without touching the filesystem.
        let Some(entities) = parse(body) else { return Vec::new() };
        if schema.is_empty() {
            return Vec::new();
        }
        diff(&entities, schema, "d.md:1")
    }

    #[test]
    fn grounded_entity_attribute_supported_and_stale_flagged() {
        let body = "erDiagram\n  CUSTOMER {\n    int id\n    string name\n    string ssn\n  }\n";
        let out = check_with(body, &schema());
        assert!(out.iter().any(|f| f.verdict == Verdict::Supported && f.claim.contains("`name`")));
        assert!(out.iter().any(|f| f.verdict == Verdict::Stale && f.detail.contains("ssn")));
    }

    #[test]
    fn unmatched_entity_is_not_flagged() {
        let body = "erDiagram\n  PRODUCT {\n    int sku\n  }\n";
        let out = check_with(body, &schema());
        assert!(out.is_empty());
    }

    #[test]
    fn relationship_registers_entities() {
        let body = "erDiagram\n  CUSTOMER ||--o{ ORDER : places\n";
        let out = check_with(body, &schema());
        // CUSTOMER grounds (one Supported); ORDER doesn't (skipped).
        assert_eq!(out.len(), 1);
        assert!(out[0].claim.contains("CUSTOMER"));
    }

    #[test]
    fn empty_schema_no_ops() {
        let body = "erDiagram\n  CUSTOMER {\n    int id\n  }\n";
        assert!(check_with(body, &Schema::default()).is_empty());
    }
}
