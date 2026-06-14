//! Minimal database-schema extraction — the ground truth ER-diagram coherence
//! checks against. v1 reads SQL `CREATE TABLE`
//! statements from `.sql` files: the cleanest, most explicit schema source.
//!
//! ORM-model detection (structs/classes tagged as entities) is **deferred**: it
//! needs decorator/derive-attribute extraction we don't yet have, and guessing
//! "which struct is a table" would break the zero-FP stance. A repo with no
//! `.sql` schema yields an empty [`Schema`], so ER grounding simply no-ops.

use std::path::{Path, PathBuf};

use walkdir::WalkDir;

/// One table: its name and column names (lowercased for case-insensitive match).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entity {
    pub name: String,
    pub fields: Vec<String>,
}

/// The repo's extracted schema. Empty when no `.sql` source is present.
#[derive(Debug, Clone, Default)]
pub struct Schema {
    pub entities: Vec<Entity>,
}

impl Schema {
    /// Extract a schema from every `.sql` file under `root`.
    pub fn extract(root: &Path) -> Schema {
        let mut entities = Vec::new();
        for sql in sql_files(root) {
            if let Ok(text) = std::fs::read_to_string(&sql) {
                entities.extend(from_sql(&text));
            }
        }
        Schema { entities }
    }

    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// The table matching `name` case-insensitively, tolerating a trailing `s`
    /// (ER `CUSTOMER` ↔ table `customers`).
    pub fn entity(&self, name: &str) -> Option<&Entity> {
        let n = name.to_ascii_lowercase();
        self.entities
            .iter()
            .find(|e| e.name == n || e.name.trim_end_matches('s') == n.trim_end_matches('s'))
    }
}

fn sql_files(root: &Path) -> Vec<PathBuf> {
    WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| !super::lang::is_skip_dir(&e.file_name().to_string_lossy()))
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("sql"))
        .collect()
}

/// Parse `CREATE TABLE` statements from one SQL text. Column names are the first
/// identifier of each top-level comma-separated item that isn't a table
/// constraint (`PRIMARY KEY`, `FOREIGN KEY`, …).
fn from_sql(text: &str) -> Vec<Entity> {
    let lower = text.to_ascii_lowercase();
    let bytes = text.as_bytes();
    let mut entities = Vec::new();
    let mut search = 0usize;

    while let Some(rel) = lower[search..].find("create table") {
        let stmt_start = search + rel;
        // Table name: the identifier after `create table [if not exists]`.
        let after = &text[stmt_start + "create table".len()..];
        let Some(paren_rel) = after.find('(') else {
            break;
        };
        let header = &after[..paren_rel];
        let name = header
            .split_whitespace()
            .rfind(|w| {
                !w.eq_ignore_ascii_case("if")
                    && !w.eq_ignore_ascii_case("not")
                    && !w.eq_ignore_ascii_case("exists")
            })
            .map(clean_ident)
            .unwrap_or_default();

        // Body: from the `(` to its matching `)`.
        let body_start = stmt_start + "create table".len() + paren_rel;
        let body_end = matching_paren(bytes, body_start);
        let body = &text[body_start + 1..body_end];
        search = body_end + 1;

        if name.is_empty() {
            continue;
        }
        let fields = split_top_level(body)
            .into_iter()
            .filter_map(|col| {
                let first = col.split_whitespace().next()?;
                let id = clean_ident(first);
                (!id.is_empty() && !is_constraint(&id)).then_some(id)
            })
            .collect();
        entities.push(Entity {
            name: name.to_ascii_lowercase(),
            fields,
        });
    }
    entities
}

/// Byte offset of the `)` matching the `(` at `open`, or the end of input.
fn matching_paren(bytes: &[u8], open: usize) -> usize {
    let mut depth = 0i32;
    for (i, &b) in bytes.iter().enumerate().skip(open) {
        match b {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return i;
                }
            }
            _ => {}
        }
    }
    bytes.len().saturating_sub(1)
}

/// Split a column list on top-level commas (ignoring commas inside type parens
/// like `DECIMAL(10,2)`).
fn split_top_level(body: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    for (i, c) in body.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                out.push(body[start..i].trim().to_string());
                start = i + 1;
            }
            _ => {}
        }
    }
    let last = body[start..].trim();
    if !last.is_empty() {
        out.push(last.to_string());
    }
    out
}

fn clean_ident(raw: &str) -> String {
    raw.trim()
        .trim_matches(|c| c == '`' || c == '"' || c == '\'' || c == '[' || c == ']')
        .rsplit('.') // strip a `schema.` qualifier
        .next()
        .unwrap_or("")
        .to_ascii_lowercase()
}

fn is_constraint(id: &str) -> bool {
    matches!(
        id,
        "primary" | "foreign" | "unique" | "key" | "constraint" | "index" | "check"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_create_table_columns() {
        let sql = "CREATE TABLE customers (\n  id INT PRIMARY KEY,\n  name VARCHAR(255),\n  balance DECIMAL(10,2),\n  PRIMARY KEY (id)\n);";
        let ents = from_sql(sql);
        assert_eq!(ents.len(), 1);
        assert_eq!(ents[0].name, "customers");
        assert_eq!(ents[0].fields, vec!["id", "name", "balance"]);
    }

    #[test]
    fn handles_if_not_exists_and_quoted_names() {
        let sql = "create table if not exists `orders` ( `order_id` int, total int );";
        let ents = from_sql(sql);
        assert_eq!(ents[0].name, "orders");
        assert_eq!(ents[0].fields, vec!["order_id", "total"]);
    }

    #[test]
    fn entity_lookup_tolerates_case_and_plural() {
        let schema = Schema {
            entities: vec![Entity {
                name: "customers".into(),
                fields: vec!["id".into()],
            }],
        };
        assert!(schema.entity("CUSTOMER").is_some());
        assert!(schema.entity("customers").is_some());
        assert!(schema.entity("product").is_none());
    }
}
