//! Behavioral-fact extraction: a per-symbol AST walk that distills the parts of
//! a definition that carry meaning — literal constants, control-flow condition
//! texts, and the declared return shape. Hashing these (not the source text, and
//! not an embedding) gives a fingerprint that moves on semantic edits (`3 -> 5`,
//! `if -> if !`) but not on renames or reformatting. See `docs/drift-detection.md`
//! (Mechanism 2).

use tree_sitter::Node;

use crate::claim::fnv1a;
use crate::code::lang::Language;
use crate::code::symbol::Facts;

/// Walk a definition subtree and collect its behavioral facts (constants +
/// predicates + return shape). `signature` is supplied by the caller (the
/// normalized declaration line) and folded into the same `Facts`.
pub fn extract(node: Node, source: &[u8], lang: Language, signature: Option<String>) -> Facts {
    let mut constants = Vec::new();
    let mut predicates = Vec::new();

    let return_shape = node
        .child_by_field_name("return_type")
        .or_else(|| node.child_by_field_name("type"))
        .and_then(|n| text(n, source))
        .map(|t| normalize(&t));

    walk(node, source, lang, &mut constants, &mut predicates);

    constants.sort();
    constants.dedup();
    predicates.sort();
    predicates.dedup();

    Facts {
        constants,
        signature: signature.map(|s| normalize(&s)),
        predicates,
        return_shape,
    }
}

/// Facts with only the signature populated — the fallback when the file failed
/// to parse into a tree, so no AST walk is possible.
pub fn extract_signature_only(signature: Option<String>) -> Facts {
    Facts {
        signature: signature.map(|s| normalize(&s)),
        ..Default::default()
    }
}

/// FNV-1a fingerprint over the canonical, order-independent rendering of a
/// symbol's facts. Stable across machines and toolchains (committed ledger).
pub fn facts_hash(f: &Facts) -> u64 {
    let canonical = format!(
        "C:{}|S:{}|P:{}|R:{}",
        f.constants.join(","),
        f.signature.as_deref().unwrap_or(""),
        f.predicates.join(","),
        f.return_shape.as_deref().unwrap_or(""),
    );
    fnv1a(&canonical)
}

/// Recursively gather constants and predicate texts from the subtree.
fn walk(
    node: Node,
    source: &[u8],
    lang: Language,
    constants: &mut Vec<String>,
    predicates: &mut Vec<String>,
) {
    let kind = node.kind();
    if is_literal(kind) {
        if let Some(t) = text(node, source) {
            constants.push(normalize(&t));
        }
    }
    if is_conditional(kind) {
        if let Some(cond) = node
            .child_by_field_name("condition")
            .or_else(|| node.child_by_field_name("value"))
            .and_then(|n| text(n, source))
        {
            predicates.push(normalize(&cond));
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        // Don't descend into nested definitions — their facts belong to them.
        if child.id() != node.id() && is_nested_definition(child.kind(), lang) {
            continue;
        }
        walk(child, source, lang, constants, predicates);
    }
}

/// Literal node kinds across the supported grammars. Matched by suffix/keyword
/// so we stay robust to per-grammar naming (`integer_literal`, `integer`,
/// `number`, `decimal_integer_literal`, …).
fn is_literal(kind: &str) -> bool {
    kind.ends_with("_literal")
        || matches!(
            kind,
            "integer"
                | "float"
                | "number"
                | "string"
                | "string_literal"
                | "true"
                | "false"
                | "none"
                | "null"
                | "boolean"
                | "char"
                | "template_string"
        )
}

/// Control-flow nodes whose condition text is a behavioral predicate.
fn is_conditional(kind: &str) -> bool {
    matches!(
        kind,
        "if_expression"
            | "if_statement"
            | "while_expression"
            | "while_statement"
            | "match_expression"
            | "switch_expression"
            | "switch_statement"
            | "conditional_expression"
            | "ternary_expression"
            | "elif_clause"
    )
}

/// Nested definitions that should not contribute their facts to the enclosing
/// symbol (the indexer records them as their own symbols).
fn is_nested_definition(kind: &str, _lang: Language) -> bool {
    matches!(
        kind,
        "function_item"
            | "function_definition"
            | "function_declaration"
            | "method_definition"
            | "method_declaration"
            | "class_declaration"
            | "class_definition"
            | "struct_item"
            | "impl_item"
            | "trait_item"
    )
}

fn text(node: Node, source: &[u8]) -> Option<String> {
    node.utf8_text(source).ok().map(|s| s.to_string())
}

/// Collapse internal whitespace runs and trim — so reflowing a condition or
/// declaration across lines does not change the fingerprint.
fn normalize(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tree_sitter::Parser;

    fn root_fn<'t>(tree: &'t tree_sitter::Tree, _src: &[u8]) -> Node<'t> {
        // The first function-ish definition under the root.
        let root = tree.root_node();
        let mut cursor = root.walk();
        for child in root.children(&mut cursor) {
            if is_nested_definition(child.kind(), Language::Rust) {
                return child;
            }
        }
        root
    }

    fn facts_of_rust(src: &str) -> Facts {
        let mut parser = Parser::new();
        parser
            .set_language(&Language::Rust.ts_language())
            .unwrap();
        let tree = parser.parse(src, None).unwrap();
        let node = root_fn(&tree, src.as_bytes());
        extract(node, src.as_bytes(), Language::Rust, Some("fn f()".into()))
    }

    #[test]
    fn captures_constants_and_is_rename_stable() {
        let a = facts_of_rust("fn f() {\n    let retries = 3;\n    let name = \"db\";\n}\n");
        assert!(a.constants.iter().any(|c| c == "3"));
        assert!(a.constants.iter().any(|c| c.contains("db")));

        // Renaming the symbol (different signature arg name) keeps body facts.
        let b = facts_of_rust("fn f() {\n    let attempts = 3;\n    let name = \"db\";\n}\n");
        assert_eq!(a.constants, b.constants);
    }

    #[test]
    fn constant_edit_changes_hash() {
        let a = facts_of_rust("fn f() {\n    let retries = 3;\n}\n");
        let b = facts_of_rust("fn f() {\n    let retries = 5;\n}\n");
        assert_ne!(facts_hash(&a), facts_hash(&b));
    }

    #[test]
    fn captures_predicate() {
        let f = facts_of_rust("fn f(x: i32) {\n    if x > 10 {\n        return;\n    }\n}\n");
        assert!(f.predicates.iter().any(|p| p.contains("x > 10")), "{f:?}");
    }
}
