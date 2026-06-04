//! Per-file extraction: symbols via tree-sitter-tags, dependency edges via a
//! small per-language import query.

use std::path::Path;

use streaming_iterator::StreamingIterator;
use tree_sitter::{Parser, Query, QueryCursor};
use tree_sitter_tags::TagsContext;

use crate::code::lang::{self, Language};
use crate::code::symbol::{DepEdge, Facts, Span, Symbol, SymbolKind, Visibility};

/// Extract symbols and dependency edges from one file. Unparseable files and
/// unsupported languages yield empty results rather than erroring.
pub fn extract_file(path: &Path, repo_root: &Path) -> (Vec<Symbol>, Vec<DepEdge>) {
    let Some(language) = Language::from_path(path) else {
        return (Vec::new(), Vec::new());
    };
    let Ok(source) = std::fs::read(path) else {
        return (Vec::new(), Vec::new());
    };
    let module = lang::module_path(path, repo_root);
    let rel = path
        .strip_prefix(repo_root)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string();
    let symbols = extract_symbols(language, &source, &module, &rel);
    let edges = extract_edges(language, &source, &module);
    (symbols, edges)
}

fn extract_symbols(language: Language, source: &[u8], module: &str, rel: &str) -> Vec<Symbol> {
    let Ok(config) = language.tags_config() else {
        return Vec::new();
    };
    let mut ctx = TagsContext::new();
    let Ok((tags, _)) = ctx.generate_tags(&config, source, None) else {
        return Vec::new();
    };

    let text = String::from_utf8_lossy(source);
    let lines: Vec<&str> = text.lines().collect();

    let mut out = Vec::new();
    for tag in tags {
        let Ok(tag) = tag else { continue };
        if !tag.is_definition {
            continue;
        }
        let name = String::from_utf8_lossy(&source[tag.name_range.clone()]).into_owned();
        let kind = map_kind(config.syntax_type_name(tag.syntax_type_id));
        let start_row = tag.span.start.row;
        let decl_line = lines.get(start_row).map(|l| l.trim().to_string());
        let visibility =
            classify_visibility(language, decl_line.as_deref().unwrap_or(""), &name);

        out.push(Symbol {
            qualified_name: format!("{module}::{name}"),
            name,
            kind,
            visibility,
            module: module.to_string(),
            span: Span {
                path: rel.to_string(),
                start_line: start_row + 1,
                end_line: tag.span.end.row + 1,
            },
            signature: decl_line,
            doc: tag.docs.clone(),
            // Behavioral-fact population is deferred to the drift-fingerprint
            // consumer, which needs a per-symbol AST walk. Plumbing is in place.
            facts: Facts::default(),
        });
    }
    out
}

fn extract_edges(language: Language, source: &[u8], module: &str) -> Vec<DepEdge> {
    let ts_lang = language.ts_language();
    let Ok(query) = Query::new(&ts_lang, language.import_query()) else {
        return Vec::new();
    };
    let mut parser = Parser::new();
    if parser.set_language(&ts_lang).is_err() {
        return Vec::new();
    }
    let Some(tree) = parser.parse(source, None) else {
        return Vec::new();
    };

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, tree.root_node(), source);
    let mut edges = Vec::new();
    while let Some(m) = matches.next() {
        for cap in m.captures {
            let raw = cap.node.utf8_text(source).unwrap_or("");
            let target = normalize_import(raw);
            if !target.is_empty() {
                edges.push(DepEdge {
                    from_module: module.to_string(),
                    to_module: target,
                });
            }
        }
    }
    edges
}

fn map_kind(name: &str) -> SymbolKind {
    match name {
        "function" => SymbolKind::Function,
        "method" | "constructor" => SymbolKind::Method,
        "struct" => SymbolKind::Struct,
        "class" => SymbolKind::Class,
        "enum" => SymbolKind::Enum,
        "trait" => SymbolKind::Trait,
        "interface" => SymbolKind::Interface,
        "module" => SymbolKind::Module,
        "constant" => SymbolKind::Constant,
        "field" | "property" | "member" => SymbolKind::Field,
        other => SymbolKind::Other(other.to_string()),
    }
}

fn classify_visibility(language: Language, decl_line: &str, name: &str) -> Visibility {
    let has_word = |w: &str| decl_line.split(|c: char| !c.is_alphanumeric()).any(|t| t == w);
    match language {
        Language::Rust => {
            if decl_line.split_whitespace().any(|w| w == "pub" || w.starts_with("pub(")) {
                Visibility::Public
            } else {
                Visibility::Private
            }
        }
        Language::Python => {
            if name.starts_with('_') {
                Visibility::Private
            } else {
                Visibility::Public
            }
        }
        Language::JavaScript | Language::TypeScript | Language::Tsx => {
            if has_word("export") {
                Visibility::Public
            } else {
                Visibility::Internal
            }
        }
        Language::Java => {
            if has_word("public") {
                Visibility::Public
            } else if has_word("private") {
                Visibility::Private
            } else {
                Visibility::Internal
            }
        }
    }
}

fn normalize_import(raw: &str) -> String {
    raw.trim()
        .trim_matches(|c| c == '"' || c == '\'' || c == '`')
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vis_of<'a>(syms: &'a [Symbol], name: &str) -> Option<Visibility> {
        syms.iter().find(|s| s.name == name).map(|s| s.visibility)
    }

    fn has_edge(edges: &[DepEdge], target_contains: &str) -> bool {
        edges.iter().any(|e| e.to_module.contains(target_contains))
    }

    #[test]
    fn rust_symbols_and_edges() {
        let src = b"pub fn foo() {}\nfn bar() {}\nuse std::fmt;\n";
        let syms = extract_symbols(Language::Rust, src, "m", "m.rs");
        assert_eq!(vis_of(&syms, "foo"), Some(Visibility::Public));
        assert_eq!(vis_of(&syms, "bar"), Some(Visibility::Private));
        let edges = extract_edges(Language::Rust, src, "m");
        assert!(has_edge(&edges, "std::fmt"));
    }

    #[test]
    fn python_symbols_and_edges() {
        let src = b"def foo():\n    pass\ndef _bar():\n    pass\nimport os\n";
        let syms = extract_symbols(Language::Python, src, "m", "m.py");
        assert_eq!(vis_of(&syms, "foo"), Some(Visibility::Public));
        assert_eq!(vis_of(&syms, "_bar"), Some(Visibility::Private));
        let edges = extract_edges(Language::Python, src, "m");
        assert!(has_edge(&edges, "os"));
    }

    #[test]
    fn javascript_symbols_and_edges() {
        let src = b"export function foo() {}\nfunction bar() {}\nimport x from \"./mod\";\n";
        let syms = extract_symbols(Language::JavaScript, src, "m", "m.js");
        assert_eq!(vis_of(&syms, "foo"), Some(Visibility::Public));
        assert_eq!(vis_of(&syms, "bar"), Some(Visibility::Internal));
        let edges = extract_edges(Language::JavaScript, src, "m");
        assert!(has_edge(&edges, "./mod"));
    }

    #[test]
    fn typescript_symbols_and_edges() {
        let src = b"export class A {}\nimport { x } from \"./mod\";\n";
        let syms = extract_symbols(Language::TypeScript, src, "m", "m.ts");
        assert_eq!(vis_of(&syms, "A"), Some(Visibility::Public));
        let edges = extract_edges(Language::TypeScript, src, "m");
        assert!(has_edge(&edges, "./mod"));
    }

    #[test]
    fn java_symbols_and_edges() {
        let src = b"import a.b.C;\npublic class A {\n  public void m() {}\n}\n";
        let syms = extract_symbols(Language::Java, src, "m", "m.java");
        assert_eq!(vis_of(&syms, "A"), Some(Visibility::Public));
        let edges = extract_edges(Language::Java, src, "m");
        assert!(has_edge(&edges, "a.b.C"));
    }
}
