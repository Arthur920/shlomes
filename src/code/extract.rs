//! Per-file extraction: symbols via tree-sitter-tags, dependency edges via a
//! small per-language import query.

use std::collections::HashMap;
use std::ops::Range;
use std::path::Path;

use streaming_iterator::StreamingIterator;
use tree_sitter::{Node, Parser, QueryCursor, Tree};
use tree_sitter_tags::TagsContext;

use crate::code::facts;
use crate::code::lang::{self, Language};
use crate::code::symbol::{DepEdge, Span, Symbol, SymbolKind, Visibility};

/// A reference whose enclosing definition has been resolved intra-file. `from`
/// is the enclosing symbol's `qualified_name` (or the module path for top-level
/// references); `name` is the referenced identifier, resolved to a target symbol
/// globally in [`CodeIndex::build`]. Internal to the extractor.
pub(crate) struct RawRef {
    pub from: String,
    pub name: String,
}

/// Extract symbols, dependency edges, and raw references from one file.
/// Unparseable files and unsupported languages yield empty results rather than
/// erroring.
pub fn extract_file(path: &Path, repo_root: &Path) -> (Vec<Symbol>, Vec<DepEdge>, Vec<RawRef>) {
    let Some(language) = Language::from_path(path) else {
        return (Vec::new(), Vec::new(), Vec::new());
    };
    let Ok(source) = std::fs::read(path) else {
        return (Vec::new(), Vec::new(), Vec::new());
    };
    let module = lang::module_path(path, repo_root);
    let rel = path
        .strip_prefix(repo_root)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string();
    // Parse the file once; the AST is shared by fact extraction and the
    // import-edge query. (tree-sitter-tags does its own internal parse for the
    // tag scan, which it doesn't expose, so that one we can't fold in.)
    let tree = parse_tree(language, &source);
    let root = tree.as_ref().map(|t| t.root_node());
    let (symbols, refs) = symbols_and_refs(language, &source, &module, &rel, root);
    let edges = import_edges(language, &source, &module, root);
    (symbols, edges, refs)
}

/// Parse `source` into a syntax tree, or `None` if the language/parse fails.
fn parse_tree(language: Language, source: &[u8]) -> Option<Tree> {
    let mut parser = Parser::new();
    parser.set_language(&language.ts_language()).ok()?;
    parser.parse(source, None)
}

fn symbols_and_refs(
    language: Language,
    source: &[u8],
    module: &str,
    rel: &str,
    root: Option<Node>,
) -> (Vec<Symbol>, Vec<RawRef>) {
    let Some(config) = language.tags_config_cached() else {
        return (Vec::new(), Vec::new());
    };
    let mut ctx = TagsContext::new();
    let Ok((tags, _)) = ctx.generate_tags(config, source, None) else {
        return (Vec::new(), Vec::new());
    };

    let text = String::from_utf8_lossy(source);
    let lines: Vec<&str> = text.lines().collect();

    // The caller hands us the shared AST. Tags give byte ranges (`tag.range`)
    // but not AST nodes; we resolve each definition's node by byte range below
    // for behavioral facts and its full body span.

    let mut symbols = Vec::new();
    // (full byte range of a definition, its qualified_name) for the innermost-
    // enclosing lookup below. Definition ranges cover the body (the tag node is
    // the whole `function_item`/`class` etc.), unlike `Tag.span` (name only).
    let mut defs: Vec<(Range<usize>, String)> = Vec::new();
    // (referenced name, byte position) — enclosing symbol resolved after the loop.
    let mut ref_sites: Vec<(String, usize)> = Vec::new();

    for tag in tags {
        let Ok(tag) = tag else { continue };
        let name = String::from_utf8_lossy(&source[tag.name_range.clone()]).into_owned();
        if !tag.is_definition {
            ref_sites.push((name, tag.name_range.start));
            continue;
        }
        let qualified_name = format!("{module}::{name}");
        let mut kind = map_kind(config.syntax_type_name(tag.syntax_type_id));
        let start_row = tag.span.start.row;
        let decl_line = lines.get(start_row).map(|l| l.trim().to_string());
        let visibility = classify_visibility(language, decl_line.as_deref().unwrap_or(""), &name);

        defs.push((tag.range.clone(), qualified_name.clone()));

        let span = Span {
            path: rel.to_string(),
            start_line: start_row + 1,
            end_line: tag.span.end.row + 1,
        };
        // Resolve the definition node (covers the body) for facts + body_span.
        let def_node = root.and_then(|r| {
            r.descendant_for_byte_range(tag.range.start, tag.range.end.saturating_sub(1))
        });
        let (body_span, fact_data) = match def_node {
            Some(node) => (
                Span {
                    path: rel.to_string(),
                    start_line: node.start_position().row + 1,
                    end_line: node.end_position().row + 1,
                },
                facts::extract(node, source, language, decl_line.clone()),
            ),
            None => (
                span.clone(),
                facts::extract_signature_only(decl_line.clone()),
            ),
        };

        // tree-sitter-tags collapses enums into the `class` tag kind, so detect
        // them from the AST node and correct the kind. Enum variants are the
        // ground truth for state-diagram grounding.
        let members = match def_node {
            Some(node) if is_enum_node(node.kind()) => {
                kind = SymbolKind::Enum;
                enum_variants(node, source, language)
            }
            Some(node) if language == Language::Python && is_python_enum(node, source) => {
                kind = SymbolKind::Enum;
                python_enum_members(node, source)
            }
            _ => Vec::new(),
        };

        symbols.push(Symbol {
            qualified_name,
            name,
            kind,
            visibility,
            module: module.to_string(),
            span,
            body_span,
            signature: decl_line,
            doc: tag.docs.clone(),
            facts: fact_data,
            calls: Vec::new(),
            members,
        });
    }

    // Ordered call list per definition: walk reference sites in source order and
    // attribute each to its innermost enclosing definition. Preserves order and
    // repetition (unlike the deduped global `ref_edges`) for sequence alignment.
    ref_sites.sort_by_key(|(_, pos)| *pos);
    let mut calls_by_def: HashMap<&str, Vec<String>> = HashMap::new();
    for (name, pos) in &ref_sites {
        if let Some(q) = enclosing(&defs, *pos) {
            calls_by_def.entry(q).or_default().push(name.clone());
        }
    }
    for s in &mut symbols {
        if let Some(c) = calls_by_def.get(s.qualified_name.as_str()) {
            s.calls = c.clone();
        }
    }

    let refs = ref_sites
        .into_iter()
        .map(|(name, pos)| RawRef {
            from: enclosing(&defs, pos).unwrap_or(module).to_string(),
            name,
        })
        .collect();

    (symbols, refs)
}

/// Whether an AST node kind denotes a first-class enum definition (Rust
/// `enum_item`, Java/TS `enum_declaration`). Python enums are class-based and
/// handled separately by [`is_python_enum`].
fn is_enum_node(kind: &str) -> bool {
    matches!(kind, "enum_item" | "enum_declaration")
}

/// Whether a Python `class_definition` derives from an enum base
/// (`Enum`/`IntEnum`/`StrEnum`/`Flag`/`IntFlag`, bare or `enum.`-qualified).
fn is_python_enum(node: tree_sitter::Node, source: &[u8]) -> bool {
    if node.kind() != "class_definition" {
        return false;
    }
    let Some(supers) = node.child_by_field_name("superclasses") else {
        return false;
    };
    let mut cursor = supers.walk();
    for c in supers.children(&mut cursor) {
        let base = c.utf8_text(source).unwrap_or("");
        let leaf = base.rsplit('.').next().unwrap_or(base);
        if matches!(leaf, "Enum" | "IntEnum" | "StrEnum" | "Flag" | "IntFlag") {
            return true;
        }
    }
    false
}

/// Member names of a Python enum: the simple `NAME = value` assignments at the
/// class body's top level. Skips dunder/private names and method definitions.
fn python_enum_members(node: tree_sitter::Node, source: &[u8]) -> Vec<String> {
    let Some(body) = node.child_by_field_name("body") else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut cursor = body.walk();
    for stmt in body.children(&mut cursor) {
        if stmt.kind() != "expression_statement" {
            continue;
        }
        let mut inner = stmt.walk();
        for child in stmt.children(&mut inner) {
            if child.kind() != "assignment" {
                continue;
            }
            let Some(left) = child.child_by_field_name("left") else {
                continue;
            };
            if left.kind() != "identifier" {
                continue;
            }
            if let Ok(name) = left.utf8_text(source) {
                if !name.starts_with('_') {
                    out.push(name.to_string());
                }
            }
        }
    }
    out
}

/// Variant names of an enum definition node. Per-language variant node kinds;
/// languages whose enum shape we don't model (e.g. Python's class-based enums)
/// yield nothing, so state-diagram grounding simply no-ops for them.
fn enum_variants(node: tree_sitter::Node, source: &[u8], lang: Language) -> Vec<String> {
    let kinds: &[&str] = match lang {
        Language::Rust => &["enum_variant"],
        Language::Java => &["enum_constant"],
        _ => &[],
    };
    if kinds.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    collect_variants(node, source, kinds, &mut out);
    out
}

fn collect_variants(node: tree_sitter::Node, source: &[u8], kinds: &[&str], out: &mut Vec<String>) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if kinds.contains(&child.kind()) {
            if let Some(name) = child
                .child_by_field_name("name")
                .and_then(|n| n.utf8_text(source).ok())
            {
                out.push(name.to_string());
            }
        } else {
            collect_variants(child, source, kinds, out);
        }
    }
}

/// The innermost definition whose byte range contains `pos`. Among containing
/// ranges the one with the largest `start` is the most deeply nested.
fn enclosing(defs: &[(Range<usize>, String)], pos: usize) -> Option<&str> {
    defs.iter()
        .filter(|(r, _)| r.start <= pos && pos < r.end)
        .max_by_key(|(r, _)| r.start)
        .map(|(_, q)| q.as_str())
}

fn import_edges(
    language: Language,
    source: &[u8],
    module: &str,
    root: Option<Node>,
) -> Vec<DepEdge> {
    let Some(query) = language.import_query_compiled() else {
        return Vec::new();
    };
    let Some(root) = root else {
        return Vec::new();
    };

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(query, root, source);
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
    let has_word = |w: &str| {
        decl_line
            .split(|c: char| !c.is_alphanumeric())
            .any(|t| t == w)
    };
    match language {
        Language::Rust => {
            if decl_line
                .split_whitespace()
                .any(|w| w == "pub" || w.starts_with("pub("))
            {
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

/// Test-only convenience wrappers that parse internally, so the unit tests can
/// drive the workers from a raw source string. Production code parses once in
/// [`extract_file`] and shares the tree.
#[cfg(test)]
fn extract_symbols_and_refs(
    language: Language,
    source: &[u8],
    module: &str,
    rel: &str,
) -> (Vec<Symbol>, Vec<RawRef>) {
    let tree = parse_tree(language, source);
    symbols_and_refs(
        language,
        source,
        module,
        rel,
        tree.as_ref().map(|t| t.root_node()),
    )
}

#[cfg(test)]
fn extract_edges(language: Language, source: &[u8], module: &str) -> Vec<DepEdge> {
    let tree = parse_tree(language, source);
    import_edges(
        language,
        source,
        module,
        tree.as_ref().map(|t| t.root_node()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vis_of(syms: &[Symbol], name: &str) -> Option<Visibility> {
        syms.iter().find(|s| s.name == name).map(|s| s.visibility)
    }

    fn has_edge(edges: &[DepEdge], target_contains: &str) -> bool {
        edges.iter().any(|e| e.to_module.contains(target_contains))
    }

    fn has_ref(refs: &[RawRef], from: &str, name: &str) -> bool {
        refs.iter().any(|r| r.from == from && r.name == name)
    }

    #[test]
    fn rust_symbols_and_edges() {
        let src = b"pub fn foo() {}\nfn bar() {}\nuse std::fmt;\n";
        let (syms, _refs) = extract_symbols_and_refs(Language::Rust, src, "m", "m.rs");
        assert_eq!(vis_of(&syms, "foo"), Some(Visibility::Public));
        assert_eq!(vis_of(&syms, "bar"), Some(Visibility::Private));
        let edges = extract_edges(Language::Rust, src, "m");
        assert!(has_edge(&edges, "std::fmt"));
    }

    #[test]
    fn python_symbols_and_edges() {
        let src = b"def foo():\n    pass\ndef _bar():\n    pass\nimport os\n";
        let (syms, _refs) = extract_symbols_and_refs(Language::Python, src, "m", "m.py");
        assert_eq!(vis_of(&syms, "foo"), Some(Visibility::Public));
        assert_eq!(vis_of(&syms, "_bar"), Some(Visibility::Private));
        let edges = extract_edges(Language::Python, src, "m");
        assert!(has_edge(&edges, "os"));
    }

    #[test]
    fn javascript_symbols_and_edges() {
        let src = b"export function foo() {}\nfunction bar() {}\nimport x from \"./mod\";\n";
        let (syms, _refs) = extract_symbols_and_refs(Language::JavaScript, src, "m", "m.js");
        assert_eq!(vis_of(&syms, "foo"), Some(Visibility::Public));
        assert_eq!(vis_of(&syms, "bar"), Some(Visibility::Internal));
        let edges = extract_edges(Language::JavaScript, src, "m");
        assert!(has_edge(&edges, "./mod"));
    }

    #[test]
    fn typescript_symbols_and_edges() {
        let src = b"export class A {}\nimport { x } from \"./mod\";\n";
        let (syms, _refs) = extract_symbols_and_refs(Language::TypeScript, src, "m", "m.ts");
        assert_eq!(vis_of(&syms, "A"), Some(Visibility::Public));
        let edges = extract_edges(Language::TypeScript, src, "m");
        assert!(has_edge(&edges, "./mod"));
    }

    #[test]
    fn java_symbols_and_edges() {
        let src = b"import a.b.C;\npublic class A {\n  public void m() {}\n}\n";
        let (syms, _refs) = extract_symbols_and_refs(Language::Java, src, "m", "m.java");
        assert_eq!(vis_of(&syms, "A"), Some(Visibility::Public));
        let edges = extract_edges(Language::Java, src, "m");
        assert!(has_edge(&edges, "a.b.C"));
    }

    #[test]
    fn call_resolves_to_enclosing_caller() {
        // `foo`'s body calls `bar` -> a reference from `m::foo` named `bar`.
        let src = b"fn bar() {}\nfn foo() {\n    bar();\n}\n";
        let (_syms, refs) = extract_symbols_and_refs(Language::Rust, src, "m", "m.rs");
        assert!(has_ref(&refs, "m::foo", "bar"));
    }

    #[test]
    fn recursive_call_is_kept_as_self_ref_site() {
        // The self-call is captured with from == name; the self-edge is dropped
        // later, globally, in `resolve_refs`.
        let src = b"fn foo() {\n    foo();\n}\n";
        let (_syms, refs) = extract_symbols_and_refs(Language::Rust, src, "m", "m.rs");
        assert!(has_ref(&refs, "m::foo", "foo"));
    }

    #[test]
    fn calls_are_captured_in_source_order() {
        // foo's body calls a, then (inside an if) b, then c -> ordered [a, b, c].
        let src = b"fn a() {}\nfn b() {}\nfn c() {}\n\
                    fn foo(x: bool) {\n    a();\n    if x { b(); }\n    c();\n}\n";
        let (syms, _refs) = extract_symbols_and_refs(Language::Rust, src, "m", "m.rs");
        let foo = syms.iter().find(|s| s.name == "foo").unwrap();
        assert_eq!(foo.calls, vec!["a", "b", "c"]);
    }

    #[test]
    fn enum_variants_are_extracted_as_members() {
        let src = b"pub enum State {\n    Idle,\n    Running,\n    Done,\n}\n";
        let (syms, _refs) = extract_symbols_and_refs(Language::Rust, src, "m", "m.rs");
        let en = syms.iter().find(|s| s.name == "State").unwrap();
        assert_eq!(en.members, vec!["Idle", "Running", "Done"]);
    }

    #[test]
    fn python_class_enum_members_are_extracted() {
        let src = b"from enum import Enum\n\
                    class State(Enum):\n    IDLE = 1\n    RUNNING = 2\n    DONE = 3\n";
        let (syms, _refs) = extract_symbols_and_refs(Language::Python, src, "m", "m.py");
        let en = syms.iter().find(|s| s.name == "State").unwrap();
        assert_eq!(en.kind, SymbolKind::Enum);
        assert_eq!(en.members, vec!["IDLE", "RUNNING", "DONE"]);
    }

    #[test]
    fn python_plain_class_is_not_an_enum() {
        let src = b"class Plain:\n    X = 1\n    def m(self):\n        pass\n";
        let (syms, _refs) = extract_symbols_and_refs(Language::Python, src, "m", "m.py");
        let cls = syms.iter().find(|s| s.name == "Plain").unwrap();
        assert_ne!(cls.kind, SymbolKind::Enum);
        assert!(cls.members.is_empty());
    }

    #[test]
    fn top_level_reference_falls_back_to_module() {
        // A call outside any definition has the module path as its `from`.
        let src = b"fn foo() {}\nconst N: usize = foo();\n";
        let (_syms, refs) = extract_symbols_and_refs(Language::Rust, src, "m", "m.rs");
        assert!(refs.iter().any(|r| r.name == "foo" && r.from == "m"));
    }
}
