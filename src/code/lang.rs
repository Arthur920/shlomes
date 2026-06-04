//! Language detection, per-language tree-sitter config, and the shared
//! code-file walker (hoisted here so the `ml` retrieval path reuses it).

use std::path::{Path, PathBuf};

use tree_sitter::Language as TsLanguage;
use tree_sitter_tags::TagsConfiguration;
use walkdir::WalkDir;

/// Source languages the extractor understands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Tsx,
    Java,
}

/// File extensions we treat as code (also gates the Layer-2 chunker).
pub const CODE_EXTS: &[&str] = &[
    "rs", "py", "ts", "tsx", "js", "jsx", "mjs", "cjs", "java", "go", "rb", "c", "h", "cpp", "hpp",
    "cc", "cs", "php", "swift", "kt", "scala", "sh", "toml", "yaml", "yml",
];

/// Directories never worth walking.
const SKIP_DIRS: &[&str] = &[".git", "target", ".shlomes", "node_modules", ".venv"];

impl Language {
    /// Map a file extension to a language the extractor can parse, if any.
    pub fn from_path(path: &Path) -> Option<Language> {
        match path.extension().and_then(|e| e.to_str())? {
            "rs" => Some(Language::Rust),
            "py" | "pyi" => Some(Language::Python),
            "js" | "jsx" | "mjs" | "cjs" => Some(Language::JavaScript),
            "ts" | "mts" | "cts" => Some(Language::TypeScript),
            "tsx" => Some(Language::Tsx),
            "java" => Some(Language::Java),
            _ => None,
        }
    }

    pub fn ts_language(self) -> TsLanguage {
        match self {
            Language::Rust => tree_sitter_rust::LANGUAGE.into(),
            Language::Python => tree_sitter_python::LANGUAGE.into(),
            Language::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
            Language::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            Language::Tsx => tree_sitter_typescript::LANGUAGE_TSX.into(),
            Language::Java => tree_sitter_java::LANGUAGE.into(),
        }
    }

    /// Build the tags configuration used to extract definitions/references.
    ///
    /// The TypeScript grammar's `TAGS_QUERY` only covers TS-specific nodes
    /// (signatures, interfaces, abstract classes); plain `class`/`function`
    /// declarations are inherited from JavaScript, so for TS/TSX we concatenate
    /// the JS tags query (the TS grammar is a JS superset, so the node types
    /// resolve).
    pub fn tags_config(self) -> Result<TagsConfiguration, tree_sitter_tags::Error> {
        let query: String = match self {
            Language::Rust => tree_sitter_rust::TAGS_QUERY.to_string(),
            Language::Python => tree_sitter_python::TAGS_QUERY.to_string(),
            Language::JavaScript => tree_sitter_javascript::TAGS_QUERY.to_string(),
            Language::TypeScript | Language::Tsx => format!(
                "{}\n{}",
                tree_sitter_javascript::TAGS_QUERY,
                tree_sitter_typescript::TAGS_QUERY
            ),
            Language::Java => tree_sitter_java::TAGS_QUERY.to_string(),
        };
        TagsConfiguration::new(self.ts_language(), &query, "")
    }

    /// A tree-sitter query that captures imported module paths, for dep edges.
    /// Node names are grammar-specific; verified by the per-language tests.
    pub fn import_query(self) -> &'static str {
        match self {
            Language::Rust => "(use_declaration argument: (_) @import)",
            Language::Python => {
                "[(import_statement name: (dotted_name) @import)
                  (import_from_statement module_name: (dotted_name) @import)]"
            }
            Language::JavaScript | Language::TypeScript | Language::Tsx => {
                "(import_statement source: (string) @import)"
            }
            Language::Java => "(import_declaration (scoped_identifier) @import)",
        }
    }
}

/// True if a path is a source file we parse/chunk.
pub fn is_code(p: &Path) -> bool {
    p.extension()
        .and_then(|s| s.to_str())
        .map(|e| CODE_EXTS.contains(&e))
        .unwrap_or(false)
}

/// Walk `repo_root` and return every code file, skipping vendored/build dirs.
pub fn code_files(repo_root: &Path) -> Vec<PathBuf> {
    WalkDir::new(repo_root)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !SKIP_DIRS.contains(&name.as_ref())
        })
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .filter(|p| is_code(p))
        .collect()
}

/// Derive a module path from a file path, relative to the repo root:
/// `src/code/symbol.rs` -> `src/code/symbol`. Used as the symbol's `module`
/// and the endpoints of dependency edges.
pub fn module_path(path: &Path, repo_root: &Path) -> String {
    let rel = path.strip_prefix(repo_root).unwrap_or(path);
    let without_ext = rel.with_extension("");
    without_ext.to_string_lossy().replace('\\', "/")
}
