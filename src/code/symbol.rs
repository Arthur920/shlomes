//! Types produced by the code extractor.
//!
//! These are the substrate every designed feature reads from — coverage gaps,
//! diagram edge-diff, architecture rules, drift provenance/fingerprints.

use serde::Serialize;

/// What kind of definition a [`Symbol`] is. Tag kind names differ per grammar;
/// unrecognized ones are preserved in `Other` rather than dropped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SymbolKind {
    Function,
    Method,
    Struct,
    Class,
    Enum,
    Trait,
    Interface,
    Module,
    Constant,
    Field,
    Other(String),
}

/// Best-effort visibility, used by coverage-gaps to scope the "documentable
/// surface" (public surface matters; private internals don't).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Visibility {
    Public,
    Internal,
    Private,
}

/// Where a symbol lives. Mirrors the `path` + line shape used by
/// `retrieve::Hit` and `findings::Finding.code_refs`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Span {
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
}

/// First-cut behavioral facts. Today just the literal constants found in a
/// symbol's body; extended later (control-flow predicates, return shape) when
/// the drift-fingerprint consumer is built. Combined with [`Symbol::signature`]
/// it forms the material the drift fingerprint will hash.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Facts {
    pub constants: Vec<String>,
}

/// A code definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Symbol {
    /// module path + enclosing scope + name, best-effort (no full scope
    /// resolution yet). Symbol identity for coverage-gaps is `(module, name, kind)`.
    pub qualified_name: String,
    pub name: String,
    pub kind: SymbolKind,
    pub visibility: Visibility,
    /// file-derived module path (e.g. `src/code/symbol` for this file).
    pub module: String,
    pub span: Span,
    pub signature: Option<String>,
    /// leading-comment documentation captured by the tags query, if any.
    pub doc: Option<String>,
    pub facts: Facts,
}

/// A module-level dependency, derived from import/use statements.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DepEdge {
    pub from_module: String,
    pub to_module: String,
}
