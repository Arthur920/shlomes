//! Types produced by the code extractor.
//!
//! These are the substrate every designed feature reads from — coverage gaps,
//! diagram edge-diff, architecture rules, drift provenance/fingerprints.

use serde::{Deserialize, Serialize};

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

impl Span {
    /// An empty span — the `body_span` fallback for synthetic symbols and the
    /// serde default if a `Symbol` is ever deserialized.
    #[allow(dead_code)]
    pub fn zero() -> Span {
        Span {
            path: String::new(),
            start_line: 0,
            end_line: 0,
        }
    }
}

/// Behavioral facts — the deterministic, model-free fingerprint of a symbol's
/// meaning. The drift layer hashes these (see `crate::code::facts::facts_hash`)
/// and flags a claim when the hash moves from its committed baseline, catching
/// small-token/high-semantic edits (`3 -> 5`, `if -> if !`) while ignoring
/// renames. Multi-valued fields are sorted + deduped so the hash is
/// order-independent.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Facts {
    /// Literal constants in the symbol's body (numbers, strings, bools).
    pub constants: Vec<String>,
    /// The declaration line (name, params, types), normalized.
    #[serde(default)]
    pub signature: Option<String>,
    /// Control-flow condition texts (`if`/`while`/`match`/`switch`), normalized.
    #[serde(default)]
    pub predicates: Vec<String>,
    /// Declared return type / shape, if the grammar exposes one.
    #[serde(default)]
    pub return_shape: Option<String>,
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
    /// Name range (identifier position) — used by reports and coverage.
    pub span: Span,
    /// Full definition range (covers the body). Drift maps git hunks to symbols
    /// by overlapping changed line ranges against this. Defaults to `span` for
    /// synthetic symbols that don't set it.
    #[serde(default = "Span::zero")]
    pub body_span: Span,
    pub signature: Option<String>,
    /// leading-comment documentation captured by the tags query, if any.
    pub doc: Option<String>,
    pub facts: Facts,
    /// Callee names this definition references, in **source order** (control flow
    /// flattened, innermost-attributed). The ordered substrate sequence-diagram
    /// alignment compares against (`docs/diagram-coherence.md`, "Ordered
    /// diagrams"); unlike `ref_edges` it preserves order and repetition.
    #[serde(default)]
    pub calls: Vec<String>,
    /// For an `Enum`, its variant names — the ground truth state-diagram grounding
    /// checks against. Empty for non-enums (and for languages whose enum-variant
    /// shape we don't extract yet).
    #[serde(default)]
    pub members: Vec<String>,
}

/// A module-level dependency, derived from import/use statements.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DepEdge {
    pub from_module: String,
    pub to_module: String,
}

/// A symbol-level reference: the enclosing definition `from_symbol` references
/// the definition `to_symbol` (a call, impl, or type use). Both endpoints are
/// `qualified_name`s. Targets are resolved by name (over-approximate on
/// collisions), so this never under-counts a symbol's callers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RefEdge {
    pub from_symbol: String,
    pub to_symbol: String,
}
