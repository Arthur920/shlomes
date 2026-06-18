//! Grounding a diagram box label to a real module — the deterministic exact tier
//! plus a dependency-free *fuzzy* tier for conceptual labels.
//!
//! Real architecture diagrams label boxes conceptually ("Auth Service", "API
//! Gateway"), which never ground to a module *path* by exact/prefix/suffix match.
//! That is why the high-value phantom-edge and missing-arrow checks fired zero
//! times across a 10-repo wild audit. Fuzzy resolution bridges that gap by token
//! overlap — but it is admitted only when it is **unique and significant** so
//! Layer 1 keeps under-reporting rather than guessing. Ambiguity resolves to
//! [`Resolution::None`].

use std::collections::HashSet;

use crate::rules::matches;

/// How a box label grounds to the real module graph.
///
/// The edge checks (phantom, missing-arrow) require a box to name **exactly one**
/// module: `Exact` and `Fuzzy` each carry that unique module path. A label that
/// matches several modules is `Ambiguous` — enough to know the box is *not* a
/// stale reference, but too imprecise to assert an edge about (so it can't drive a
/// phantom/missing-arrow false positive). This is what the wild audit demanded:
/// `matches` is segment/suffix-based, so a short label like `auth` otherwise
/// grounds to every `…/auth/…` module across a monorepo and cascades into noise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Resolution {
    /// Deterministically grounds to one module (its unique path).
    Exact(String),
    /// Fuzzily grounds to one module (its unique path).
    Fuzzy(String),
    /// Grounds to more than one module — real, but no single identity.
    Ambiguous,
    None,
}

impl Resolution {
    /// True if the box names real code at all (any tier) — so it is not stale.
    pub(super) fn grounds(&self) -> bool {
        !matches!(self, Resolution::None)
    }

    /// The single module path, only when the box grounds to exactly one module by
    /// the deterministic exact tier. Used for edge checks (phantom, missing-arrow).
    pub(super) fn exact_module(&self) -> Option<&str> {
        match self {
            Resolution::Exact(s) => Some(s),
            _ => None,
        }
    }
}

/// Strip a trailing source-file extension from a diagram box label. The code
/// index stores modules extension-stripped (`foo/bar`), but authors routinely
/// draw boxes as `foo/bar.ts` or `auth.py`; without this, a box that names a
/// real file fails to ground and is wrongly reported stale.
pub(super) fn ground_label(label: &str) -> &str {
    const EXTS: &[&str] = &[
        ".rs", ".py", ".pyi", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".java",
    ];
    for ext in EXTS {
        if let Some(stripped) = label.strip_suffix(ext) {
            return stripped;
        }
    }
    label
}

/// Diagram filler that carries no module identity. A closed set, kept small and
/// conservative: dropping these lets "Auth Service" align with `src/auth` without
/// every `…/service` module becoming a candidate.
fn is_stopword(t: &str) -> bool {
    matches!(
        t,
        "service"
            | "module"
            | "component"
            | "layer"
            | "bridge"
            | "client"
            | "server"
            | "system"
            | "the"
            | "a"
            | "an"
            | "of"
            | "and"
            | "to"
    )
}

/// Normalize a trailing plural `s` (`services` → `service`) so plural/singular
/// label and module spellings align. Left untouched for short tokens.
fn singular(t: &str) -> String {
    if t.len() > 3 && t.ends_with('s') && !t.ends_with("ss") {
        t[..t.len() - 1].to_string()
    } else {
        t.to_string()
    }
}

/// Split a token on camelCase boundaries (`AuthService` → `auth`, `service`;
/// `maxRetries` → `max`, `retries`), mirroring the case-shape idiom in
/// `constswap::is_code_shaped`.
fn split_camel(s: &str, out: &mut Vec<String>) {
    let chars: Vec<char> = s.chars().collect();
    let mut cur = String::new();
    for (i, &c) in chars.iter().enumerate() {
        if c.is_ascii_uppercase() && i > 0 {
            let prev = chars[i - 1];
            let next_lower = chars.get(i + 1).is_some_and(|n| n.is_ascii_lowercase());
            // boundary at lower→Upper (`authService`) or Upper→Upper→lower (`APIGateway`)
            if (prev.is_ascii_lowercase() || (prev.is_ascii_uppercase() && next_lower))
                && !cur.is_empty()
            {
                out.push(std::mem::take(&mut cur));
            }
        }
        cur.push(c);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
}

/// Tokenize a label into lowercase, singularized, stopword-free tokens. Splits on
/// whitespace, path/namespace separators, and camelCase boundaries.
pub(super) fn label_tokens(label: &str) -> Vec<String> {
    let g = ground_label(label);
    let mut raw = Vec::new();
    for part in g.split(|c: char| c.is_whitespace() || matches!(c, '-' | '_' | '/' | '.' | ':')) {
        if !part.is_empty() {
            split_camel(part, &mut raw);
        }
    }
    raw.into_iter()
        .map(|t| singular(&t.to_ascii_lowercase()))
        .filter(|t| !t.is_empty() && !is_stopword(t))
        .collect()
}

/// Precompute each real module path's token set once per run.
pub(super) fn module_token_index(modules: &HashSet<String>) -> Vec<(String, HashSet<String>)> {
    modules
        .iter()
        .map(|m| (m.clone(), label_tokens(m).into_iter().collect()))
        .collect()
}

/// Significant tokens are those long enough to discriminate; a lone generic token
/// like `db`/`ui` carries no fuzzy weight (it grounds only via the exact tier).
fn significant(tokens: &[String]) -> Vec<&str> {
    tokens
        .iter()
        .filter(|t| t.len() >= 3)
        .map(|s| s.as_str())
        .collect()
}

/// Ground a box label: exact tier first, then a unique-and-significant fuzzy tier.
pub(super) fn resolve(
    label: &str,
    modules: &HashSet<String>,
    module_tokens: &[(String, HashSet<String>)],
) -> Resolution {
    let g = ground_label(label);
    // Exact path identity wins outright, even when submodules share the prefix.
    if modules.contains(g) {
        return Resolution::Exact(g.to_string());
    }
    // Otherwise count modules this label matches (segment/suffix/prefix). Unique →
    // Exact identity; several → Ambiguous (real but no single edge identity).
    let mut hits = modules.iter().filter(|m| matches(m, g));
    if let Some(first) = hits.next() {
        return match hits.next() {
            None => Resolution::Exact(first.clone()),
            Some(_) => Resolution::Ambiguous,
        };
    }

    let tokens = label_tokens(label);
    let sig = significant(&tokens);
    if sig.is_empty() {
        return Resolution::None; // nothing discriminating to match on
    }

    // Modules whose token set contains every significant label token.
    let mut candidates: Vec<&str> = module_tokens
        .iter()
        .filter(|(_, mtoks)| sig.iter().all(|t| mtoks.contains(*t)))
        .map(|(m, _)| m.as_str())
        .collect();
    candidates.sort_unstable();
    candidates.dedup();

    match candidates.as_slice() {
        [one] => Resolution::Fuzzy(one.to_string()),
        [] => Resolution::None,
        many => {
            // Multiple matches are admissible only if they share one subtree root
            // that is itself a candidate (e.g. `src/auth` is an ancestor of every
            // `src/auth/*` match). Otherwise the label is ambiguous → drop it.
            match many
                .iter()
                .find(|root| many.iter().all(|m| is_subtree(m, root)))
            {
                Some(root) => Resolution::Fuzzy(root.to_string()),
                None => Resolution::None,
            }
        }
    }
}

/// True if `module` is `root` itself or lives under it (`root/…`).
fn is_subtree(module: &str, root: &str) -> bool {
    module == root || module.starts_with(&format!("{root}/"))
}
