//! Model-free evidence selection for the Layer 3 judge.
//!
//! Layer 2's embedding retrieval re-discovers, by cosine similarity, code that
//! Layer 1 has often *already* resolved: a claim's backtick tokens are grounded
//! to exact symbols/modules during extraction ([`crate::judge::candidate_claims`]
//! -> `ground_claim`). This module turns that grounding into the NLI premise
//! directly — read the resolved symbol bodies — and falls back to a lexical
//! (idf-weighted) match over the public symbol table when a claim grounded to
//! nothing. No embedding model, no whole-corpus pass.
//!
//! Embedding stays available behind `STALEGUARD_EMBED_RETRIEVE=1` for claims whose
//! relevant code is genuinely semantic (named by behaviour, not by identifier).

use std::collections::HashMap;
use std::path::Path;

use crate::claim::Provenance;
use crate::code::symbol::Visibility;
use crate::code::{CodeIndex, SymbolLookup};

/// One premise chunk for the judge: code text plus where it came from.
pub struct Evidence {
    pub text: String,
    pub path: String,
    pub start_line: usize,
    pub score: f32,
}

/// Cap on body lines pulled per symbol. The NLI cross-encoder truncates the pair
/// to its token window anyway; this keeps tokenization bounded for huge bodies.
const MAX_BODY_LINES: usize = 50;
/// Below this lexical score a fallback match is too weak to be evidence.
const MIN_LEXICAL_SCORE: f32 = 0.5;

/// Lazily-read file lines, shared across claims (`None` = unreadable).
pub type FileCache = HashMap<String, Option<Vec<String>>>;

/// Gather evidence for one claim: grounded symbols/modules first, then a lexical
/// fallback over the public symbol table if nothing grounded. Returns up to `k`,
/// best first.
#[allow(clippy::too_many_arguments)] // cohesive evidence-gathering inputs
pub fn gather(
    claim_text: &str,
    prov: &Provenance,
    index: &CodeIndex,
    lookup: &SymbolLookup,
    lexicon: &Lexicon,
    root: &Path,
    k: usize,
    files: &mut FileCache,
) -> Vec<Evidence> {
    let mut out: Vec<Evidence> = Vec::new();
    let mut seen: std::collections::HashSet<(String, usize)> = std::collections::HashSet::new();

    // 1. Symbols the claim grounded to (exact, preferred).
    for qn in &prov.symbols {
        for sym in lookup.by_qualified(qn) {
            if let Some(ev) = symbol_evidence(sym, root, 3.0, files) {
                if seen.insert((ev.path.clone(), ev.start_line)) {
                    out.push(ev);
                }
            }
        }
    }

    // 2. Modules the claim grounded to: their top-level public symbols.
    for m in &prov.modules {
        for sym in lookup.public_in_module(m).take(k) {
            if let Some(ev) = symbol_evidence(sym, root, 2.0, files) {
                if seen.insert((ev.path.clone(), ev.start_line)) {
                    out.push(ev);
                }
            }
        }
    }

    // 3. Lexical fallback only when grounding produced nothing.
    if out.is_empty() {
        for (idx, score) in lexicon.top(claim_text, k) {
            if let Some(sym) = index.symbols.get(idx) {
                if let Some(ev) = symbol_evidence(sym, root, score, files) {
                    if seen.insert((ev.path.clone(), ev.start_line)) {
                        out.push(ev);
                    }
                }
            }
        }
    }

    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out.truncate(k);
    out
}

/// Read a symbol's body via its `body_span`, capped to [`MAX_BODY_LINES`].
fn symbol_evidence(
    sym: &crate::code::symbol::Symbol,
    root: &Path,
    score: f32,
    files: &mut FileCache,
) -> Option<Evidence> {
    let span = &sym.body_span;
    if span.path.is_empty() || span.start_line == 0 || span.end_line < span.start_line {
        return None;
    }
    let lines = files
        .entry(span.path.clone())
        .or_insert_with(|| {
            std::fs::read_to_string(root.join(&span.path))
                .ok()
                .map(|c| c.lines().map(str::to_string).collect())
        })
        .as_ref()?;

    let s = span.start_line.saturating_sub(1);
    let e = span.end_line.min(lines.len());
    if s >= e {
        return None;
    }
    let end = e.min(s + MAX_BODY_LINES);
    let text = lines[s..end].join("\n");
    if text.trim().is_empty() {
        return None;
    }
    Some(Evidence {
        text,
        path: span.path.clone(),
        start_line: span.start_line,
        score,
    })
}

// ---- lexical fallback (idf-weighted symbol-table match) --------------------

/// Pre-tokenized, idf-weighted view of the public symbol table for cheap lexical
/// retrieval. Built once per run; scoring a claim is a set intersection.
pub struct Lexicon {
    /// (symbol index in `index.symbols`, its name tokens, its full tokens).
    entries: Vec<(usize, Vec<String>, Vec<String>)>,
    /// token -> inverse document frequency.
    idf: HashMap<String, f32>,
}

impl Lexicon {
    /// Build from the index's public symbols (docs describe public API; private
    /// helpers are noise and bloat the table).
    pub fn build(index: &CodeIndex) -> Lexicon {
        let mut entries = Vec::new();
        let mut df: HashMap<String, usize> = HashMap::new();
        for (i, s) in index.symbols.iter().enumerate() {
            if s.visibility != Visibility::Public {
                continue;
            }
            let name_toks = tokens(&s.name);
            let mut full: Vec<String> = name_toks.clone();
            full.extend(tokens(&s.qualified_name));
            full.extend(tokens(&s.module));
            if let Some(sig) = &s.signature {
                full.extend(tokens(sig));
            }
            if let Some(doc) = &s.doc {
                full.extend(tokens(doc));
            }
            full.sort();
            full.dedup();
            for t in &full {
                // Clone the token into the table only on first sight; repeat
                // tokens (the common case across the symbol set) just bump a count.
                if let Some(c) = df.get_mut(t) {
                    *c += 1;
                } else {
                    df.insert(t.clone(), 1);
                }
            }
            entries.push((i, name_toks, full));
        }
        let n = entries.len().max(1) as f32;
        let idf = df
            .into_iter()
            .map(|(t, d)| (t, (1.0 + n / (1.0 + d as f32)).ln()))
            .collect();
        Lexicon { entries, idf }
    }

    /// Top-`k` (symbol index, score) for a claim, idf-weighted, name matches
    /// boosted. Empty when nothing clears [`MIN_LEXICAL_SCORE`].
    fn top(&self, claim: &str, k: usize) -> Vec<(usize, f32)> {
        let q: std::collections::HashSet<String> = tokens(claim).into_iter().collect();
        if q.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(usize, f32)> = Vec::new();
        for (idx, name_toks, full) in &self.entries {
            let mut score = 0.0f32;
            for t in full {
                if q.contains(t) {
                    let w = self.idf.get(t).copied().unwrap_or(1.0);
                    let boost = if name_toks.contains(t) { 2.0 } else { 1.0 };
                    score += w * boost;
                }
            }
            if score >= MIN_LEXICAL_SCORE {
                scored.push((*idx, score));
            }
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}

/// Split text into lowercased subword tokens: break on non-alphanumerics, then
/// on camelCase and digit boundaries, drop very short tokens and stopwords.
fn tokens(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in s.split(|c: char| !c.is_alphanumeric()) {
        split_subwords(raw, &mut out);
    }
    out.retain(|t| t.len() >= 3 && !STOPWORDS.contains(&t.as_str()));
    out
}

/// Split a single `[A-Za-z0-9]+` run on camelCase / snake boundaries.
fn split_subwords(raw: &str, out: &mut Vec<String>) {
    if raw.is_empty() {
        return;
    }
    let chars: Vec<char> = raw.chars().collect();
    let mut start = 0;
    for i in 1..chars.len() {
        let prev = chars[i - 1];
        let cur = chars[i];
        let next_lower = chars.get(i + 1).map(|c| c.is_lowercase()).unwrap_or(false);
        let boundary = (prev.is_lowercase() && cur.is_uppercase()) // camelCase
            || (prev.is_uppercase() && cur.is_uppercase() && next_lower) // JSONSchema -> JSON|Schema
            || (prev.is_alphabetic() && cur.is_ascii_digit())
            || (prev.is_ascii_digit() && cur.is_alphabetic());
        if boundary {
            out.push(chars[start..i].iter().collect::<String>().to_lowercase());
            start = i;
        }
    }
    out.push(chars[start..].iter().collect::<String>().to_lowercase());
    // Keep the whole run too (so `model_construct` matches an exact `model_construct`).
    let whole = raw.to_lowercase();
    if !out.last().map(|l| l == &whole).unwrap_or(false) {
        out.push(whole);
    }
}

const STOPWORDS: &[&str] = &[
    "the", "and", "for", "with", "this", "that", "when", "then", "are", "was", "has", "have",
    "not", "but", "all", "any", "can", "you", "use", "used", "uses", "via", "per", "its", "from",
    "into", "only", "must", "may", "should", "would", "will", "does", "doc", "docs", "code",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_split_camel_and_snake() {
        // `_` is split by the outer non-alnum pass, so `model_construct` yields
        // its subwords (which still align with a same-named symbol's subwords).
        let t = tokens("`model_construct` and `JSONSchema` with extraData2");
        assert!(t.contains(&"model".to_string()));
        assert!(t.contains(&"construct".to_string()));
        // Acronym boundary: JSONSchema -> json + schema.
        assert!(t.contains(&"json".to_string()));
        assert!(t.contains(&"schema".to_string()));
        assert!(t.contains(&"extra".to_string()));
        assert!(t.contains(&"data".to_string()));
    }

    #[test]
    fn stopwords_dropped() {
        let t = tokens("the cache invalidates on write");
        assert!(!t.contains(&"the".to_string()));
        assert!(t.contains(&"cache".to_string()));
        assert!(t.contains(&"invalidates".to_string()));
    }

    // ===== Layer-2 retrieval recall harness ==================================
    //
    // The two judge harnesses (src/judge.rs) measure Layer 3 with the *correct*
    // evidence handed to it. In production that evidence is chosen by Layer 2 —
    // grounding + the model-free lexical fallback ([`gather`]) by default, or the
    // embedding retriever behind STALEGUARD_EMBED_RETRIEVE. A perfect judge still
    // returns `unverifiable` if Layer 2 surfaced the wrong code, so the pipeline is
    // only as strong as this feeder, and until now it had no measured signal.
    //
    // This harness closes that gap. It builds a small fixture repo, runs the real
    // claim pipeline (candidate_claims -> grounding -> gather), and measures
    // recall@k: for each labelled claim, did the file that actually decides it land
    // in the top-k evidence? The default path needs no model, so its harness is a
    // normal CI test (a regression gate on the weakest link); the embedding variant
    // loads the ~160 MB jina model and is #[ignore]d like the judge harnesses.

    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// How many evidence chunks Layer 3 is fed per claim (the shipped default `k`).
    const RECALL_K: usize = 5;

    /// Fixture library code. Each file is a small module of public symbols with
    /// doc comments — enough surface for grounding and lexical/embedding retrieval
    /// to have something to match, plus distractors so recall isn't trivial.
    const FIXTURE_FILES: &[(&str, &str)] = &[
        (
            "src/net.rs",
            "/// Attempt a failing operation a fixed number of times before giving up.\n\
             pub fn retry<T>(mut op: impl FnMut() -> Result<T, String>) -> Result<T, String> {\n    \
                 let mut last = String::new();\n    \
                 for _ in 0..3 {\n        \
                     match op() {\n            \
                         Ok(v) => return Ok(v),\n            \
                         Err(e) => last = e,\n        \
                     }\n    \
                 }\n    \
                 Err(last)\n\
             }\n\n\
             /// Parse a port, returning an error when the number exceeds the valid range.\n\
             pub fn parse_port(s: &str) -> Result<u16, String> {\n    \
                 let n: u32 = s.parse().map_err(|_| \"bad port\".to_string())?;\n    \
                 if n > 65535 {\n        \
                     return Err(format!(\"port {n} out of range\"));\n    \
                 }\n    \
                 Ok(n as u16)\n\
             }\n\n\
             /// Open a connection using the configured port, defaulting to the standard database port 5432.\n\
             pub fn connect(host: &str, port: Option<u16>) -> Result<(), String> {\n    \
                 let _p = port.unwrap_or(5432);\n    \
                 let _ = host;\n    \
                 Ok(())\n\
             }\n",
        ),
        (
            "src/auth.rs",
            "/// Hash the raw password with bcrypt before it is stored.\n\
             pub fn hash_password(raw: &str) -> String {\n    \
                 format!(\"bcrypt${raw}\")\n\
             }\n\n\
             /// Verify a stored password against its bcrypt hash on login.\n\
             pub fn verify_password(raw: &str, stored: &str) -> bool {\n    \
                 stored == format!(\"bcrypt${raw}\")\n\
             }\n\n\
             /// Return true only for users whose role is administrator.\n\
             pub fn is_admin(role: &str) -> bool {\n    \
                 role == \"admin\"\n\
             }\n",
        ),
        (
            "src/cache.rs",
            "/// Insert a value, evicting the least recently used entry when the cache is at capacity.\n\
             pub fn cache_put(map: &mut Vec<(String, String)>, cap: usize, k: String, v: String) {\n    \
                 if map.len() >= cap {\n        \
                     map.remove(0);\n    \
                 }\n    \
                 map.push((k, v));\n\
             }\n",
        ),
        (
            "src/store.rs",
            "/// Write all buffered records to disk before returning.\n\
             pub fn flush(buf: &mut Vec<u8>) -> std::io::Result<()> {\n    \
                 buf.clear();\n    \
                 Ok(())\n\
             }\n\n\
             /// Trim surrounding whitespace and lowercase the input string.\n\
             pub fn normalize(s: &str) -> String {\n    \
                 s.trim().to_lowercase()\n\
             }\n",
        ),
        // Distractors: plausible public API the retriever can be lured by.
        (
            "src/util.rs",
            "/// Clamp a value into an inclusive range.\n\
             pub fn clamp(v: i64, lo: i64, hi: i64) -> i64 {\n    \
                 v.max(lo).min(hi)\n\
             }\n\n\
             /// Turn a title into a url-safe slug.\n\
             pub fn slugify(title: &str) -> String {\n    \
                 title.to_lowercase().replace(' ', \"-\")\n\
             }\n",
        ),
        (
            "src/log.rs",
            "/// Append a structured event to the log buffer.\n\
             pub fn log_event(buf: &mut Vec<String>, msg: &str) {\n    \
                 buf.push(msg.to_string());\n\
             }\n",
        ),
    ];

    /// Labelled recall corpus: `(claim, gold_file)`. `gold_file` is the repo file
    /// whose code actually decides the claim — recall is "did that file surface in
    /// the top-k evidence?". Mirrors real docs: some claims name the symbol in
    /// backticks (grounding should resolve them exactly); others describe the
    /// behaviour and backtick a non-symbol word, so the lexical/embedding step has
    /// to do the work; a couple use low-overlap phrasing on purpose, the kind of
    /// paraphrase the feeder is expected to struggle with.
    const RECALL_CORPUS: &[(&str, &str)] = &[
        // -- grounded: the claim names the deciding symbol in backticks --
        (
            "The `retry` helper attempts a failing operation a few times before returning the error.",
            "src/net.rs",
        ),
        (
            "`parse_port` returns an error when the port number exceeds the valid range.",
            "src/net.rs",
        ),
        (
            "`hash_password` hashes the raw password with bcrypt before it is stored.",
            "src/auth.rs",
        ),
        (
            "`is_admin` returns true only for users whose role is administrator.",
            "src/auth.rs",
        ),
        (
            "The `flush` method writes all buffered records to disk before returning.",
            "src/store.rs",
        ),
        (
            "`normalize` trims the surrounding whitespace and lowercases the input string.",
            "src/store.rs",
        ),
        // -- behavioural: backtick a non-symbol word, so the feeder must match on meaning --
        (
            "The cache evicts the least recently used `entry` when it is at capacity.",
            "src/cache.rs",
        ),
        (
            "A stored password is verified against its bcrypt `hash` during login.",
            "src/auth.rs",
        ),
        (
            "A connection falls back to the default database `port` when none is configured.",
            "src/net.rs",
        ),
        // -- hard: behaviour described with low lexical overlap with the code --
        (
            "Flaky requests are attempted again a handful of times when the `network` misbehaves.",
            "src/net.rs",
        ),
    ];

    fn write_fixture_repo() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("staleguard-l2recall-{nanos}"));
        for (rel, content) in FIXTURE_FILES {
            let path = root.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, content).unwrap();
        }
        root
    }

    /// Run the real extraction step and return the single grounded claim for a
    /// one-line doc; panics if the corpus line stops being a valid candidate claim
    /// (a drift in candidate_claims this harness should catch).
    fn claim_for(line: &str, lookup: &SymbolLookup) -> crate::judge::ProseClaim {
        let mut claims = crate::judge::candidate_claims(line, "DOC.md", lookup);
        assert_eq!(
            claims.len(),
            1,
            "corpus line is no longer a single candidate claim: {line:?}"
        );
        claims.pop().unwrap()
    }

    fn recall_at_k(paths: &[String], gold: &str) -> bool {
        paths.iter().take(RECALL_K).any(|p| p == gold)
    }

    /// Pretty per-case + aggregate report; returns `(recall@k, recall@1)`.
    fn report(label: &str, hits: &[(bool, bool, &str)]) -> (f32, f32) {
        let n = hits.len() as f32;
        let at_k = hits.iter().filter(|(k, _, _)| *k).count() as f32 / n;
        let at_1 = hits.iter().filter(|(_, one, _)| *one).count() as f32 / n;
        eprintln!("\n[layer2 {label}] {} cases", hits.len());
        for (k, one, claim) in hits {
            let mark = if *k { "ok " } else { "MISS" };
            let top = if *one { "@1" } else { "  " };
            eprintln!("  {mark} {top} {claim}");
        }
        eprintln!("  recall@{RECALL_K} = {at_k:.3}   recall@1 = {at_1:.3}");
        (at_k, at_1)
    }

    /// Default Layer 2 (grounding + lexical fallback, no model). Runs in CI: it is
    /// the shipped default feeder and the deciding link for every ML-path verdict,
    /// so a recall regression here should fail the build.
    #[test]
    fn layer2_recall_model_free() {
        let root = write_fixture_repo();
        let index = CodeIndex::build(&root);
        let lexicon = Lexicon::build(&index);
        let lookup = SymbolLookup::build(&index);
        let mut files = FileCache::new();

        let mut hits: Vec<(bool, bool, &str)> = Vec::new();
        for (line, gold) in RECALL_CORPUS {
            let claim = claim_for(line, &lookup);
            let ev = gather(
                &claim.text,
                &claim.provenance,
                &index,
                &lookup,
                &lexicon,
                &root,
                RECALL_K,
                &mut files,
            );
            let paths: Vec<String> = ev.iter().map(|e| e.path.clone()).collect();
            hits.push((
                recall_at_k(&paths, gold),
                paths.first().map(|p| p == gold).unwrap_or(false),
                line,
            ));
        }
        let _ = std::fs::remove_dir_all(&root);

        let (at_k, at_1) = report("model-free", &hits);
        // Measured baselines (gather is deterministic): the grounded + clearly
        // lexical claims resolve; the low-overlap paraphrase is the documented miss.
        // Gate just under measured so a real recall regression trips this.
        assert!(
            at_k >= 0.85,
            "model-free recall@{RECALL_K} regressed: {at_k:.3}"
        );
        assert!(at_1 >= 0.70, "model-free recall@1 regressed: {at_1:.3}");
    }

    /// Embedding retriever (STALEGUARD_EMBED_RETRIEVE path). Loads the ~160 MB jina
    /// model, so it is #[ignore]d and run on demand, the same as the judge
    /// harnesses:  cargo test --features ml layer2_recall_embedding -- --ignored --nocapture
    #[test]
    #[ignore = "loads the ~160 MB embedding model; run on demand"]
    fn layer2_recall_embedding() {
        let root = write_fixture_repo();
        let index = CodeIndex::build(&root);
        let queries: Vec<String> = RECALL_CORPUS.iter().map(|(c, _)| c.to_string()).collect();
        let per_query =
            crate::retrieve::retrieve(&root, &index, &queries, RECALL_K).expect("retrieve");
        let _ = std::fs::remove_dir_all(&root);

        let mut hits: Vec<(bool, bool, &str)> = Vec::new();
        for ((line, gold), res) in RECALL_CORPUS.iter().zip(&per_query) {
            let paths: Vec<String> = res.iter().map(|h| h.path.clone()).collect();
            hits.push((
                recall_at_k(&paths, gold),
                paths.first().map(|p| p == gold).unwrap_or(false),
                line,
            ));
        }
        let (at_k, _at_1) = report("embedding", &hits);
        // Measured: recall@5 = recall@1 = 1.00 — the semantic retriever even recovers
        // the low-overlap paraphrase the lexical fallback misses. Gate a little under
        // measured so model/version drift trips this rather than corpus noise.
        assert!(
            at_k >= 0.85,
            "embedding recall@{RECALL_K} regressed: {at_k:.3}"
        );
    }
}
