//! Layer 2 retrieval: local code embeddings via fastembed + the jina code model.
//!
//! Fully offline after the first model download (`jina-embeddings-v2-base-code`,
//! ~160 MB, cached by fastembed). Code is chunked on **symbol boundaries** from
//! the tree-sitter index (falling back to overlapping line windows for files
//! with no extractable symbols), embedded, and queried by cosine similarity.
//!
//! Embeddings are cached on disk under `.shlomes/` keyed by a content hash, so
//! unchanged chunks (and unchanged queries) are free on re-run; the model is
//! only loaded when something actually needs embedding. An optional reranker
//! ([`crate::rerank`]) sharpens the top-k before it reaches the judge.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::Result;
use fastembed::{
    InitOptionsUserDefined, Pooling, TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel,
};
use hf_hub::api::sync::Api;
use serde::{Deserialize, Serialize};

use crate::claim::fnv1a;
use crate::code::lang;
use crate::code::CodeIndex;

/// Default HuggingFace repo + the int8-quantized ONNX (162 MB vs the 642 MB fp32
/// that fastembed's built-in `JinaEmbeddingsV2BaseCode` variant would download).
/// Both are overridable via `SHLOMES_EMBED_REPO` / `SHLOMES_EMBED_ONNX` so the
/// embedding model can be swapped (e.g. for a smaller, faster one) without a
/// rebuild; the on-disk cache is tagged with the repo so a swap invalidates it.
/// A swapped model must expose the same tokenizer file layout and use mean
/// pooling (same as jina v2).
const DEFAULT_MODEL_REPO: &str = "jinaai/jina-embeddings-v2-base-code";
const DEFAULT_ONNX: &str = "onnx/model_quantized.onnx";

fn model_repo() -> String {
    std::env::var("SHLOMES_EMBED_REPO").unwrap_or_else(|_| DEFAULT_MODEL_REPO.to_string())
}

fn model_onnx() -> String {
    std::env::var("SHLOMES_EMBED_ONNX").unwrap_or_else(|_| DEFAULT_ONNX.to_string())
}

/// Fallback line-window chunking (files with no extractable symbols).
const CHUNK_LINES: usize = 40;
const CHUNK_OVERLAP: usize = 10;
/// A symbol body longer than this is split into windows so a giant function
/// doesn't become one unwieldy (and token-limit-busting) chunk.
const MAX_SYMBOL_LINES: usize = 80;

struct Chunk {
    path: String,
    start_line: usize,
    text: String,
}

/// A retrieved code chunk ranked by similarity to the query.
pub struct Hit {
    pub path: String,
    pub start_line: usize,
    pub score: f32,
    /// Chunk body — the evidence the Layer 3 judge ([`crate::judge`]) reads.
    pub text: String,
}

// ---- chunking -------------------------------------------------------------

/// Overlapping line-window chunks — the fallback when a file yields no symbols.
fn window_chunks(path: &str, lines: &[&str], base_line: usize, out: &mut Vec<Chunk>) {
    if lines.is_empty() {
        return;
    }
    let step = CHUNK_LINES.saturating_sub(CHUNK_OVERLAP).max(1);
    let mut start = 0;
    while start < lines.len() {
        let end = (start + CHUNK_LINES).min(lines.len());
        let text = lines[start..end].join("\n");
        if !text.trim().is_empty() {
            out.push(Chunk {
                path: path.to_string(),
                start_line: base_line + start,
                text,
            });
        }
        if end == lines.len() {
            break;
        }
        start += step;
    }
}

/// Reduce a file's symbol body spans to top-level ones: drop any span strictly
/// contained within another (a method inside its class), so chunks don't nest.
fn top_level_spans(mut spans: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    spans.sort_by_key(|&(s, e)| (s, std::cmp::Reverse(e)));
    let mut kept: Vec<(usize, usize)> = Vec::new();
    for (s, e) in spans {
        let contained = kept.iter().any(|&(ks, ke)| ks <= s && e <= ke);
        if !contained {
            kept.push((s, e));
        }
    }
    kept
}

/// Chunk one file's covered symbol spans; large bodies are windowed.
fn symbol_chunks(path: &str, lines: &[&str], spans: &[(usize, usize)], out: &mut Vec<Chunk>) {
    for &(start, end) in spans {
        // body_span lines are 1-based and inclusive.
        let (s, e) = (start.saturating_sub(1), end.min(lines.len()));
        if s >= e {
            continue;
        }
        let body = &lines[s..e];
        if e - s <= MAX_SYMBOL_LINES {
            let text = body.join("\n");
            if !text.trim().is_empty() {
                out.push(Chunk {
                    path: path.to_string(),
                    start_line: start,
                    text,
                });
            }
        } else {
            window_chunks(path, body, start, out);
        }
    }
}

/// Collect chunks for the whole repo: symbol-aligned where the index has
/// symbols for a file, line-windowed otherwise.
fn collect_chunks(repo_root: &Path, index: &CodeIndex) -> Vec<Chunk> {
    let mut spans_by_file: HashMap<&str, Vec<(usize, usize)>> = HashMap::new();
    for sym in &index.symbols {
        let span = &sym.body_span;
        if span.path.is_empty() || span.start_line == 0 || span.end_line < span.start_line {
            continue;
        }
        spans_by_file
            .entry(span.path.as_str())
            .or_default()
            .push((span.start_line, span.end_line));
    }

    let mut chunks = Vec::new();
    for p in lang::code_files(repo_root) {
        let rel = p
            .strip_prefix(repo_root)
            .unwrap_or(&p)
            .to_string_lossy()
            .to_string();
        // Docs describe the library's own API, so tests/benchmarks/examples are
        // noise as *evidence* — and on a real repo they are ~half the corpus, the
        // dominant embedding cost. Drop them from the retrieval set (the symbol
        // index still sees them). Disable with `SHLOMES_EMBED_INCLUDE_TESTS=1`.
        if !include_tests() && is_non_library(&rel) {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(&p) else {
            continue;
        };
        let lines: Vec<&str> = content.lines().collect();
        match spans_by_file.remove(rel.as_str()) {
            Some(spans) if !spans.is_empty() => {
                symbol_chunks(&rel, &lines, &top_level_spans(spans), &mut chunks)
            }
            _ => window_chunks(&rel, &lines, 1, &mut chunks),
        }
    }
    chunks
}

fn include_tests() -> bool {
    std::env::var_os("SHLOMES_EMBED_INCLUDE_TESTS").is_some()
}

/// Heuristic: is this a test / benchmark / example file (not library code the
/// docs would describe)? Matched on path segments and filename conventions
/// across the common ecosystems (pytest, go, rust, js).
fn is_non_library(rel: &str) -> bool {
    let lower = rel.to_ascii_lowercase();
    let seg = |s: &str| {
        lower.starts_with(&format!("{s}/")) || lower.contains(&format!("/{s}/"))
    };
    if ["tests", "test", "testing", "benchmarks", "benchmark", "bench", "examples", "example", "e2e", "fixtures", "__tests__"]
        .iter()
        .any(|d| seg(d))
    {
        return true;
    }
    let name = lower.rsplit('/').next().unwrap_or(&lower);
    name.starts_with("test_")
        || name.starts_with("conftest")
        || name.ends_with("_test.py")
        || name.ends_with("_test.go")
        || name.ends_with("_test.rs")
        || name.ends_with(".test.ts")
        || name.ends_with(".test.js")
        || name.ends_with(".spec.ts")
        || name.ends_with(".spec.js")
}

// ---- embedding cache ------------------------------------------------------

/// On-disk embedding cache keyed by content hash. Vectors are stored already
/// normalized. Tagged with the model id so switching models invalidates it.
#[derive(Default, Serialize, Deserialize)]
struct EmbedCache {
    model: String,
    /// hex(fnv1a(text)) -> normalized embedding.
    vectors: HashMap<String, Vec<f32>>,
    #[serde(skip)]
    dirty: bool,
}

impl EmbedCache {
    fn path(repo_root: &Path) -> PathBuf {
        repo_root.join(".shlomes").join("embeddings.json")
    }

    fn load(repo_root: &Path) -> EmbedCache {
        let cache: Option<EmbedCache> = std::fs::read(Self::path(repo_root))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok());
        match cache {
            // A cache from a different model lives in a different vector space.
            Some(c) if c.model == model_repo() => c,
            _ => EmbedCache {
                model: model_repo(),
                ..Default::default()
            },
        }
    }

    fn get(&self, text: &str) -> Option<&Vec<f32>> {
        self.vectors.get(&key(text))
    }

    fn insert(&mut self, text: &str, vec: Vec<f32>) {
        self.vectors.insert(key(text), vec);
        self.dirty = true;
    }

    fn save(&self, repo_root: &Path) {
        if !self.dirty {
            return;
        }
        let path = Self::path(repo_root);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(bytes) = serde_json::to_vec(self) {
            let _ = std::fs::write(path, bytes);
        }
    }
}

fn key(text: &str) -> String {
    format!("{:016x}", fnv1a(text))
}

// ---- vector math ----------------------------------------------------------

fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Dot product of two already-normalized vectors == cosine similarity.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// ---- model ----------------------------------------------------------------

/// Load jina-embeddings-v2-base-code from its int8-quantized ONNX. fastembed's
/// built-in variant hardcodes the fp32 file, so we fetch the quantized weights
/// (and tokenizer files) via hf-hub and load them as a user-defined model. Jina
/// v2 uses mean pooling; quantization stays `None` because the int8 is baked
/// into the graph, not applied by fastembed.
fn new_model() -> Result<TextEmbedding> {
    let repo = Api::new()?.model(model_repo());
    let read = |path: &str| -> Result<Vec<u8>> { Ok(std::fs::read(repo.get(path)?)?) };

    let onnx = read(&model_onnx())?;
    let tokenizer_files = TokenizerFiles {
        tokenizer_file: read("tokenizer.json")?,
        config_file: read("config.json")?,
        special_tokens_map_file: read("special_tokens_map.json")?,
        tokenizer_config_file: read("tokenizer_config.json")?,
    };

    let model = UserDefinedEmbeddingModel::new(onnx, tokenizer_files).with_pooling(Pooling::Mean);
    // Cap the embedded sequence length. fastembed pads each batch to its longest
    // member and runs batches sequentially, so on a CPU the per-run cost scales
    // with this length. Code chunks carry their retrieval signal (signature +
    // opening lines) well within ~128 tokens; the default 512 quadruples the work
    // for no recall gain. Overridable via `SHLOMES_EMBED_MAX_TOKENS`.
    let max_tokens = std::env::var("SHLOMES_EMBED_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let opts = InitOptionsUserDefined::new()
        .with_max_length(max_tokens)
        // fastembed runs batches sequentially, so ONNX intra-op threads are the
        // only parallelism for embedding. Default leaves cores idle (~4/10 here);
        // pin it to all available cores. Overridable via `SHLOMES_ORT_THREADS`.
        .with_intra_threads(ort_threads());
    TextEmbedding::try_new_from_user_defined(model, opts)
}

/// Fetch (and cache) the embedding model so the first real run is offline.
/// Triggers the Hub download and a full load, surfacing any auth/network error
/// up front. Used by `shlomes setup`.
pub fn prefetch_model() -> Result<()> {
    new_model()?;
    Ok(())
}

/// ONNX intra-op thread count: `SHLOMES_ORT_THREADS` if set, else every core.
pub(crate) fn ort_threads() -> usize {
    std::env::var("SHLOMES_ORT_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        })
}

/// Embed `texts`, using the cache for hits and the model for misses. The model
/// is loaded only if at least one text is missing. Returns a vector per input,
/// parallel to `texts`, all normalized.
fn embed_cached(texts: &[String], cache: &mut EmbedCache) -> Result<Vec<Vec<f32>>> {
    // Unique missing texts (dedup so duplicate chunks embed once).
    let mut needed: Vec<&str> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for t in texts {
        if cache.get(t).is_none() && seen.insert(key(t)) {
            needed.push(t.as_str());
        }
    }
    if !needed.is_empty() {
        let mut model = new_model()?;
        let mut vecs = model.embed(needed.clone(), None)?;
        for (t, v) in needed.iter().zip(vecs.iter_mut()) {
            normalize(v);
            cache.insert(t, v.clone());
        }
    }
    Ok(texts
        .iter()
        .map(|t| cache.get(t).cloned().unwrap_or_default())
        .collect())
}

// ---- retrieval ------------------------------------------------------------

/// Build the code index, chunk it, then return the top-`k` chunks for each query
/// by cosine similarity. If a reranker is configured ([`crate::rerank`]),
/// over-fetch by cosine and rerank down to `k`. Result is parallel to `queries`.
pub fn retrieve(
    repo_root: &Path,
    index: &CodeIndex,
    queries: &[String],
    k: usize,
) -> Result<Vec<Vec<Hit>>> {
    let t = std::time::Instant::now();
    let chunks = collect_chunks(repo_root, index);
    crate::judge::timing(format!("  retrieve: chunk ({} chunks)", chunks.len()), t);
    if chunks.is_empty() {
        return Ok(queries.iter().map(|_| Vec::new()).collect());
    }

    let t = std::time::Instant::now();
    let mut cache = EmbedCache::load(repo_root);
    crate::judge::timing("  retrieve: cache load", t);
    let chunk_texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let t = std::time::Instant::now();
    let chunk_vecs = embed_cached(&chunk_texts, &mut cache)?;
    crate::judge::timing("  retrieve: embed corpus", t);
    let query_vecs = embed_cached(queries, &mut cache)?;
    let t = std::time::Instant::now();
    cache.save(repo_root);
    crate::judge::timing("  retrieve: cache save", t);

    let mut reranker = crate::rerank::Reranker::from_env()?;
    // Over-fetch before reranking so the reranker can promote chunks cosine ranked
    // just outside the top-k.
    let fetch = if reranker.is_some() { (k * 4).max(k) } else { k };

    let mut results = Vec::with_capacity(queries.len());
    for (qi, qv) in query_vecs.iter().enumerate() {
        let mut scored: Vec<(f32, usize)> = chunk_vecs
            .iter()
            .enumerate()
            .map(|(i, cv)| (cosine(qv, cv), i))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(fetch);

        let mut hits: Vec<Hit> = scored
            .into_iter()
            .map(|(score, i)| Hit {
                path: chunks[i].path.clone(),
                start_line: chunks[i].start_line,
                score,
                text: chunks[i].text.clone(),
            })
            .collect();

        if let Some(rr) = reranker.as_mut() {
            let passages: Vec<String> = hits.iter().map(|h| h.text.clone()).collect();
            let scores = rr.scores(&queries[qi], &passages)?;
            for (h, s) in hits.iter_mut().zip(scores) {
                h.score = s;
            }
            hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        }
        hits.truncate(k);
        results.push(hits);
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_level_spans_drop_nested() {
        // class (1..20) with two methods inside -> only the class survives.
        let spans = vec![(1, 20), (3, 8), (10, 18)];
        assert_eq!(top_level_spans(spans), vec![(1, 20)]);
    }

    #[test]
    fn top_level_spans_keep_siblings() {
        let spans = vec![(1, 5), (7, 12)];
        let kept = top_level_spans(spans);
        assert!(kept.contains(&(1, 5)) && kept.contains(&(7, 12)));
    }

    #[test]
    fn symbol_chunk_slices_body_lines() {
        let lines: Vec<&str> = "a\nb\nc\nd\ne".lines().collect();
        let mut out = Vec::new();
        symbol_chunks("f.rs", &lines, &[(2, 4)], &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].start_line, 2);
        assert_eq!(out[0].text, "b\nc\nd");
    }

    #[test]
    fn non_library_files_detected() {
        for p in [
            "tests/test_main.py",
            "src/foo/test_x.py",
            "pkg/foo_test.go",
            "benchmarks/run.py",
            "docs/examples/demo.py",
            "web/__tests__/a.test.ts",
            "conftest.py",
        ] {
            assert!(is_non_library(p), "should be non-library: {p}");
        }
        for p in [
            "src/main.rs",
            "pydantic/main.py",
            "lib/contest.py", // 'contest' must not match 'conftest'
            "src/latest/mod.rs", // 'latest' must not match the 'test' segment
        ] {
            assert!(!is_non_library(p), "should be library: {p}");
        }
    }

    #[test]
    fn cache_round_trips_by_content() {
        let mut c = EmbedCache {
            model: DEFAULT_MODEL_REPO.to_string(),
            ..Default::default()
        };
        assert!(c.get("hello").is_none());
        c.insert("hello", vec![1.0, 0.0]);
        assert_eq!(c.get("hello"), Some(&vec![1.0, 0.0]));
        assert!(c.dirty);
    }
}
