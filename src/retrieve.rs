//! Layer 2 retrieval: local code embeddings via fastembed + the jina code model.
//!
//! Fully offline after the first model download (`jina-embeddings-v2-base-code`,
//! ~160 MB, cached by fastembed). Code is chunked into overlapping line windows,
//! embedded, and queried by cosine similarity. AST/tree-sitter chunking and a
//! content-hash vector cache are the planned next steps.

use std::path::Path;

use anyhow::Result;
use fastembed::{
    InitOptionsUserDefined, Pooling, TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel,
};
use hf_hub::api::sync::Api;
use walkdir::WalkDir;

/// HuggingFace repo + the int8-quantized ONNX (162 MB vs the 642 MB fp32 that
/// fastembed's built-in `JinaEmbeddingsV2BaseCode` variant would download).
const MODEL_REPO: &str = "jinaai/jina-embeddings-v2-base-code";
const QUANTIZED_ONNX: &str = "onnx/model_quantized.onnx";

const CODE_EXTS: &[&str] = &[
    "rs", "py", "ts", "tsx", "js", "jsx", "go", "java", "rb", "c", "h", "cpp", "hpp", "cc", "cs",
    "php", "swift", "kt", "scala", "sh", "toml", "yaml", "yml",
];
const CHUNK_LINES: usize = 40;
const CHUNK_OVERLAP: usize = 10;

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
    /// Chunk body — evidence passed to the Layer 3 judge (not printed yet).
    #[allow(dead_code)]
    pub text: String,
}

fn is_code(p: &Path) -> bool {
    p.extension()
        .and_then(|s| s.to_str())
        .map(|e| CODE_EXTS.contains(&e))
        .unwrap_or(false)
}

fn chunk_file(path: &str, content: &str) -> Vec<Chunk> {
    let lines: Vec<&str> = content.lines().collect();
    let mut chunks = Vec::new();
    if lines.is_empty() {
        return chunks;
    }
    let step = CHUNK_LINES.saturating_sub(CHUNK_OVERLAP).max(1);
    let mut start = 0;
    while start < lines.len() {
        let end = (start + CHUNK_LINES).min(lines.len());
        let text = lines[start..end].join("\n");
        if !text.trim().is_empty() {
            chunks.push(Chunk {
                path: path.to_string(),
                start_line: start + 1,
                text,
            });
        }
        if end == lines.len() {
            break;
        }
        start += step;
    }
    chunks
}

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

fn collect_chunks(repo_root: &Path) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    for e in WalkDir::new(repo_root)
        .into_iter()
        .filter_entry(|e| {
            let n = e.file_name().to_string_lossy();
            n != ".git" && n != "target" && n != ".doc-aligner"
        })
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let p = e.path();
        if !is_code(p) {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(p) else {
            continue;
        };
        let rel = p
            .strip_prefix(repo_root)
            .unwrap_or(p)
            .to_string_lossy()
            .to_string();
        chunks.extend(chunk_file(&rel, &content));
    }
    chunks
}

/// Load jina-embeddings-v2-base-code from its int8-quantized ONNX. fastembed's
/// built-in variant hardcodes the fp32 file, so we fetch the quantized weights
/// (and tokenizer files) via hf-hub and load them as a user-defined model. Jina
/// v2 uses mean pooling; quantization stays `None` because the int8 is baked
/// into the graph, not applied by fastembed.
fn new_model() -> Result<TextEmbedding> {
    let repo = Api::new()?.model(MODEL_REPO.to_string());
    let read = |path: &str| -> Result<Vec<u8>> { Ok(std::fs::read(repo.get(path)?)?) };

    let onnx = read(QUANTIZED_ONNX)?;
    let tokenizer_files = TokenizerFiles {
        tokenizer_file: read("tokenizer.json")?,
        config_file: read("config.json")?,
        special_tokens_map_file: read("special_tokens_map.json")?,
        tokenizer_config_file: read("tokenizer_config.json")?,
    };

    let model = UserDefinedEmbeddingModel::new(onnx, tokenizer_files).with_pooling(Pooling::Mean);
    Ok(TextEmbedding::try_new_from_user_defined(
        model,
        InitOptionsUserDefined::new(),
    )?)
}

/// Embed every code chunk in the repo, then return the top-`k` chunks for each
/// query by cosine similarity. Result is parallel to `queries`.
pub fn retrieve(repo_root: &Path, queries: &[String], k: usize) -> Result<Vec<Vec<Hit>>> {
    let chunks = collect_chunks(repo_root);
    if chunks.is_empty() {
        return Ok(queries.iter().map(|_| Vec::new()).collect());
    }

    let mut model = new_model()?;

    let chunk_texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
    let mut chunk_vecs = model.embed(chunk_texts, None)?;
    for v in chunk_vecs.iter_mut() {
        normalize(v);
    }

    let mut query_vecs = model.embed(queries, None)?;
    for v in query_vecs.iter_mut() {
        normalize(v);
    }

    let mut results = Vec::with_capacity(queries.len());
    for qv in &query_vecs {
        let mut scored: Vec<(f32, usize)> = chunk_vecs
            .iter()
            .enumerate()
            .map(|(i, cv)| (cosine(qv, cv), i))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        results.push(
            scored
                .into_iter()
                .map(|(score, i)| Hit {
                    path: chunks[i].path.clone(),
                    start_line: chunks[i].start_line,
                    score,
                    text: chunks[i].text.clone(),
                })
                .collect(),
        );
    }
    Ok(results)
}
