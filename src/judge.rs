//! Layer 3 verification: an NLI cross-encoder as the coherence judge.
//!
//! For doc claims that survive the deterministic layers (behavioural prose like
//! "the cache invalidates on write"), Layer 2 retrieves the most relevant code
//! chunks and this layer renders the verdict. The judge is a natural-language
//! inference (NLI) cross-encoder: it reads `(premise = code evidence,
//! hypothesis = doc claim)` and classifies entailment / contradiction / neutral,
//! which map onto [`Verdict::Supported`] / [`Verdict::Contradicted`] /
//! [`Verdict::Unverifiable`].
//!
//! It is a *classifier*, not a generative LLM — no API, no per-token cost, code
//! never leaves the machine. The model is an int8-quantized ONNX
//! (`nli-deberta-v3-xsmall`, ~20 MB) loaded via `ort`, mirroring the offline
//! model-download path Layer 2 already uses. Repo, ONNX file, and the decision
//! threshold are overridable via `SHLOMES_NLI_REPO`, `SHLOMES_NLI_ONNX`, and
//! `SHLOMES_NLI_THRESHOLD`.
//!
//! Unlike embeddings, a cross-encoder can separate `supported` from
//! `contradicted`: negation ("does X" vs "does *not* X") barely moves an
//! embedding vector but flips an NLI verdict. That contradiction axis is the
//! whole reason this layer exists.

use std::borrow::Cow;
use std::collections::HashSet;
use std::path::Path;
use std::sync::OnceLock;

use anyhow::{anyhow, Result};
use hf_hub::api::sync::Api;
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use regex::Regex;
use serde_json::Value as Json;
use tokenizers::{Encoding, Tokenizer, TruncationParams};

use crate::claim::Provenance;
use crate::code::CodeIndex;
use crate::findings::{Finding, Verdict};
use crate::retrieve;

const DEFAULT_REPO: &str = "Xenova/nli-deberta-v3-xsmall";
const DEFAULT_ONNX: &str = "onnx/model_quantized.onnx";
const DEFAULT_THRESHOLD: f32 = 0.5;
const MAX_TOKENS: usize = 256;

/// Top-k code chunks retrieved per claim and fed to the judge as evidence.
pub const EVIDENCE_K: usize = 5;
/// Upper bound on prose claims judged per run — one forward pass per
/// (claim, evidence) pair, so this bounds model cost.
pub const MAX_CLAIMS: usize = 300;

/// A behavioural doc claim awaiting a Layer 3 verdict: the prose under test, its
/// `path:line` origin, and the code [`Provenance`] its backtick tokens ground to
/// (so a `Supported` verdict is ledgerable and re-opens via the Layer 0
/// fingerprint flag when that code changes).
pub struct ProseClaim {
    pub text: String,
    pub doc_ref: String,
    pub provenance: Provenance,
}

/// Which output index corresponds to each NLI class. Read from the model's
/// `config.json` (`id2label`) rather than hardcoded, since the index order
/// differs across checkpoints (sentence-transformers vs MoritzLaurer, etc.).
struct Labels {
    entail: usize,
    contra: usize,
    neutral: usize,
}

impl Labels {
    fn from_config(cfg: &Json) -> Result<Labels> {
        let map = cfg
            .get("id2label")
            .and_then(Json::as_object)
            .ok_or_else(|| anyhow!("model config.json has no id2label map"))?;
        let (mut entail, mut contra, mut neutral) = (None, None, None);
        for (idx, label) in map {
            let i: usize = idx.parse().map_err(|_| anyhow!("non-numeric label id {idx}"))?;
            let name = label.as_str().unwrap_or_default().to_ascii_lowercase();
            if name.contains("entail") {
                entail = Some(i);
            } else if name.contains("contrad") {
                contra = Some(i);
            } else if name.contains("neutral") {
                neutral = Some(i);
            }
        }
        match (entail, contra, neutral) {
            (Some(entail), Some(contra), Some(neutral)) => Ok(Labels {
                entail,
                contra,
                neutral,
            }),
            _ => Err(anyhow!("id2label is not a 3-class NLI head (need entailment/contradiction/neutral)")),
        }
    }
}

/// A loaded NLI cross-encoder ready to judge `(evidence, claim)` pairs.
pub struct Judge {
    session: Session,
    tokenizer: Tokenizer,
    labels: Labels,
    /// Some DeBERTa ONNX exports omit `token_type_ids`; only pass it if declared.
    needs_token_types: bool,
    threshold: f32,
}

impl Judge {
    /// Fetch (once, then cached) and load the NLI model, tokenizer, and label map.
    pub fn load() -> Result<Judge> {
        let repo_name = env_or("SHLOMES_NLI_REPO", DEFAULT_REPO);
        let onnx_rel = env_or("SHLOMES_NLI_ONNX", DEFAULT_ONNX);
        let threshold = std::env::var("SHLOMES_NLI_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_THRESHOLD);

        let repo = Api::new()?.model(repo_name);
        let onnx = repo.get(&onnx_rel)?;
        let tok = repo.get("tokenizer.json")?;
        let cfg = repo.get("config.json")?;

        let session = Session::builder()?
            .with_intra_threads(retrieve::ort_threads())
            .map_err(|e| anyhow!("set intra threads: {e}"))?
            .commit_from_file(onnx)?;

        let mut tokenizer =
            Tokenizer::from_file(tok).map_err(|e| anyhow!("load tokenizer: {e}"))?;
        // Cross-encoder context is short; truncate the pair so long code chunks
        // don't blow past the model's max position embeddings.
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_TOKENS,
                ..Default::default()
            }))
            .map_err(|e| anyhow!("set truncation: {e}"))?;

        let cfg: Json = serde_json::from_slice(&std::fs::read(cfg)?)?;
        let labels = Labels::from_config(&cfg)?;
        let needs_token_types = session.inputs().iter().any(|i| i.name() == "token_type_ids");

        Ok(Judge {
            session,
            tokenizer,
            labels,
            needs_token_types,
            threshold,
        })
    }

    /// One batched forward pass over many `(premise, hypothesis)` pairs. Pairs
    /// are padded to the batch's longest sequence (mask 0 on pad positions).
    /// Returns one `[p_contradiction, p_entailment, p_neutral]` per pair,
    /// reordered into class semantics regardless of the model's native order.
    fn classify_batch(&mut self, pairs: &[(&str, &str)]) -> Result<Vec<[f32; 3]>> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        let encs: Vec<Encoding> = pairs
            .iter()
            .map(|(p, h)| self.tokenizer.encode((*p, *h), true))
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| anyhow!("tokenize: {e}"))?;

        let n = encs.len();
        let max_len = encs.iter().map(|e| e.get_ids().len()).max().unwrap_or(0).max(1);
        let mut ids = vec![0i64; n * max_len];
        let mut mask = vec![0i64; n * max_len];
        let mut types = vec![0i64; n * max_len];
        for (row, e) in encs.iter().enumerate() {
            let base = row * max_len;
            for (c, &id) in e.get_ids().iter().enumerate() {
                ids[base + c] = id as i64;
            }
            for (c, &m) in e.get_attention_mask().iter().enumerate() {
                mask[base + c] = m as i64;
            }
            for (c, &t) in e.get_type_ids().iter().enumerate() {
                types[base + c] = t as i64;
            }
        }

        let shape = vec![n as i64, max_len as i64];
        let mut inputs = ort::inputs![
            "input_ids" => Tensor::from_array((shape.clone(), ids))?,
            "attention_mask" => Tensor::from_array((shape.clone(), mask))?,
        ];
        if self.needs_token_types {
            inputs.push((
                Cow::from("token_type_ids"),
                SessionInputValue::from(Tensor::from_array((shape, types))?),
            ));
        }

        let outputs = self.session.run(inputs)?;
        let (_, logits) = outputs[0].try_extract_tensor::<f32>()?;
        let n_labels = logits.len() / n;
        let mut out = Vec::with_capacity(n);
        for row in 0..n {
            let probs = softmax(&logits[row * n_labels..(row + 1) * n_labels]);
            out.push([
                probs[self.labels.contra],
                probs[self.labels.entail],
                probs[self.labels.neutral],
            ]);
        }
        Ok(out)
    }

    /// Judge a claim against its retrieved evidence in one batched pass.
    /// Max-pools entailment and contradiction across the chunks, then
    /// [`decide`]s. Returns the verdict and the confidence behind it.
    pub fn judge(&mut self, claim: &str, evidence: &[String]) -> Result<(Verdict, f32)> {
        if evidence.is_empty() {
            return Ok((Verdict::Unverifiable, 0.0));
        }
        let pairs: Vec<(&str, &str)> = evidence.iter().map(|ev| (ev.as_str(), claim)).collect();
        let (mut best_entail, mut best_contra) = (0.0f32, 0.0f32);
        for [contra, entail, _neutral] in self.classify_batch(&pairs)? {
            best_entail = best_entail.max(entail);
            best_contra = best_contra.max(contra);
        }
        Ok(decide(best_entail, best_contra, self.threshold))
    }
}

/// The verdict rule (pure, model-free, unit-tested). Contradiction is the
/// differentiating signal, so it must clear the threshold *and* out-weigh
/// entailment before we flag drift; otherwise a strong entailment supports the
/// claim, and anything weaker is `unverifiable`.
fn decide(best_entail: f32, best_contra: f32, threshold: f32) -> (Verdict, f32) {
    if best_contra >= threshold && best_contra >= best_entail {
        (Verdict::Contradicted, best_contra)
    } else if best_entail >= threshold {
        (Verdict::Supported, best_entail)
    } else {
        (Verdict::Unverifiable, best_entail.max(best_contra))
    }
}

/// Layer 3 entry point: retrieve evidence (Layer 2) for each claim, then judge.
/// Emits `Supported` claims (ledgered, not reported) and `Contradicted` /
/// `Unverifiable` problems, all tagged `layer = 3` and anchored to the evidence
/// files so drift lineage can re-open them when that code changes.
pub fn check(root: &Path, index: &CodeIndex, claims: &[ProseClaim], k: usize) -> Result<Vec<Finding>> {
    if claims.is_empty() {
        return Ok(Vec::new());
    }

    let texts: Vec<String> = claims.iter().map(|c| c.text.clone()).collect();
    let t = std::time::Instant::now();
    let retrieved = retrieve::retrieve(root, index, &texts, k)?;
    timing(format!("retrieve ({} claims)", claims.len()), t);
    let t = std::time::Instant::now();
    let mut judge = Judge::load()?;
    timing("judge model load", t);

    let t = std::time::Instant::now();
    let mut findings = Vec::new();
    for (claim, hits) in claims.iter().zip(retrieved) {
        let evidence: Vec<String> = hits.iter().map(|h| h.text.clone()).collect();
        let refs: Vec<String> = hits
            .iter()
            .map(|h| format!("{}:{}", h.path, h.start_line))
            .collect();
        // Prefer the claim's own grounding (symbols/modules — survives moves and
        // feeds the fingerprint flag); fall back to the evidence files only when
        // the claim grounded to nothing.
        let prov = if claim.provenance.is_empty() {
            Provenance {
                paths: hits.iter().map(|h| h.path.clone()).collect(),
                ..Default::default()
            }
        } else {
            claim.provenance.clone()
        };

        let (verdict, conf) = judge.judge(&claim.text, &evidence)?;
        let mut finding = match verdict {
            Verdict::Supported => Finding::supported(claim.text.clone(), claim.doc_ref.clone(), prov),
            v => Finding::problem(
                v,
                claim.text.clone(),
                claim.doc_ref.clone(),
                format!("{} (NLI confidence {conf:.2})", detail_for(v)),
            )
            .anchored(prov)
            .with_refs(refs),
        };
        finding.layer = 3;
        findings.push(finding);
    }
    timing(format!("judge {} claims", claims.len()), t);
    Ok(findings)
}

/// Print elapsed time for a phase when `SHLOMES_TIMING` is set; no-op otherwise.
pub(crate) fn timing(label: impl AsRef<str>, since: std::time::Instant) {
    if std::env::var_os("SHLOMES_TIMING").is_some() {
        eprintln!("[timing] {}: {:.2}s", label.as_ref(), since.elapsed().as_secs_f32());
    }
}

/// Pull candidate behavioural claims from doc prose: lines that reference code
/// (an inline backtick span) and read like a sentence. Each claim's backtick
/// tokens are grounded to the code index. Deliberately heuristic — the NLI judge
/// is the filter, and the confidence threshold keeps weak verdicts out of the
/// report. Skips fenced code, headings, and table rows.
pub fn candidate_claims(text: &str, doc_path: &str, index: &CodeIndex) -> Vec<ProseClaim> {
    let modules = index.module_set();
    let mut out = Vec::new();
    let mut in_fence = false;
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.starts_with("```") || line.starts_with("~~~") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence || line.is_empty() || line.starts_with('#') || line.starts_with('|') {
            continue;
        }
        if !line.contains('`') {
            continue;
        }
        let cleaned = line
            .trim_start_matches(['-', '*', '>', ' ', '\t'])
            .trim()
            .to_string();
        if cleaned.split_whitespace().count() < 6 {
            continue;
        }
        let provenance = ground_claim(&cleaned, index, &modules);
        out.push(ProseClaim {
            text: cleaned,
            doc_ref: format!("{doc_path}:{}", i + 1),
            provenance,
        });
    }
    out
}

/// Ground a claim's backtick tokens to code: each token that names an indexed
/// symbol becomes a symbol anchor (preferred — survives moves), else a module
/// anchor if it matches a real module path. Tokens that match neither (paths,
/// commands, prose) are ignored.
fn ground_claim(line: &str, index: &CodeIndex, modules: &HashSet<String>) -> Provenance {
    let mut prov = Provenance::default();
    for tok in backtick_tokens(line) {
        if let Some(sym) = index.symbols.iter().find(|s| {
            s.qualified_name == tok || s.name == tok || s.qualified_name.ends_with(&format!("::{tok}"))
        }) {
            if !prov.symbols.contains(&sym.qualified_name) {
                prov.symbols.push(sym.qualified_name.clone());
            }
        } else if let Some(m) = modules.iter().find(|m| crate::rules::matches(m, &tok)) {
            if !prov.modules.contains(m) {
                prov.modules.push(m.clone());
            }
        }
    }
    prov
}

/// All backtick-quoted tokens in a line.
fn backtick_tokens(line: &str) -> Vec<String> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"`([^`]+)`").unwrap());
    re.captures_iter(line).map(|c| c[1].to_string()).collect()
}

fn detail_for(verdict: Verdict) -> &'static str {
    match verdict {
        Verdict::Contradicted => "code evidence contradicts this doc claim",
        _ => "no code evidence supports or refutes this claim",
    }
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// Numerically stable softmax over a small logit slice.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return vec![0.0; logits.len()];
    }
    exps.iter().map(|v| v / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn labels_resolved_from_config_by_name() {
        let cfg = json!({ "id2label": { "0": "CONTRADICTION", "1": "ENTAILMENT", "2": "NEUTRAL" } });
        let l = Labels::from_config(&cfg).unwrap();
        assert_eq!((l.contra, l.entail, l.neutral), (0, 1, 2));

        // MoritzLaurer-style ordering must resolve to the same semantics.
        let cfg = json!({ "id2label": { "0": "entailment", "1": "neutral", "2": "contradiction" } });
        let l = Labels::from_config(&cfg).unwrap();
        assert_eq!((l.entail, l.neutral, l.contra), (0, 1, 2));
    }

    #[test]
    fn non_nli_head_is_rejected() {
        let cfg = json!({ "id2label": { "0": "POSITIVE", "1": "NEGATIVE" } });
        assert!(Labels::from_config(&cfg).is_err());
    }

    #[test]
    fn softmax_sums_to_one() {
        let p = softmax(&[2.0, 1.0, 0.1]);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(p[0] > p[1] && p[1] > p[2]);
    }

    #[test]
    fn candidate_claims_filters_noise() {
        let md = "\
# Heading with `code`
The cache invalidates on `write` and never on read.
- short `x`
```
let in_fence = `ignored`;
```
| col | `cell` |
plain sentence without code tokens here at all
";
        let claims = candidate_claims(md, "DOC.md", &CodeIndex::default());
        let texts: Vec<&str> = claims.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, vec!["The cache invalidates on `write` and never on read."]);
        assert_eq!(claims[0].doc_ref, "DOC.md:2");
    }

    #[test]
    fn decide_contradiction_must_win_and_clear_threshold() {
        // Strong contradiction, weaker entailment -> contradicted.
        assert_eq!(decide(0.3, 0.8, 0.5).0, Verdict::Contradicted);
        // Strong entailment -> supported.
        assert_eq!(decide(0.9, 0.2, 0.5).0, Verdict::Supported);
        // Both below threshold -> unverifiable.
        assert_eq!(decide(0.4, 0.45, 0.5).0, Verdict::Unverifiable);
        // Entailment beats an above-threshold contradiction -> supported.
        assert_eq!(decide(0.85, 0.6, 0.5).0, Verdict::Supported);
    }

    #[test]
    fn ground_claim_prefers_symbol_then_module() {
        use crate::code::symbol::{Facts, Span, Symbol, SymbolKind, Visibility};
        let mut index = CodeIndex::default();
        index.symbols.push(Symbol {
            qualified_name: "src/cache::invalidate".into(),
            name: "invalidate".into(),
            kind: SymbolKind::Function,
            visibility: Visibility::Public,
            module: "src/cache".into(),
            span: Span::zero(),
            body_span: Span::zero(),
            signature: None,
            doc: None,
            facts: Facts::default(),
            calls: Vec::new(),
            members: Vec::new(),
        });
        let modules = index.module_set();

        // Symbol token grounds to the symbol; module token grounds to the module.
        let prov = ground_claim("`invalidate` lives in `cache`", &index, &modules);
        assert_eq!(prov.symbols, vec!["src/cache::invalidate"]);
        assert_eq!(prov.modules, vec!["src/cache"]);

        // A token matching nothing is ignored.
        let prov = ground_claim("see `nonexistent_thing` here", &index, &modules);
        assert!(prov.is_empty());
    }
}
