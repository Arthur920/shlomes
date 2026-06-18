//! Layer 3 verification: an NLI cross-encoder as the coherence judge.
//!
//! The judge is a **code-aware** NLI cross-encoder: `staleguard`,
//! a `microsoft/unixcoder-base` fine-tune over `(code premise, prose claim)` pairs
//! that predicts `{entailment, neutral, contradiction}`. UniXcoder's code-aware
//! pretraining keeps real code *in*-distribution as the premise, which is exactly
//! the failure mode a text-NLI model (MNLI/SNLI on prose) hit — overconfident
//! false contradictions on genuine claims. The model is overridable via
//! `STALEGUARD_NLI_REPO`, with `STALEGUARD_NLI_ONNX` / `STALEGUARD_NLI_THRESHOLD` /
//! `STALEGUARD_NLI_MARGIN` for the artifact and decision knobs.
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
//! (`model_quantized.onnx`, ~121 MB) loaded via `ort`, mirroring the offline
//! model-download path Layer 2 already uses.
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

const DEFAULT_REPO: &str = "Arthur920/staleguard";
const DEFAULT_ONNX: &str = "model_quantized.onnx";
const DEFAULT_THRESHOLD: f32 = 0.5;
/// How far contradiction must out-score entailment *within a single evidence
/// chunk* before that chunk counts as contradicting. Guards against the OOD
/// failure mode where a text-NLI model, fed code it never trained on, scatters
/// near-equal mass onto contradiction and entailment. Override: `STALEGUARD_NLI_MARGIN`.
const DEFAULT_MARGIN: f32 = 0.15;
const MAX_TOKENS: usize = 256;

/// Top-k code chunks retrieved per claim and fed to the judge as evidence.
pub const EVIDENCE_K: usize = 5;
/// Default upper bound on prose claims judged per run — one forward pass per
/// (claim, evidence) pair, so this bounds model cost. See [`max_claims`].
pub const DEFAULT_MAX_CLAIMS: usize = 300;

/// Upper bound on prose claims judged per run, from `STALEGUARD_NLI_MAX_CLAIMS`
/// (default [`DEFAULT_MAX_CLAIMS`]). `0` means no cap — judge every candidate
/// claim, trading runtime for coverage. The judge cost is ~linear in this, so
/// it's the main knob for the Layer-3 time/coverage trade-off.
pub fn max_claims() -> usize {
    std::env::var("STALEGUARD_NLI_MAX_CLAIMS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MAX_CLAIMS)
}

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
            let i: usize = idx
                .parse()
                .map_err(|_| anyhow!("non-numeric label id {idx}"))?;
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
            _ => Err(anyhow!(
                "id2label is not a 3-class NLI head (need entailment/contradiction/neutral)"
            )),
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
    margin: f32,
}

impl Judge {
    /// Fetch (once, then cached) and load the NLI model, tokenizer, and label map.
    pub fn load() -> Result<Judge> {
        let repo_name = env_or("STALEGUARD_NLI_REPO", DEFAULT_REPO);
        let onnx_rel = env_or("STALEGUARD_NLI_ONNX", DEFAULT_ONNX);
        let threshold = std::env::var("STALEGUARD_NLI_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_THRESHOLD);
        let margin = std::env::var("STALEGUARD_NLI_MARGIN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_MARGIN);

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
        let needs_token_types = session
            .inputs()
            .iter()
            .any(|i| i.name() == "token_type_ids");

        Ok(Judge {
            session,
            tokenizer,
            labels,
            needs_token_types,
            threshold,
            margin,
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
        let max_len = encs
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0)
            .max(1);
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

    /// Judge a claim against its retrieved evidence in one batched pass: classify
    /// every `(evidence, claim)` pair, then [`decide`] over the per-chunk scores.
    /// Returns the verdict and the confidence behind it.
    pub fn judge(&mut self, claim: &str, evidence: &[String]) -> Result<(Verdict, f32)> {
        if evidence.is_empty() {
            return Ok((Verdict::Unverifiable, 0.0));
        }
        let pairs: Vec<(&str, &str)> = evidence.iter().map(|ev| (ev.as_str(), claim)).collect();
        let scores = self.classify_batch(&pairs)?;
        Ok(decide(&scores, self.threshold, self.margin))
    }
}

/// The verdict rule (pure, model-free, unit-tested), over per-chunk
/// `[contra, entail, neutral]` probabilities.
///
/// Entailment is pooled as a plain max — any chunk that clearly entails supports
/// the claim. Contradiction is the differentiating signal but also the OOD
/// failure mode (a text-NLI model fed code dumps near-equal mass onto contra and
/// entail), so a chunk only *counts* as contradicting when contradiction is the
/// dominant class **within that chunk**: it beats entailment by `margin` and at
/// least matches neutral. Naively max-pooling contradiction across chunks let one
/// noisy chunk fire a false verdict even when that same chunk was, on balance,
/// more entailing than contradicting. A qualifying contradiction then still has
/// to clear `threshold` and out-weigh the best entailment before we flag drift.
fn decide(scores: &[[f32; 3]], threshold: f32, margin: f32) -> (Verdict, f32) {
    let best_entail = scores.iter().map(|s| s[1]).fold(0.0_f32, f32::max);
    let best_contra = scores
        .iter()
        .filter(|s| s[0] >= s[1] + margin && s[0] >= s[2])
        .map(|s| s[0])
        .fold(0.0_f32, f32::max);

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
pub fn check(
    root: &Path,
    index: &CodeIndex,
    claims: &[ProseClaim],
    k: usize,
) -> Result<Vec<Finding>> {
    if claims.is_empty() {
        return Ok(Vec::new());
    }

    // Evidence selection. Default: Layer-1 grounding + a model-free lexical
    // fallback ([`crate::evidence`]) — no corpus embedding. `STALEGUARD_EMBED_RETRIEVE`
    // restores the embedding retriever. Each entry is (text, path, start_line).
    let t = std::time::Instant::now();
    let per_claim: Vec<Vec<(String, String, usize)>> =
        if std::env::var_os("STALEGUARD_EMBED_RETRIEVE").is_some() {
            let texts: Vec<String> = claims.iter().map(|c| c.text.clone()).collect();
            retrieve::retrieve(root, index, &texts, k)?
                .into_iter()
                .map(|hits| {
                    hits.into_iter()
                        .map(|h| (h.text, h.path, h.start_line))
                        .collect()
                })
                .collect()
        } else {
            let lexicon = crate::evidence::Lexicon::build(index);
            let mut files = crate::evidence::FileCache::new();
            claims
                .iter()
                .map(|c| {
                    crate::evidence::gather(
                        &c.text,
                        &c.provenance,
                        index,
                        &lexicon,
                        root,
                        k,
                        &mut files,
                    )
                    .into_iter()
                    .map(|e| (e.text, e.path, e.start_line))
                    .collect()
                })
                .collect()
        };
    timing(format!("evidence ({} claims)", claims.len()), t);
    let t = std::time::Instant::now();
    let mut judge = Judge::load()?;
    timing("judge model load", t);

    let t = std::time::Instant::now();
    let mut findings = Vec::new();
    for (claim, ev) in claims.iter().zip(per_claim) {
        let evidence: Vec<String> = ev.iter().map(|(text, _, _)| text.clone()).collect();
        let refs: Vec<String> = ev
            .iter()
            .map(|(_, path, line)| format!("{path}:{line}"))
            .collect();
        // Prefer the claim's own grounding (symbols/modules — survives moves and
        // feeds the fingerprint flag); fall back to the evidence files only when
        // the claim grounded to nothing.
        let prov = if claim.provenance.is_empty() {
            Provenance {
                paths: ev.iter().map(|(_, path, _)| path.clone()).collect(),
                ..Default::default()
            }
        } else {
            claim.provenance.clone()
        };

        let (verdict, conf) = judge.judge(&claim.text, &evidence)?;
        let mut finding = match verdict {
            Verdict::Supported => {
                Finding::supported(claim.text.clone(), claim.doc_ref.clone(), prov)
            }
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

/// Print elapsed time for a phase when `STALEGUARD_TIMING` is set; no-op otherwise.
pub(crate) fn timing(label: impl AsRef<str>, since: std::time::Instant) {
    if std::env::var_os("STALEGUARD_TIMING").is_some() {
        eprintln!(
            "[timing] {}: {:.2}s",
            label.as_ref(),
            since.elapsed().as_secs_f32()
        );
    }
}

/// Pull candidate behavioural claims from doc prose: complete sentences/bullets
/// that reference code (an inline backtick span) and read like an assertion. Each
/// claim's backtick tokens are grounded to the code index. Deliberately heuristic
/// — the NLI judge is the filter — but the NLI model is text-trained and brittle,
/// so we only hand it *propositions*: soft-wrapped lines are reassembled into one
/// logical claim (so it isn't judged as a truncated fragment), and sentence
/// fragments and quoted illustrative examples are dropped. Skips fenced code,
/// headings, and table rows.
pub fn candidate_claims(text: &str, doc_path: &str, index: &CodeIndex) -> Vec<ProseClaim> {
    let modules = index.module_set();
    let mut out = Vec::new();
    for (start, block) in logical_lines(text) {
        let cleaned = block
            .trim_start_matches(['-', '*', '>', ' ', '\t'])
            .trim()
            .to_string();
        if cleaned.split_whitespace().count() < 6 || !cleaned.contains('`') {
            continue;
        }
        // The NLI judge can only rule on a complete proposition. Drop the shapes
        // that aren't one: mid-clause fragments (soft-wrap / list continuations),
        // quoted examples of some other rule, `**Bold** — gloss` feature entries,
        // and lowercase-leading list continuations. All skew to false verdicts.
        if is_fragment(&cleaned)
            || is_quoted_example(&cleaned)
            || is_feature_entry(&cleaned)
            || starts_lowercase(&cleaned)
        {
            continue;
        }
        let provenance = ground_claim(&cleaned, index, &modules);
        out.push(ProseClaim {
            text: cleaned,
            doc_ref: format!("{doc_path}:{}", start + 1),
            provenance,
        });
    }
    out
}

/// Collapse markdown prose into logical lines for claim extraction: each list
/// item or paragraph becomes one `(start_line, joined_text)`, with soft-wrapped
/// continuation lines folded in. Fenced code, blank lines, headings, and table
/// rows act as separators (and are never emitted). `start_line` is 0-based.
fn logical_lines(text: &str) -> Vec<(usize, String)> {
    let mut blocks: Vec<(usize, String)> = Vec::new();
    let mut cur: Option<(usize, String)> = None;
    let mut in_fence = false;
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.starts_with("```") || line.starts_with("~~~") {
            in_fence = !in_fence;
            if let Some(b) = cur.take() {
                blocks.push(b);
            }
            continue;
        }
        if in_fence {
            continue;
        }
        // Separators close the current block without starting a new one.
        if line.is_empty() || line.starts_with('#') || line.starts_with('|') {
            if let Some(b) = cur.take() {
                blocks.push(b);
            }
            continue;
        }
        let starts_item = line.starts_with('-')
            || line.starts_with('*')
            || line.starts_with('>')
            || is_numbered_item(line);
        if starts_item {
            if let Some(b) = cur.take() {
                blocks.push(b);
            }
            cur = Some((i, line.to_string()));
        } else if let Some((_, buf)) = cur.as_mut() {
            // Soft-wrapped continuation of the current paragraph/item.
            buf.push(' ');
            buf.push_str(line);
        } else {
            cur = Some((i, line.to_string()));
        }
    }
    if let Some(b) = cur {
        blocks.push(b);
    }
    blocks
}

/// A markdown ordered-list marker: `1.` / `2)` etc. at the start of the line.
fn is_numbered_item(line: &str) -> bool {
    let digits: String = line.chars().take_while(|c| c.is_ascii_digit()).collect();
    !digits.is_empty() && line[digits.len()..].starts_with(['.', ')'])
}

/// True when the line ends mid-clause — a trailing comma/semicolon or a dangling
/// conjunction/preposition/article — i.e. it is a fragment, not a full assertion.
fn is_fragment(s: &str) -> bool {
    let trimmed = s.trim_end_matches(|c: char| c.is_whitespace());
    if trimmed.ends_with(',') || trimmed.ends_with(';') || trimmed.ends_with(':') {
        return true;
    }
    let last = trimmed
        .rsplit(char::is_whitespace)
        .next()
        .unwrap_or("")
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_ascii_lowercase();
    DANGLING.contains(&last.as_str())
}

/// Words that, ending a line, mean the sentence was cut off.
const DANGLING: &[&str] = &[
    "and", "or", "but", "with", "that", "which", "the", "a", "an", "of", "to", "for", "in", "on",
    "at", "by", "from", "as", "into", "than", "then", "if", "when", "where", "while", "via",
];

/// A definition/feature-list entry: a `**bold lead-in**` immediately followed by
/// an em/en-dash gloss (`Local & offline** — the model runs locally`). These
/// describe a feature; they are not checkable propositions about specific code.
/// (The leading `**` is already stripped as a list marker, so we match the close.)
fn is_feature_entry(s: &str) -> bool {
    ["** —", "**—", "** –", "**–"]
        .iter()
        .any(|sep| s.contains(sep))
}

/// A claim that opens with a lowercase letter is a list continuation or sentence
/// fragment — a real assertion opens with a capital or a code span. Leading
/// emphasis markers are unwrapped first; a leading backtick code span is kept.
fn starts_lowercase(s: &str) -> bool {
    let t = s.trim_start_matches(['*', '_', ' ']);
    matches!(t.chars().next(), Some(c) if c.is_ascii_lowercase())
}

/// True when the line's code spans live inside a double-quoted illustrative
/// example (e.g. a rule shown by example: `"`controllers` must not import `db`"`),
/// which asserts nothing about *this* codebase.
fn is_quoted_example(s: &str) -> bool {
    let mut in_quote = false;
    let mut quoted_backtick = false;
    for c in s.chars() {
        match c {
            '"' => {
                if in_quote && quoted_backtick {
                    return true;
                }
                in_quote = !in_quote;
                quoted_backtick = false;
            }
            '`' if in_quote => quoted_backtick = true,
            _ => {}
        }
    }
    false
}

/// Ground a claim's backtick tokens to code: each token that names an indexed
/// symbol becomes a symbol anchor (preferred — survives moves), else a module
/// anchor if it matches a real module path. Tokens that match neither (paths,
/// commands, prose) are ignored.
fn ground_claim(line: &str, index: &CodeIndex, modules: &HashSet<String>) -> Provenance {
    let mut prov = Provenance::default();
    for tok in backtick_tokens(line) {
        if let Some(sym) = index.symbols.iter().find(|s| {
            s.qualified_name == tok
                || s.name == tok
                || s.qualified_name.ends_with(&format!("::{tok}"))
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
        let cfg =
            json!({ "id2label": { "0": "CONTRADICTION", "1": "ENTAILMENT", "2": "NEUTRAL" } });
        let l = Labels::from_config(&cfg).unwrap();
        assert_eq!((l.contra, l.entail, l.neutral), (0, 1, 2));

        // MoritzLaurer-style ordering must resolve to the same semantics.
        let cfg =
            json!({ "id2label": { "0": "entailment", "1": "neutral", "2": "contradiction" } });
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
        assert_eq!(
            texts,
            vec!["The cache invalidates on `write` and never on read."]
        );
        assert_eq!(claims[0].doc_ref, "DOC.md:2");
    }

    #[test]
    fn candidate_claims_reassembles_soft_wrapped_lines() {
        // A soft-wrapped bullet must be judged as one whole proposition, not as a
        // truncated fragment ending in "and".
        let md = "\
- The `judge` reads the resolved symbol body as the premise and
  classifies the `claim` against it, emitting one verdict.
";
        let claims = candidate_claims(md, "DOC.md", &CodeIndex::default());
        assert_eq!(claims.len(), 1);
        assert_eq!(
            claims[0].text,
            "The `judge` reads the resolved symbol body as the premise and classifies the `claim` against it, emitting one verdict."
        );
        assert_eq!(claims[0].doc_ref, "DOC.md:1");
    }

    #[test]
    fn candidate_claims_drops_fragments_and_examples() {
        // A genuinely truncated fragment (no continuation to fold in) and a quoted
        // illustrative rule are both dropped; the real assertion survives.
        let md = "\
commands (`npm run`, `make`, `cargo --bin`) with no matching script, target,

forbidden imports — \"`controllers` must not import `db`\"

The `check` command resolves `Manifests` from the nearest ancestor directory.
";
        let claims = candidate_claims(md, "DOC.md", &CodeIndex::default());
        let texts: Vec<&str> = claims.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(
            texts,
            vec!["The `check` command resolves `Manifests` from the nearest ancestor directory."]
        );
    }

    #[test]
    fn candidate_claims_drops_feature_entries_and_continuations() {
        // A `**Bold** — gloss` feature entry and a lowercase list continuation are
        // not propositions; a capital- or code-led assertion survives.
        let md = "\
- Local & offline** — the `judge` and embedder run on the machine, no API calls.
- commands (`npm run`, `make`) with no matching script or target are flagged.
- `--diff <ref>` re-checks only what changed since a git ref in the working tree.
- The `check` command grounds each `claim` to an indexed symbol before judging.
";
        let claims = candidate_claims(md, "DOC.md", &CodeIndex::default());
        let texts: Vec<&str> = claims.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(
            texts,
            vec![
                "`--diff <ref>` re-checks only what changed since a git ref in the working tree.",
                "The `check` command grounds each `claim` to an indexed symbol before judging.",
            ]
        );
    }

    #[test]
    fn decide_contradiction_must_win_and_clear_threshold() {
        // chunk = [contra, entail, neutral]; default margin 0.15.
        let m = 0.15;
        // Strong, dominant contradiction -> contradicted.
        assert_eq!(decide(&[[0.8, 0.3, 0.0]], 0.5, m).0, Verdict::Contradicted);
        // Strong entailment -> supported.
        assert_eq!(decide(&[[0.2, 0.9, 0.0]], 0.5, m).0, Verdict::Supported);
        // Both below threshold -> unverifiable.
        assert_eq!(
            decide(&[[0.45, 0.4, 0.15]], 0.5, m).0,
            Verdict::Unverifiable
        );
        // Entailment dominates the chunk -> supported (contra filtered by margin).
        assert_eq!(decide(&[[0.6, 0.85, 0.0]], 0.5, m).0, Verdict::Supported);
    }

    #[test]
    fn decide_requires_per_chunk_dominance() {
        let m = 0.15;
        // One chunk entails strongly; another has high contra but its own entail
        // is within the margin, so it does NOT count -> supported, not contradicted.
        let scores = [[0.1, 0.9, 0.0], [0.6, 0.55, 0.0]];
        assert_eq!(decide(&scores, 0.5, m).0, Verdict::Supported);

        // A chunk where contra clears threshold AND dominates by margin -> contradicted.
        let scores = [[0.2, 0.7, 0.1], [0.78, 0.5, 0.0]];
        assert_eq!(decide(&scores, 0.5, m).0, Verdict::Contradicted);

        // No evidence-equivalent: empty -> unverifiable, zero confidence.
        assert_eq!(decide(&[], 0.5, m), (Verdict::Unverifiable, 0.0));
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

    // ---- Layer-3 decision-rule eval harness --------------------------------
    //
    // Drives `decide` over a checked-in labeled corpus of NLI score
    // distributions and reports a confusion matrix. This measures the verdict
    // POLICY (the rule that turns model scores into reported drift) without the
    // 121 MB model, so it runs in normal CI. A *false contradiction* — a verdict
    // of Contradicted where gold is not — is a wrongly-reported drift finding,
    // the Layer-3 analog of the Layer-1 zero-false-positive contract, and is a
    // hard failure here. Overall accuracy is ratcheted. Tuning threshold/margin
    // moves these numbers, so this is the substrate for that trade-off.

    fn verdict_from_gold(s: &str) -> Verdict {
        match s {
            "contradicted" => Verdict::Contradicted,
            "supported" => Verdict::Supported,
            "unverifiable" => Verdict::Unverifiable,
            other => panic!("unknown gold verdict {other:?}"),
        }
    }

    #[test]
    fn nli_decision_corpus_accuracy_and_zero_false_contradictions() {
        let corpus = include_str!("../tests/fixtures/nli_decision_corpus.jsonl");
        let (mut correct, mut total) = (0usize, 0usize);
        let mut false_contras: Vec<String> = Vec::new();

        for (i, raw) in corpus.lines().enumerate() {
            let raw = raw.trim();
            if raw.is_empty() || raw.starts_with("//") {
                continue;
            }
            let v: serde_json::Value = serde_json::from_str(raw)
                .unwrap_or_else(|e| panic!("corpus line {}: {e}\n{raw}", i + 1));
            let chunks: Vec<[f32; 3]> = v["chunks"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| {
                    let a = c.as_array().unwrap();
                    [
                        a[0].as_f64().unwrap() as f32,
                        a[1].as_f64().unwrap() as f32,
                        a[2].as_f64().unwrap() as f32,
                    ]
                })
                .collect();
            let gold = verdict_from_gold(v["gold"].as_str().unwrap());
            let tag = v["tag"].as_str().unwrap_or("?");

            let (got, _) = decide(&chunks, DEFAULT_THRESHOLD, DEFAULT_MARGIN);
            total += 1;
            if got == gold {
                correct += 1;
            } else {
                eprintln!("  MISS line {} [{tag}]: gold {gold:?} got {got:?}", i + 1);
            }
            // A false contradiction is the cardinal Layer-3 sin.
            if got == Verdict::Contradicted && gold != Verdict::Contradicted {
                false_contras.push(format!("  line {} [{tag}]: gold {gold:?}", i + 1));
            }
        }

        let accuracy = correct as f64 / total as f64;
        eprintln!("nli decision eval: correct={correct}/{total} accuracy={accuracy:.3}");

        assert!(
            false_contras.is_empty(),
            "false contradiction(s) — wrongly-reported drift:\n{}",
            false_contras.join("\n")
        );
        assert!(
            accuracy >= 0.95,
            "decision-rule accuracy regression: {accuracy:.3} < 0.95 floor ({correct}/{total})"
        );
    }

    // ---- Layer-3 END-TO-END eval harness (real model) ----------------------
    //
    // Unlike the decision-rule eval above, this drives the *loaded model* over a
    // labeled (claim, evidence) corpus and reports its actual contradiction
    // precision/recall. It is a deliberately ADVERSARIAL recall probe: hard
    // minimal-pair negations and constant swaps (high lexical overlap, opposite
    // meaning) — the slice cross-encoders are worst at. It is NOT an overall
    // accuracy benchmark and does not supersede the disjoint-holdout precision
    // measured at training time; it is the in-tree recall tripwire and the
    // regression baseline for model work.
    //
    // It downloads the 121 MB model, so it is #[ignore]d and runs on demand:
    //     cargo test --features ml e2e -- --ignored --nocapture
    //
    // The floors below are a *measured baseline*, not a target: they lock in the
    // current model's behaviour on this adversarial slice so a future retrain that
    // regresses fails loudly, and they sit just under what the model achieves
    // today. As of the current `Arthur920/staleguard` checkpoint this probe
    // measures roughly: contradiction recall ~0.14, false contradictions = 1.
    // Low recall here reflects the hard minimal-pair slice (the model reads
    // negations with high lexical overlap as supported), NOT the holdout-measured
    // precision — see the module-level note and DETAILS.md. Raise these floors as
    // the model gets better at subtle drift.

    #[derive(serde::Deserialize)]
    struct E2eCase {
        claim: String,
        evidence: Vec<String>,
        gold: String,
        tag: String,
    }

    #[test]
    #[ignore = "downloads the 121 MB NLI model; run with --ignored"]
    fn nli_e2e_corpus_precision_recall() {
        let corpus = include_str!("../tests/fixtures/nli_e2e_corpus.jsonl");
        let mut judge = Judge::load().expect("load NLI model");

        let (mut correct, mut total) = (0usize, 0usize);
        // Contradiction-class confusion: tp = gold&got contra, fp = false contra,
        // fn = missed contra.
        let (mut tp, mut fp, mut fn_) = (0usize, 0usize, 0usize);
        let mut false_contras: Vec<String> = Vec::new();

        for (i, raw) in corpus.lines().enumerate() {
            let raw = raw.trim();
            if raw.is_empty() || raw.starts_with("//") {
                continue;
            }
            let case: E2eCase = serde_json::from_str(raw)
                .unwrap_or_else(|e| panic!("corpus line {}: {e}\n{raw}", i + 1));
            let gold = verdict_from_gold(&case.gold);

            let (got, conf) = judge.judge(&case.claim, &case.evidence).expect("judge");
            total += 1;
            if got == gold {
                correct += 1;
            } else {
                eprintln!(
                    "  MISS line {} [{}]: gold {gold:?} got {got:?} (conf {conf:.2})",
                    i + 1,
                    case.tag
                );
            }
            match (gold == Verdict::Contradicted, got == Verdict::Contradicted) {
                (true, true) => tp += 1,
                (false, true) => {
                    fp += 1;
                    false_contras.push(format!("  line {} [{}]: gold {gold:?}", i + 1, case.tag));
                }
                (true, false) => fn_ += 1,
                (false, false) => {}
            }
        }

        let accuracy = correct as f64 / total as f64;
        let precision = if tp + fp == 0 {
            1.0
        } else {
            tp as f64 / (tp + fp) as f64
        };
        let recall = if tp + fn_ == 0 {
            1.0
        } else {
            tp as f64 / (tp + fn_) as f64
        };
        eprintln!(
            "nli e2e eval: accuracy={accuracy:.3} ({correct}/{total}) | \
             contradiction precision={precision:.3} recall={recall:.3} (tp={tp} fp={fp} fn={fn_})"
        );

        // Baseline floors (see comment above): the goal is fp == 0, but the current
        // model false-fires once, so the regression gate allows <= 1 and the printed
        // list names which. Drive this to 0 on the next retrain.
        assert!(
            fp <= 1,
            "false contradictions {fp} > 1 baseline — wrongly-reported drift regressed:\n{}",
            false_contras.join("\n")
        );
        assert!(
            recall >= 0.10,
            "contradiction recall {recall:.3} below 0.10 baseline — model went blinder to drift (tp={tp} fn={fn_})"
        );
        assert!(
            accuracy >= 0.50,
            "e2e accuracy {accuracy:.3} below 0.50 baseline ({correct}/{total})"
        );
    }

    // ---- Layer-3 ABILITY benchmark (real model, holdout slice) -------------
    //
    // The companion to the adversarial probe above. Drives the loaded model over
    // a deterministic, class-balanced slice of the CodingNLI repo-disjoint holdout
    // split — the same generalization-to-unseen-repos data the project's headline
    // contradiction-precision number was measured on. This is the faithful "what
    // can the model do" benchmark; the minimal-pair probe is the "where is it
    // blind" tripwire. Together they bound the model's real ability.
    //
    // The sample is NOT vendored: its rows are code snippets harvested from many
    // third-party OSS repos under mixed licenses, and this repo is public —
    // redistributing them here would be an attribution/licensing problem. Instead
    // generate it locally from the (private) training data and point the harness
    // at it; absent that, the benchmark skips:
    //     python3 tools/gen_holdout_sample.py --src <CodingNLI>/data/test \
    //         --out /tmp/nli_holdout_sample.jsonl
    //     STALEGUARD_NLI_HOLDOUT=/tmp/nli_holdout_sample.jsonl \
    //         cargo test --features ml holdout -- --ignored --nocapture
    //
    // Each row is a single `(premise = code, hypothesis = claim, label)`. The
    // headline metric is contradiction PRECISION (false contradictions are the
    // cardinal sin); recall and per-class accuracy are also reported. Floors are a
    // measured baseline — run once to set them, then ratchet up on retrain.

    #[derive(serde::Deserialize)]
    struct HoldoutCase {
        premise: String,
        hypothesis: String,
        label: String,
    }

    fn verdict_from_nli_label(s: &str) -> Verdict {
        match s {
            "entailment" => Verdict::Supported,
            "contradiction" => Verdict::Contradicted,
            "neutral" => Verdict::Unverifiable,
            other => panic!("unknown NLI label {other:?}"),
        }
    }

    #[test]
    #[ignore = "downloads the 121 MB NLI model; run with --ignored"]
    fn nli_holdout_precision_recall() {
        let Some(path) = std::env::var_os("STALEGUARD_NLI_HOLDOUT") else {
            eprintln!(
                "skipping holdout benchmark: set STALEGUARD_NLI_HOLDOUT to a sample \
                 generated by tools/gen_holdout_sample.py (not vendored — third-party OSS)"
            );
            return;
        };
        let corpus = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read holdout sample {path:?}: {e}"));
        let mut judge = Judge::load().expect("load NLI model");

        let (mut correct, mut total) = (0usize, 0usize);
        let (mut tp, mut fp, mut fn_) = (0usize, 0usize, 0usize);

        for raw in corpus.lines() {
            let raw = raw.trim();
            if raw.is_empty() || raw.starts_with("//") {
                continue;
            }
            let case: HoldoutCase = serde_json::from_str(raw).expect("parse holdout row");
            let gold = verdict_from_nli_label(&case.label);
            // One premise per row, mirroring how the model was evaluated.
            let (got, _) = judge
                .judge(&case.hypothesis, &[case.premise.clone()])
                .expect("judge");
            total += 1;
            if got == gold {
                correct += 1;
            }
            match (gold == Verdict::Contradicted, got == Verdict::Contradicted) {
                (true, true) => tp += 1,
                (false, true) => fp += 1,
                (true, false) => fn_ += 1,
                (false, false) => {}
            }
        }

        let accuracy = correct as f64 / total as f64;
        let precision = if tp + fp == 0 {
            1.0
        } else {
            tp as f64 / (tp + fp) as f64
        };
        let recall = if tp + fn_ == 0 {
            1.0
        } else {
            tp as f64 / (tp + fn_) as f64
        };
        eprintln!(
            "nli holdout eval (n={total}): accuracy={accuracy:.3} | \
             contradiction precision={precision:.3} recall={recall:.3} (tp={tp} fp={fp} fn={fn_})"
        );

        // Headline gate: contradiction precision on unseen repos. Baseline set from
        // the first measured run — ratchet up as the model improves.
        // Baselines sit just under the first measured run (this sample and the
        // model are deterministic): precision 0.894, recall 0.917, accuracy 0.825.
        assert!(
            precision >= 0.85,
            "contradiction precision {precision:.3} below 0.85 baseline (tp={tp} fp={fp})"
        );
        assert!(
            recall >= 0.85,
            "contradiction recall {recall:.3} below 0.85 baseline (tp={tp} fn={fn_})"
        );
        assert!(
            accuracy >= 0.78,
            "holdout 3-class accuracy {accuracy:.3} below 0.78 baseline ({correct}/{total})"
        );
    }
}
