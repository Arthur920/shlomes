<p align="center">
  <img src="shlomes.png" alt="shlomes logo" width="320">
</p>

# shlomes

Catch **documentation drift** — places where your `CLAUDE.md`, READMEs, and
`*.md` docs claim something the code no longer (or never did) backs up. shlomes
checks docs against the actual codebase and reports what's stale, wrong, or
missing.

Fully local and offline. The deterministic core needs no model and is tuned for
**zero false positives** — it under-reports rather than ever crying wolf.

## What it catches

**Broken references**
- file/dir paths quoted in docs that don't exist in the repo
- commands (`npm run`, `make`, `cargo --bin`) with no matching script, target,
  or binary in `package.json` / `Makefile` / `Cargo.toml`
- env vars and CLI flags documented but never read in the code
- qualified code refs (`module::symbol`, `Type.method`) that resolve to no symbol

**Architecture violations** — rules parsed straight from prose, checked against
the real import graph:
- forbidden imports — "`controllers` must not import `db`"
- layering — "`domain` depends on nothing", "`api` may only depend on `domain`"
- independence — "`core` is independent of `infra`"
- forbidden symbols — "no direct `os.environ` outside `config`" (text scan +
  resolved references)

**Behavioral contradictions** — a local NLI cross-encoder (no LLM API) judges
prose the deterministic layer can't, e.g. "the cache invalidates on write":
- verdicts `supported` / `contradicted` / `unverifiable`, each with a confidence
- claims ground to symbols, so a verdict re-opens when that code changes

**Coverage gaps**
- public code surface that no doc describes, risk-ranked by fan-in, churn, and
  complexity

**Diagram coherence**
- Mermaid / PlantUML / Graphviz diagrams diffed against the real dependency
  graph — phantom edges, stale boxes, missing arrows

**Drift over time**
- `--diff <ref>` re-checks only what changed since a git ref
- a per-module and repo-wide **alignment score**, with a CI **regression gate**
- fingerprint staleness: a previously-verified claim is flagged when the code
  behind it changes

## Output

- `text` (human) or `json` (machine-readable) findings
- exits non-zero on any reportable finding or a score regression — drop-in for CI

## Usage

```bash
shlomes check                 # full repo (layer 1, deterministic)
shlomes check --diff main     # only what changed vs main
shlomes check --format json   # machine-readable findings
shlomes check --layer 1       # deterministic only (no model needed)
shlomes check --layer 3       # + retrieval + NLI judge (requires the `ml` build)

shlomes index                 # code symbols + module/reference edges (tree-sitter)
shlomes coverage              # public code surface that no doc describes

# Local semantic code search (requires the `ml` build)
shlomes retrieve "where is auth handled" --k 5
```

## Performance & footprint

- **Local & offline** — the jina embedding model (~160 MB) and the NLI
  cross-encoder (`nli-deberta-v3-xsmall`, int8 ONNX ~20 MB) download once, then
  run on-device. Code never leaves the machine.
- **Content-hash caches** — unchanged files and code chunks are free on re-run;
  the embedding model loads only when something actually needs embedding.
- **Symbol-aligned chunking** — code is chunked on tree-sitter symbol boundaries
  (line-window fallback), with an optional reranker (`SHLOMES_RERANK_REPO`).
- **Lean default build** — Layer 1 pulls no ML dependencies. Embeddings and the
  judge live behind the `ml` feature.
- Models and thresholds are overridable via `SHLOMES_NLI_*` / `SHLOMES_RERANK_*`.

## How it works

A 3-layer hybrid: each layer is cheaper and higher-signal than escalating to the
next, so most drift is caught before any model runs.

```
 Layer 1 — DETERMINISTIC  (no ML, zero false positives)
   paths exist? commands real? config keys present? entry points valid?
   architecture rules from prose vs the import graph → contradicted
 Layer 2 — RETRIEVAL  (local embeddings + optional reranker)
   for each surviving claim, fetch the most-relevant code chunks
 Layer 3 — VERIFICATION  (local NLI cross-encoder)
   (evidence, claim) → supported | contradicted | unverifiable
```

Underneath sits a **drift ledger** (Layer 0): it makes runs incremental, scores
alignment, and gates CI on regressions.

## Build

```bash
cargo build                          # debug binary at target/debug/shlomes
cargo test                           # unit tests
cargo run -- check .                 # run against this repo

# Layers 2-3 (local embeddings + NLI judge) are behind the `ml` feature:
cargo build --features ml
cargo run --features ml -- check --layer 3

cargo install --path . --features ml # install the `shlomes` binary (with ml)
```

Layer 1 (deterministic) builds with no extra features. Layers 2 (retrieval) and 3
(NLI judge) live behind the `ml` feature so the default build stays lean.
