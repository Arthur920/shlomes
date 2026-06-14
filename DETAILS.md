# shlomes — details

Full breakdown of what shlomes detects, how the layers work, and its performance
profile. For setup, CI, and editor/agent integration see the [README](README.md).

## What it detects

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

## How the layers work

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

## Commands

```bash
shlomes check                 # full repo (layer 1, deterministic)
shlomes check --diff main     # only what changed vs main
shlomes check --format json   # machine-readable findings
shlomes check --layer 3       # + retrieval + NLI judge (requires the `ml` build)
shlomes check --doc README.md # restrict to one doc (cheaper; scopes layer 3)

shlomes index                 # code symbols + module/reference edges (tree-sitter)
shlomes coverage              # public code surface that no doc describes
shlomes retrieve "where is auth handled" --k 5   # local semantic code search (ml)
```

Output is `text` (human) or `json` (machine-readable). `check` exits non-zero on
any reportable finding or a score regression — drop-in for CI.

## Performance & footprint

- **Fast** — the deterministic Layer 1 scans a ~330k-line repo (1,363 source
  files) in **~1.2s** for a full `check` (~0.7s warm) and **under a second**
  for `index`, at ~100 MB peak memory. Per-file parsing runs in parallel
  (rayon) and tree-sitter queries are compiled once and cached, so throughput
  scales with cores.
- **Local & offline** — the jina embedding model (~160 MB) and the code-aware
  NLI cross-encoder (`code-doc-coherence-shlomes`, a UniXcoder fine-tune, int8
  ONNX ~121 MB) download once from the Hub, then run on-device. Code never leaves
  the machine.
- **Content-hash caches** — unchanged files and code chunks are free on re-run;
  the embedding model loads only when something actually needs embedding.
- **Symbol-aligned chunking** — code is chunked on tree-sitter symbol boundaries
  (line-window fallback), with an optional reranker (`SHLOMES_RERANK_REPO`).
- **Lean default build** — Layer 1 pulls no ML dependencies. Embeddings and the
  judge live behind the `ml` feature.

## Layer 3 cost

Layer 3 is the heaviest pass — one cross-encoder forward per (claim × evidence)
chunk, ~0.14s/claim on CPU. It caps at `SHLOMES_NLI_MAX_CLAIMS` claims per run
(default 300; `0` = no cap), the main time/coverage knob. On a large repo a
capped run is ~45s; uncapped scales linearly with claim count.

## Environment overrides

| Variable | Effect |
|---|---|
| `SHLOMES_NLI_REPO` | NLI judge model repo (default `Arthur920/code-doc-coherence-shlomes`) |
| `SHLOMES_NLI_ONNX` | ONNX artifact within the repo |
| `SHLOMES_NLI_THRESHOLD` / `SHLOMES_NLI_MARGIN` | decision thresholds |
| `SHLOMES_NLI_MAX_CLAIMS` | per-run claim budget (default 300; `0` = no cap) |
| `SHLOMES_EMBED_REPO` / `SHLOMES_EMBED_ONNX` | Layer 2 embedding model |
| `SHLOMES_RERANK_REPO` | optional reranker |
| `SHLOMES_ORT_THREADS` | ONNX intra-op threads (default: all cores) |

## Status

- **Layers 1–2** are the stable core and what most runs should rely on. Layer 1
  is deterministic and tuned to under-report rather than false-alarm.
- **Layer 3** (the NLI judge) is newer. The default model,
  [`code-doc-coherence-shlomes`](https://huggingface.co/Arthur920/code-doc-coherence-shlomes),
  is a `microsoft/unixcoder-base` fine-tune trained specifically for this
  task — code-aware, so real code stays in-distribution as the premise (an
  earlier text-NLI model did not, and produced overconfident false
  contradictions). Treat its verdicts as advisory and review contradictions
  before acting.

## About this project

shlomes is a personal project, and its development was **heavily AI-assisted** —
most of the implementation was written with AI coding tools, with me directing
the architecture, the layer design, the evaluation, and the call on what was good
enough to keep. The custom Layer 3 model was likewise trained and evaluated as
part of that loop. I'm flagging this plainly rather than dressing it up: the
ideas and the judgment calls are mine, a large share of the code is not
hand-typed, and the deterministic core is built to be auditable precisely because
I don't expect anyone (including me) to take generated code on faith.
