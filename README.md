# doc-aligner

A CLI that sanity-checks your `CLAUDE.md`, project docs (`*.md`), and the actual
codebase against each other to surface **coherence drift** — places where the
docs claim something the code no longer (or never did) backs up.

> Status: early scaffold. The deterministic layer works; retrieval + LLM layers
> are stubbed with clear interfaces.

## Architecture: a 3-layer hybrid

```
docs (.md, CLAUDE.md)                         codebase
        │                                         │
        ▼                                         ▼
 ┌──────────────┐                        ┌────────────────┐
 │  extract     │  atomic claims         │  index         │  files, AST facts,
 │  claims      │ ───────────────┐       │  (tree-sitter, │  commands, config keys
 └──────────────┘                │       │   ctags, glob) │
        │                        │       └────────────────┘
        ▼                        │                │
 ┌──────────────────────────────┴────────────────┴───────┐
 │ Layer 1 — DETERMINISTIC  (no ML, zero false positives) │
 │   file paths exist? commands real? env vars/config     │
 │   keys present? entry points valid?                    │
 ├────────────────────────────────────────────────────────┤
 │ Layer 2 — RETRIEVAL  (embeddings + optional reranker)  │
 │   for each surviving claim, fetch most-relevant code   │
 ├────────────────────────────────────────────────────────┤
 │ Layer 3 — VERIFICATION  (LLM-as-judge / NLI)           │
 │   claim + evidence → supported | contradicted | stale  │
 └────────────────────────────────────────────────────────┘
        │
        ▼
   findings report (text / json / sarif)
```

### Layer 1 — Deterministic checks
The cheapest, highest-signal layer. Many doc claims are concrete and verifiable
without any model:
- file/dir paths quoted in docs that don't exist
- commands (`npm test`, `make build`) with no matching script/target
- referenced env vars, config keys, flags
- stated entry points / module paths

Runs in milliseconds, no API cost, no false positives. Catches a large share of
real drift on its own.

### Layer 2 — Retrieval (this is where embeddings belong)
For claims that aren't deterministically checkable ("the cache invalidates on
write"), embed doc claims and code chunks, retrieve the top-k relevant code via
cosine similarity, optionally rerank. Embeddings + AST chunking, cached by
content hash.

### Layer 3 — Verification (LLM-as-judge)
The actual coherence judgment: hand the LLM `(claim, retrieved evidence)` and ask
it to classify `supported / contradicted / stale / unverifiable` with a citation.
This is RAG-style fact-checking, the part embeddings *cannot* do alone.

## Performance / cost

- **Content-hash cache** for embeddings and claim extraction — unchanged files
  are free on re-run.
- **`--diff` mode**: scope a run to files changed vs a git ref, so CI checks only
  touch what moved.

## Usage (planned)

```bash
doc-aligner check                 # full repo
doc-aligner check --diff main     # only what changed vs main
doc-aligner check --format json   # machine-readable findings
doc-aligner check --layer 1       # deterministic only (no API key needed)
```

## Install (dev)

```bash
pip install -e ".[dev]"
pytest
```
