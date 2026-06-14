<p align="center">
  <img src="shlomes.png" alt="shlomes logo" width="320">
</p>

# shlomes

Catch **documentation drift** — places where your `CLAUDE.md`, READMEs, and
`*.md` docs claim something the code no longer backs up. shlomes checks docs
against the actual codebase and reports what's stale, wrong, or missing.

Fully local and offline. The deterministic core needs no model and is tuned for
**zero false positives**.

**What it catches**, broadly: broken references (paths, commands, env vars,
flags, code symbols), architecture-rule violations parsed from prose, behavioral
contradictions, undocumented public surface, stale diagrams, and drift over time
with a CI alignment score. Full breakdown in [DETAILS.md](DETAILS.md).

## The three layers

```
 Layer 1 — DETERMINISTIC  (no ML, zero false positives)
   paths exist? commands real? config keys present? architecture rules hold?
 Layer 2 — RETRIEVAL  (local embeddings)
   for each surviving claim, fetch the most-relevant code chunks
 Layer 3 — VERIFICATION  (local code-aware NLI cross-encoder)
   (evidence, claim) → supported | contradicted | unverifiable
```

Layer 1 is instant and needs nothing. Layers 2–3 run local ONNX models (no API,
code never leaves the machine) and live behind the `ml` build feature.

## Setup

```bash
# Layer 1 only (no models):
cargo install --path .

# All layers (downloads the Layer 2/3 models on setup):
cargo install --path . --features ml
shlomes setup        # fetch + load every model, fully offline thereafter
```

`shlomes setup` prepares all layers and surfaces any model download error up
front. Then:

```bash
shlomes check                 # full repo (Layer 1)
shlomes check --layer 3       # all three layers
```

The Layer 3 judge is the
[`code-doc-coherence-shlomes`](https://huggingface.co/Arthur920/code-doc-coherence-shlomes)
model on Hugging Face (a `microsoft/unixcoder-base` fine-tune); it downloads on
`setup` / first run. Override any model or threshold via `SHLOMES_*` env vars —
see [DETAILS.md](DETAILS.md#environment-overrides).

## CI integration

`shlomes check` exits non-zero on any finding or a score regression, so it drops
straight into a pipeline. Commit a baseline on your main branch, then gate PRs
against it:

```bash
# once, on the base branch — records the alignment baseline under .shlomes/
shlomes check --write-ledger

# in CI on each PR — fail only if alignment regressed below the baseline
shlomes check --fail-on-regression
```

Example GitHub Actions step:

```yaml
- name: doc-coherence
  run: |
    cargo install --path .          # Layer 1 is fast and dependency-light
    shlomes check --fail-on-regression --format json
```

For behavioral checks in CI, build `--features ml` and run `shlomes setup` (cache
the model download between runs).

## Use it in AI-assisted coding (MCP / agents)

shlomes is a CLI with `--format json`, so any coding agent can run it and read
the findings back. Two ways to wire it in:

**1. As a tool the agent runs directly.** In Claude Code (or any agent with shell
access), just let it call:

```bash
shlomes check --format json --diff main
```

A good standing instruction in `CLAUDE.md`: *"After editing code or docs, run
`shlomes check --format json` and fix any reported drift before finishing."*

**2. As an MCP server.** Expose shlomes over the Model Context Protocol with a
thin command-runner MCP (e.g. a generic "run this CLI" server), mapping a
`check_doc_drift` tool to `shlomes check --format json`. The agent then calls the
tool and receives the structured findings as context — no shell access needed.
The JSON output (one object per finding: layer, verdict, doc ref, code anchor,
detail) is the contract to map onto MCP tool results.

Either way, point the agent at the JSON output and feed `contradicted` / `stale`
findings back as fixes.

## Build

```bash
cargo build                          # debug binary at target/debug/shlomes
cargo test                           # unit tests
cargo build --features ml            # with Layers 2-3
cargo run --features ml -- check --layer 3
```

---

Heavily AI-assisted personal project — see [DETAILS.md](DETAILS.md#about-this-project).
Full feature breakdown, performance, and env overrides also in
[DETAILS.md](DETAILS.md).
