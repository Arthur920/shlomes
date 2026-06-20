<p align="center">
  <img src="staleguard.png" alt="Staleguard logo" width="200">
</p>

# Staleguard

<p align="center">
  <a href="https://github.com/Arthur920/Staleguard/actions/workflows/ci.yml"><img src="https://github.com/Arthur920/Staleguard/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/Arthur920/Staleguard/releases"><img src="https://img.shields.io/github/v/release/Arthur920/Staleguard?sort=semver&color=blue" alt="Release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Arthur920/Staleguard?color=green" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/rust-2021-orange?logo=rust" alt="Rust 2021">
  <img src="https://img.shields.io/badge/analyzes-Rust%20%7C%20Python%20%7C%20JS%20%7C%20TS%20%7C%20Java-informational" alt="Languages analyzed">
</p>

Catch **documentation drift** — places where your READMEs, `CLAUDE.md`, and
`*.md` docs claim something the code no longer backs up. Staleguard checks docs
against the actual codebase and reports what's stale, wrong, or missing.

Fully local and offline. The core is **deterministic** — no model, no API — and
tuned for **zero false positives**: every finding points at a concrete path,
command, symbol, or import edge that the docs got wrong.

**What it catches**, broadly: broken references (paths, commands, env vars,
flags, code symbols), architecture-rule violations parsed from prose,
undocumented public surface, stale diagrams, and drift over time with a CI
alignment score. Full breakdown in [DETAILS.md](DETAILS.md).

## How it works

The deterministic core (**Layer 1**) is the tool. It checks docs against the real
codebase — paths, commands, config keys, entry points, and architecture rules
parsed from prose versus the actual import graph — and reports only what it can
prove wrong. It needs no models, runs in ~1.2s on a 330k-line repo, and is what
every install ships with by default.

```
 Layer 1 — DETERMINISTIC  (no ML, zero false positives)
   paths exist? commands real? config keys present? architecture rules hold?
```

### Experimental ML layers (opt-in)

Behind the `ml` build feature sit two local-ONNX layers that try to catch
*behavioral* drift the deterministic core can't see — prose like "the cache
invalidates on write":

```
 Layer 2 — RETRIEVAL  (local embeddings)        for each claim, fetch relevant code
 Layer 3 — VERIFICATION  (NLI cross-encoder)    (evidence, claim) → supported | contradicted | unverifiable
```

These are **experimental and advisory** — treat a `contradicted` verdict as a
high-precision hint to go look, not a gate. If you just want a dependable check,
Layer 1 alone is the recommended use. Method and measured numbers for the curious
are in [DETAILS.md](DETAILS.md#status).

## Install

```bash
# Homebrew (macOS / Linux)
brew install Arthur920/tap/staleguard

# or the install script (macOS / Linux)
curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/Arthur920/Staleguard/releases/latest/download/staleguard-installer.sh | sh

# or from source
cargo install --git https://github.com/Arthur920/Staleguard
```

(Windows: a PowerShell installer is attached to each
[release](https://github.com/Arthur920/Staleguard/releases). Recent Homebrew
versions prompt to trust third-party taps — run `brew trust arthur920/tap` if
asked.)

All of these give you **Layer 1** — the deterministic, zero-false-positive core,
which needs no models. Then:

```bash
staleguard check                 # full repo (Layer 1)
```

### Experimental ML layers (Layers 2–3)

Layers 2–3 are opt-in and advisory (see [the eval](DETAILS.md#status) before you
rely on them). They run local ONNX models and need the `ml` feature, which the
prebuilt binaries omit (the ONNX + embedding deps are large). Two ways to get an
ml-enabled build — both compile from source, then fetch models at runtime:

```bash
# Homebrew (compiles with the ml feature; conflicts with the plain `staleguard`)
brew install Arthur920/tap/staleguard-ml

# or with cargo
cargo install --git https://github.com/Arthur920/Staleguard --features ml
```

Then:

```bash
staleguard setup                 # fetch + load every model, offline thereafter
staleguard check --layer 3       # all three layers
```

`staleguard setup` prepares all layers and surfaces any model download error up
front. (The model files are always a runtime download — there's no separate
"install the models" step.)

The Layer 3 judge is the
[`staleguard`](https://huggingface.co/Arthur920/staleguard)
model on Hugging Face (a `microsoft/unixcoder-base` fine-tune); it downloads on
`setup` / first run. Override any model or threshold via `STALEGUARD_*` env vars —
see [DETAILS.md](DETAILS.md#environment-overrides).

## CI integration

`staleguard check` exits non-zero on any finding or a score regression, so it drops
straight into a pipeline. Commit a baseline on your main branch, then gate PRs
against it:

```bash
# once, on the base branch — records the alignment baseline under .staleguard/
staleguard check --write-ledger

# in CI on each PR — fail only if alignment regressed below the baseline
staleguard check --fail-on-regression
```

### GitHub Action

A reusable action installs the binary and runs the check for you:

```yaml
- uses: Arthur920/Staleguard@v0.2.1
  with:
    args: --fail-on-regression       # passed through to `staleguard check`
```

To get findings as inline PR annotations and entries in the **Security → Code
scanning** tab, emit SARIF and upload it:

```yaml
- uses: Arthur920/Staleguard@v0.2.1
  id: staleguard
  with:
    format: sarif
    args: --min-severity warning   # recommended for first adoption (see below)
- uses: github/codeql-action/upload-sarif@v3
  if: always()
  with:
    sarif_file: ${{ steps.staleguard.outputs.sarif-file }}
```

**First adoption — start at `--min-severity warning`.** A fresh scan of a large
repo surfaces a lot of `undocumented` findings (public surface no doc mentions);
those are `note`-level and advisory. `--min-severity warning` drops them and
keeps only provable drift — broken references and contradictions. Severity ranks
`note` < `warning` < `error`; raise to `--min-severity error` for the strictest
gate, or drop the flag (default `note`) once you want the full coverage report.

Action inputs: `args`, `format` (`text`/`json`/`sarif`), `version`, and
`working-directory`. Or call the binary directly:

```yaml
- run: |
    brew install Arthur920/tap/staleguard   # or: cargo install --git https://github.com/Arthur920/Staleguard
    staleguard check --fail-on-regression --format sarif > staleguard.sarif
```

For behavioral checks in CI, build `--features ml` and run `staleguard setup` (cache
the model download between runs).

### Pre-commit hook

Run the deterministic check locally whenever a doc changes, via
[pre-commit](https://pre-commit.com):

```yaml
# .pre-commit-config.yaml
- repo: https://github.com/Arthur920/Staleguard
  rev: v0.2.1
  hooks:
    - id: staleguard
```

### Configuration (`.staleguard.toml`)

Drop a `.staleguard.toml` at the repo root to tune a run (all keys optional):

```toml
# Doc paths to skip, as globs relative to the repo root
# (`*` within a segment, `**` across segments; a bare name matches in any dir).
exclude = ["docs/legacy/**", "vendor/**", "NOTES.md"]

# Verdict categories to drop from the report and the failing set. One or more of:
# "contradicted", "stale", "unverifiable", "undocumented".
suppress = ["undocumented"]

# Drop everything below this severity (note < warning < error). Same effect as
# `--min-severity`, which overrides it. `warning` hides the undocumented notes.
min_severity = "warning"
```

Neither suppression nor the severity threshold touches the alignment score —
they only filter what gets reported and what gates CI.

## Use it in AI-assisted coding (MCP / agents)

Staleguard is a CLI with `--format json`, so any coding agent can run it and read
the findings back. Two ways to wire it in:

**1. As a tool the agent runs directly.** In Claude Code (or any agent with shell
access), just let it call:

```bash
staleguard check --format json --diff main
```

A good standing instruction in `CLAUDE.md`: *"After editing code or docs, run
`staleguard check --format json` and fix any reported drift before finishing."*

**2. As an MCP server.** Expose staleguard over the Model Context Protocol with a
thin command-runner MCP (e.g. a generic "run this CLI" server), mapping a
`check_doc_drift` tool to `staleguard check --format json`. The agent then calls the
tool and receives the structured findings as context — no shell access needed.
The JSON output (one object per finding: layer, verdict, doc ref, code anchor,
detail) is the contract to map onto MCP tool results.

Either way, point the agent at the JSON output and feed `contradicted` / `stale`
findings back as fixes.

## Build

```bash
cargo build                          # debug binary at target/debug/staleguard
cargo test                           # unit tests
cargo build --features ml            # with Layers 2-3
cargo run --features ml -- check --layer 3
```

---

Heavily AI-assisted personal project — see [DETAILS.md](DETAILS.md#about-this-project).
Full feature breakdown, performance, and env overrides also in
[DETAILS.md](DETAILS.md).
