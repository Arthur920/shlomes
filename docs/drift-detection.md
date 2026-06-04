# Drift detection (Layer 0)

A cheap, LLM-free layer that sits *underneath* Layers 1–3: it makes coherence
**incremental** (only re-check what changed) and produces a **comparable
alignment score** so CI can flag regressions. Mechanisms borrowed from three
unrelated domains, sharing one data structure.

Origins:
- **Provenance / incremental view maintenance** — databases
- **Semantic (behavioral-fact) baseline** — file-integrity monitoring / Tripwire (security)
- **Score-regression gate** — code-coverage CI gates (codecov-style base-vs-PR)

## Decisions (locked)

| Concern | Decision | Why |
|---|---|---|
| Ledger location | **Committed, facts-only** (no embeddings) | Reproducible across devs, works cold in CI, no thrash on model bumps |
| Fingerprint | **Behavioral facts** (not embeddings) | Catches small-token/high-semantic edits (`3→5`, `if→if!`, dropped `.sort()`); ignores renames |
| Provenance anchor | **Symbol-anchored** (qualified names) | Survives moves/renames that silently break line ranges |
| Temporal | **No SPC.** Alignment score as artifact, CI compares base vs head | No statistical footing needed; works at any volume |

## The shared data structure: the claim ledger

A **committed** sidecar (`.doc-aligner/ledger.json`) with one record per claim,
carried across runs. Deterministic facts only — no embedding vectors (those live
in the gitignored Layer-2 vector cache, recomputed on demand), so the ledger is
commit-safe and merge-friendly.

```
ClaimRecord {
  id:          stable hash of (doc_path + normalized claim text)
  doc_ref:     path:line of the claim
  provenance:  [ Symbol { qualified_name, kind } ]   # symbol-anchored
  facts:       behavioral facts of those symbols (constants, signature,
               control-flow predicates, return shape)
  facts_hash:  hash of `facts` at last verification
  verdict:     supported | contradicted | stale | unverifiable
  commit:      git sha at last verification
}
```

`provenance` and `facts` are a byproduct of the existing layers + a tree-sitter
pass shared with Layer 2's chunker:
- Layer 1 path/symbol claims → the symbol is the reference itself.
- Layer 2 retrieval → the retrieved chunks resolve to their enclosing symbols.
- Layer 3 → the evidence it judged on.

## Mechanism 1 — provenance / lineage invalidation (databases)

A doc is a **materialized view over the code**. DB engines invalidate only the
views whose inputs changed, via lineage. We do the same, **symbol-anchored**:

1. From `git diff` (with rename detection), map changed line ranges → changed
   **symbols** (tree-sitter), not just files.
2. A claim is **dirty** if any of its `provenance` symbols changed *or moved*.
3. Unchanged claims carry forward their last verdict.

Symbol anchoring is what makes this survive a function moving files — the common
refactor that silently defeats line-range lineage.

**Known blind spot (by design):** lineage only sees code a claim already points
at. It cannot catch drift in **net-new code** or **negative/absence claims**
("we never log PII", "no DB access from controllers") — there's no positive span
to anchor. Those need a separate forbidden-pattern / AST-query check; out of
scope for Layer 0 but tracked. Because of this, lineage is a *narrowing
optimization that sits behind a periodic full scan*, never a replacement for it.

## Mechanism 2 — behavioral-fact baseline (Tripwire / FIM)

Tripwire alarms when a file drifts from a known-good baseline. We swap the
*textual* hash for a **behavioral-fact** hash — deliberately **not** an embedding,
which has inverted sensitivity for this task (embeddings barely move on `3→5` or
`if→if!`, yet jump on harmless renames).

For each provenance symbol, extract and hash the facts that actually carry
meaning:
- literal **constants** (`retries = 3`, timeouts, limits)
- **signature** (name, params, types)
- **control-flow predicates** (the conditions, including negation)
- **return shape / type**

A dirty claim recomputes its `facts_hash`; if it differs from the baseline →
flag cheaply (no LLM). Only genuinely ambiguous changes escalate to Layer 2/3.
Deterministic, model-independent, and sensitive to exactly the edits that break
coherence. Fact extraction is the same tree-sitter pass used for provenance.

## Mechanism 3 — alignment score + CI regression gate (coverage gates)

No control charts. Each run produces a single **alignment score** as an artifact
(`.doc-aligner/score.json`): per-module and repo-level, e.g. severity-weighted
`supported / total` over all claims.

The score is the signal:
- **CI** computes the score on the PR head and compares it to the base branch's
  committed score. A regression beyond tolerance **fails the check** — exactly the
  codecov pattern.
- Carry-forward verdicts (Mechanism 1) keep this cheap: only dirty claims are
  re-judged, the aggregate is recomputed, and the scalar is still comparable.

No i.i.d. assumption, no variance estimation, no minimum sample size — it works
on a 3-claim repo and a 3000-claim one. The comparison *is* the early warning.

## Evolutionary coupling — a no-LLM staleness prior

Borrowed from architecture-inspection tools that mine **change coupling** — files
that historically change together (sentrux's `git_stats` reports
`coupling_pairs`; the "logical coupling" of mining-software-repositories
research). Flip it for coherence:

> A doc and the code it describes that *used to co-change but no longer do* = drift.

If `auth.rs` has churned 20× while `docs/auth.md` sits frozen, that's a strong
staleness prior — computed purely from git history, **no model, no embeddings, no
LLM**. It feeds the `stale` verdict as a prior (and helps triage which claims to
re-judge first), independent of the lineage/fingerprint path. See
[architecture-rules.md](architecture-rules.md) for the broader set of
inspection-tool transfers.

## Run pipeline

```
1. load ledger; diff vs ledger.commit (or --diff <ref>) → changed symbols
2. lineage: claims whose provenance symbols changed/moved → dirty      (mech 1)
3. for dirty claims:
     re-extract behavioral facts; facts_hash changed?
       → flag/recheck cheaply; escalate ambiguous to Layer 2/3         (mech 2)
   unchanged claims → carry forward last verdict
4. compute alignment score (per-module + repo)                         (mech 3)
5. write back ledger (facts, hashes, verdicts, commit) + score artifact
6. CI: compare score vs base; regression beyond tolerance → fail
7. report findings + score delta
```

Steps 1–3 are O(diff + dirty); the LLM runs only on the ambiguous remainder.

## Open questions

- **Negative / net-new code**: the lineage blind spot above needs a separate
  forbidden-pattern check. Design separately.
- **Fact extraction coverage**: which behavioral facts per language? Start with
  constants + signatures, expand to predicates/return shape.
- **Score formula + tolerance**: severity weights? absolute vs relative
  regression threshold? per-module gates vs repo-level?
- **Claim identity across edits**: editing a claim's text retires the old record
  (loses its history). Acceptable for a base-vs-head gate, since both sides are
  recomputed — confirm that's fine before relying on history elsewhere.
