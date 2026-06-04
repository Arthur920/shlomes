# Drift detection (Layer 0)

A cheap, LLM-free layer that sits *underneath* Layers 1–3 and makes coherence
**incremental** (only re-check what changed) and **temporal** (track drift over
time, not just a snapshot). Three mechanisms, borrowed from three unrelated
domains, that share one data structure.

Origins:
- **Provenance / incremental view maintenance** — databases
- **Semantic-hash baseline** — file-integrity monitoring / Tripwire (security)
- **SPC control charts (EWMA / CUSUM)** — manufacturing / Six Sigma

## The shared data structure: the claim ledger

A persisted sidecar (`.doc-aligner/ledger.*`, gitignored) with one record per
claim, carried across runs:

```
ClaimRecord {
  id:           stable hash of (doc_path + normalized claim text)
  doc_ref:      path:line of the claim
  provenance:   [ CodeSpan { path, start_line, end_line } ]   # what it was derived from
  fingerprint:  semantic hash of the provenance at last verification
  verdict:      supported | contradicted | stale | unverifiable
  commit:       git sha at last verification
  score:        per-claim coherence score (for SPC)
}
```

`provenance` and `fingerprint` are produced as a *byproduct* of the existing
layers — no extra work:
- Layer 1 path/symbol claims → the span is the referenced file/symbol.
- Layer 2 retrieval → the retrieved top-k chunks **are** the provenance.
- Layer 3 → the evidence it judged on.

(The `Finding.code_refs` field already exists for this.)

## Mechanism 1 — provenance / lineage invalidation (databases)

A doc is a **materialized view over the code** (the base tables). DB engines
don't recompute a view on every write — they track lineage and invalidate only
the views whose inputs changed.

On a run scoped to a diff (`--diff <ref>`):
1. Get changed line ranges per file from `git diff`.
2. Intersect each claim's `provenance` spans with those ranges.
3. Claims whose lineage touched the diff → **dirty**; everything else is known
   coherent from the last run and skipped.

Turns coherence from "re-scan everything" into "invalidate by lineage." The
instant underlying code changes, the dependent claims are surfaced.

## Mechanism 2 — semantic-hash baseline (Tripwire / FIM)

Host-intrusion tools keep a known-good baseline of file hashes and alarm on
drift. We swap *textual* hashes for **semantic** ones. Each claim records the
fingerprint of the code version it described.

For a dirty claim, recompute the fingerprint of its provenance spans and compare
to the stored baseline:
- **AST fingerprint** — structural hash of the symbol (normalized
  signature + body). Exact, cheap, catches structural change.
- **Embedding fingerprint** — the chunk vector (reuses Layer 2). Drift =
  cosine distance from the stored vector; catches *semantic* change even when a
  refactor barely touches the text.

If the fingerprint moved past a threshold → flag `stale` **immediately, no LLM**.
Only claims that genuinely changed escalate to Layer 2/3. This is the main cost
lever — the expensive LLM judge runs on a tiny fraction of claims.

## Mechanism 3 — SPC over time (manufacturing)

Every other check is a snapshot. SPC monitors a *process*. Append each per-module
coherence score to a time series (one point per commit) and run:
- **EWMA** — exponentially-weighted moving average + control limits; smooths
  noise, flags sustained shifts.
- **CUSUM** — accumulates small deviations; detects **drift onset** (a module
  starting to trend out of spec) faster than a snapshot threshold ever could.

Emits early warnings — "auth/ coherence has been trending down over the last 6
commits" — before anything is grossly wrong. Coherence becomes a monitored
signal with early warning, not a binary pass/fail gate.

## Run pipeline

```
1. load ledger, determine diff vs ledger.commit (or --diff <ref>)
2. lineage invalidation      → mark dirty claims            (mechanism 1)
3. for dirty claims:
     recompute semantic fingerprint
     drifted past threshold?  → verdict = stale, cheaply    (mechanism 2)
     ambiguous?               → escalate to Layer 2/3
4. write back verdicts + fingerprints + commit + scores
5. append per-module scores to time series; run EWMA/CUSUM  (mechanism 3)
6. report: findings + any drift-onset warnings
```

Cost: steps 1–2 are O(diff), step 3's hash is O(dirty), LLM only on the
ambiguous remainder. Steps 5–6 are arithmetic over a small series.

## Open questions

- **Claim identity across edits**: hashing normalized claim text means editing a
  claim retires the old record and starts a new one (losing its history). Accept
  that, or fuzzy-match to carry history forward?
- **Fingerprint threshold**: cosine cutoff for "drifted." Per-language? Learned
  from a repo's own change distribution?
- **Store**: start with JSON; move to SQLite if the time-series/history grows.
- **Cold start**: first run establishes the baseline (record everything, no
  alarms); alarms begin on run two.
- **Score definition**: per-module `supported / total`, or weight by verdict
  severity?
