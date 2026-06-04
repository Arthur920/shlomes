# Coverage gaps (the code → doc traversal)

Everything else in shlomes runs **doc → code**: extract a claim from the
docs, verify it against the code (claim ledger, lineage, fingerprints, the
judge). A documentation **gap** is the opposite traversal — **code → doc**:
here's code/behavior that *should* be described; is it?

This is the missing half of the tool. It's a **first-class second traversal**, a
code→doc coverage index — not a Layer-2 afterthought — and it's mostly
deterministic.

## Why it needs its own traversal

The doc→code pipeline starts at a doc assertion, so it structurally cannot see
something the docs never mention. Gap detection must start from the **code's
documentable surface** and ask whether each element is covered. Same machinery,
opposite direction.

## 1. Surface extraction (shared primitive)

Enumerate the *documentable surface* — not every line, the things that warrant
docs. From the same tree-sitter extractor that diagram-diff, rule-checks,
provenance, and fingerprints all need:

- public / exported symbols (pub fns, classes, traits)
- CLI commands & flags
- HTTP routes / RPC endpoints
- config keys, env vars, feature flags
- error / exception types

Gap detection is just one more consumer of this extractor.

## 2. Coverage cross-ref (deterministic, no LLM)

For each surface element: is it referenced/described in any doc? This is the
**exact inverse** of the existence check already built —

| Direction | Check |
|---|---|
| doc → code (built) | a doc names a thing → does it exist in code? |
| code → doc (this)  | a thing exists in code → does any doc name it? |

A surface element with no doc reference → **`undocumented`** verdict. Fully
deterministic, no model.

## 3. Risk-weighting (borrowed from `test_gaps`)

The part that makes this *usable* instead of noise. sentrux's `test_gaps` doesn't
flag every untested file — it cross-references **complexity + import-graph
fan-in** to surface the *risky* untested code. Do the same for docs:

> a doc-gap matters when the undocumented surface is **public AND (complex OR
> high-churn OR high fan-in)**.

Otherwise users drown in "this private helper has no docstring." Report
risk-weighted gaps, ranked — not an exhaustive list. Signals:
- visibility (public > internal > private)
- cyclomatic complexity
- churn (git history)
- fan-in / number of dependents (from the DSM / import graph)

## 4. Co-change / net-new (no LLM)

A symbol added recently that has no doc which ever co-changed with it → a fresh
gap. Reuses the evolutionary-coupling signal from
[drift-detection.md](drift-detection.md), and directly addresses the net-new-code
blind spot noted there.

## 5. Adequacy — the ceiling (LLM)

"Is the documentation *adequate*?" — are error cases, preconditions, and side
effects covered? That's semantic completeness; only the judge can rule on it.
This is a **separate, softer** verdict (`under-documented`) from the deterministic
`undocumented`. Keep them distinct: presence is free and exact; adequacy needs
the LLM and is opt-in.

## New verdicts

The existing vocabulary (missing / stale / structural-mismatch / broken-example /
unsupported / contradicted) has **no verdict for "exists but undocumented"** —
the tell that gaps weren't first-class. Add:

| Verdict | Meaning | Needs LLM? |
|---|---|---|
| `undocumented` | surface element no doc references | ❌ deterministic |
| `under-documented` | documented but missing key behavior (errors, side effects) | ✅ judge |

## Score integration

Coverage becomes its **own dimension** of the alignment score. A PR that adds a
public API with no docs *lowers the score* and trips the base-vs-head regression
gate (see [drift-detection.md](drift-detection.md)) — symmetric with how a broken
claim lowers it. Surface coverage % is reported per-module.

## Open questions

- **Documentable-surface policy**: where's the visibility/complexity cutoff for
  "should be documented"? Per-language defaults + repo override?
- **What counts as "documented"**: a bare name-drop, or a described section? Tie
  to a minimum (e.g. surface element appears in a heading or has surrounding
  prose), else it's `under-documented`, not covered.
- **Reference resolution**: matching a symbol to its mention — exact name, then
  qualified name, then Layer-2 fuzzy for renamed/aliased references.
- **Generated/trivial surface**: exclude generated code, trivial
  getters/setters, test fixtures from the surface.
