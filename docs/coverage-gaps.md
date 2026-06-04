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

### Reachability, not just name-presence

Flat "does a doc name this symbol?" is the cheap version. The richer model treats
coverage as a **bipartite graph**: doc-concept nodes on one side, code symbols on
the other, edges where a doc references a symbol. Layered on top of the code's own
**dependency edges** (which the extractor already emits), this lets us
*disambiguate* the two reasons a symbol can be an orphan — which flat presence
cannot:

| Inbound **code** refs | Inbound **doc** refs | Diagnosis |
|---|---|---|
| yes | yes | covered |
| yes | no  | **`undocumented`** — live code no doc reaches |
| no  | no  | **dead code**, not a doc gap (route to architecture-rules, not a finding) |
| no  | yes | doc describes code nothing calls — stale doc / removed feature |

So an orphan with no doc path *and* no callers isn't a documentation gap — it's
dead code, and flagging it as "undocumented" would be noise. The dependency graph
is what tells the two apart. Reachability also generalizes "documented": a symbol
reachable only *through* a documented neighbor (transitively) is weaker coverage
than a directly-described one — a future ranking signal, not a v1 verdict.

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

## 6. Terminology drift (glossary coherence)

`undocumented` means *no doc mentions this concept*. The adjacent failure is when
the doc and the code **both** describe a concept but under **different names** —
docs say "Account," the public API says `Customer`; docs say "tenant," the code
says `Workspace`. The concept is "covered" by a name-presence check yet the
vocabulary has diverged, which is its own coherence defect (and a real source of
onboarding friction).

Scope this **narrowly** to keep it usable:
- only **public / domain** terms — exported types, modules, endpoints, config
  keys — never every identifier. (A hard lexicon gate over all identifiers — the
  strong "Whorfian linter" — is a non-starter: internal helpers and locals would
  drown it in false positives.)
- a **soft finding**, never a build gate.

Detection: build a term set from the documentable surface and a term set from the
docs' domain prose; a public concept whose doc-side synonym scores high on
Layer-2 similarity but mismatches on exact/qualified name → **`term-drift`**.
This deliberately reuses the same exact → qualified → Layer-2-fuzzy ladder as
reference resolution (see Open questions); term-drift is the case where the fuzzy
match succeeds but the exact one didn't.

## New verdicts

The existing vocabulary (missing / stale / structural-mismatch / broken-example /
unsupported / contradicted) has **no verdict for "exists but undocumented"** —
the tell that gaps weren't first-class. Add:

| Verdict | Meaning | Needs LLM? |
|---|---|---|
| `undocumented` | surface element no doc references | ❌ deterministic |
| `under-documented` | documented but missing key behavior (errors, side effects) | ✅ judge |
| `term-drift` | public concept covered, but doc & code names diverge | ⚠️ Layer-2 |

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
