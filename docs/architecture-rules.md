# Architecture rules & inspection inspirations

Architecture-inspection tools (sentrux, ArchUnit, dependency-cruiser, NDepend,
Structure101) check **code against rules**. shlomes checks **code against
docs**. They converge the moment the docs *contain* the rules — which
`CLAUDE.md` and architecture docs constantly do ("controllers must not touch the
DB directly", "the domain layer has no framework deps").

Reframed: **shlomes is partly an architecture-inspection tool whose rules are
sourced from the docs instead of a separate config file.** That reframing extends
the LLM-free tier substantially and closes a blind spot we couldn't otherwise
close.

## The key transfer: rules-as-fitness-functions → closes the negative-claim blind spot

Drift-detection's lineage layer has a structural blind spot (see
[drift-detection.md](drift-detection.md)): it can't see **negative/absence
claims** ("we never log PII", "infrastructure doesn't import domain") because
there's no positive code span to anchor provenance to.

Architecture tools solve exactly this with **fitness functions** — declared
invariants verified against the dependency graph. So:

1. **Extract** architectural rules from prose docs:
   - forbidden edge: "controllers must not import the DB layer"
   - layering: "domain depends on nothing; infra depends on domain"
   - forbidden call/symbol: "no `eval`", "no direct `os.environ` outside config"
2. **Compile** each to a dependency-graph / AST query.
3. **Verify** deterministically → a violation is a hard `contradicted` verdict,
   **no LLM**.

This is deterministic contradiction detection for the architectural-claim slice,
and it's exactly the forbidden-pattern check flagged as the one real gap.

### Rule sources
- **Explicit** — a rules file (sentrux uses `.sentrux/rules.toml`; ArchUnit uses
  test code). Hand-authored, precise.
- **Extracted from prose** — the novel bit. Parse architectural assertions out of
  docs and synthesize the rule. Lower precision, so: high-confidence patterns run
  as deterministic rules; ambiguous ones fall through to the Layer-3 judge.

## Reuse, don't rebuild: the dependency graph (DSM)

A Design/Dependency Structure Matrix *is* the ground-truth import graph that both
the rule checks above and **diagram-coherence** (Mermaid edge-diff) need. On this
repo sentrux's DSM reports `size 5, edge_count 1, "Clean layering: all
dependencies flow downward", 0 back-edges` — i.e. it already extracts modules,
edges, cycles, and layering. Borrow (or shell out to) that extractor as the
shared substrate rather than reimplementing AST import-walking three times.

## Other transfers

| Architecture-tool concept | shlomes use |
|---|---|
| **DSM / dependency graph** | substrate for diagram edge-diff + rule checks |
| **Fitness functions** (`check_rules`) | deterministic checks for architectural prose claims; negative-claim handling |
| **Health grading** (A–F, multi-dimensional) | template for the alignment score: per-dimension sub-scores, severity weights, regression gate |
| **Change coupling** (`git_stats`) | evolutionary-coupling staleness signal — see [drift-detection.md](drift-detection.md) |
| **Dead code / cycles** | doc describing dead code → stale-feature candidate; diagram drawing a clean layer over a real cycle → structural contradiction |
| **Test gaps** (`test_gaps`) | direct analog of doc-coverage (undocumented surface = "doc gap") |

## Positioning

The LLM-free tier now reaches: existence + structural + version + broken-example
+ **architectural-rule compliance** + co-change staleness + the regression gate.
The LLM stays confined to the prose-contradiction slice that *isn't* expressible
as a graph/AST rule.

## Open questions
- Rule extraction from prose: which assertion patterns are reliable enough to run
  deterministically vs. defer to the judge? Start with explicit forbidden-edge /
  layering phrasings.
- Rule DSL: adopt an existing format (sentrux `rules.toml`, dependency-cruiser
  config) or define our own keyed to doc provenance?
- Build our own graph extractor vs. depend on an external tool's output?
