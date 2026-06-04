# Diagram coherence

Design note for checking diagrams in docs against the actual code. Diagrams are
structured data, so most of this is **deterministic (Layer 1)** — unlike prose,
we don't need an LLM just to read a diagram.

## Scope

**Supported formats:** text-based only.

| Format | Source | Nodes | Edges |
|---|---|---|---|
| Mermaid | ` ```mermaid ` blocks | ✅ | ✅ |
| PlantUML | `@startuml` / ` ```plantuml ` | ✅ | ✅ |
| Graphviz DOT | ` ```dot ` blocks, `*.dot` / `*.gv` | ✅ | ✅ |

**Out of scope:** SVG, raster images (PNG/JPG), excalidraw, screenshots. No XML
geometry parsing, no OCR, no vision model. If we ever revisit, it would be a
separate opt-in path.

## What "coherent" means, by diagram type

| Diagram | Coherent with code means | Ground truth |
|---|---|---|
| Architecture / component / flowchart | boxes = real modules/services; arrows = real dependencies | import/dependency graph (tree-sitter) |
| Sequence | participants = real components; messages = real calls, **in order** (→ sequence alignment) | call graph / route table |
| Class | classes/methods/relations exist | AST types |
| ER | entities/fields = schema | DB models / migrations |
| State | states/transitions = a real state machine | enum + transition code |

First target: **architecture / dependency diagrams vs the real import graph** —
highest drift rate, cleanest grounding, mostly deterministic.

## The core check: graph diff

For Mermaid/PlantUML/DOT:

1. Parse the diagram → declared `{nodes, edges}`.
2. Build the **actual** module dependency graph from code (tree-sitter
   extracts imports / `use` / `require`).
3. Set-diff:
   - node in diagram, no matching module → **stale box**
   - module exists, not in diagram → **undocumented component**
   - edge in diagram, no real import → **phantom dependency** (most dangerous)
   - real import not drawn → **missing arrow**

Edge diffs are exactly what humans miss by eye.

The graph-diff above is right for **graph-shaped** diagrams (architecture,
component, class, ER) — order doesn't matter, only the node/edge sets. It is the
wrong tool for **ordered** diagrams, below.

## Ordered diagrams: sequence alignment

A `sequenceDiagram` is not a set of edges — it's an **ordered list of messages**
(A→B, B→C, …). Set-diff throws away order, which is the entire point of a
sequence diagram. The right tool is **sequence alignment** (an adapted
Needleman–Wunsch / Smith–Waterman), the one technique that compares two ordered
sequences while handling insertions and deletions gracefully.

Align the diagram's message sequence to the code's actual call sequence in the
relevant function:

- **match** — step present, right order → coherent
- **deletion** (gap in code) — diagram shows a step the code doesn't do →
  **missing step** / `contradicted`
- **insertion** (gap in diagram) — code makes a call the diagram omits →
  **undrawn step**
- **mismatch / transposition** — wrong call, or right calls out of order

The alignment trace *is* the explanation: "diagram step 3 `validate()` has no
matching call; code inserts `rateLimit()` between steps 2 and 3" — far better UX
than a score.

### The adaptation (open-vocabulary tokens)

Genomic alignment uses a fixed substitution matrix over a tiny alphabet. Our
"symbols" are calls / messages — open-vocabulary and fuzzy — so:

- **Substitution score = semantic similarity**, not a fixed matrix: tokens match
  by resolved symbol identity, or (when names differ) by **embedding cosine**, so
  "diagram: *authenticate user*" aligns to code `auth.login()` with a soft score.
  This is Layer 2's second use.
- **Affine gaps** (open vs extend penalty) so a run of ignorable calls (logging,
  metrics) doesn't tank the alignment — same reason bio uses them for indels.
- **Global (NW)** for "the whole documented flow should match the whole code
  path"; **local (SW)** for "find where this documented subsequence appears in a
  large handler."

The same matcher serves **procedural / install / runbook docs** (ordered steps
vs the real setup script / Dockerfile / CI) even though those aren't diagrams —
it's a shared engine for any ordered artifact, not just sequence diagrams.

**Not** for graph-shaped diagrams: the right relative there is graph edit
distance, never sequence alignment. Don't align a DSM.

**Prerequisite + priority:** needs an **ordered call-sequence extractor**
(tree-sitter, with control flow) that doesn't exist yet — a phase-2 refinement on
top of the core extractor, not foundational. The DP itself is trivial (O(nm) on
5–50-step sequences); the real work is canonicalizing calls into comparable
tokens, which sequence-diagram support needs anyway.

## Layer mapping

- **Layer 1 (deterministic):** parse the diagram; match node labels to real
  files/modules/symbols; diff edges against the import graph. No `ml` needed.
- **Layer 2 (retrieval):** fuzzy labels ("Auth Service", "Worker") with no exact
  name match → embed + retrieve the most likely code unit.
- **Layer 3 (LLM judge):** semantic structure mismatches — "diagram shows
  API → DB direct, but code routes through a cache." (Sequence *ordering* is
  handled deterministically by alignment once calls resolve; the judge only
  enters for genuinely semantic mismatches.)

`ml` only enters for fuzzy label→code matching, never for parsing the diagram.

## Implementation sketch (Rust)

- New module `src/diagram/` with per-format parsers behind a `Diagram { nodes,
  edges, kind, source_span }` type.
- Reuse the markdown walker to pull fenced ` ```mermaid|plantuml|dot ` blocks;
  add `*.dot`/`*.gv` to the file collector.
- Mermaid/DOT: small hand-written parsers for the graph subset (nodes + edges);
  defer full grammar.
- Import graph: tree-sitter per language → emit module→module edges (shared with
  the Layer 2 chunker work).
- Sequence diagrams: an ordered call-sequence extractor (tree-sitter + control
  flow) + an alignment module (adapted NW/SW, affine gaps, similarity-scored
  substitution) — reused for procedural-doc checks. Phase 2.
- Emit `Finding`s with the existing verdict types (`Stale`, `Contradicted`,
  `Unverifiable`).

## Open questions

- Node-label → module matching: exact, then normalized (case/spaces), then
  Layer 2 fuzzy. Where to set the confidence cutoff before reporting?
- Cross-file diagrams: a diagram in `README.md` describing the whole repo vs a
  diagram in a subpackage doc — scope the import graph accordingly?
- Multiple diagrams claiming overlapping structure — reconcile or check each
  independently?

## Dogfood

The README's architecture diagram is currently ASCII (not machine-checkable).
Converting it to a Mermaid `graph` makes shlomes able to verify its own
architecture (`extract` / `retrieve` / `verify` modules and their `use` edges)
against `src/` — a clean first fixture.
