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
| Sequence | participants = real components; messages = real calls/endpoints | call graph / route table |
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

## Layer mapping

- **Layer 1 (deterministic):** parse the diagram; match node labels to real
  files/modules/symbols; diff edges against the import graph. No `ml` needed.
- **Layer 2 (retrieval):** fuzzy labels ("Auth Service", "Worker") with no exact
  name match → embed + retrieve the most likely code unit.
- **Layer 3 (LLM judge):** semantic structure mismatches — "diagram shows
  API → DB direct, but code routes through a cache"; sequence-message ordering.

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
Converting it to a Mermaid `graph` makes doc-aligner able to verify its own
architecture (`extract` / `retrieve` / `verify` modules and their `use` edges)
against `src/` — a clean first fixture.
