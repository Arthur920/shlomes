#!/usr/bin/env python3
"""Synthetic mutation eval for the Layer-1 diagram-drift detector.

Ground truth is the repo's own real module import graph (`staleguard index`).
We build a correct-by-construction Mermaid flowchart over an induced subgraph
(drawing EVERY real edge among the chosen nodes, so it is genuinely drift-free),
then inject single, known defects and check that `staleguard check` reports
exactly the expected finding. Three classes:

  * recall   -- missing arrow, phantom edge, stale box are each caught
  * precision -- the drift-free diagram is silent
  * grounding -- real wild false-positive shapes (file extensions, URL routes,
                 decision-node labels) stay silent

The grounding cases pin the fixes made after a 10-repo wild audit, where every
diagram-coherence finding emitted in the wild was a false positive driven by
label noise. The detector's set-diff logic itself was already sound.

Usage:  cargo build --release && python3 tools/diagram_synth_eval.py
Exits non-zero if any case misbehaves (suitable for CI).
"""
import json, subprocess, os, sys, collections

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SG = os.environ.get("STALEGUARD_BIN", os.path.join(REPO, "target", "release", "staleguard"))


def index():
    out = subprocess.run([SG, "index", "--format", "json"], cwd=REPO,
                         capture_output=True, text=True).stdout
    d = json.loads(out)
    edges = {(e["from_module"], e["to_module"]) for e in d["module_edges"]}
    mods = sorted({s["module"] for s in d["symbols"]} |
                  {e["from_module"] for e in d["module_edges"]} |
                  {e["to_module"] for e in d["module_edges"]})
    return edges, mods


def mermaid(nodes, drawn_edges, extra_nodes=()):
    ids = {lab: f"n{i}" for i, lab in enumerate(list(nodes) + list(extra_nodes))}
    lines = ["```mermaid", "flowchart LR"]
    for lab in list(nodes) + list(extra_nodes):
        lines.append(f"  {ids[lab]}[{lab}]")
    for a, b in drawn_edges:
        lines.append(f"  {ids[a]} --> {ids[b]}")
    lines.append("```")
    return "\n".join(lines)


def run_on(markdown):
    """Write a temp doc into the repo, run check, return findings for that doc."""
    path = os.path.join(REPO, "__synth_diag__.md")
    with open(path, "w") as f:
        f.write("# synth\n\n" + markdown + "\n")
    try:
        out = subprocess.run([SG, "check", "--format", "json"], cwd=REPO,
                             capture_output=True, text=True).stdout
        d = json.loads(out)
        fs = d.get("findings", []) if isinstance(d, dict) else d
        return [f for f in fs if f.get("doc_path", "").startswith("__synth_diag__.md")]
    finally:
        os.remove(path)


def problems(fs):
    return [f for f in fs if f.get("verdict") in ("contradicted", "stale", "undocumented")]


def main():
    if not os.path.exists(SG):
        sys.exit(f"binary not found: {SG} (run `cargo build --release` first)")
    edges, mods = index()
    seed = [m for m in mods if m.startswith("src/diagram/")][:6]
    nodes = set(seed)
    for a, b in edges:
        if a in seed:
            nodes.add(b)
    nodes = sorted(nodes)
    induced = sorted((a, b) for (a, b) in edges if a in nodes and b in nodes)
    print(f"induced subgraph: {len(nodes)} nodes, {len(induced)} real edges")
    if len(induced) < 3:
        sys.exit("need a few edges for a meaningful test")

    phantom = next(((a, b) for a in nodes for b in nodes
                    if a != b and (a, b) not in edges and (b, a) not in edges), None)

    failures = 0
    cases = [
        ("correct",       mermaid(nodes, induced), {}),
        ("missing_arrow", mermaid(nodes, induced[1:]), {"undocumented": 1}),
        ("phantom_edge",  mermaid(nodes, induced + [phantom]), {"contradicted": 1}),
        ("stale_box",     mermaid(nodes, induced, extra_nodes=["src/removed/widget"]), {"stale": 1}),
    ]
    print("\n--- representative cases ---")
    for name, md, expected in cases:
        got = dict(collections.Counter(f["verdict"] for f in problems(run_on(md))))
        ok = got == expected
        failures += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: expected {expected or '{}'}, got {got or '{}'}")

    print("\n--- recall sweep (one defect per trial) ---")
    miss = sum(collections.Counter(f["verdict"] for f in problems(
        run_on(mermaid(nodes, induced[:i] + induced[i+1:])))) == {"undocumented": 1}
        for i in range(len(induced)))
    print(f"missing_arrow : {miss}/{len(induced)} caught")
    failures += miss != len(induced)
    nonedges = [(a, b) for a in nodes for b in nodes
                if a != b and (a, b) not in edges and (b, a) not in edges]
    phan = sum(collections.Counter(f["verdict"] for f in problems(
        run_on(mermaid(nodes, induced + [ne])))).get("contradicted") == 1 for ne in nonedges)
    print(f"phantom_edge  : {phan}/{len(nonedges)} caught")
    failures += phan != len(nonedges)

    print("\n--- wild false-positive reproductions (must be silent) ---")
    real_mod = nodes[0]
    fp_cases = [
        ("extension_suffix", [real_mod + ".rs"], "real module written with an extension"),
        ("url_route", ["/items/public/"], "REST URL path in a flow diagram"),
        ("br_tag_label", ["full_tests_needed=False<br/>ok"], "decision-node label with <br/>"),
    ]
    for name, boxes, desc in fp_cases:
        probs = problems(run_on(mermaid([], [], extra_nodes=boxes)))
        ok = not probs
        failures += not ok
        verdicts = dict(collections.Counter(f["verdict"] for f in probs))
        print(f"[{'OK (silent)' if ok else 'FALSE POSITIVE'}] {name}: {desc} -> {verdicts or 'silent'}")

    # ----- fuzzy grounding: conceptual labels never drive edge findings --------
    # Real diagrams label boxes conceptually ("Align", not "src/diagram/align").
    # Fuzzy grounding only *suppresses* stale-box findings; it must NEVER drive a
    # phantom or missing-arrow (conceptual/behavioral arrows aren't import claims —
    # the novu wild-audit false positives). So a Title-cased conceptual phantom
    # pair must stay silent, and an ambiguous segment label must stay silent.
    print("\n--- fuzzy grounding (conceptual labels stay silent for edges) ---")
    # Count modules that contain a given path component, approximating the Rust
    # token overlap: a segment is unambiguous iff exactly one module carries it.
    def norm(seg):  # mirror the Rust token normalization (lowercase + singular)
        s = seg.lower()
        return s[:-1] if len(s) > 3 and s.endswith("s") and not s.endswith("ss") else s

    comp_count = collections.Counter(norm(c) for m in mods for c in m.split("/"))
    fuzzy_nodes = [n for n in nodes if comp_count[norm(n.rsplit("/", 1)[-1])] == 1]
    concept = {n: n.rsplit("/", 1)[-1].replace("_", " ").title() for n in fuzzy_nodes}

    # A drawn edge between two conceptually-labelled boxes (no real import) must
    # NOT produce a phantom — conceptual labels are fuzzy and fuzzy can't drive edges.
    fuzzy_phantom = next(((a, b) for a in fuzzy_nodes for b in fuzzy_nodes
                          if a != b and (a, b) not in edges and (b, a) not in edges), None)
    if fuzzy_phantom and concept[fuzzy_phantom[0]] != concept[fuzzy_phantom[1]]:
        labels = [concept[fuzzy_phantom[0]], concept[fuzzy_phantom[1]]]
        md = mermaid(labels, [(labels[0], labels[1])])
        probs = problems(run_on(md))
        ok = not probs
        failures += not ok
        verdicts = dict(collections.Counter(f["verdict"] for f in probs))
        print(f"[{'OK (silent)' if ok else 'FALSE POSITIVE'}] conceptual phantom "
              f"({labels[0]!r}->{labels[1]!r}) -> {verdicts or 'silent'}")
    else:
        print("[skip] no distinct conceptual phantom pair in this subgraph")

    # Ambiguity probe: a label whose token is shared by >=2 modules must be silent.
    shared = next((seg for seg, c in comp_count.items()
                   if c >= 2 and len(seg) >= 3 and seg not in mods), None)
    # `shared` is already normalized; re-derive a representative original spelling.
    if shared:
        probs = problems(run_on(mermaid([shared.title()], [])))
        ok = not probs
        failures += not ok
        verdicts = dict(collections.Counter(f["verdict"] for f in probs))
        print(f"[{'OK (silent)' if ok else 'FALSE POSITIVE'}] ambiguous label "
              f"{shared.title()!r} (in {comp_count[shared]} modules) -> {verdicts or 'silent'}")
    else:
        print("[skip] no shared segment to probe ambiguity")

    print(f"\n{'ALL GREEN' if not failures else str(failures) + ' FAILURE(S)'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
