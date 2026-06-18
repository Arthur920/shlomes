#!/usr/bin/env python3
"""Build a deterministic, class-balanced slice of the CodingNLI holdout split for
the Layer-3 ability benchmark (`nli_holdout_precision_recall` in src/judge.rs).

Source: the model's repo-disjoint eval split (`data/test/`), the same data the
held-out contradiction-precision headline was measured on. Takes a fixed
per-(label) sample, spread evenly across the sorted ids of every language, so the
benchmark is reproducible and balanced.

IMPORTANT: the output is NOT checked in. Its rows are code snippets from many
third-party OSS repos under mixed licenses, and Staleguard is a public repo —
redistributing them would be an attribution/licensing problem. Write it to a
local path and point the harness at it via STALEGUARD_NLI_HOLDOUT:

    python3 tools/gen_holdout_sample.py \
        --src ~/Documents/Personal/CodingNLI/data/test \
        --out /tmp/nli_holdout_sample.jsonl --per-label 120
    STALEGUARD_NLI_HOLDOUT=/tmp/nli_holdout_sample.jsonl \
        cargo test --features ml holdout -- --ignored --nocapture
"""
import argparse, glob, json, os

KEEP = ("premise", "hypothesis", "label", "lang", "repo", "mutation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-label", type=int, default=120)
    args = ap.parse_args()

    by_label = {"entailment": [], "contradiction": [], "neutral": []}
    for path in sorted(glob.glob(os.path.join(os.path.expanduser(args.src), "**", "*.jsonl"), recursive=True)):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                lab = o.get("label")
                if lab in by_label:
                    by_label[lab].append(o)

    rows = []
    for lab, items in by_label.items():
        items.sort(key=lambda o: o["id"])  # deterministic order
        n = len(items)
        k = max(1, n // args.per_label)
        picked = items[::k][: args.per_label]
        rows.extend(picked)

    rows.sort(key=lambda o: (o["label"], o.get("lang", ""), o["premise"]))
    with open(args.out, "w") as f:
        f.write("// Layer-3 ABILITY benchmark: a deterministic, class-balanced slice of the\n")
        f.write("// CodingNLI repo-disjoint holdout split (data/test) — the same generalization\n")
        f.write("// set the held-out contradiction-precision headline was measured on. Generated\n")
        f.write("// by tools/gen_holdout_sample.py; do not hand-edit. label: entailment ->\n")
        f.write("// supported, contradiction -> contradicted, neutral -> unverifiable.\n")
        for o in rows:
            f.write(json.dumps({k: o.get(k) for k in KEEP}, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}")
    for lab, items in by_label.items():
        print(f"  {lab}: pool {len(items)} -> sampled {min(args.per_label, len(items))}")


if __name__ == "__main__":
    main()
