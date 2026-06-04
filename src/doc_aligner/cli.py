"""doc-aligner command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from doc_aligner import __version__
from doc_aligner.extract import extract_path_claims
from doc_aligner.findings import Finding
from doc_aligner.verify import check_paths

DOC_GLOBS = ("**/*.md", "**/*.markdown")


def _collect_docs(root: Path) -> list[Path]:
    seen: list[Path] = []
    for pattern in DOC_GLOBS:
        seen.extend(p for p in root.glob(pattern) if ".doc-aligner" not in p.parts)
    return sorted(set(seen))


def _run_check(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for doc in _collect_docs(root):
        text = doc.read_text(encoding="utf-8", errors="replace")
        rel = str(doc.relative_to(root))
        claims = extract_path_claims(text, rel)
        findings.extend(check_paths(claims, root))
    return findings


def _report(findings: list[Finding], fmt: str) -> None:
    if fmt == "json":
        json.dump([f.to_dict() for f in findings], sys.stdout, indent=2)
        sys.stdout.write("\n")
        return
    if not findings:
        print("✓ no coherence issues found")
        return
    for f in findings:
        print(f"[{f.verdict.value}] {f.doc_path}: {f.detail}")
    print(f"\n{len(findings)} finding(s)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="doc-aligner", description=__doc__)
    parser.add_argument("--version", action="version", version=f"doc-aligner {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="check docs against code for coherence drift")
    check.add_argument("path", nargs="?", default=".", help="repo root (default: cwd)")
    check.add_argument("--format", choices=("text", "json"), default="text")
    check.add_argument(
        "--layer", type=int, default=1,
        help="max layer to run: 1 deterministic, 2 +retrieval, 3 +LLM (1 only for now)",
    )

    args = parser.parse_args(argv)

    if args.command == "check":
        root = Path(args.path).resolve()
        if args.layer > 1:
            print("note: layers 2-3 are not implemented yet; running layer 1.", file=sys.stderr)
        findings = _run_check(root)
        _report(findings, args.format)
        return 1 if findings else 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
