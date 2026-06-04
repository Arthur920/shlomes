"""Verification layers.

Layer 1 (deterministic) is implemented. Layers 2 (retrieval) and 3 (LLM judge)
are declared here as the integration points.
"""

from __future__ import annotations

import os
from pathlib import Path

from doc_aligner.extract import PathClaim
from doc_aligner.findings import Finding, Verdict


def check_paths(claims: list[PathClaim], repo_root: Path) -> list[Finding]:
    """Layer 1: every path a doc names by backtick should exist in the repo."""
    findings: list[Finding] = []
    for c in claims:
        target = repo_root / c.raw
        # Tolerate paths anchored at repo root with or without a leading ./
        exists = target.exists() or any(repo_root.glob(f"**/{c.raw}"))
        if not exists:
            findings.append(
                Finding(
                    verdict=Verdict.STALE,
                    claim=f"references `{c.raw}`",
                    doc_path=f"{c.doc_path}:{c.line}",
                    detail=f"Path `{c.raw}` is named in docs but does not exist in the repo.",
                    layer=1,
                )
            )
    return findings


def retrieve_evidence(*args, **kwargs):  # noqa: D401 - stub
    """Layer 2: embed claims + code, return top-k relevant code chunks.

    Not yet implemented. Will live behind the `[ml]` extra (numpy + an embedding
    backend) and cache vectors by content hash.
    """
    raise NotImplementedError("Layer 2 (retrieval) is not implemented yet.")


def judge_claim(*args, **kwargs):  # noqa: D401 - stub
    """Layer 3: LLM-as-judge over (claim, evidence) -> Verdict.

    Not yet implemented. Will live behind the `[ml]` extra (anthropic SDK).
    """
    raise NotImplementedError("Layer 3 (LLM verification) is not implemented yet.")


def has_ml_extra() -> bool:
    return os.environ.get("ANTHROPIC_API_KEY") is not None
