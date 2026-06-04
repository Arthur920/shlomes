"""Shared finding type used by every layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Verdict(str, Enum):
    CONTRADICTED = "contradicted"   # doc claim disagrees with code
    STALE = "stale"                 # doc refers to something that no longer exists
    UNVERIFIABLE = "unverifiable"   # could not gather evidence either way
    SUPPORTED = "supported"         # claim backed by code (not reported by default)


@dataclass
class Finding:
    verdict: Verdict
    claim: str                      # the doc assertion under test
    doc_path: str                   # where the claim came from
    detail: str                     # human-readable explanation
    layer: int                      # 1 deterministic | 2 retrieval | 3 llm
    code_refs: list[str] = field(default_factory=list)  # supporting/conflicting code

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "claim": self.claim,
            "doc_path": self.doc_path,
            "detail": self.detail,
            "layer": self.layer,
            "code_refs": self.code_refs,
        }
