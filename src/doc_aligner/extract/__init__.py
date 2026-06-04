"""Pull verifiable claims out of markdown docs.

For now this only surfaces the deterministically checkable claims (paths and
commands). Layer-3 free-text claim extraction (LLM) plugs in alongside these.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# `backtick-quoted` tokens that look like a relative file or dir path.
_PATH_RE = re.compile(r"`([\w./\-]+/[\w./\-]+|[\w\-]+\.[\w]{1,5})`")


@dataclass
class PathClaim:
    raw: str          # the quoted token, e.g. "src/index.ts"
    doc_path: str     # markdown file it appeared in
    line: int


def extract_path_claims(markdown: str, doc_path: str) -> list[PathClaim]:
    """Find backtick-quoted tokens that look like paths the repo should contain."""
    claims: list[PathClaim] = []
    for lineno, line in enumerate(markdown.splitlines(), start=1):
        for m in _PATH_RE.finditer(line):
            token = m.group(1)
            # Skip obvious non-paths (URLs, version specifiers, globs).
            if "://" in token or token.startswith("*") or token.endswith("/*"):
                continue
            claims.append(PathClaim(raw=token, doc_path=doc_path, line=lineno))
    return claims
