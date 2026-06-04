from pathlib import Path

from doc_aligner.extract import extract_path_claims
from doc_aligner.verify import check_paths


def test_missing_path_is_flagged(tmp_path: Path):
    (tmp_path / "real.py").write_text("x = 1\n")
    md = "Entry point is `real.py`, config in `does/not/exist.toml`."
    claims = extract_path_claims(md, "README.md")
    findings = check_paths(claims, tmp_path)

    flagged = {f.claim for f in findings}
    assert "references `does/not/exist.toml`" in flagged
    assert "references `real.py`" not in flagged  # real file -> no finding


def test_clean_repo_has_no_findings(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')\n")
    md = "See `src/main.py`."
    claims = extract_path_claims(md, "README.md")
    assert check_paths(claims, tmp_path) == []
