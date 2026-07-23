from __future__ import annotations

from hashlib import sha256
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_PATH = REPO_ROOT / "paper.pdf"
EXPECTED_SIZE_BYTES = 18_550_329
EXPECTED_SHA256 = "a3a0a444d34b657541409f7d09a6d8ac58fef82d617c59f47a0ac8b3cfcff36b"


def test_public_paper_matches_reviewed_pdf() -> None:
    payload = PAPER_PATH.read_bytes()

    assert len(payload) == EXPECTED_SIZE_BYTES
    assert payload.startswith(b"%PDF-1.7")
    assert payload.rstrip().endswith(b"%%EOF")
    assert sha256(payload).hexdigest() == EXPECTED_SHA256
