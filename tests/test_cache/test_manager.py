import pytest
import json
from pathlib import Path
from citeguard.cache.manager import CacheManager
from citeguard.models import (
    VerificationResult, ClaimRecord, Verdict, Severity, TokenUsage,
)


def make_result(citation_id: str, verdict: Verdict = Verdict.SUPPORTED) -> VerificationResult:
    return VerificationResult(
        citation_id=citation_id,
        timestamp="2026-01-01T00:00:00Z",
        claim=ClaimRecord(citation_id=citation_id, claim_text="test claim"),
        verdict=verdict,
        confidence=0.9,
        severity=Severity.LOW,
        reasoning="Well supported by evidence.",
    )


def make_cache(tmp_path) -> CacheManager:
    return CacheManager(
        checkpoints_dir=tmp_path / "ckpts",
        index_cache_dir=tmp_path / "idx",
    )


def test_dirs_created_on_init(tmp_path):
    cm = make_cache(tmp_path)
    assert (tmp_path / "ckpts").exists()
    assert (tmp_path / "idx").exists()


def test_save_and_load_checkpoint(tmp_path):
    cm = make_cache(tmp_path)
    cm.save_checkpoint(make_result("ref_001"))
    loaded = cm.load_checkpoint("ref_001")
    assert loaded is not None
    assert loaded.verdict == Verdict.SUPPORTED
    assert loaded.confidence == 0.9


def test_load_nonexistent_returns_none(tmp_path):
    cm = make_cache(tmp_path)
    assert cm.load_checkpoint("ref_999") is None


def test_completed_ids(tmp_path):
    cm = make_cache(tmp_path)
    cm.save_checkpoint(make_result("ref_001"))
    cm.save_checkpoint(make_result("ref_002"))
    ids = cm.completed_citation_ids()
    assert "ref_001" in ids
    assert "ref_002" in ids


def test_checkpoint_is_valid_json(tmp_path):
    cm = make_cache(tmp_path)
    cm.save_checkpoint(make_result("ref_001"))
    path = tmp_path / "ckpts" / "ref_001.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["verdict"] == "SUPPORTED"
    assert data["citation_id"] == "ref_001"


def test_all_results_returns_all(tmp_path):
    cm = make_cache(tmp_path)
    cm.save_checkpoint(make_result("ref_001"))
    cm.save_checkpoint(make_result("ref_002"))
    results = cm.all_results()
    assert len(results) == 2


def test_index_cache_path(tmp_path):
    cm = make_cache(tmp_path)
    path = cm.index_cache_path("pdfs/smith2023 nature.pdf")
    assert path.suffix == ".pkl"
    assert path.parent == tmp_path / "idx"


def test_index_cache_exists(tmp_path):
    cm = make_cache(tmp_path)
    pdf_path = "test.pdf"
    assert not cm.index_cache_exists(pdf_path)
    cache_path = cm.index_cache_path(pdf_path)
    cache_path.write_bytes(b"data")
    assert cm.index_cache_exists(pdf_path)
