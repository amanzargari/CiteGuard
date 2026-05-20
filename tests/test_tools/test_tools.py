import pytest
from pathlib import Path
from citeguard.tools.session import VerificationSession
from citeguard.tools.retrieve import make_retrieve_tools
from citeguard.tools.verdict import make_verdict_tools
from citeguard.models import CitationRecord, ClaimRecord, CitationFormat, Verdict, Severity
from citeguard.cache.manager import CacheManager
from citeguard.retrieval.bm25 import BM25Retriever
from citeguard.models import ChunkRecord


def make_session() -> VerificationSession:
    citation = CitationRecord(
        id="ref_001", raw_marker="[1]",
        format=CitationFormat.NUMERIC, position=0,
        matched_pdf="test.pdf",
    )
    claim = ClaimRecord(citation_id="ref_001", claim_text="test claim about neural networks")
    return VerificationSession(citation, claim)


def make_retriever() -> BM25Retriever:
    r = BM25Retriever()
    chunks = [
        ChunkRecord(pdf_path="test.pdf", chunk_id="c1", text="neural networks achieve high accuracy", page=1),
        ChunkRecord(pdf_path="test.pdf", chunk_id="c2", text="unrelated weather data", page=2),
    ]
    r.index(chunks)
    return r


def test_session_defaults():
    s = make_session()
    assert s.verdict is None
    assert s.re_query_count == 0
    assert s.evidence_chunks == []


def test_retrieve_chunks_returns_results():
    session = make_session()
    retriever = make_retriever()
    retrieve_chunks, _ = make_retrieve_tools(retriever, session)
    result = retrieve_chunks("neural network accuracy")
    assert "chunks" in result
    assert len(result["chunks"]) > 0
    assert result["chunks"][0]["text"] != ""


def test_retrieve_chunks_no_retriever():
    session = make_session()
    retrieve_chunks, _ = make_retrieve_tools(None, session)
    result = retrieve_chunks("anything")
    assert "error" in result
    assert result["chunks"] == []


def test_retrieve_chunks_extends_evidence():
    session = make_session()
    retriever = make_retriever()
    retrieve_chunks, _ = make_retrieve_tools(retriever, session)
    retrieve_chunks("neural network")
    assert len(session.evidence_chunks) > 0


def test_re_query_increments_count():
    session = make_session()
    retriever = make_retriever()
    _, re_query = make_retrieve_tools(retriever, session, max_requery_rounds=3)
    re_query("refined query about accuracy")
    assert session.re_query_count == 1


def test_re_query_blocked_at_max():
    session = make_session()
    session.re_query_count = 3
    retriever = make_retriever()
    _, re_query = make_retrieve_tools(retriever, session, max_requery_rounds=3)
    result = re_query("another query")
    assert "error" in result
    assert result["chunks"] == []


def test_mark_verdict_records_state():
    session = make_session()
    cache = CacheManager.__new__(CacheManager)  # don't create dirs
    mark_verdict, _ = make_verdict_tools(session, cache)
    result = mark_verdict(
        verdict="SUPPORTED",
        confidence=0.92,
        reasoning="Evidence clearly supports the claim.",
        issues=[],
        severity="LOW",
    )
    assert result["verdict"] == "SUPPORTED"
    assert session.verdict == Verdict.SUPPORTED
    assert session.confidence == 0.92
    assert session.severity == Severity.LOW


def test_mark_verdict_invalid_enum_defaults_to_ambiguous():
    session = make_session()
    cache = CacheManager.__new__(CacheManager)
    mark_verdict, _ = make_verdict_tools(session, cache)
    mark_verdict(verdict="INVALID_VERDICT", confidence=0.5, reasoning="test")
    assert session.verdict == Verdict.AMBIGUOUS


def test_save_checkpoint_writes_file(tmp_path):
    session = make_session()
    cache = CacheManager(
        checkpoints_dir=tmp_path / "ckpts",
        index_cache_dir=tmp_path / "idx",
    )
    mark_verdict, save_checkpoint = make_verdict_tools(session, cache)
    mark_verdict("PARTIAL", 0.7, "Some evidence but incomplete")
    result = save_checkpoint()
    assert result["status"] == "saved"
    assert result["citation_id"] == "ref_001"
    loaded = cache.load_checkpoint("ref_001")
    assert loaded is not None
    assert loaded.verdict == Verdict.PARTIAL
