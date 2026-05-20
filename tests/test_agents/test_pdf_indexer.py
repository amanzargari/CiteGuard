import pytest
from pathlib import Path
from citeguard.agents.pdf_indexer import PDFIndexer, _similarity
from citeguard.config import Settings
from citeguard.models import CitationRecord, CitationFormat
from citeguard.cache.manager import CacheManager


def make_indexer(tmp_path=None) -> PDFIndexer:
    return PDFIndexer(Settings())


def make_citation(marker: str, ref_text: str = "") -> CitationRecord:
    return CitationRecord(
        id="ref_001", raw_marker=marker,
        format=CitationFormat.NUMERIC, position=0,
        reference_text=ref_text,
    )


def test_similarity_identical():
    assert _similarity("hello world", "hello world") == 1.0


def test_similarity_empty():
    assert _similarity("", "something") == 0.0
    assert _similarity("something", "") == 0.0


def test_similarity_different():
    score = _similarity("neural networks", "weather forecasting")
    assert 0.0 <= score < 0.5


def test_index_all_with_real_pdf(tmp_path, sample_pdf):
    """Uses the session-scoped sample_pdf fixture from conftest.py."""
    cache = CacheManager(
        checkpoints_dir=tmp_path / "ckpts",
        index_cache_dir=tmp_path / "idx",
    )
    indexer = PDFIndexer(Settings())
    indexer.index_all([sample_pdf], cache, show_progress=False)
    retriever = indexer.get_retriever(str(sample_pdf))
    assert retriever is not None
    results = retriever.query("accuracy benchmark", top_k=3)
    assert len(results) > 0


def test_index_cached_pdf(tmp_path, sample_pdf):
    """Second index call should use cache, not re-extract."""
    cache = CacheManager(
        checkpoints_dir=tmp_path / "ckpts",
        index_cache_dir=tmp_path / "idx",
    )
    indexer = PDFIndexer(Settings())
    indexer.index_all([sample_pdf], cache, show_progress=False)
    # Second indexer instance should load from cache
    indexer2 = PDFIndexer(Settings())
    indexer2.index_all([sample_pdf], cache, show_progress=False)
    retriever = indexer2.get_retriever(str(sample_pdf))
    assert retriever is not None


def test_get_retriever_none_for_unknown():
    indexer = make_indexer()
    assert indexer.get_retriever("nonexistent.pdf") is None


def test_match_citations_to_pdfs_no_match(tmp_path):
    """Citations with no reference text should not match any PDF."""
    indexer = make_indexer()
    citations = [make_citation("[1]", ref_text="")]
    updated = indexer.match_citations_to_pdfs(citations, {}, [])
    assert updated[0].matched_pdf is None


def test_match_citations_assigns_reference_text(tmp_path, sample_pdf):
    indexer = make_indexer()
    ref_entries = {"[1]": "Smith et al. Citation verification. 2023."}
    citations = [make_citation("[1]")]
    updated = indexer.match_citations_to_pdfs(citations, ref_entries, [sample_pdf])
    assert updated[0].reference_text == "Smith et al. Citation verification. 2023."
