import pytest
from pathlib import Path
from citeguard.pdf.extractor import PDFExtractor
from citeguard.pdf.chunker import TextChunker


def test_extractor_returns_pages(sample_pdf):
    extractor = PDFExtractor()
    pages = extractor.extract(sample_pdf)
    assert len(pages) >= 1
    assert all(isinstance(p["text"], str) for p in pages)
    assert all(isinstance(p["page"], int) for p in pages)


def test_extractor_captures_content(sample_pdf):
    extractor = PDFExtractor()
    pages = extractor.extract(sample_pdf)
    all_text = " ".join(p["text"] for p in pages)
    assert "94.3%" in all_text or "citation" in all_text.lower()


def test_extractor_nonexistent_raises(tmp_path):
    extractor = PDFExtractor()
    with pytest.raises(RuntimeError):
        extractor.extract(tmp_path / "missing.pdf")


def test_chunker_produces_multiple_chunks():
    chunker = TextChunker(chunk_size=20, overlap=5)
    pages = [{"text": "word " * 100, "page": 1}]
    chunks = chunker.chunk(pages, pdf_path="test.pdf")
    assert len(chunks) > 1


def test_chunker_respects_size():
    chunker = TextChunker(chunk_size=20, overlap=5)
    pages = [{"text": "alpha beta gamma delta epsilon " * 20, "page": 1}]
    chunks = chunker.chunk(pages, pdf_path="test.pdf")
    for chunk in chunks:
        assert len(chunk.text.split()) <= 25  # allow overlap slack


def test_chunker_assigns_unique_ids():
    chunker = TextChunker(chunk_size=30, overlap=5)
    pages = [{"text": "word " * 200, "page": 1}]
    chunks = chunker.chunk(pages, pdf_path="test.pdf")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunker_sets_pdf_path():
    chunker = TextChunker(chunk_size=50, overlap=10)
    pages = [{"text": "some content words " * 10, "page": 2}]
    chunks = chunker.chunk(pages, pdf_path="myfile.pdf")
    assert all(c.pdf_path == "myfile.pdf" for c in chunks)
    assert all(c.page == 2 for c in chunks)


def test_chunker_empty_pages():
    chunker = TextChunker(chunk_size=50, overlap=10)
    chunks = chunker.chunk([], pdf_path="test.pdf")
    assert chunks == []
