import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from citeguard.agents.pdf_indexer import PDFIndexer
from citeguard.agents.orchestrator import CiteGuardRunner
from citeguard.agents.verifier import VerificationAgent
from citeguard.cache.manager import CacheManager
from citeguard.config import Settings
from citeguard.models import CitationRecord, CitationFormat, ClaimRecord, Verdict
from citeguard.parsing.citations import CitationExtractor
from citeguard.parsing.claims import ClaimSegmenter, BibliographyParser
from citeguard.parsing.manuscript import ManuscriptParser


class FakeProgress:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def add_task(self, *args, **kwargs):
        return 1

    def update(self, *args, **kwargs):
        return None

    def advance(self, *args, **kwargs):
        return None


def make_citation(cid: str, marker: str, matched_pdf: str | None = "test.pdf") -> CitationRecord:
    return CitationRecord(
        id=cid,
        raw_marker=marker,
        format=CitationFormat.NUMERIC,
        position=0,
        matched_pdf=matched_pdf,
    )


def build_runner_with_mocks(monkeypatch):
    runner = CiteGuardRunner(Settings())

    # Patch progress to avoid rich rendering
    monkeypatch.setattr("citeguard.agents.orchestrator.Progress", lambda *a, **k: FakeProgress())

    # Intentional: patch internal collaborators to isolate orchestration behavior.
    runner._parser = MagicMock(spec_set=ManuscriptParser)
    runner._citation_extractor = MagicMock(spec_set=CitationExtractor)
    runner._bib_parser = MagicMock(spec_set=BibliographyParser)
    runner._indexer = MagicMock(spec_set=PDFIndexer)
    runner._verifier = MagicMock(spec_set=VerificationAgent)
    runner._cache = MagicMock(spec_set=CacheManager)
    runner._claim_segmenter = MagicMock(spec_set=ClaimSegmenter)

    runner._parser.parse.return_value = "Some manuscript text"
    runner._bib_parser.parse.return_value = {"[1]": "Ref 1"}

    return runner


@pytest.fixture
def runner(monkeypatch):
    return build_runner_with_mocks(monkeypatch)


@pytest.fixture
def manuscript_pdf(tmp_path):
    manuscript = tmp_path / "paper.md"
    manuscript.write_text("text")
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "a.pdf").write_text("dummy")
    return SimpleNamespace(manuscript=manuscript, pdf_dir=pdf_dir)


def test_run_happy_path_calls_all_steps(runner, manuscript_pdf):
    citations = [make_citation("ref_0001", "[1]"), make_citation("ref_0002", "[2]")]
    runner._citation_extractor.extract.return_value = citations
    runner._indexer.match_citations_to_pdfs.return_value = citations
    runner._cache.completed_citation_ids.return_value = set()
    runner._cache.all_results.return_value = ["result"]

    session = SimpleNamespace(verdict=Verdict.SUPPORTED)
    runner._verifier.verify.return_value = session
    runner._indexer.get_retriever.return_value = object()

    pdf_paths = sorted(manuscript_pdf.pdf_dir.glob("*.pdf"))
    results = runner.run(manuscript_pdf.manuscript, manuscript_pdf.pdf_dir, resume=True)

    runner._parser.parse.assert_called_once_with(manuscript_pdf.manuscript)
    runner._citation_extractor.extract.assert_called_once_with("Some manuscript text")
    runner._bib_parser.parse.assert_called_once_with("Some manuscript text")
    runner._indexer.match_citations_to_pdfs.assert_called_once_with(
        citations,
        {"[1]": "Ref 1"},
        pdf_paths,
    )
    runner._indexer.index_all.assert_called_once_with(pdf_paths, runner._cache)
    assert runner._verifier.verify.call_count == 2
    assert results == ["result"]


def test_run_resume_skips_completed(runner, manuscript_pdf):
    citations = [make_citation("ref_0001", "[1]"), make_citation("ref_0002", "[2]")]
    runner._citation_extractor.extract.return_value = citations
    runner._indexer.match_citations_to_pdfs.return_value = citations
    runner._cache.completed_citation_ids.return_value = {"ref_0001"}
    runner._cache.all_results.return_value = []

    session = SimpleNamespace(verdict=Verdict.SUPPORTED)
    runner._verifier.verify.return_value = session
    runner._indexer.get_retriever.return_value = object()

    runner.run(manuscript_pdf.manuscript, manuscript_pdf.pdf_dir, resume=True)

    # Only one remaining citation should be verified
    assert runner._verifier.verify.call_count == 1


def test_run_unmatched_pdf_passes_none_retriever(runner, manuscript_pdf):
    citations = [make_citation("ref_0001", "[1]", matched_pdf=None)]
    runner._citation_extractor.extract.return_value = citations
    runner._indexer.match_citations_to_pdfs.return_value = citations
    runner._cache.completed_citation_ids.return_value = set()
    runner._cache.all_results.return_value = []

    session = SimpleNamespace(verdict=Verdict.AMBIGUOUS)
    runner._verifier.verify.return_value = session

    runner.run(manuscript_pdf.manuscript, manuscript_pdf.pdf_dir, resume=True)

    args, kwargs = runner._verifier.verify.call_args
    assert kwargs["retriever"] is None
    runner._indexer.get_retriever.assert_not_called()


def test_run_verifier_exception_saves_error_checkpoint(runner, manuscript_pdf):
    citations = [make_citation("ref_0001", "[1]")]
    runner._citation_extractor.extract.return_value = citations
    runner._indexer.match_citations_to_pdfs.return_value = citations
    runner._cache.completed_citation_ids.return_value = set()
    runner._cache.all_results.return_value = []

    runner._verifier.verify.side_effect = RuntimeError("boom")
    runner._indexer.get_retriever.return_value = object()
    runner._claim_segmenter.segment.return_value = ClaimRecord(
        citation_id="ref_0001", claim_text="some claim"
    )

    runner.run(manuscript_pdf.manuscript, manuscript_pdf.pdf_dir, resume=True)

    assert runner._cache.save_checkpoint.called
    saved = runner._cache.save_checkpoint.call_args.args[0]
    assert saved.citation_id == "ref_0001"
    assert saved.verdict == Verdict.ERROR


def test_on_progress_receives_found_event(runner, manuscript_pdf):
    citations = [make_citation("ref_0001", "[1]"), make_citation("ref_0002", "[2]")]
    runner._citation_extractor.extract.return_value = citations
    runner._indexer.match_citations_to_pdfs.return_value = citations
    runner._cache.completed_citation_ids.return_value = set()
    runner._cache.all_results.return_value = []
    runner._verifier.verify.return_value = SimpleNamespace(verdict=Verdict.SUPPORTED, confidence=0.9)
    runner._indexer.get_retriever.return_value = object()

    events = []
    runner.run(manuscript_pdf.manuscript, manuscript_pdf.pdf_dir, on_progress=events.append)

    types = [e["type"] for e in events]
    assert "found" in types
    found = next(e for e in events if e["type"] == "found")
    assert found["citations"] == 2


def test_on_progress_receives_verdict_events(runner, manuscript_pdf):
    citations = [make_citation("ref_0001", "[1]"), make_citation("ref_0002", "[2]")]
    runner._citation_extractor.extract.return_value = citations
    runner._indexer.match_citations_to_pdfs.return_value = citations
    runner._cache.completed_citation_ids.return_value = set()
    runner._cache.all_results.return_value = []
    runner._verifier.verify.return_value = SimpleNamespace(verdict=Verdict.SUPPORTED, confidence=0.9)
    runner._indexer.get_retriever.return_value = object()

    events = []
    runner.run(manuscript_pdf.manuscript, manuscript_pdf.pdf_dir, on_progress=events.append)

    verdict_events = [e for e in events if e["type"] == "verdict"]
    assert len(verdict_events) == 2
    assert verdict_events[0]["verdict"] == "SUPPORTED"


def test_on_progress_none_does_not_raise(runner, manuscript_pdf):
    citations = [make_citation("ref_0001", "[1]")]
    runner._citation_extractor.extract.return_value = citations
    runner._indexer.match_citations_to_pdfs.return_value = citations
    runner._cache.completed_citation_ids.return_value = set()
    runner._cache.all_results.return_value = []
    runner._verifier.verify.return_value = SimpleNamespace(verdict=Verdict.SUPPORTED, confidence=0.9)
    runner._indexer.get_retriever.return_value = object()

    # Should not raise even with no callback
    runner.run(manuscript_pdf.manuscript, manuscript_pdf.pdf_dir, on_progress=None)


def test_on_progress_receives_done_event(runner, manuscript_pdf):
    citations = [make_citation("ref_0001", "[1]")]
    runner._citation_extractor.extract.return_value = citations
    runner._indexer.match_citations_to_pdfs.return_value = citations
    runner._cache.completed_citation_ids.return_value = set()
    runner._cache.all_results.return_value = []
    runner._verifier.verify.return_value = SimpleNamespace(verdict=Verdict.SUPPORTED, confidence=0.9)
    runner._indexer.get_retriever.return_value = object()

    events = []
    runner.run(manuscript_pdf.manuscript, manuscript_pdf.pdf_dir, on_progress=events.append)

    assert any(e["type"] == "done" for e in events)
