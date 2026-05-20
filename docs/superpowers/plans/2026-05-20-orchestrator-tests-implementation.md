# Orchestrator Unit Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add lightweight, deterministic unit tests for `CiteGuardRunner.run()` and ensure orchestrator saves ERROR checkpoints when verifier throws.

**Architecture:** Tests will mock all orchestrator collaborators to validate control flow without invoking ADK, PDF parsing, or retrieval. A minimal error-handling path will be added to `CiteGuardRunner` to persist an ERROR checkpoint when verification fails unexpectedly.

**Tech Stack:** Python 3.11+, pytest, unittest.mock, rich

---

## File Map

| File | Responsibility |
|---|---|
| `tests/test_agents/test_orchestrator.py` | Unit tests for `CiteGuardRunner.run()` using mocks |
| `citeguard/agents/orchestrator.py` | Orchestrator logic; add ERROR checkpoint on verifier exception |

---

### Task 1: Add orchestrator unit tests (mocked)

**Files:**
- Create: `tests/test_agents/test_orchestrator.py`

- [ ] **Step 1: Write failing tests**

```python
import pytest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock

from citeguard.agents.orchestrator import CiteGuardRunner
from citeguard.config import Settings
from citeguard.models import CitationRecord, CitationFormat, Verdict


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

    runner._parser = MagicMock()
    runner._citation_extractor = MagicMock()
    runner._bib_parser = MagicMock()
    runner._indexer = MagicMock()
    runner._verifier = MagicMock()
    runner._cache = MagicMock()

    runner._parser.parse.return_value = "Some manuscript text"
    runner._bib_parser.parse.return_value = {"[1]": "Ref 1"}

    return runner


def test_run_happy_path_calls_all_steps(monkeypatch, tmp_path):
    runner = build_runner_with_mocks(monkeypatch)

    citations = [make_citation("ref_0001", "[1]"), make_citation("ref_0002", "[2]")]
    runner._citation_extractor.extract.return_value = citations
    runner._indexer.match_citations_to_pdfs.return_value = citations
    runner._cache.completed_citation_ids.return_value = set()
    runner._cache.all_results.return_value = ["result"]

    session = SimpleNamespace(verdict=Verdict.SUPPORTED)
    runner._verifier.verify.return_value = session
    runner._indexer.get_retriever.return_value = object()

    manuscript = tmp_path / "paper.md"
    manuscript.write_text("text")
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "a.pdf").write_text("dummy")

    results = runner.run(manuscript, pdf_dir, resume=True)

    runner._parser.parse.assert_called_once_with(manuscript)
    runner._citation_extractor.extract.assert_called_once()
    runner._bib_parser.parse.assert_called_once()
    runner._indexer.match_citations_to_pdfs.assert_called_once()
    runner._indexer.index_all.assert_called_once()
    assert runner._verifier.verify.call_count == 2
    assert results == ["result"]


def test_run_resume_skips_completed(monkeypatch, tmp_path):
    runner = build_runner_with_mocks(monkeypatch)

    citations = [make_citation("ref_0001", "[1]"), make_citation("ref_0002", "[2]")]
    runner._citation_extractor.extract.return_value = citations
    runner._indexer.match_citations_to_pdfs.return_value = citations
    runner._cache.completed_citation_ids.return_value = {"ref_0001"}
    runner._cache.all_results.return_value = []

    session = SimpleNamespace(verdict=Verdict.SUPPORTED)
    runner._verifier.verify.return_value = session
    runner._indexer.get_retriever.return_value = object()

    manuscript = tmp_path / "paper.md"
    manuscript.write_text("text")
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "a.pdf").write_text("dummy")

    runner.run(manuscript, pdf_dir, resume=True)

    # Only one remaining citation should be verified
    assert runner._verifier.verify.call_count == 1


def test_run_unmatched_pdf_passes_none_retriever(monkeypatch, tmp_path):
    runner = build_runner_with_mocks(monkeypatch)

    citations = [make_citation("ref_0001", "[1]", matched_pdf=None)]
    runner._citation_extractor.extract.return_value = citations
    runner._indexer.match_citations_to_pdfs.return_value = citations
    runner._cache.completed_citation_ids.return_value = set()
    runner._cache.all_results.return_value = []

    session = SimpleNamespace(verdict=Verdict.AMBIGUOUS)
    runner._verifier.verify.return_value = session

    manuscript = tmp_path / "paper.md"
    manuscript.write_text("text")
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "a.pdf").write_text("dummy")

    runner.run(manuscript, pdf_dir, resume=True)

    args, kwargs = runner._verifier.verify.call_args
    assert kwargs["retriever"] is None


def test_run_verifier_exception_saves_error_checkpoint(monkeypatch, tmp_path):
    runner = build_runner_with_mocks(monkeypatch)

    citations = [make_citation("ref_0001", "[1]")]
    runner._citation_extractor.extract.return_value = citations
    runner._indexer.match_citations_to_pdfs.return_value = citations
    runner._cache.completed_citation_ids.return_value = set()
    runner._cache.all_results.return_value = []

    runner._verifier.verify.side_effect = RuntimeError("boom")
    runner._indexer.get_retriever.return_value = object()

    manuscript = tmp_path / "paper.md"
    manuscript.write_text("text")
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "a.pdf").write_text("dummy")

    runner.run(manuscript, pdf_dir, resume=True)

    assert runner._cache.save_checkpoint.called
    saved = runner._cache.save_checkpoint.call_args.args[0]
    assert saved.citation_id == "ref_0001"
    assert saved.verdict == Verdict.ERROR
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_agents/test_orchestrator.py -v`

Expected: FAIL (verifier exception test should fail because orchestrator does not save ERROR checkpoints yet).

---

### Task 2: Add ERROR checkpoint on verifier exceptions

**Files:**
- Modify: `citeguard/agents/orchestrator.py`

- [ ] **Step 1: Update orchestrator exception handling**

```python
from datetime import datetime, timezone
from citeguard.models import CitationRecord, Verdict, VerificationResult, Severity

# inside the loop, in the exception block:
except Exception as e:
    console.print(f"  [red]ERROR[/red] {citation.id}: {e}")
    verdict = Verdict.ERROR
    error_result = VerificationResult(
        citation_id=citation.id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        claim=claim,
        matched_pdf=citation.matched_pdf,
        verdict=Verdict.ERROR,
        confidence=0.0,
        severity=Severity.HIGH,
        reasoning=str(e),
        issues=[str(e)],
        re_query_count=0,
    )
    self._cache.save_checkpoint(error_result)
```

- [ ] **Step 2: Run tests to verify pass**

Run: `pytest tests/test_agents/test_orchestrator.py -v`

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add citeguard/agents/orchestrator.py tests/test_agents/test_orchestrator.py
git commit -m "test: add orchestrator unit tests and error checkpoint handling"
```

---

## Self-Review Checklist

- [ ] All tests are deterministic and mock external dependencies.
- [ ] Orchestrator error handling persists ERROR checkpoints.
- [ ] Tests pass locally and do not require network access.
- [ ] No changes to runtime behavior outside error checkpoint handling.
