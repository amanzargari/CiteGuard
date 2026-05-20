# Orchestrator Unit Tests — Design Spec

**Date:** 2026-05-20
**Status:** Draft
**Project:** CiteGuard

---

## 1. Goal

Add lightweight unit tests for `CiteGuardRunner.run()` that validate orchestration behavior without invoking real ADK, PDF parsing, or retrieval. Tests should be fast, deterministic, and focused on control flow and collaborator interactions.

## 2. Scope

### In Scope
- Unit tests that mock collaborators (`ManuscriptParser`, `CitationExtractor`, `BibliographyParser`, `PDFIndexer`, `VerificationAgent`, `CacheManager`).
- Verification of call order and arguments at a high level.
- Resume behavior with cached checkpoints.
- Handling of missing PDFs (`matched_pdf=None`).
- Handling of verifier exceptions without crashing the run.

### Out of Scope
- End-to-end integration of parsing/indexing/retrieval.
- ADK execution and tool loop behavior.
- LLM calls or network access.
- PDF extraction or chunking fidelity.

## 3. Design Overview

We will add a new test module that instantiates a `CiteGuardRunner`, then replaces its collaborators on the instance with mocks. This preserves the public `run()` logic while preventing expensive or external operations. We will also stub the progress display to avoid terminal-specific behavior in CI.

### Collaborator Mocking
- Patch these instance attributes on `CiteGuardRunner`:
  - `_parser`
  - `_citation_extractor`
  - `_bib_parser`
  - `_indexer`
  - `_verifier`
  - `_cache`

Each mock returns deterministic values so we can validate orchestration flow and calls.

### Progress UI
- Patch `rich.progress.Progress` to a lightweight fake context manager to avoid rendering issues and to keep tests deterministic.

## 4. Test Cases

1. **Happy path orchestration**
   - `parse()` returns text; `extract()` returns citations; `match_citations_to_pdfs()` returns matched citations.
   - `index_all()` is called with PDF paths.
   - `verify()` called once per pending citation.
   - `run()` returns `cache.all_results()`.

2. **Resume logic honored**
   - `completed_citation_ids()` returns a subset of citation IDs.
   - `verify()` is only called for remaining citations.

3. **Unmatched PDFs handled**
   - Citation `matched_pdf=None`.
   - `verify()` still called with `retriever=None`.

4. **Verifier exception handling**
   - `verify()` raises an exception for one citation.
   - `run()` continues and returns `cache.all_results()`.

## 5. Files

- **Create:** `tests/test_agents/test_orchestrator.py`
  - Contains unit tests with mocks and a small helper for fake citations.

## 6. Risks & Mitigations

- **Risk:** Over-mocking may miss integration issues.
  - **Mitigation:** Keep unit tests focused; integration tests can be added later if needed.

- **Risk:** Progress rendering may cause flaky tests.
  - **Mitigation:** Patch `Progress` with a no-op fake context manager.

## 7. Acceptance Criteria

- New test file passes reliably in isolation.
- All tests are deterministic and do not require network or external APIs.
- No changes to orchestrator runtime behavior required.

---

**Next Step:** After approval, write the implementation plan and execute using TDD.
