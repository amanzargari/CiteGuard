# CiteGuard Web UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local FastAPI web UI to CiteGuard with real-time SSE progress streaming, so users can run verifications and watch results in a browser.

**Architecture:** FastAPI serves a single HTML page; a background thread runs `CiteGuardRunner`; a `queue.Queue` bridges the thread to an SSE endpoint (`GET /stream`) that the browser listens to with `EventSource`. The orchestrator gains an optional `on_progress` callback — existing CLI behavior is unchanged.

**Tech Stack:** FastAPI, uvicorn, Jinja2, python-multipart, vanilla JS + EventSource (no npm, no build step).

---

## File Map

| File | Action |
|---|---|
| `pyproject.toml` | Modify — add 4 new dependencies |
| `requirements.txt` | Modify — add same 4 dependencies |
| `citeguard/agents/orchestrator.py` | Modify — add `on_progress` callback to `run()` |
| `citeguard/web/__init__.py` | Create — empty package marker |
| `citeguard/web/app.py` | Create — FastAPI app, routes, SSE, job state |
| `citeguard/web/templates/index.html` | Create — single-page UI |
| `citeguard/cli.py` | Modify — add `web` command |

---

## Task 1: Add dependencies

**Files:**
- Modify: `pyproject.toml`
- Modify: `requirements.txt`

- [ ] **Step 1: Add to pyproject.toml**

In `pyproject.toml`, add these four lines to the `dependencies` list (after the existing entries):

```toml
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.29.0",
    "jinja2>=3.1.0",
    "python-multipart>=0.0.9",
```

The full `dependencies` block becomes:

```toml
dependencies = [
    "google-adk>=1.0.0",
    "litellm>=1.40.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "pymupdf>=1.24.0",
    "rank-bm25>=0.2.2",
    "sentence-transformers>=3.0.0",
    "python-docx>=1.1.0",
    "click>=8.1.0",
    "rich>=13.0.0",
    "reportlab>=4.0.0",
    "PyYAML>=6.0.0",
    "tenacity>=8.0.0",
    "httpx>=0.27.0",
    "python-dotenv>=1.0.0",
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.29.0",
    "jinja2>=3.1.0",
    "python-multipart>=0.0.9",
]
```

- [ ] **Step 2: Add to requirements.txt**

Append to `requirements.txt`:

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
jinja2>=3.1.0
python-multipart>=0.0.9
```

- [ ] **Step 3: Install**

```bash
pip install fastapi "uvicorn[standard]" jinja2 python-multipart
```

Expected: installs without errors.

- [ ] **Step 4: Smoke test**

```bash
python -c "import fastapi, uvicorn, jinja2, multipart; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml requirements.txt
git commit -m "chore: add fastapi, uvicorn, jinja2, python-multipart dependencies"
```

---

## Task 2: Add on_progress callback to orchestrator

**Files:**
- Modify: `citeguard/agents/orchestrator.py`
- Modify: `tests/test_agents/test_orchestrator.py`

The orchestrator's `run()` gains an optional `on_progress: Callable[[dict], None] | None = None` parameter. An internal `_emit` helper calls it at each milestone. All existing `console.print` calls stay — the callback is purely additive.

- [ ] **Step 1: Write failing tests**

Add these tests to `tests/test_agents/test_orchestrator.py` (append after the existing tests):

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_agents/test_orchestrator.py::test_on_progress_receives_found_event -v
```

Expected: FAIL with `TypeError: run() got an unexpected keyword argument 'on_progress'`

- [ ] **Step 3: Update orchestrator**

Replace `citeguard/agents/orchestrator.py` with:

```python
from __future__ import annotations
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TaskProgressColumn, TimeElapsedColumn,
)

from citeguard.config import Settings
from citeguard.models import Verdict, VerificationResult, Severity
from citeguard.parsing.manuscript import ManuscriptParser
from citeguard.parsing.citations import CitationExtractor
from citeguard.parsing.claims import ClaimSegmenter, BibliographyParser
from citeguard.agents.pdf_indexer import PDFIndexer
from citeguard.agents.verifier import VerificationAgent
from citeguard.cache.manager import CacheManager

console = Console()

_VERDICT_STYLE: dict[str, str] = {
    Verdict.SUPPORTED: "green",
    Verdict.PARTIAL: "yellow",
    Verdict.UNSUPPORTED: "red",
    Verdict.EXAGGERATED: "orange3",
    Verdict.FABRICATED: "bold red",
    Verdict.AMBIGUOUS: "dim",
    Verdict.UNVERIFIABLE: "blue",
    Verdict.ERROR: "bold magenta",
}


class CiteGuardRunner:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cache = CacheManager(
            checkpoints_dir=Path(settings.output.checkpoints_dir),
            index_cache_dir=Path(settings.output.index_cache_dir),
        )
        self._indexer = PDFIndexer(settings)
        self._verifier = VerificationAgent(settings)
        self._parser = ManuscriptParser()
        self._citation_extractor = CitationExtractor()
        self._claim_segmenter = ClaimSegmenter(window=2)
        self._bib_parser = BibliographyParser()

    def run(
        self,
        manuscript_path: Path,
        pdf_folder: Path,
        resume: bool = True,
        on_progress: Callable[[dict], None] | None = None,
    ) -> list:
        def _emit(event_type: str, **data: object) -> None:
            if on_progress is not None:
                on_progress({"type": event_type, **data})

        console.rule("[bold blue]CiteGuard")
        console.print(
            f"Manuscript: [cyan]{manuscript_path}[/cyan]  "
            f"PDFs: [cyan]{pdf_folder}[/cyan]  "
            f"Strictness: [yellow]{self._settings.strictness}[/yellow]  "
            f"Backend: [yellow]{self._settings.retrieval_backend}[/yellow]"
        )

        # 1. Parse manuscript
        console.print("\n[bold]Step 1/4:[/bold] Parsing manuscript...")
        _emit("step", message="Parsing manuscript...")
        text = self._parser.parse(manuscript_path)
        citations = self._citation_extractor.extract(text)
        bib_entries = self._bib_parser.parse(text)
        console.print(f"  Found [green]{len(citations)}[/green] citations")

        # 2. Discover PDFs
        pdf_paths = sorted(pdf_folder.glob("*.pdf"))
        console.print(f"  Found [green]{len(pdf_paths)}[/green] PDFs in {pdf_folder}")

        # 3. Match citations to PDFs
        console.print("\n[bold]Step 2/4:[/bold] Matching citations to PDFs...")
        _emit("step", message="Matching citations to PDFs...")
        citations = self._indexer.match_citations_to_pdfs(citations, bib_entries, pdf_paths)
        matched = sum(1 for c in citations if c.matched_pdf)
        console.print(f"  Matched [green]{matched}[/green] / {len(citations)} citations to PDFs")
        _emit("found", citations=len(citations), pdfs=len(pdf_paths), matched=matched)

        # 4. Index PDFs
        console.print("\n[bold]Step 3/4:[/bold] Indexing PDFs...")
        _emit("step", message=f"Indexing {len(pdf_paths)} PDFs...")
        self._indexer.index_all(pdf_paths, self._cache)
        _emit("step", message="Indexing complete")

        # 5. Determine which citations to verify
        done_ids = self._cache.completed_citation_ids() if resume else set()
        pending = [c for c in citations if c.id not in done_ids]

        if done_ids:
            console.print(
                f"\n[bold]Resuming:[/bold] {len(done_ids)} already verified, "
                f"{len(pending)} remaining"
            )

        # 6. Verify pending citations
        console.print(f"\n[bold]Step 4/4:[/bold] Verifying {len(pending)} citations...\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Verifying", total=len(pending))

            for i, citation in enumerate(pending):
                progress.update(
                    task,
                    description=f"Verifying [cyan]{citation.id}[/cyan] {citation.raw_marker[:30]}",
                )
                _emit(
                    "verifying",
                    current=i + 1,
                    total=len(pending),
                    citation_id=citation.id,
                    marker=citation.raw_marker,
                )
                claim = self._claim_segmenter.segment(text, citation)
                retriever = (
                    self._indexer.get_retriever(citation.matched_pdf)
                    if citation.matched_pdf else None
                )
                confidence = 0.0
                try:
                    session = self._verifier.verify(
                        citation=citation,
                        claim=claim,
                        retriever=retriever,
                        cache=self._cache,
                        pdf_paths=[str(p) for p in pdf_paths],
                    )
                    verdict = session.verdict or Verdict.ERROR
                    confidence = session.confidence
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

                style = _VERDICT_STYLE.get(verdict, "white")
                console.print(
                    f"  [{style}]{verdict.value:<14}[/{style}] "
                    f"[dim]{citation.id}[/dim] {citation.raw_marker[:40]}"
                )
                _emit(
                    "verdict",
                    citation_id=citation.id,
                    marker=citation.raw_marker,
                    verdict=verdict.value,
                    confidence=confidence,
                )
                progress.advance(task)

        results = self._cache.all_results()
        _emit(
            "done",
            total=len(pending),
            verdicts=dict(Counter(r.verdict.value for r in results)),
        )
        return results
```

- [ ] **Step 4: Run all orchestrator tests**

```bash
python -m pytest tests/test_agents/test_orchestrator.py -v
```

Expected: All tests pass (7 original + 4 new = 11 total).

- [ ] **Step 5: Run full suite**

```bash
python -m pytest tests/ -q
```

Expected: All tests pass (101 total).

- [ ] **Step 6: Commit**

```bash
git add citeguard/agents/orchestrator.py tests/test_agents/test_orchestrator.py
git commit -m "feat: add on_progress callback to CiteGuardRunner.run()"
```

---

## Task 3: Create FastAPI web app

**Files:**
- Create: `citeguard/web/__init__.py`
- Create: `citeguard/web/app.py`

- [ ] **Step 1: Create package marker**

Create `citeguard/web/__init__.py` as an empty file.

- [ ] **Step 2: Write failing import test**

```bash
python -c "from citeguard.web.app import app"
```

Expected: `ModuleNotFoundError: No module named 'citeguard.web'`

- [ ] **Step 3: Create citeguard/web/app.py**

```python
from __future__ import annotations
import json
import queue
import threading
from collections import Counter
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from citeguard.config import load_settings
from citeguard.agents.orchestrator import CiteGuardRunner
from citeguard.reporting.json_report import generate_json_report
from citeguard.reporting.pdf_report import generate_pdf_report

app = FastAPI(title="CiteGuard")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

_job: dict = {
    "state": "idle",   # idle | running | done | error
    "queue": queue.Queue(),
    "thread": None,
    "result_json": None,
    "result_pdf": None,
    "error": None,
}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/run")
async def run_job(
    manuscript_path: str = Form(...),
    pdf_folder: str = Form(...),
    strictness: str = Form("balanced"),
    retrieval_backend: str = Form("bm25"),
    model: str = Form("anthropic/claude-sonnet-4-5"),
) -> dict:
    if _job["state"] == "running":
        raise HTTPException(status_code=409, detail="A run is already in progress")

    manuscript = Path(manuscript_path)
    pdfs = Path(pdf_folder)

    if not manuscript.exists():
        raise HTTPException(status_code=400, detail=f"Manuscript not found: {manuscript_path}")
    if not pdfs.exists() or not pdfs.is_dir():
        raise HTTPException(status_code=400, detail=f"PDF folder not found: {pdf_folder}")

    _job["state"] = "running"
    _job["queue"] = queue.Queue()
    _job["result_json"] = None
    _job["result_pdf"] = None
    _job["error"] = None

    def _run() -> None:
        try:
            settings = load_settings()
            settings = settings.model_copy(update={
                "strictness": strictness,
                "retrieval_backend": retrieval_backend,
                "models": settings.models.model_copy(update={"strong": model}),
            })

            runner = CiteGuardRunner(settings)
            results = runner.run(
                manuscript_path=manuscript,
                pdf_folder=pdfs,
                on_progress=_job["queue"].put,
            )

            out = Path(settings.output.dir)
            json_path = out / "citeguard_results.json"
            pdf_path = out / "citeguard_report.pdf"

            generate_json_report(results, settings, manuscript_path, pdf_folder, json_path)
            generate_pdf_report(results, pdf_path, manuscript.name, strictness)

            _job["result_json"] = json_path
            _job["result_pdf"] = pdf_path
            _job["state"] = "done"
        except Exception as exc:
            _job["error"] = str(exc)
            _job["state"] = "error"
            _job["queue"].put({"type": "error", "message": str(exc)})

    _job["thread"] = threading.Thread(target=_run, daemon=True)
    _job["thread"].start()
    return {"ok": True}


@app.get("/stream")
async def stream() -> StreamingResponse:
    def _generate():
        while True:
            try:
                event = _job["queue"].get(timeout=30)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("done", "error"):
                    break
            except queue.Empty:
                yield 'data: {"type":"ping"}\n\n'
                if _job["state"] not in ("running",):
                    break

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/status")
async def status() -> dict:
    return {"state": _job["state"], "error": _job["error"]}


@app.get("/report/pdf")
async def report_pdf() -> FileResponse:
    path = _job["result_pdf"]
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="PDF report not available yet")
    return FileResponse(path, filename="citeguard_report.pdf", media_type="application/pdf")


@app.get("/report/json")
async def report_json() -> FileResponse:
    path = _job["result_json"]
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="JSON report not available yet")
    return FileResponse(path, filename="citeguard_results.json", media_type="application/json")
```

- [ ] **Step 4: Smoke test imports**

```bash
python -c "from citeguard.web.app import app; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Write FastAPI route tests**

Create `tests/test_web/__init__.py` (empty) and `tests/test_web/test_app.py`:

```python
import pytest
from fastapi.testclient import TestClient
from citeguard.web import app as web_module
from citeguard.web.app import app, _job


@pytest.fixture(autouse=True)
def reset_job():
    """Reset global job state before each test."""
    _job["state"] = "idle"
    _job["result_json"] = None
    _job["result_pdf"] = None
    _job["error"] = None
    import queue
    _job["queue"] = queue.Queue()
    yield


@pytest.fixture
def client():
    return TestClient(app)


def test_index_returns_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "CiteGuard" in resp.text


def test_status_idle(client):
    resp = client.get("/status")
    assert resp.status_code == 200
    assert resp.json()["state"] == "idle"


def test_run_missing_manuscript_returns_400(client, tmp_path):
    resp = client.post("/run", data={
        "manuscript_path": str(tmp_path / "nonexistent.pdf"),
        "pdf_folder": str(tmp_path),
    })
    assert resp.status_code == 400
    assert "not found" in resp.json()["detail"].lower()


def test_run_missing_pdf_folder_returns_400(client, tmp_path):
    ms = tmp_path / "paper.pdf"
    ms.write_bytes(b"%PDF-1.4")
    resp = client.post("/run", data={
        "manuscript_path": str(ms),
        "pdf_folder": str(tmp_path / "nonexistent"),
    })
    assert resp.status_code == 400


def test_run_while_running_returns_409(client, tmp_path):
    _job["state"] = "running"
    ms = tmp_path / "paper.pdf"
    ms.write_bytes(b"%PDF-1.4")
    resp = client.post("/run", data={
        "manuscript_path": str(ms),
        "pdf_folder": str(tmp_path),
    })
    assert resp.status_code == 409


def test_report_pdf_not_ready_returns_404(client):
    resp = client.get("/report/pdf")
    assert resp.status_code == 404


def test_report_json_not_ready_returns_404(client):
    resp = client.get("/report/json")
    assert resp.status_code == 404
```

- [ ] **Step 6: Run web tests**

```bash
python -m pytest tests/test_web/ -v
```

Expected: 7 tests pass.

- [ ] **Step 7: Run full suite**

```bash
python -m pytest tests/ -q
```

Expected: All tests pass.

- [ ] **Step 8: Commit**

```bash
git add citeguard/web/__init__.py citeguard/web/app.py tests/test_web/
git commit -m "feat: FastAPI web app with SSE streaming and job state"
```

---

## Task 4: Create HTML template

**Files:**
- Create: `citeguard/web/templates/index.html`

- [ ] **Step 1: Create templates directory and index.html**

```bash
mkdir -p citeguard/web/templates
```

Create `citeguard/web/templates/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CiteGuard</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, -apple-system, sans-serif; background: #f0f2f5; color: #1a1a2e; min-height: 100vh; }

    header { background: #1a1a2e; color: #fff; padding: 1.25rem 2rem; display: flex; align-items: baseline; gap: 1rem; }
    header h1 { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.5px; }
    header p { opacity: 0.55; font-size: 0.85rem; }

    main { max-width: 780px; margin: 2rem auto; padding: 0 1rem; }

    .card { background: #fff; border-radius: 10px; padding: 1.5rem; box-shadow: 0 1px 6px rgba(0,0,0,0.08); margin-bottom: 1.25rem; }

    h2 { font-size: 1.1rem; font-weight: 700; margin-bottom: 1rem; }

    label { display: block; font-size: 0.8rem; font-weight: 600; color: #6c757d; margin-bottom: 0.3rem; margin-top: 1rem; text-transform: uppercase; letter-spacing: 0.04em; }
    label:first-of-type { margin-top: 0; }
    input[type=text], select {
      width: 100%; padding: 0.55rem 0.8rem; border: 1.5px solid #dee2e6;
      border-radius: 7px; font-size: 0.95rem; background: #fff;
      transition: border-color 0.15s, box-shadow 0.15s;
    }
    input[type=text]:focus, select:focus {
      outline: none; border-color: #4361ee;
      box-shadow: 0 0 0 3px rgba(67,97,238,0.15);
    }
    input[type=text]:disabled, select:disabled { background: #f8f9fa; color: #adb5bd; cursor: not-allowed; }

    .row3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.75rem; margin-top: 1rem; }

    .btn {
      display: inline-flex; align-items: center; gap: 0.4rem;
      padding: 0.6rem 1.4rem; border: none; border-radius: 7px;
      font-size: 0.9rem; font-weight: 600; cursor: pointer;
      text-decoration: none; transition: filter 0.15s;
    }
    .btn:hover:not(:disabled) { filter: brightness(0.92); }
    .btn:disabled { opacity: 0.45; cursor: not-allowed; }
    .btn-primary  { background: #4361ee; color: #fff; }
    .btn-secondary { background: #6c757d; color: #fff; }
    .btn-outline  { background: transparent; border: 1.5px solid #4361ee; color: #4361ee; }
    .btn-row { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-top: 1.25rem; }

    .step-label { font-size: 0.95rem; font-weight: 600; color: #4361ee; margin-bottom: 0.75rem; }
    .progress-wrap { background: #e9ecef; border-radius: 99px; height: 8px; overflow: hidden; margin-bottom: 0.35rem; }
    .progress-fill { height: 100%; background: linear-gradient(90deg,#4361ee,#7b9eff); border-radius: 99px; width: 0%; transition: width 0.4s ease; }
    .progress-text { font-size: 0.8rem; color: #adb5bd; margin-bottom: 1rem; }

    #verdict-log { max-height: 340px; overflow-y: auto; border: 1.5px solid #f0f2f5; border-radius: 8px; background: #fafbfc; }
    .vrow { display: flex; align-items: center; gap: 0.65rem; padding: 0.42rem 0.75rem; border-bottom: 1px solid #f0f2f5; font-size: 0.85rem; }
    .vrow:last-child { border-bottom: none; }

    .badge {
      display: inline-block; padding: 0.18rem 0.55rem; border-radius: 5px;
      font-size: 0.72rem; font-weight: 700; color: #fff;
      min-width: 106px; text-align: center; letter-spacing: 0.03em;
    }
    .vmarker { font-family: monospace; color: #495057; font-size: 0.82rem; flex: 1; }
    .vconf { font-size: 0.78rem; color: #adb5bd; white-space: nowrap; }

    .summary-wrap { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.75rem 0 1.25rem; }
    .sbadge { padding: 0.35rem 0.85rem; border-radius: 6px; font-size: 0.82rem; font-weight: 700; color: #fff; }

    #error-banner {
      background: #fff5f5; border: 1.5px solid #fc8181; border-radius: 8px;
      padding: 0.85rem 1rem; color: #c53030; margin-bottom: 1rem;
      font-size: 0.9rem; display: none;
    }
    [hidden] { display: none !important; }
  </style>
</head>
<body>

<header>
  <h1>CiteGuard</h1>
  <p>Academic citation verification</p>
</header>

<main>
  <div id="error-banner"></div>

  <!-- State 1: Config form -->
  <section id="form-section">
    <div class="card">
      <h2>Configuration</h2>
      <form id="run-form">
        <label>Manuscript path</label>
        <input type="text" id="manuscript_path" name="manuscript_path"
               placeholder="/path/to/paper.pdf or paper.docx" required>

        <label>PDF folder</label>
        <input type="text" id="pdf_folder" name="pdf_folder"
               placeholder="/path/to/pdfs/" required>

        <div class="row3">
          <div>
            <label>Strictness</label>
            <select id="strictness" name="strictness">
              <option value="lenient">Lenient</option>
              <option value="balanced" selected>Balanced</option>
              <option value="strict">Strict</option>
            </select>
          </div>
          <div>
            <label>Retrieval backend</label>
            <select id="retrieval_backend" name="retrieval_backend">
              <option value="bm25" selected>BM25</option>
              <option value="local_embeddings">Local Embeddings</option>
              <option value="api_embeddings">API Embeddings</option>
            </select>
          </div>
          <div>
            <label>Model (OpenRouter)</label>
            <input type="text" id="model" name="model"
                   value="anthropic/claude-sonnet-4-5">
          </div>
        </div>

        <div class="btn-row">
          <button type="submit" class="btn btn-primary" id="run-btn">&#9654; Run Verification</button>
        </div>
      </form>
    </div>
  </section>

  <!-- State 2: Progress -->
  <section id="progress-section" hidden>
    <div class="card">
      <div class="step-label" id="step-label">Starting...</div>
      <div class="progress-wrap"><div class="progress-fill" id="progress-fill"></div></div>
      <div class="progress-text" id="progress-text"></div>
      <div id="verdict-log"></div>
    </div>
  </section>

  <!-- State 3: Done -->
  <section id="done-section" hidden>
    <div class="card">
      <h2 style="color:#2dc653">&#10003; Verification Complete</h2>
      <div class="summary-wrap" id="summary-wrap"></div>
      <div class="btn-row">
        <a href="/report/pdf" download class="btn btn-primary">&#8659; Download PDF Report</a>
        <a href="/report/json" download class="btn btn-secondary">&#8659; Download JSON</a>
        <button class="btn btn-outline" id="run-another-btn">&#8617; Run Another</button>
      </div>
    </div>
  </section>
</main>

<script>
  const COLORS = {
    SUPPORTED:    '#2dc653',
    PARTIAL:      '#f4a261',
    UNSUPPORTED:  '#e63946',
    EXAGGERATED:  '#e07c24',
    FABRICATED:   '#9d0208',
    AMBIGUOUS:    '#adb5bd',
    UNVERIFIABLE: '#4895ef',
    ERROR:        '#7b2d8b',
  };

  const $ = id => document.getElementById(id);
  let _total = 0, _current = 0, _es = null;

  function showError(msg) {
    const b = $('error-banner');
    b.textContent = '⚠ ' + msg;
    b.style.display = 'block';
  }
  function clearError() {
    $('error-banner').textContent = '';
    $('error-banner').style.display = 'none';
  }

  function setProgress(current, total) {
    _current = current; _total = total;
    const pct = total > 0 ? (current / total * 100) : 0;
    $('progress-fill').style.width = pct + '%';
    $('progress-text').textContent = total > 0 ? current + ' / ' + total : '';
  }

  function appendVerdict(ev) {
    const color = COLORS[ev.verdict] || '#adb5bd';
    const row = document.createElement('div');
    row.className = 'vrow';
    const conf = ev.confidence != null ? Math.round(ev.confidence * 100) + '%' : '';
    row.innerHTML =
      '<span class="badge" style="background:' + color + '">' + ev.verdict + '</span>' +
      '<span class="vmarker">' + escHtml(ev.marker) + '</span>' +
      '<span class="vconf">' + conf + '</span>';
    const log = $('verdict-log');
    log.appendChild(row);
    log.scrollTop = log.scrollHeight;
  }

  function showDone(ev) {
    $('progress-section').hidden = true;
    $('done-section').hidden = false;
    const wrap = $('summary-wrap');
    wrap.innerHTML = '';
    Object.entries(ev.verdicts || {}).forEach(function([v, n]) {
      const el = document.createElement('span');
      el.className = 'sbadge';
      el.style.background = COLORS[v] || '#adb5bd';
      el.textContent = v + '  ' + n;
      wrap.appendChild(el);
    });
  }

  function escHtml(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  function lockForm(locked) {
    ['manuscript_path','pdf_folder','strictness','retrieval_backend','model'].forEach(function(id) {
      $(id).disabled = locked;
    });
    $('run-btn').disabled = locked;
  }

  function startStream() {
    if (_es) { _es.close(); _es = null; }
    _es = new EventSource('/stream');
    _es.onmessage = function(e) {
      const ev = JSON.parse(e.data);
      if (ev.type === 'ping') return;
      if (ev.type === 'step') {
        $('step-label').textContent = ev.message;
      } else if (ev.type === 'found') {
        $('step-label').textContent =
          'Found ' + ev.citations + ' citations — ' +
          ev.matched + ' matched to PDFs';
        setProgress(0, ev.citations);
      } else if (ev.type === 'verifying') {
        $('step-label').textContent = 'Verifying ' + ev.current + ' / ' + ev.total;
        setProgress(ev.current - 1, ev.total);
      } else if (ev.type === 'verdict') {
        appendVerdict(ev);
        setProgress(_current + 1, _total);
      } else if (ev.type === 'done') {
        _es.close(); _es = null;
        showDone(ev);
      } else if (ev.type === 'error') {
        _es.close(); _es = null;
        $('progress-section').hidden = true;
        $('form-section').hidden = false;
        lockForm(false);
        showError(ev.message);
      }
    };
    _es.onerror = function() {
      if (_es && _es.readyState !== EventSource.CLOSED) {
        showError('Lost connection to server.');
      }
    };
  }

  $('run-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    clearError();
    const fd = new FormData(e.target);
    try {
      const res = await fetch('/run', { method: 'POST', body: fd });
      if (!res.ok) {
        const err = await res.json();
        showError(err.detail || 'Failed to start run');
        return;
      }
      lockForm(true);
      $('form-section').hidden = true;
      $('progress-section').hidden = false;
      $('step-label').textContent = 'Starting...';
      setProgress(0, 0);
      startStream();
    } catch(err) {
      showError('Could not reach server: ' + err.message);
    }
  });

  $('run-another-btn').addEventListener('click', function() {
    $('done-section').hidden = true;
    $('form-section').hidden = false;
    $('verdict-log').innerHTML = '';
    $('summary-wrap').innerHTML = '';
    setProgress(0, 0);
    lockForm(false);
    clearError();
  });
</script>
</body>
</html>
```

- [ ] **Step 2: Verify template is discovered**

```bash
python -c "
from citeguard.web.app import app, templates
print(templates.env.loader.searchpath)
"
```

Expected: prints a path ending in `citeguard/web/templates`

- [ ] **Step 3: Commit**

```bash
git add citeguard/web/templates/index.html
git commit -m "feat: single-page web UI with SSE progress streaming"
```

---

## Task 5: Add web CLI command and smoke test

**Files:**
- Modify: `citeguard/cli.py`

- [ ] **Step 1: Add web command to citeguard/cli.py**

Append this to `citeguard/cli.py` (after the `status` command):

```python
@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Host to bind to")
@click.option("--port", default=8000, show_default=True, type=int, help="Port to listen on")
def web(host: str, port: int) -> None:
    """Start the CiteGuard web UI."""
    import uvicorn
    console.print(
        f"\n[bold green]CiteGuard Web UI[/bold green]  "
        f"→  [cyan]http://{host}:{port}[/cyan]\n"
        f"Press [bold]Ctrl+C[/bold] to stop.\n"
    )
    uvicorn.run("citeguard.web.app:app", host=host, port=port, reload=False)
```

- [ ] **Step 2: Test help output**

```bash
python -m citeguard.cli web --help
```

Expected output contains:
```
Usage: cli web [OPTIONS]
  Start the CiteGuard web UI.
Options:
  --host TEXT     Host to bind to  [default: 127.0.0.1]
  --port INTEGER  Port to listen on  [default: 8000]
```

- [ ] **Step 3: Smoke test — server starts and serves the page**

```bash
python -m citeguard.cli web --port 8765 &
sleep 2
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8765/
kill %1
```

Expected: `200`

- [ ] **Step 4: Run full test suite**

```bash
python -m pytest tests/ -q
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add citeguard/cli.py
git commit -m "feat: citeguard web command to start local web UI"
```

---

## Self-Review

**Spec coverage:**
- ✅ FastAPI + SSE + vanilla JS (Approach A)
- ✅ Three UI states: idle form → running progress → done with downloads
- ✅ All form fields: manuscript path, PDF folder, strictness, retrieval backend, model
- ✅ SSE event types: step, found, verifying, verdict, done, error
- ✅ All API endpoints: GET /, POST /run, GET /stream, GET /status, GET /report/pdf, GET /report/json
- ✅ HTTP 409 on double-run
- ✅ HTTP 400 on invalid paths
- ✅ `on_progress` callback in orchestrator — existing CLI unchanged
- ✅ `citeguard web --host --port` CLI command
- ✅ Dependencies added to pyproject.toml and requirements.txt

**No placeholders found.**

**Type consistency:** `on_progress: Callable[[dict], None] | None` defined in Task 2 and used identically in Task 3 (`on_progress=_job["queue"].put`). ✅
