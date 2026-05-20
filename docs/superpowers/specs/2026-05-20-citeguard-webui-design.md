# CiteGuard Web UI Design

## Goal

Add a local web interface to CiteGuard so users can set paths, configure the run, start verification, and watch real-time progress — all in a browser — without touching the CLI.

## Constraints

- Local only (no auth, no cloud deployment)
- No Streamlit, Gradio, or similar Python UI frameworks
- One run at a time
- Existing CLI behavior must remain unchanged

## Architecture

```
citeguard/web/
    __init__.py
    app.py              ← FastAPI app: routes, SSE, job state
    templates/
        index.html      ← single-page UI (HTML + CSS + vanilla JS)
```

New entry point added to CLI: `citeguard web [--host 127.0.0.1] [--port 8000]`

New dependencies: `fastapi>=0.110.0`, `uvicorn[standard]>=0.29.0`, `jinja2>=3.1.0`, `python-multipart>=0.0.9`

## UI: Three States

**State 1 — Idle**
Form with all inputs visible and editable:
- Manuscript path (text input)
- PDF folder path (text input)
- Strictness (select: lenient / balanced / strict, default: balanced)
- Retrieval backend (select: bm25 / local_embeddings / api_embeddings, default: bm25)
- Model (text input, default: `anthropic/claude-sonnet-4-5`)
- Run button

**State 2 — Running**
Form is locked. Progress panel shows:
- Current step label (e.g. "Step 3/4: Verifying 24 citations")
- Progress bar (current / total citations)
- Scrolling live log: one row per verdict as it arrives, color-coded by verdict type

**State 3 — Done**
- Summary counts per verdict (colored badges)
- Download PDF Report button (`/report/pdf`)
- Download JSON button (`/report/json`)
- "Run Another" button resets to State 1

## Progress Streaming

**Mechanism:** Server-Sent Events (SSE) — a persistent HTTP connection over which the server pushes newline-delimited JSON. The browser uses `EventSource` to receive events without polling.

**Bridge:** A `queue.Queue` connects the background thread running `CiteGuardRunner.run()` to the SSE endpoint. The orchestrator calls `on_progress(event_dict)` at each milestone; the web layer's callback puts the event into the queue; the SSE endpoint reads and formats it.

**Event types:**

| Type | Payload fields |
|---|---|
| `step` | `message: str` |
| `found` | `citations: int, pdfs: int, matched: int` |
| `verifying` | `current: int, total: int, citation_id: str, marker: str` |
| `verdict` | `citation_id: str, marker: str, verdict: str, confidence: float` |
| `done` | `total: int, verdicts: dict[str, int]` |
| `error` | `message: str` |

## API Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Serve `index.html` |
| `POST` | `/run` | Accept form data, start background job, return `{ok: true}` or error |
| `GET` | `/stream` | SSE stream; browser connects after POST /run |
| `GET` | `/status` | Returns `{state: "idle" | "running" | "done"}` |
| `GET` | `/report/pdf` | Stream the PDF report file as download |
| `GET` | `/report/json` | Stream the JSON report file as download |

`POST /run` form fields: `manuscript_path`, `pdf_folder`, `strictness`, `retrieval_backend`, `model`

If a run is already in progress, `POST /run` returns HTTP 409.

## Orchestrator Change

`CiteGuardRunner.run()` gains an optional parameter:

```python
def run(
    self,
    manuscript_path: Path,
    pdf_folder: Path,
    resume: bool = True,
    on_progress: Callable[[dict], None] | None = None,
) -> list:
```

An internal `_emit(event_type, **data)` helper calls `on_progress({"type": event_type, **data})` when the callback is set. All existing `console.print` and Rich Progress calls are kept — the callback is additive only.

Milestones that call `_emit`:
- After manuscript parsed: `found` event
- Each PDF indexed: `indexing` event
- Before each citation verification: `verifying` event
- After each verdict: `verdict` event
- After all done: `done` event
- On exception in verification loop: `error` event

## Job State (in-memory)

```python
_job = {
    "state": "idle",          # idle | running | done | error
    "queue": queue.Queue(),   # SSE event queue
    "thread": None,           # background Thread
    "result_json": None,      # Path to JSON report
    "result_pdf": None,       # Path to PDF report
    "error": None,            # str if state == error
}
```

No database. Restarting the server resets state.

## Files Changed / Created

| File | Action |
|---|---|
| `citeguard/web/__init__.py` | Create (empty) |
| `citeguard/web/app.py` | Create |
| `citeguard/web/templates/index.html` | Create |
| `citeguard/agents/orchestrator.py` | Modify (add `on_progress` callback) |
| `citeguard/cli.py` | Modify (add `web` command) |
| `pyproject.toml` | Modify (add 4 new dependencies) |
| `requirements.txt` | Modify (add same 4 dependencies) |

## Error Handling

- Invalid paths: caught by FastAPI before starting thread, returned as HTTP 400 with message shown in UI
- Orchestrator exception mid-run: emits `error` event, sets state to `error`, UI shows red banner
- SSE client disconnect: queue is drained silently; thread continues to completion
- Second `POST /run` while running: HTTP 409, UI shows "already running" message
