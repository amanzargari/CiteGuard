from __future__ import annotations
import json
import queue
import threading
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
    return templates.TemplateResponse(request, "index.html")


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
