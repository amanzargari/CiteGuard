import queue
import pytest
from fastapi.testclient import TestClient
from citeguard.web.app import app, _job


@pytest.fixture(autouse=True)
def reset_job():
    _job["state"] = "idle"
    _job["result_json"] = None
    _job["result_pdf"] = None
    _job["error"] = None
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
