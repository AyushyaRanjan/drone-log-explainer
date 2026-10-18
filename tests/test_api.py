"""HTTP contract for demo, upload validation, and missing reports."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

REQUIRED_KEYS = {
    "report_id",
    "file_name",
    "generated_at",
    "verdict",
    "summary",
    "sections",
    "timeseries",
    "checklist",
    "explanation",
    "parse_warnings",
}


def test_demo_returns_report():
    response = client.post("/api/demo")
    assert response.status_code == 200
    body = response.json()
    assert REQUIRED_KEYS <= set(body)
    assert body["sections"]["battery"]["status"] == "critical"


def test_analyze_rejects_text_file():
    response = client.post(
        "/api/analyze",
        files={"file": ("notes.txt", b"not a log", "text/plain")},
    )
    assert response.status_code == 400
    body = response.json()
    assert "error" in body and "detail" in body


def test_unknown_report_is_404():
    response = client.get("/api/reports/unknown")
    assert response.status_code == 404
