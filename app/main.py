"""FastAPI app: upload a .BIN, or run the built-in demo flight."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app import config
from app.demo import generate_demo_flight
from app.parser import ParseError, parse_bin
from app.report import build_report, to_builtin

app = FastAPI(title="Drone Flight-Log Explainer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:8000",
        "http://127.0.0.1",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

REPORTS: dict[str, dict] = {}
STATIC_INDEX = Path(__file__).resolve().parent / "static" / "index.html"


def _error(status: int, error: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": error, "detail": detail})


def _store(report: dict) -> dict:
    report = to_builtin(report)
    REPORTS[report["report_id"]] = report
    try:
        config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = config.REPORT_DIR / f"{report['report_id']}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
    except OSError:
        pass
    return report


@app.get("/")
def index():
    return FileResponse(STATIC_INDEX)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/demo")
def demo():
    parsed = generate_demo_flight(seed=42)
    report = build_report(parsed, file_name="demo-flight.bin")
    return _store(report)


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    name = Path(file.filename or "").name
    if not name.lower().endswith(".bin"):
        return _error(400, "Unsupported file type", "Upload an ArduPilot .bin flight log.")
    payload = await file.read()
    if len(payload) > config.MAX_UPLOAD_BYTES:
        return _error(400, "File too large", "Logs must be 100 MB or smaller.")
    if not payload:
        return _error(400, "Empty file", "The uploaded log is empty.")

    tmp_dir = config.ROOT / "data" / "uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{uuid4().hex}.bin"
    try:
        tmp_path.write_bytes(payload)
        try:
            parsed = parse_bin(str(tmp_path))
        except ParseError as exc:
            return _error(400, "Could not parse log", str(exc))
        except Exception as exc:  # noqa: BLE001
            return _error(400, "Could not parse log", str(exc))
        report = build_report(parsed, file_name=name)
        return _store(report)
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/api/reports/{report_id}")
def get_report(report_id: str):
    report = REPORTS.get(report_id)
    if report is None:
        path = config.REPORT_DIR / f"{report_id}.json"
        if path.is_file():
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
                REPORTS[report_id] = report
            except (OSError, json.JSONDecodeError):
                report = None
    if report is None:
        return _error(404, "Report not found", f"No report with id {report_id}.")
    return report
