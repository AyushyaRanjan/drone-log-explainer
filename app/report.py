"""Turn analyzer output into the report JSON the UI and API share."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from app import config
from app.analyzers.battery import analyze_battery
from app.analyzers.gps import analyze_gps
from app.analyzers.summary import summarize
from app.analyzers.vibration import analyze_vibration
from app.analyzers.warnings import analyze_warnings
from app.explainer import explain

MESSAGE_TYPES = ("BAT", "VIBE", "GPS", "MSG", "ERR", "MODE", "BARO", "POS")

CHECKLIST_ITEMS = {
    "battery.low_cell": "Inspect battery cells and connectors before next flight",
    "battery.sag": "Test the battery under load or replace it; voltage sag looks high",
    "battery.fast_drain": "Confirm pack capacity and look for a reason the battery drained quickly",
    "battery.temp": "Let the battery cool and check it for heat damage before charging",
    "vibration.high": "Check prop balance and motor mounts; vibration was above 60 m/s^2",
    "vibration.level": "Check props and motor mounts; vibration sat above 30 m/s^2",
    "vibration.clip": "Reduce vibration until the accelerometer stops clipping",
    "gps.fix": "Verify GPS antenna placement and wait for a 3D fix before arming",
    "gps.nsats": "Wait for more satellites and check that the GPS antenna can see the sky",
    "gps.hdop": "GPS accuracy was poor. Check antenna placement and fly from a more open spot",
    "warnings.msg": "Read the autopilot messages and clear each one before flying again",
    "warnings.err": "Review the autopilot error before the next flight",
    "warnings.unplanned_mode": "Review failsafe settings: the log shows a switch to RTL or LAND",
}


def build_report(parsed: dict, file_name: str, report_id: str | None = None) -> dict:
    frames = {name: _as_frame(parsed.get(name)) for name in MESSAGE_TYPES}
    parse_warnings = list(parsed.get("parse_warnings") or [])

    battery = analyze_battery(frames["BAT"])
    vibration = analyze_vibration(frames["VIBE"])
    gps = analyze_gps(frames["GPS"])
    warnings = analyze_warnings(frames["MSG"], frames["ERR"], frames["MODE"])
    sections = {
        "battery": battery,
        "vibration": vibration,
        "gps": gps,
        "warnings": warnings,
    }
    rolled = summarize(frames, sections)
    checklist = build_checklist(sections, warnings.get("metrics", {}).get("mode_timeline", []))
    timeseries = build_timeseries(frames)

    report = {
        "report_id": report_id or str(uuid4()),
        "file_name": file_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": rolled["verdict"],
        "summary": rolled["summary"],
        "sections": sections,
        "timeseries": timeseries,
        "checklist": checklist,
        "explanation": {"source": "template", "markdown": ""},
        "parse_warnings": parse_warnings,
    }
    report["explanation"] = explain(
        {
            "verdict": report["verdict"],
            "summary": report["summary"],
            "sections": {
                name: {
                    "status": section["status"],
                    "metrics": section["metrics"],
                    "findings": section["findings"],
                }
                for name, section in sections.items()
            },
            "checklist": checklist,
        }
    )
    return to_builtin(report)


def build_checklist(sections: dict, mode_timeline: list[dict]) -> list[dict]:
    items: list[dict] = []
    modes = {event.get("mode") for event in mode_timeline}
    saw_failsafe = False
    for section in sections.values():
        for item in section.get("findings", []):
            text = f"{item.get('title', '')} {item.get('detail', '')}".lower()
            if "failsafe" in text:
                saw_failsafe = True
    for section in sections.values():
        for found in section.get("findings", []):
            severity = found.get("severity")
            if severity == "critical":
                priority = 1
            elif severity == "warning":
                priority = 2
            else:
                continue
            fid = str(found.get("id", ""))
            base_id = fid if fid in CHECKLIST_ITEMS else fid.rsplit(".", 1)[0] if fid.count(".") > 1 else fid
            # vibration.imbalance.x is info and already skipped. err ids are warnings.err.N.N
            if fid.startswith("warnings.err"):
                base_id = "warnings.err"
            if fid.startswith("vibration.imbalance"):
                base_id = "vibration.imbalance"
            label = CHECKLIST_ITEMS.get(base_id) or CHECKLIST_ITEMS.get(fid) or found.get("title") or "Review this finding"
            if fid.startswith("warnings.msg") and saw_failsafe and ("RTL" in modes or "LAND" in modes or "SMART_RTL" in modes):
                label = "Review failsafe settings: the log shows an RTL triggered by battery failsafe"
            if fid.startswith("vibration.imbalance"):
                axis = fid.split(".")[-1].upper()
                label = f"Check prop balance and motor mounts on the {axis} axis"
            items.append({"priority": priority, "item": label, "reason": found.get("detail") or found.get("title") or ""})
    items.sort(key=lambda row: row["priority"])
    return items


def build_timeseries(frames: dict[str, pd.DataFrame], max_points: int = config.TIMESERIES_MAX_POINTS) -> dict:
    sources = {
        "voltage": (frames["BAT"], "Volt"),
        "current": (frames["BAT"], "Curr"),
        "rem_pct": (frames["BAT"], "RemPct"),
        "vibe_x": (frames["VIBE"], "VibeX"),
        "vibe_y": (frames["VIBE"], "VibeY"),
        "vibe_z": (frames["VIBE"], "VibeZ"),
        "nsats": (frames["GPS"], "NSats"),
        "hdop": (frames["GPS"], "HDop"),
        "alt": (frames["BARO"] if frames["BARO"] is not None and not frames["BARO"].empty else frames["POS"], "Alt"),
    }
    grid = _time_grid(frames, max_points)
    series = {"time_s": [round(float(value), 3) for value in grid]}
    for name, (frame, column) in sources.items():
        series[name] = _align(frame, column, grid)
    return series


def _time_grid(frames: dict[str, pd.DataFrame], max_points: int) -> np.ndarray:
    chunks = []
    for frame in frames.values():
        if frame is None or frame.empty or "time_s" not in frame.columns:
            continue
        values = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            chunks.append(values)
    if not chunks:
        return np.array([], dtype=float)
    # Prefer the densest series so charts follow the flight rather than sparse events.
    base = max(chunks, key=lambda values: values.size)
    base = np.unique(np.sort(base))
    if base.size > max_points:
        step = int(math.ceil(base.size / max_points))
        base = base[::step]
    return base


def _align(frame: pd.DataFrame | None, column: str, grid: np.ndarray) -> list:
    if frame is None or frame.empty or column not in frame.columns or grid.size == 0 or "time_s" not in frame.columns:
        return [None] * int(grid.size)
    time_s = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(time_s) & np.isfinite(values)
    if not mask.any():
        return [None] * int(grid.size)
    x = time_s[mask]
    y = values[mask]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    keep = np.concatenate([[True], np.diff(x) > 1e-6])
    x = x[keep]
    y = y[keep]
    if x.size == 1:
        filled = np.where(np.abs(grid - x[0]) <= 0.5, y[0], np.nan)
    else:
        filled = np.interp(grid, x, y)
        filled[(grid < x[0]) | (grid > x[-1])] = np.nan
    return [None if not np.isfinite(value) else round(float(value), 4) for value in filled]


def _as_frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    return pd.DataFrame()


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
