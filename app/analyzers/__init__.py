"""Shared helpers for rule-based flight-log analyzers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "finding",
    "rollup",
    "empty_result",
    "num_series",
    "time_values",
    "r",
    "sustained_above",
]


def finding(
    fid: str,
    severity: str,
    title: str,
    detail: str,
    time_s: float | None = None,
    evidence: dict | None = None,
) -> dict[str, Any]:
    ts = None
    if time_s is not None:
        try:
            ts = float(time_s)
        except (TypeError, ValueError):
            ts = None
        if ts is not None and (math.isnan(ts) or math.isinf(ts)):
            ts = None
        if ts is not None:
            ts = round(ts, 3)
    return {
        "id": fid,
        "severity": severity,
        "title": title,
        "detail": detail,
        "time_s": ts,
        "evidence": evidence or {},
    }


def rollup(findings: list[dict], has_data: bool) -> str:
    if not has_data:
        return "no_data"
    severities = {item["severity"] for item in findings}
    if "critical" in severities:
        return "critical"
    if "warning" in severities:
        return "warning"
    return "ok"


def empty_result(metrics: dict | None = None) -> dict[str, Any]:
    return {"status": "no_data", "findings": [], "metrics": metrics or {}}


def num_series(df: pd.DataFrame | None, col: str) -> pd.Series | None:
    if df is None or df.empty or col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce")


def time_values(df: pd.DataFrame) -> np.ndarray:
    if "time_s" in df.columns:
        t = pd.to_numeric(df["time_s"], errors="coerce").to_numpy(dtype=float)
    else:
        t = np.arange(len(df), dtype=float)
    return t


def r(value: Any, digits: int = 4) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return round(number, digits)


def sustained_above(
    time_s: np.ndarray,
    values: np.ndarray,
    threshold: float,
    min_duration: float,
) -> list[tuple[float, float]]:
    """Runs where `values` stay above `threshold` for more than `min_duration` seconds."""
    if time_s.size == 0:
        return []
    mask = np.isfinite(time_s) & np.isfinite(values) & (values > threshold)
    runs: list[tuple[float, float]] = []
    start: int | None = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            duration = float(time_s[i - 1] - time_s[start])
            if duration > min_duration:
                runs.append((float(time_s[start]), duration))
            start = None
    if start is not None and np.isfinite(time_s[-1]) and np.isfinite(time_s[start]):
        duration = float(time_s[-1] - time_s[start])
        if duration > min_duration:
            runs.append((float(time_s[start]), duration))
    return runs
