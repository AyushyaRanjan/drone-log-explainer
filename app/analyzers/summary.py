"""Flight summary and overall verdict."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.analyzers import num_series, r


def _span(frames: dict[str, pd.DataFrame]) -> float:
    hi = 0.0
    found = False
    for frame in frames.values():
        if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        if "time_s" not in frame.columns:
            continue
        values = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            found = True
            hi = max(hi, float(np.max(finite)))
    return hi if found else 0.0


def _max_alt(baro: pd.DataFrame | None, pos: pd.DataFrame | None) -> float | None:
    for frame in (baro, pos):
        if frame is None or frame.empty:
            continue
        alt = num_series(frame, "Alt")
        if alt is None:
            continue
        values = alt.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            return float(np.max(finite))
    return None


def summarize(frames: dict, sections: dict) -> dict:
    counts = {"critical": 0, "warning": 0, "info": 0}
    for section in sections.values():
        for item in section.get("findings", []):
            severity = item.get("severity")
            if severity in counts:
                counts[severity] += 1
    if counts["critical"]:
        verdict = "critical"
    elif counts["warning"]:
        verdict = "warning"
    else:
        verdict = "ok"

    mode_changes = sections.get("warnings", {}).get("metrics", {}).get("mode_changes", 0)
    max_alt = _max_alt(frames.get("BARO"), frames.get("POS"))
    return {
        "verdict": verdict,
        "summary": {
            "duration_s": r(_span(frames), 1),
            "max_alt_m": r(max_alt, 1),
            "mode_changes": int(mode_changes or 0),
            "findings_count": counts,
        },
    }
