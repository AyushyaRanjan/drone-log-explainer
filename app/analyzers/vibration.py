"""Vibration level, sustained spikes, clipping, and axis imbalance."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app import config
from app.analyzers import empty_result, finding, num_series, r, rollup, sustained_above, time_values

AXES = ("VibeX", "VibeY", "VibeZ")
CLIPS = ("Clip0", "Clip1", "Clip2")


def analyze_vibration(df: pd.DataFrame | None) -> dict:
    if df is None or df.empty:
        return empty_result()
    series = {name: num_series(df, name) for name in AXES}
    if all(item is None for item in series.values()):
        return empty_result()

    time_s = time_values(df)
    arrays: dict[str, np.ndarray] = {}
    for name, col in series.items():
        if col is None:
            arrays[name] = np.full(len(df), np.nan)
        else:
            arrays[name] = col.to_numpy(dtype=float)

    magnitude = np.sqrt(
        np.nansum(np.vstack([arrays[name] ** 2 for name in AXES]), axis=0)
    )
    # If every axis is NaN on a row, nansum yields 0. Blank those rows back out.
    all_missing = np.all(np.vstack([~np.isfinite(arrays[name]) for name in AXES]), axis=0)
    magnitude = magnitude.astype(float)
    magnitude[all_missing] = np.nan

    metrics: dict = {"magnitude_mean": r(np.nanmean(magnitude), 2), "magnitude_p95": r(np.nanpercentile(magnitude, 95), 2)}
    for name in AXES:
        values = arrays[name]
        metrics[f"{name}_mean"] = r(np.nanmean(values), 2) if np.isfinite(values).any() else None
        metrics[f"{name}_p95"] = r(np.nanpercentile(values, 95), 2) if np.isfinite(values).any() else None

    findings: list[dict] = []
    runs: list[tuple[str, float, float]] = []
    for name in AXES:
        for start, duration in sustained_above(time_s, np.abs(arrays[name]), config.VIBE_WARN_MAX, config.VIBE_SUSTAINED_S):
            runs.append((name, start, duration))
    for start, duration in sustained_above(time_s, magnitude, config.VIBE_WARN_MAX, config.VIBE_SUSTAINED_S):
        runs.append(("magnitude", start, duration))

    peak = float(np.nanmax(magnitude)) if np.isfinite(magnitude).any() else 0.0
    p95 = float(np.nanpercentile(magnitude, 95)) if np.isfinite(magnitude).any() else 0.0
    axis_p95 = {
        name: float(np.nanpercentile(np.abs(arrays[name]), 95))
        for name in AXES
        if np.isfinite(arrays[name]).any()
    }

    if runs:
        runs.sort(key=lambda item: item[1])
        name, start, duration = runs[0]
        where = "overall vibration" if name == "magnitude" else name
        findings.append(
            finding(
                "vibration.high",
                "critical",
                f"{where} stayed above 60 m/s^2",
                f"{where} stayed above 60 m/s^2 for {duration:.0f} s. "
                "That is high enough to need a mechanical inspection before the next flight.",
                time_s=start,
                evidence={"axis": name, "duration_s": r(duration, 2), "threshold": config.VIBE_WARN_MAX},
            )
        )
    elif peak > config.VIBE_WARN_MAX or any(value > config.VIBE_WARN_MAX for value in axis_p95.values()):
        findings.append(
            finding(
                "vibration.high",
                "critical",
                "Vibration exceeded 60 m/s^2",
                f"Peak vibration magnitude was {peak:.0f} m/s^2. Levels above 60 m/s^2 can saturate the sensors "
                "and hide mechanical problems.",
                time_s=_time_of_peak(time_s, magnitude),
                evidence={"peak": r(peak, 2)},
            )
        )
    elif p95 >= config.VIBE_GOOD_MAX or any(value >= config.VIBE_GOOD_MAX for value in axis_p95.values()):
        findings.append(
            finding(
                "vibration.level",
                "warning",
                "Vibration sat in the 30 to 60 m/s^2 band",
                f"The 95th percentile of vibration magnitude was {p95:.0f} m/s^2. "
                "Under 30 is a comfortable range; 30 to 60 deserves a look at props and mounts.",
                evidence={"magnitude_p95": r(p95, 2)},
            )
        )

    clip_notes = []
    for name in CLIPS:
        col = num_series(df, name)
        if col is None:
            continue
        values = col.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        start_value = float(finite[0])
        peak_value = float(np.max(finite))
        metrics[f"{name}_max"] = r(peak_value, 0)
        if peak_value > start_value:
            clip_notes.append(f"{name} rose from {start_value:.0f} to {peak_value:.0f}")
    if clip_notes:
        findings.append(
            finding(
                "vibration.clip",
                "warning",
                "Accelerometer clipping detected",
                "Accelerometer clipping detected, vibration is saturating the sensor. "
                + "; ".join(clip_notes)
                + ".",
                evidence={"clips": clip_notes},
            )
        )

    means = {}
    for name in AXES:
        values = np.abs(arrays[name])
        if np.isfinite(values).any():
            means[name] = float(np.nanmean(values))
    for name, mean in means.items():
        others = [value for key, value in means.items() if key != name]
        if not others:
            continue
        if mean > config.VIBE_IMBALANCE_RATIO * max(others):
            axis = name.replace("Vibe", "")
            findings.append(
                finding(
                    f"vibration.imbalance.{axis.lower()}",
                    "info",
                    f"The {axis} axis is vibrating more than the others",
                    f"Mean |{name}| was {mean:.1f} m/s^2, more than twice the other axes. "
                    f"Check prop balance and motor mounts on the {axis} axis.",
                    evidence={"axis": axis, "mean": r(mean, 2), "other_means": {k: r(v, 2) for k, v in means.items()}},
                )
            )

    return {"status": rollup(findings, True), "findings": findings, "metrics": metrics}


def _time_of_peak(time_s: np.ndarray, values: np.ndarray) -> float | None:
    valid = np.isfinite(time_s) & np.isfinite(values)
    if not valid.any():
        return None
    idx = np.flatnonzero(valid)
    peak_i = idx[int(np.argmax(values[idx]))]
    return float(time_s[peak_i])
