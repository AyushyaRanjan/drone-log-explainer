"""Battery sag, drain, current, and temperature rules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app import config
from app.analyzers import empty_result, finding, num_series, r, rollup, time_values


def _primary_battery(df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    if "Instance" not in df.columns:
        return df, []
    instance = pd.to_numeric(df["Instance"], errors="coerce")
    present = sorted({int(v) for v in instance.dropna().unique()})
    primary_id = 0 if 0 in present else (present[0] if present else 0)
    others = [item for item in present if item != primary_id]
    chosen = df.loc[instance == primary_id].copy()
    if chosen.empty:
        return df, others
    return chosen, others


def _cell_count(volt: np.ndarray, time_s: np.ndarray) -> int:
    if time_s.size == 0 or not np.isfinite(time_s).any():
        start_max = float(np.nanmax(volt))
    else:
        t0 = np.nanmin(time_s)
        window = volt[np.isfinite(time_s) & (time_s <= t0 + config.START_WINDOW_S) & np.isfinite(volt)]
        start_max = float(np.max(window)) if window.size else float(np.nanmax(volt))
    cells = int(round(start_max / config.CELL_NOMINAL_V))
    return max(config.CELL_COUNT_MIN, min(config.CELL_COUNT_MAX, cells))


def _drain_baseline(time_s: np.ndarray, rem: np.ndarray, duration_s: float) -> float | None:
    """Median consumption rate (percent per minute) across the first third of the flight."""
    if duration_s <= 0:
        return None
    t0 = float(np.nanmin(time_s))
    cutoff = t0 + duration_s / 3.0
    rates: list[float] = []
    bucket_s = 5.0
    left = t0
    while left < cutoff - 1e-6:
        right = min(left + bucket_s, cutoff)
        in_bucket = np.isfinite(time_s) & np.isfinite(rem) & (time_s >= left) & (time_s <= right)
        if int(in_bucket.sum()) >= 2:
            idx = np.flatnonzero(in_bucket)
            dt_min = (time_s[idx[-1]] - time_s[idx[0]]) / 60.0
            if dt_min > 0:
                rates.append(float(rem[idx[0]] - rem[idx[-1]]) / dt_min)
        left = right
    if not rates:
        return None
    return float(np.median(rates))


def analyze_battery(df: pd.DataFrame | None) -> dict:
    if df is None or df.empty:
        return empty_result()

    frame, other_instances = _primary_battery(df)
    volt_s = num_series(frame, "Volt")
    if volt_s is None or not np.isfinite(volt_s.to_numpy(dtype=float)).any():
        return empty_result({"other_instances": other_instances})

    time_s = time_values(frame)
    volt = volt_s.to_numpy(dtype=float)
    finite_v = volt[np.isfinite(volt)]
    voltage_min = float(np.min(finite_v))
    voltage_max = float(np.max(finite_v))
    voltage_mean = float(np.mean(finite_v))
    cells = _cell_count(volt, time_s)
    min_cell_v = voltage_min / cells

    metrics: dict = {
        "voltage_min": r(voltage_min, 3),
        "voltage_max": r(voltage_max, 3),
        "voltage_mean": r(voltage_mean, 3),
        "cells": cells,
        "min_cell_v": r(min_cell_v, 3),
        "other_instances": other_instances,
        "curr_max": None,
        "resistance_ohm_per_cell": None,
        "drain_rate_pct_per_min": None,
        "temp_max_c": None,
        "rem_drop_pct": None,
    }
    findings: list[dict] = []

    if other_instances:
        findings.append(
            finding(
                "battery.other_instances",
                "info",
                "More than one battery was logged",
                "Analysis uses battery instance 0 when it is present, otherwise the lowest instance. "
                f"Other instances in this log: {other_instances}.",
                evidence={"other_instances": other_instances},
            )
        )

    if min_cell_v < config.CELL_CRITICAL_V:
        severity = "critical"
        title = "Cell voltage fell below 3.3 V under load"
    elif min_cell_v < config.CELL_WARN_V:
        severity = "warning"
        title = "Cell voltage fell below 3.5 V under load"
    else:
        severity = None
        title = ""
    if severity:
        findings.append(
            finding(
                "battery.low_cell",
                severity,
                title,
                f"Lowest pack voltage was {voltage_min:.2f} V. Estimated cell count is {cells}, "
                f"so the lowest cell voltage was about {min_cell_v:.2f} V.",
                time_s=_time_at_extreme(time_s, volt, np.nanmin),
                evidence={"voltage_min": r(voltage_min, 3), "cells": cells, "min_cell_v": r(min_cell_v, 3)},
            )
        )

    curr_s = num_series(frame, "Curr")
    if curr_s is not None:
        curr = curr_s.to_numpy(dtype=float)
        finite_c = curr[np.isfinite(curr)]
        if finite_c.size:
            curr_max = float(np.max(finite_c))
            metrics["curr_max"] = r(curr_max, 2)
            limit = config.PEAK_CURRENT_REFERENCE_A * config.PEAK_CURRENT_FRACTION
            if curr_max > limit:
                findings.append(
                    finding(
                        "battery.peak_current",
                        "info",
                        "Peak current was a large share of the reference load",
                        f"Peak current was {curr_max:.1f} A, above {limit:.0f} A "
                        f"({config.PEAK_CURRENT_FRACTION:.0%} of the {config.PEAK_CURRENT_REFERENCE_A:.0f} A reference).",
                        time_s=_time_at_extreme(time_s, curr, np.nanmax),
                        evidence={"curr_max": r(curr_max, 2), "reference_a": config.PEAK_CURRENT_REFERENCE_A},
                    )
                )
            resistance = _resistance_per_cell(volt, curr, cells)
            metrics["resistance_ohm_per_cell"] = r(resistance, 4)
            if resistance is not None and resistance > config.SAG_RESISTANCE_OHM_PER_CELL:
                findings.append(
                    finding(
                        "battery.sag",
                        "warning",
                        "Voltage sag under load looks high",
                        "A straight-line fit of voltage against current (samples above "
                        f"{config.SAG_MIN_CURRENT_A:.0f} A) estimates {resistance:.3f} ohm per cell. "
                        f"Above {config.SAG_RESISTANCE_OHM_PER_CELL:.3f} ohm per cell suggests an aging or failing pack, "
                        "or a poor connection.",
                        evidence={
                            "resistance_ohm_per_cell": r(resistance, 4),
                            "threshold_ohm": config.SAG_RESISTANCE_OHM_PER_CELL,
                        },
                    )
                )

    rem_s = num_series(frame, "RemPct")
    if rem_s is not None:
        rem = rem_s.to_numpy(dtype=float)
        valid = np.isfinite(time_s) & np.isfinite(rem)
        if int(valid.sum()) >= 2:
            t = time_s[valid]
            remaining = rem[valid]
            duration_s = float(t[-1] - t[0])
            drop = float(remaining[0] - remaining[-1])
            metrics["rem_drop_pct"] = r(drop, 2)
            if duration_s > 0:
                drain_rate = drop / (duration_s / 60.0)
                metrics["drain_rate_pct_per_min"] = r(drain_rate, 2)
                baseline = _drain_baseline(t, remaining, duration_s)
                reasons = []
                if (
                    baseline is not None
                    and baseline > 0
                    and drain_rate > config.DRAIN_RATE_RATIO * baseline
                ):
                    reasons.append(
                        f"the flight used about {drain_rate:.1f} percent per minute, more than "
                        f"{config.DRAIN_RATE_RATIO:.1f} times the {baseline:.1f} percent per minute "
                        "typical of the first third"
                    )
                if drop > config.FAST_DRAIN_DROP_PCT and duration_s < config.FAST_DRAIN_MAX_DURATION_S:
                    reasons.append(
                        f"remaining capacity fell {drop:.0f} percentage points during a flight shorter than 6 minutes"
                    )
                if reasons:
                    findings.append(
                        finding(
                            "battery.fast_drain",
                            "warning",
                            "The battery drained quickly",
                            "Fast drain: " + "; ".join(reasons) + ".",
                            evidence={
                                "drain_rate_pct_per_min": r(drain_rate, 2),
                                "baseline_pct_per_min": r(baseline, 2),
                                "rem_drop_pct": r(drop, 2),
                                "duration_s": r(duration_s, 1),
                            },
                        )
                    )

    temp_s = num_series(frame, "Temp")
    if temp_s is not None:
        temp = temp_s.to_numpy(dtype=float)
        finite_t = temp[np.isfinite(temp)]
        if finite_t.size:
            # Some Pixhawk logs store temperature in centidegrees.
            scale = 100.0 if float(np.median(finite_t)) > 200 else 1.0
            temp_c = finite_t / scale
            temp_max = float(np.max(temp_c))
            metrics["temp_max_c"] = r(temp_max, 1)
            if temp_max > config.BATTERY_TEMP_WARN_C:
                findings.append(
                    finding(
                        "battery.temp",
                        "warning",
                        "Battery temperature exceeded 60 C",
                        f"The highest battery temperature in the log was {temp_max:.0f} C.",
                        evidence={"temp_max_c": r(temp_max, 1)},
                    )
                )

    return {"status": rollup(findings, True), "findings": findings, "metrics": metrics}


def _time_at_extreme(time_s: np.ndarray, values: np.ndarray, reducer) -> float | None:
    valid = np.isfinite(time_s) & np.isfinite(values)
    if not valid.any():
        return None
    target = reducer(values[valid])
    idx = np.flatnonzero(valid & (values == target))
    if idx.size == 0:
        return None
    return float(time_s[idx[0]])


def _resistance_per_cell(volt: np.ndarray, curr: np.ndarray, cells: int) -> float | None:
    mask = np.isfinite(volt) & np.isfinite(curr) & (curr > config.SAG_MIN_CURRENT_A)
    if int(mask.sum()) < 10:
        return None
    current = curr[mask]
    voltage = volt[mask]
    if float(np.std(current)) < 1.0:
        return None
    slope, _intercept = np.polyfit(current, voltage, 1)
    # Slope is volts per amp. A negative slope is the pack's internal resistance.
    pack_ohm = -float(slope)
    if pack_ohm <= 0 or cells <= 0:
        return None
    return pack_ohm / cells
