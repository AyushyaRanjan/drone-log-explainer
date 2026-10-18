"""GPS fix, satellite count, and HDOP rules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app import config
from app.analyzers import empty_result, finding, num_series, r, rollup, sustained_above, time_values


def _fraction(mask: np.ndarray, valid: np.ndarray) -> float:
    count = int(valid.sum())
    if count == 0:
        return 0.0
    return float(np.count_nonzero(mask & valid) / count)


def analyze_gps(df: pd.DataFrame | None) -> dict:
    if df is None or df.empty:
        return empty_result()

    status_s = num_series(df, "Status")
    nsats_s = num_series(df, "NSats")
    hdop_s = num_series(df, "HDop")
    if status_s is None and nsats_s is None and hdop_s is None:
        return empty_result()

    time_s = time_values(df)
    findings: list[dict] = []
    metrics: dict = {
        "fix_below_3d_fraction": None,
        "nsats_min": None,
        "nsats_below_6_fraction": None,
        "hdop_max": None,
        "hdop_above_2_fraction": None,
        "first_fix_loss_s": None,
    }

    if status_s is not None:
        status = status_s.to_numpy(dtype=float)
        valid = np.isfinite(status)
        lost = valid & (status < config.GPS_FIX_3D)
        fraction = _fraction(lost, valid)
        metrics["fix_below_3d_fraction"] = r(fraction, 4)
        if lost.any():
            first = float(time_s[np.flatnonzero(lost)[0]]) if np.isfinite(time_s[lost]).any() else None
            metrics["first_fix_loss_s"] = r(first, 3)
            if fraction > config.GPS_FIX_CRITICAL_FRACTION:
                severity = "critical"
                title = "3D GPS fix was missing for more than 5 percent of the flight"
            else:
                severity = "warning"
                title = "GPS lost its 3D fix during the flight"
            findings.append(
                finding(
                    "gps.fix",
                    severity,
                    title,
                    f"A status below 3 means the receiver did not have a 3D fix. "
                    f"That happened on {fraction * 100:.1f} percent of GPS samples"
                    + (f", first at {first:.0f} s." if first is not None else "."),
                    time_s=first,
                    evidence={"fraction": r(fraction, 4), "first_fix_loss_s": r(first, 3)},
                )
            )

    if nsats_s is not None:
        nsats = nsats_s.to_numpy(dtype=float)
        valid = np.isfinite(nsats)
        if valid.any():
            low = valid & (nsats < config.GPS_NSATS_MIN)
            fraction = _fraction(low, valid)
            nsats_min = float(np.min(nsats[valid]))
            metrics["nsats_min"] = r(nsats_min, 1)
            metrics["nsats_below_6_fraction"] = r(fraction, 4)
            if fraction > config.GPS_NSATS_WARN_FRACTION:
                first = float(time_s[np.flatnonzero(low)[0]]) if low.any() and np.isfinite(time_s[low]).any() else None
                findings.append(
                    finding(
                        "gps.nsats",
                        "warning",
                        "Satellite count stayed below 6",
                        f"Fewer than 6 satellites on {fraction * 100:.1f} percent of samples. "
                        f"The lowest count was {nsats_min:.0f}.",
                        time_s=first,
                        evidence={"fraction": r(fraction, 4), "nsats_min": r(nsats_min, 1)},
                    )
                )

    if hdop_s is not None:
        hdop = hdop_s.to_numpy(dtype=float)
        valid = np.isfinite(hdop)
        if valid.any():
            high = valid & (hdop > config.GPS_HDOP_WARN)
            fraction = _fraction(high, valid)
            hdop_max = float(np.max(hdop[valid]))
            metrics["hdop_max"] = r(hdop_max, 2)
            metrics["hdop_above_2_fraction"] = r(fraction, 4)
            sustained = sustained_above(time_s, hdop, config.GPS_HDOP_CRITICAL, config.GPS_HDOP_SUSTAINED_S)
            if sustained:
                start, duration = sustained[0]
                findings.append(
                    finding(
                        "gps.hdop",
                        "critical",
                        "GPS accuracy was very poor (HDOP above 5)",
                        "HDOP is a unitless accuracy figure: lower is better, and above 2 is already loose. "
                        f"HDOP stayed above 5 for {duration:.0f} s. The worst value was {hdop_max:.1f}.",
                        time_s=start,
                        evidence={"hdop_max": r(hdop_max, 2), "duration_s": r(duration, 2)},
                    )
                )
            elif fraction > config.GPS_HDOP_WARN_FRACTION:
                first = float(time_s[np.flatnonzero(high)[0]]) if high.any() and np.isfinite(time_s[high]).any() else None
                findings.append(
                    finding(
                        "gps.hdop",
                        "warning",
                        "GPS accuracy was loose (HDOP above 2)",
                        "HDOP is a unitless accuracy figure: lower is better. "
                        f"It was above 2 on {fraction * 100:.1f} percent of samples. The worst value was {hdop_max:.1f}.",
                        time_s=first,
                        evidence={"fraction": r(fraction, 4), "hdop_max": r(hdop_max, 2)},
                    )
                )

    return {"status": rollup(findings, True), "findings": findings, "metrics": metrics}
