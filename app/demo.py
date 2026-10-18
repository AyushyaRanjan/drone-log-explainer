"""Synthetic flights that exercise the same DataFrame shape as the parser."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _frame(time_s: np.ndarray, columns: dict[str, np.ndarray]) -> pd.DataFrame:
    data = {"time_s": np.asarray(time_s, dtype=float)}
    data.update({key: np.asarray(value) for key, value in columns.items()})
    return pd.DataFrame(data)


def generate_demo_flight(seed: int = 42) -> dict:
    """Four-minute copter flight with battery, vibration, GPS, and failsafe events.

    The prompt's sag plateau is 14.1 V. A five-second dip to 13.12 V is added so a
    4S pack crosses the 3.3 V/cell critical rule. The 20 s GPS degradation is under
    10% of a 4-minute log, so an 8 s loss of 3D fix is included to trip the GPS warning.
    """
    rng = np.random.default_rng(seed)
    dt = 0.1
    time_s = np.round(np.arange(0.0, 240.0, dt), 4)
    n = len(time_s)

    volt = 16.72 - 0.0004 * time_s + rng.normal(0.0, 0.01, n)
    curr = np.full(n, 10.0)
    sag = (time_s >= 130.0) & (time_s < 160.0)
    volt[sag] = 14.1 + rng.normal(0.0, 0.01, int(sag.sum()))
    curr[sag] = 32.0
    dip = (time_s >= 155.0) & (time_s < 160.0)
    volt[dip] = 13.12
    curr[dip] = 36.0

    rem = np.empty(n)
    early = time_s < 150.0
    rem[early] = 95.0 - (7.0 * time_s[early] / 150.0)
    rem[~early] = 88.0 - (45.0 * (time_s[~early] - 150.0) / 90.0)
    temp = np.full(n, 34.0)

    vibe_x = np.full(n, 8.0)
    vibe_y = np.full(n, 7.5)
    vibe_z = np.full(n, 11.0)
    spike = (time_s >= 80.0) & (time_s <= 86.0)
    vibe_x[spike] = 70.0
    clip0 = np.cumsum(spike.astype(float))
    clip1 = np.zeros(n)
    clip2 = np.zeros(n)

    status = np.full(n, 3.0)
    nsats = np.full(n, 16.0)
    hdop = np.full(n, 0.8)
    degraded = (time_s >= 180.0) & (time_s < 200.0)
    nsats[degraded] = 4.0
    hdop[degraded] = 3.2
    lost_fix = (time_s >= 180.0) & (time_s < 188.0)
    status[lost_fix] = 2.0

    alt = 42.0 * np.sin(np.pi * np.clip(time_s, 0, 220) / 220.0)

    frames = {
        "BAT": _frame(time_s, {"Volt": volt, "Curr": curr, "RemPct": rem, "Temp": temp, "Instance": np.zeros(n)}),
        "VIBE": _frame(time_s, {"VibeX": vibe_x, "VibeY": vibe_y, "VibeZ": vibe_z, "Clip0": clip0, "Clip1": clip1, "Clip2": clip2}),
        "GPS": _frame(time_s, {"Status": status, "NSats": nsats, "HDop": hdop}),
        "MSG": _frame(np.array([210.0]), {"Message": np.array(["Battery failsafe triggered"])}),
        "ERR": _frame(np.array([210.0]), {"Subsys": np.array([5]), "ECode": np.array([1])}),
        "MODE": _frame(
            np.array([0.0, 30.0, 210.0]),
            {"Mode": np.array([0, 5, 6]), "ModeNum": np.array([0, 5, 6])},
        ),
        "BARO": _frame(time_s, {"Alt": alt}),
        "POS": pd.DataFrame(),
        "parse_warnings": [],
    }
    return frames


def generate_clean_flight(seed: int = 1) -> dict:
    """Eight-minute flight inside every warning threshold. Used by tests."""
    del seed  # deterministic; kept so callers can pass a seed like the demo
    dt = 0.2
    time_s = np.round(np.arange(0.0, 480.0, dt), 4)
    n = len(time_s)
    volt = np.linspace(16.6, 16.2, n)
    curr = np.full(n, 8.0)
    rem = np.linspace(92.0, 78.0, n)
    temp = np.full(n, 32.0)
    frames = {
        "BAT": _frame(time_s, {"Volt": volt, "Curr": curr, "RemPct": rem, "Temp": temp}),
        "VIBE": _frame(
            time_s,
            {
                "VibeX": np.full(n, 8.0),
                "VibeY": np.full(n, 8.2),
                "VibeZ": np.full(n, 9.0),
                "Clip0": np.zeros(n),
                "Clip1": np.zeros(n),
                "Clip2": np.zeros(n),
            },
        ),
        "GPS": _frame(time_s, {"Status": np.full(n, 3.0), "NSats": np.full(n, 14.0), "HDop": np.full(n, 0.9)}),
        "MSG": pd.DataFrame(columns=["time_s", "Message"]),
        "ERR": pd.DataFrame(columns=["time_s", "Subsys", "ECode"]),
        "MODE": _frame(np.array([0.0, 20.0]), {"Mode": np.array([0, 5]), "ModeNum": np.array([0, 5])}),
        "BARO": _frame(time_s, {"Alt": 25.0 * np.sin(np.pi * np.clip(time_s, 0, 400) / 400.0)}),
        "POS": pd.DataFrame(),
        "parse_warnings": [],
    }
    return frames
