"""Parse an ArduPilot dataflash .BIN into one DataFrame per message type."""

from __future__ import annotations

import contextlib
import os

import numpy as np
import pandas as pd
from pymavlink import mavutil

WANTED = ("BAT", "VIBE", "GPS", "MSG", "ERR", "MODE", "BARO", "POS")


class ParseError(Exception):
    """The file could not be opened as a dataflash log."""


def empty_frames() -> dict[str, pd.DataFrame]:
    return {name: pd.DataFrame() for name in WANTED}


def parse_bin(path: str) -> dict:
    """Return DataFrames plus a `parse_warnings` list.

    Time is seconds from the earliest TimeUS (or TimeMS) sample. A missing
    message type is an empty DataFrame. Bad messages are counted and skipped.
    """
    buckets: dict[str, list[dict]] = {name: [] for name in WANTED}
    skipped = 0
    warnings: list[str] = []

    mlog = None
    try:
        # pymavlink prints one line per bad byte. Silence that so a partial log
        # does not flood the server, and always close the mmap so Windows can
        # delete the temp file afterwards.
        with open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stderr(devnull):
            try:
                mlog = mavutil.mavlink_connection(path)
            except Exception as exc:  # noqa: BLE001 - partial logs must not crash the request
                raise ParseError(f"Could not open flight log: {exc}") from exc
            while True:
                try:
                    msg = mlog.recv_match(blocking=False)
                except Exception as exc:  # noqa: BLE001
                    skipped += 1
                    if len(warnings) < 8:
                        warnings.append(f"Stopped reading after a decode error: {exc}")
                    break
                if msg is None:
                    break
                try:
                    mtype = msg.get_type()
                    if mtype not in buckets:
                        continue
                    row = msg.to_dict()
                    row.pop("mavpackettype", None)
                    buckets[mtype].append(row)
                except Exception as exc:  # noqa: BLE001
                    skipped += 1
                    if len(warnings) < 8:
                        warnings.append(f"Skipped an unreadable message: {exc}")
    finally:
        if mlog is not None:
            try:
                mlog.close()
            except Exception:
                pass

    if skipped:
        warnings.append(f"Skipped {skipped} unreadable message(s).")

    frames = _frames_from_buckets(buckets)
    if not any(not frame.empty for frame in frames.values()):
        warnings.append("No BAT, VIBE, GPS, MSG, ERR, MODE, BARO, or POS messages were found.")
    frames["parse_warnings"] = warnings
    return frames


def _frames_from_buckets(buckets: dict[str, list[dict]]) -> dict[str, pd.DataFrame]:
    built: dict[str, pd.DataFrame] = {}
    clock: dict[str, pd.Series] = {}
    for name, rows in buckets.items():
        if not rows:
            built[name] = pd.DataFrame()
            continue
        frame = pd.DataFrame(rows)
        built[name] = frame
        clock[name] = _clock_us(frame)

    present = [series.dropna() for series in clock.values() if not series.dropna().empty]
    origin = float(min(series.min() for series in present)) if present else 0.0

    frames: dict[str, pd.DataFrame] = {}
    for name, frame in built.items():
        if frame.empty:
            frames[name] = frame
            continue
        stamps = clock.get(name)
        if stamps is None:
            seconds = pd.Series(0.0, index=frame.index)
        else:
            seconds = (stamps - origin) / 1_000_000.0
            seconds = seconds.fillna(0.0)
        frame = frame.copy()
        frame.insert(0, "time_s", seconds)
        frames[name] = frame.sort_values("time_s", kind="mergesort").reset_index(drop=True)
    return frames


def _clock_us(frame: pd.DataFrame) -> pd.Series:
    """Microseconds used only to align messages. TimeMS is scaled up to that unit."""
    if "TimeUS" in frame.columns:
        return pd.to_numeric(frame["TimeUS"], errors="coerce")
    if "TimeMS" in frame.columns:
        return pd.to_numeric(frame["TimeMS"], errors="coerce") * 1000.0
    return pd.Series(np.nan, index=frame.index)
