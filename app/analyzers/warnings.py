"""Failsafe text, ERR rows, and flight-mode changes."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from app import config
from app.analyzers import empty_result, finding, num_series, r, rollup, time_values


def mode_name(raw_mode, raw_num) -> str:
    for candidate in (raw_mode, raw_num):
        if candidate is None or (isinstance(candidate, float) and pd.isna(candidate)):
            continue
        if isinstance(candidate, str):
            text = candidate.strip()
            if not text:
                continue
            try:
                number = int(float(text))
            except ValueError:
                return text.upper()
            return config.COPTER_MODES.get(number, f"MODE {number}")
        try:
            number = int(candidate)
        except (TypeError, ValueError):
            continue
        return config.COPTER_MODES.get(number, f"MODE {number}")
    return "UNKNOWN"


def analyze_warnings(msg: pd.DataFrame | None, err: pd.DataFrame | None, mode: pd.DataFrame | None) -> dict:
    has_any = any(frame is not None and not frame.empty for frame in (msg, err, mode))
    if not has_any:
        return empty_result()

    findings: list[dict] = []
    messages: list[dict] = []
    if msg is not None and not msg.empty and "Message" in msg.columns:
        time_s = time_values(msg)
        texts = msg["Message"].astype(str)
        grouped: dict[str, dict] = {}
        for i, text in enumerate(texts):
            cleaned = " ".join(text.split())
            if not cleaned or cleaned.lower() in {"nan", "none"}:
                continue
            bucket = grouped.setdefault(cleaned, {"count": 0, "time_s": None})
            bucket["count"] += 1
            if bucket["time_s"] is None and i < len(time_s) and pd.notna(time_s[i]):
                bucket["time_s"] = float(time_s[i])
        for text, info in grouped.items():
            lowered = text.lower()
            if not any(keyword in lowered for keyword in config.MSG_KEYWORDS):
                messages.append({"time_s": r(info["time_s"], 3), "message": text, "count": info["count"]})
                continue
            severity = "critical" if any(keyword in lowered for keyword in config.MSG_CRITICAL_KEYWORDS) else "warning"
            count_bit = f" It appears {info['count']} times." if info["count"] > 1 else ""
            findings.append(
                finding(
                    "warnings.msg",
                    severity,
                    text if len(text) <= 80 else text[:77] + "...",
                    f"Autopilot message: \"{text}\".{count_bit}",
                    time_s=info["time_s"],
                    evidence={"message": text, "count": info["count"]},
                )
            )
            messages.append({"time_s": r(info["time_s"], 3), "message": text, "count": info["count"], "matched": True})

    errors: list[dict] = []
    if err is not None and not err.empty:
        time_s = time_values(err)
        subsys = num_series(err, "Subsys")
        ecode = num_series(err, "ECode")
        if subsys is not None:
            seen: Counter[tuple[int, int]] = Counter()
            first_time: dict[tuple[int, int], float | None] = {}
            for i in range(len(err)):
                if pd.isna(subsys.iloc[i]):
                    continue
                sub = int(subsys.iloc[i])
                code = int(ecode.iloc[i]) if ecode is not None and pd.notna(ecode.iloc[i]) else 0
                key = (sub, code)
                seen[key] += 1
                if key not in first_time:
                    ts = time_s[i] if i < len(time_s) and pd.notna(time_s[i]) else None
                    first_time[key] = None if ts is None else float(ts)
            for (sub, code), count in seen.items():
                name = config.ERR_SUBSYSTEMS.get(sub, f"subsystem {sub}")
                severity = "critical" if "FAILSAFE" in name.upper() else "warning"
                findings.append(
                    finding(
                        f"warnings.err.{sub}.{code}",
                        severity,
                        f"Autopilot error: {name}",
                        f"ERR subsystem {sub} ({name}), code {code}, seen {count} time(s).",
                        time_s=first_time[(sub, code)],
                        evidence={"subsys": sub, "ecode": code, "name": name, "count": count},
                    )
                )
                errors.append({"subsys": sub, "ecode": code, "name": name, "count": count, "time_s": r(first_time[(sub, code)], 3)})

    timeline: list[dict] = []
    if mode is not None and not mode.empty:
        time_s = time_values(mode)
        mode_col = mode["Mode"] if "Mode" in mode.columns else pd.Series([None] * len(mode))
        num_col = mode["ModeNum"] if "ModeNum" in mode.columns else pd.Series([None] * len(mode))
        for i in range(len(mode)):
            name = mode_name(mode_col.iloc[i], num_col.iloc[i])
            ts = float(time_s[i]) if i < len(time_s) and pd.notna(time_s[i]) else None
            if timeline and timeline[-1]["mode"] == name:
                continue
            timeline.append({"time_s": r(ts, 3), "mode": name})

        for index, event in enumerate(timeline):
            if event["mode"] not in config.UNPLANNED_MODES:
                continue
            later = [item["mode"] for item in timeline[index + 1 :]]
            if any(name in config.MANUAL_MODES for name in later):
                continue
            findings.append(
                finding(
                    "warnings.unplanned_mode",
                    "info",
                    f"Mode changed to {event['mode']}",
                    f"The log shows a change to {event['mode']} with no later manual mode. "
                    "This is only a hint: the pilot may have selected it, or a failsafe may have.",
                    time_s=event["time_s"],
                    evidence={"mode": event["mode"], "later_modes": later},
                )
            )

    metrics = {
        "message_count": len(messages),
        "error_count": sum(item["count"] for item in errors) if errors else 0,
        "mode_timeline": timeline,
        "mode_changes": max(0, len(timeline) - 1),
        "matched_messages": [item for item in messages if item.get("matched")],
        "errors": errors,
    }
    return {"status": rollup(findings, True), "findings": findings, "metrics": metrics}
