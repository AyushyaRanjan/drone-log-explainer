"""Plain-English explanation. Uses Claude when a key is set, otherwise a template."""

from __future__ import annotations

import json

from app import config

SYSTEM_PROMPT = (
    "You are a drone diagnostics assistant explaining a flight log to a hobbyist pilot. "
    "Use ONLY the numbers and findings provided in the JSON. Do not invent values, causes, "
    "or measurements. If something is not in the data, say it is unknown. Use plain language, "
    "avoid jargon, define a term the first time you use it. Output Markdown with these headings: "
    "Summary, Battery, Vibration, GPS, Warnings, What to inspect before the next flight. "
    "Keep it under 500 words. Be calm and specific. Do not tell the pilot the drone is safe to fly "
    "unless the verdict is ok."
)


def explain(payload: dict) -> dict:
    if not config.ANTHROPIC_API_KEY:
        return {"source": "template", "markdown": render_template(payload)}
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        message = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=1200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(payload, default=str)}],
        )
        text = "".join(getattr(block, "text", "") for block in message.content).strip()
        if not text:
            return {"source": "template", "markdown": render_template(payload)}
        return {"source": "llm", "markdown": text}
    except Exception:
        return {"source": "template", "markdown": render_template(payload)}


def render_template(payload: dict) -> str:
    summary = payload.get("summary") or {}
    sections = payload.get("sections") or {}
    verdict = payload.get("verdict") or "ok"
    duration = _fmt_duration(summary.get("duration_s"))
    alt = summary.get("max_alt_m")
    alt_text = "unknown" if alt is None else f"{alt} m"
    counts = summary.get("findings_count") or {}
    lines = [
        "## Summary",
        "",
        f"This flight lasted {duration} and reached about {alt_text}. "
        f"There were {summary.get('mode_changes', 0)} mode changes. "
        f"The checks found {counts.get('critical', 0)} critical item(s), "
        f"{counts.get('warning', 0)} warning(s), and {counts.get('info', 0)} note(s). "
        f"Overall verdict: {verdict}.",
        "",
    ]
    if verdict == "ok":
        lines.append(
            "The automated checks did not find a warning or a critical issue. "
            "That is not a certificate that the aircraft is ready to fly."
        )
    else:
        lines.append(
            "Treat the critical items as grounds to inspect the aircraft before another flight."
        )
    lines.append("")
    lines.extend(_section("Battery", sections.get("battery"), _battery_blurb))
    lines.extend(_section("Vibration", sections.get("vibration"), _vibration_blurb))
    lines.extend(_section("GPS", sections.get("gps"), _gps_blurb))
    lines.extend(_section("Warnings", sections.get("warnings"), _warnings_blurb))
    lines.append("## What to inspect before the next flight")
    lines.append("")
    checklist = payload.get("checklist") or []
    if not checklist:
        lines.append("No inspection items were generated from these rules.")
    else:
        for index, item in enumerate(checklist, start=1):
            lines.append(f"{index}. {item.get('item')} — {item.get('reason')}")
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def _section(title: str, section: dict | None, blurb) -> list[str]:
    lines = [f"## {title}", ""]
    if not section or section.get("status") == "no_data":
        lines.append(f"No {title.lower()} data was found in this log, so this section is unknown.")
        lines.append("")
        return lines
    lines.append(blurb(section.get("metrics") or {}))
    findings = section.get("findings") or []
    if not findings:
        lines.append("No rule in this section fired.")
    else:
        for item in findings:
            when = _fmt_duration(item.get("time_s")) if item.get("time_s") is not None else None
            prefix = f"At {when}: " if when else ""
            lines.append(f"- **{item.get('severity', 'info')}** — {prefix}{item.get('title')}. {item.get('detail')}")
    lines.append("")
    return lines


def _battery_blurb(metrics: dict) -> str:
    return (
        f"Pack voltage ran from {metrics.get('voltage_min')} V to {metrics.get('voltage_max')} V "
        f"(average {metrics.get('voltage_mean')} V). Estimated cells: {metrics.get('cells')}. "
        f"Lowest per-cell voltage: {metrics.get('min_cell_v')} V. "
        f"Peak current: {metrics.get('curr_max')} A. "
        f"Estimated resistance: {metrics.get('resistance_ohm_per_cell')} ohm per cell. "
        f"Remaining-percent drain: {metrics.get('drain_rate_pct_per_min')} percent per minute. "
        f"Peak temperature: {metrics.get('temp_max_c')} C."
    )


def _vibration_blurb(metrics: dict) -> str:
    return (
        "Vibration is reported in meters per second squared (m/s^2); under 30 is a comfortable range. "
        f"Mean X/Y/Z: {metrics.get('VibeX_mean')} / {metrics.get('VibeY_mean')} / {metrics.get('VibeZ_mean')}. "
        f"95th percentile X/Y/Z: {metrics.get('VibeX_p95')} / {metrics.get('VibeY_p95')} / {metrics.get('VibeZ_p95')}."
    )


def _gps_blurb(metrics: dict) -> str:
    fraction = metrics.get("fix_below_3d_fraction")
    fraction_text = "unknown" if fraction is None else f"{round(float(fraction) * 100, 1)}%"
    return (
        "A 3D fix (status 3 or higher) means the GPS has enough satellites to estimate position and height. "
        f"Share of samples without a 3D fix: {fraction_text}. "
        f"Fewest satellites: {metrics.get('nsats_min')}. Worst HDOP: {metrics.get('hdop_max')} "
        "(HDOP is a unitless accuracy figure; lower is better)."
    )


def _warnings_blurb(metrics: dict) -> str:
    timeline = metrics.get("mode_timeline") or []
    if not timeline:
        modes = "No mode changes were logged."
    else:
        bits = []
        for event in timeline:
            bits.append(f"{_fmt_duration(event.get('time_s'))} {event.get('mode')}")
        modes = "Mode timeline: " + ", ".join(bits) + "."
    return f"Matched autopilot messages: {len(metrics.get('matched_messages') or [])}. {modes}"


def _fmt_duration(seconds) -> str:
    if seconds is None:
        return "unknown"
    try:
        total = int(round(float(seconds)))
    except (TypeError, ValueError):
        return "unknown"
    sign = "-" if total < 0 else ""
    total = abs(total)
    return f"{sign}{total // 60}:{total % 60:02d}"
