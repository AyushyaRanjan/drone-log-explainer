# Drone Flight-Log Explainer

A small web app for hobbyist ArduCopter / Pixhawk pilots. Upload a dataflash `.BIN` log, or run a built-in demo flight, and get a plain-English diagnostic report covering battery, vibration, GPS, and autopilot warnings.

The analyzer produces every number in the report. The language model only explains those findings. If `ANTHROPIC_API_KEY` is missing or the call fails, a template writes the same sections so the app still works offline.

## Architecture

```
browser (index.html + Chart.js)
    |  POST /api/analyze  or  POST /api/demo
    v
FastAPI (app/main.py)
    |-- parser.py      pymavlink DF reader -> pandas DataFrame per message
    |-- analyzers/     battery, vibration, GPS, warnings, summary
    |-- report.py      verdict, checklist, downsampled timeseries
    |-- explainer.py   Claude, or the template fallback
    v
in-memory report store, plus JSON under data/reports/
```

No database. Reports are kept in process memory and written as JSON when the disk is writable.

## Run it

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://localhost:8000 and click **Try demo flight**.

Three-line check for judges, from the project directory, with the server already running:

```bash
curl -s -X POST http://localhost:8000/api/demo -o demo-report.json
python -c "import json; r=json.load(open('demo-report.json')); print(r['verdict'], r['explanation']['source'], r['summary'])"
```

Health check: `GET /api/health`.

## Environment

Copy `.env.example` to `.env`.

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Optional. When empty, explanations use the template. |
| `ANTHROPIC_MODEL` | Claude model id. Defaults to `claude-sonnet-5-5`. |

The model never receives the raw log. It receives the summary, section metrics, findings, and checklist as JSON.

## Thresholds

| Check | Rule |
|---|---|
| Cell count | `round(max voltage in the first 15 s / 4.2)`, clamped to 2–12 |
| Low cell | warning below 3.5 V/cell, critical below 3.3 V/cell |
| Sag | linear fit of `Volt` vs `Curr` where current > 5 A; warning if ohms per cell > 0.025 |
| Fast drain | warning if percent-per-minute is > 1.5× the median 5 s rate in the first third, or remaining % drops more than 30 points in a flight under 6 minutes |
| Peak current | info above 80% of 40 A |
| Temperature | warning above 60 C |
| Vibration | under 30 m/s² good, 30–60 warning, above 60 critical; a run above 60 for more than 2 s is critical |
| Clipping | warning if `Clip0/1/2` increases during the flight |
| Imbalance | info if one axis mean is more than twice the others |
| GPS fix | status < 3: warning if any sample, critical if more than 5% of samples |
| Satellites | warning if `NSats` < 6 for more than 10% of samples |
| HDOP | warning if above 2.0 for more than 10% of samples; critical if above 5.0 for more than 2 s |
| Messages | case-insensitive match on failsafe, crash, low battery, EKF, compass, gps glitch, bad gyro, error |
| ERR | one finding per subsystem/code; critical when the subsystem name contains FAILSAFE |
| Mode | info hint when the log changes to RTL or LAND and no later mode is a manual mode |

## Assumptions

- Time is `TimeUS` in microseconds, converted to seconds from the first sample. `TimeMS` is accepted if `TimeUS` is absent.
- Battery instance 0 is the pack under test. If instance 0 is missing, the lowest instance is used and the others are noted.
- Cell count is estimated from the highest voltage in the first 15 seconds, not from a parameter.
- Temperature values whose median is above 200 are treated as centidegrees.
- Mode numbers use the ArduCopter map. Other vehicle types show as `MODE n`.
- An RTL or LAND with no later manual mode is only a hint. The log does not record who commanded it.
- The demo sag sits at 14.1 V, then dips to 13.12 V for five seconds so a 4S pack crosses the 3.3 V/cell critical rule. The 20 s GPS window is under 10% of a 4-minute flight, so the demo also drops GPS status below 3 for 8 seconds to produce a GPS warning.
- Timeseries charts share one downsampled clock of at most 2000 points, interpolated per series.
- Info-only findings leave a section `ok`. `critical` outranks `warning`.

## Limitations

These rules are heuristics, not certified diagnostics. A quiet log is not an airworthiness sign-off. Vibration units, battery temperature scaling, and failsafe causes vary by firmware and vehicle. The resistance estimate is a straight-line fit, not a lab measurement of internal resistance. Always confirm with a visual inspection and the official ArduPilot tools before flying.

## Tests

```bash
pytest -q
```
