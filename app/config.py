"""Environment loading and diagnostic thresholds."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "").strip() or "claude-sonnet-5-5"

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
TIMESERIES_MAX_POINTS = 2000

# Battery
CELL_NOMINAL_V = 4.2
CELL_COUNT_MIN = 2
CELL_COUNT_MAX = 12
CELL_WARN_V = 3.5
CELL_CRITICAL_V = 3.3
SAG_MIN_CURRENT_A = 5.0
SAG_RESISTANCE_OHM_PER_CELL = 0.025
DRAIN_RATE_RATIO = 1.5
FAST_DRAIN_DROP_PCT = 30.0
FAST_DRAIN_MAX_DURATION_S = 6 * 60
PEAK_CURRENT_REFERENCE_A = 40.0
PEAK_CURRENT_FRACTION = 0.80
BATTERY_TEMP_WARN_C = 60.0
START_WINDOW_S = 15.0

# Vibration (m/s^2)
VIBE_GOOD_MAX = 30.0
VIBE_WARN_MAX = 60.0
VIBE_SUSTAINED_S = 2.0
VIBE_IMBALANCE_RATIO = 2.0

# GPS
GPS_FIX_3D = 3
GPS_FIX_CRITICAL_FRACTION = 0.05
GPS_NSATS_MIN = 6
GPS_NSATS_WARN_FRACTION = 0.10
GPS_HDOP_WARN = 2.0
GPS_HDOP_WARN_FRACTION = 0.10
GPS_HDOP_CRITICAL = 5.0
GPS_HDOP_SUSTAINED_S = 2.0

# Message keywords matched case-insensitively against MSG.Message
MSG_KEYWORDS = (
    "failsafe",
    "crash",
    "low battery",
    "ekf",
    "compass",
    "gps glitch",
    "bad gyro",
    "error",
)
MSG_CRITICAL_KEYWORDS = ("failsafe", "crash", "low battery")

ERR_SUBSYSTEMS = {
    1: "Main",
    2: "Radio",
    3: "Compass",
    4: "Optical flow",
    5: "Failsafe battery",
    6: "Failsafe GCS",
    7: "Failsafe EKF",
    8: "GPS",
    9: "Crash check",
    10: "Flip",
    11: "Autotune",
    12: "Parachute",
}

# ArduCopter mode numbers. Plane/rover logs fall back to "MODE n".
COPTER_MODES = {
    0: "STABILIZE",
    1: "ACRO",
    2: "ALT_HOLD",
    3: "AUTO",
    4: "GUIDED",
    5: "LOITER",
    6: "RTL",
    7: "CIRCLE",
    8: "LAND",
    9: "DRIFT",
    10: "SPORT",
    11: "FLIP",
    13: "POSHOLD",
    14: "BRAKE",
    15: "THROW",
    17: "AVOID_ADSB",
    18: "GUIDED_NOGPS",
    19: "SMART_RTL",
    20: "FLOWHOLD",
    21: "FOLLOW",
    22: "ZIGZAG",
    23: "SYSTEMID",
    24: "AUTOROTATE",
    25: "AUTO_RTL",
}

MANUAL_MODES = {
    "STABILIZE",
    "ACRO",
    "ALT_HOLD",
    "LOITER",
    "POSHOLD",
    "SPORT",
    "DRIFT",
    "BRAKE",
    "THROW",
}

UNPLANNED_MODES = {"RTL", "LAND", "SMART_RTL", "AUTO_RTL"}

REPORT_DIR = ROOT / "data" / "reports"
