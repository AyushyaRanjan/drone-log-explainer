"""Analyzer behavior on a clean flight, the demo flight, and missing data."""

from app.analyzers.battery import analyze_battery
from app.analyzers.gps import analyze_gps
from app.analyzers.vibration import analyze_vibration
from app.analyzers.warnings import analyze_warnings
from app.demo import generate_clean_flight, generate_demo_flight


def test_clean_flight_is_ok():
    flight = generate_clean_flight()
    battery = analyze_battery(flight["BAT"])
    vibration = analyze_vibration(flight["VIBE"])
    gps = analyze_gps(flight["GPS"])
    warnings = analyze_warnings(flight["MSG"], flight["ERR"], flight["MODE"])
    assert battery["status"] == "ok"
    assert vibration["status"] == "ok"
    assert gps["status"] == "ok"
    assert warnings["status"] == "ok"


def test_anomalous_flight_matches_expected_severity():
    flight = generate_demo_flight()
    battery = analyze_battery(flight["BAT"])
    vibration = analyze_vibration(flight["VIBE"])
    gps = analyze_gps(flight["GPS"])
    warnings = analyze_warnings(flight["MSG"], flight["ERR"], flight["MODE"])

    assert battery["status"] == "critical"
    assert any(item["severity"] == "critical" for item in battery["findings"])
    assert any(item["severity"] in {"warning", "critical"} for item in vibration["findings"])
    assert any(item["severity"] == "warning" for item in gps["findings"])
    assert any("failsafe" in f"{item['title']} {item['detail']}".lower() for item in warnings["findings"])


def test_missing_message_type_is_no_data():
    assert analyze_gps(None)["status"] == "no_data"
    assert analyze_battery(None)["findings"] == []
    assert analyze_vibration(None)["status"] == "no_data"
