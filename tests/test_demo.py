"""The synthetic demo flight contains the injected anomalies."""

from app.demo import generate_demo_flight
from app.report import build_report


def test_demo_flight_contains_injected_anomalies():
    report = build_report(generate_demo_flight(seed=42), file_name="demo-flight.bin")
    battery = report["sections"]["battery"]["findings"]
    vibration = report["sections"]["vibration"]["findings"]
    gps = report["sections"]["gps"]["findings"]
    warnings = report["sections"]["warnings"]["findings"]

    assert any(item["severity"] == "critical" for item in battery)
    assert len(vibration) >= 1
    assert any(item["severity"] == "warning" for item in gps)
    assert any("failsafe" in f"{item['title']} {item['detail']}".lower() for item in warnings)
    assert report["explanation"]["source"] in {"template", "llm"}
    assert "## Summary" in report["explanation"]["markdown"]
    assert report["verdict"] == "critical"
