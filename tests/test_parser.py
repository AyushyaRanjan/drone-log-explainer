"""A hand-built dataflash log proves the parser reads real .BIN bytes."""

import struct

from app.parser import parse_bin


def _fmt(mtype: int, length: int, name: str, fmt: str, columns: str) -> bytes:
    body = struct.pack("<BB4s16s64s", mtype, length, name.encode(), fmt.encode(), columns.encode())
    return b"\xa3\x95\x80" + body


def test_parse_minimal_bin(tmp_path):
    fmt = _fmt(0x20, 27, "BAT", "Qffff", "TimeUS,Volt,Curr,RemPct,Temp")
    assert len(fmt) == 89
    rows = b""
    for index, volt in enumerate((16.8, 15.5)):
        rows += b"\xa3\x95\x20" + struct.pack("<Qffff", 2_000_000 + index * 1_000_000, volt, 12.0, 80.0 - index, 31.0)
    path = tmp_path / "mini.bin"
    path.write_bytes(fmt + rows + b"\xff" * 40)

    parsed = parse_bin(str(path))
    battery = parsed["BAT"]
    assert len(battery) == 2
    assert abs(battery["Volt"].iloc[0] - 16.8) < 0.01
    assert abs(battery["time_s"].iloc[0]) < 1e-6
    assert abs(battery["time_s"].iloc[1] - 1.0) < 1e-3
    assert parsed["GPS"].empty
    assert parsed["parse_warnings"] == []
