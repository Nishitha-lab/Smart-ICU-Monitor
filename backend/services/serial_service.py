"""Non-blocking ESP32 serial reader for TBP_final.ino."""
from __future__ import annotations
import os, re, threading, time
from datetime import datetime, timezone
from typing import Any
import serial
from serial import SerialException

LINE_PATTERN = re.compile(r"Fluid:\s*([\d.]+)\s*mL\s*\|\s*BPM:\s*([\d.]+)\s*\|\s*SpO2:\s*([\d.]+)%\s*\|\s*STATUS:\s*(\w+)")

class SerialMonitor:
    def __init__(self, port: str | None = None, baudrate: int = 115200):
        self.port = port or os.getenv("SMART_ICU_SERIAL_PORT", "COM9")
        self.baudrate = int(os.getenv("SMART_ICU_BAUDRATE", baudrate))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._serial: serial.Serial | None = None
        self._data: dict[str, Any] = {
            "connected": False, "port": self.port, "baudrate": self.baudrate,
            "fluid_ml": None, "bpm": None, "spo2": None,
            "device_status": None, "connection_status": "DISCONNECTED",
            "last_update": None, "last_error": None,
            "validity": {"fluid": False, "bpm": False, "spo2": False}
        }

    def start(self) -> None:
        if self._thread and self._thread.is_alive(): return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="esp32-serial-reader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._serial and self._serial.is_open:
            try: self._serial.close()
            except Exception: pass
        if self._thread: self._thread.join(timeout=2)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            result = dict(self._data)
            result["validity"] = dict(self._data["validity"])
        updated = result.get("last_update")
        if updated:
            try:
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(updated)).total_seconds()
                if age > 5:
                    result["connected"] = False
                    result["connection_status"] = "STALE"
            except ValueError: pass
        return result

    def _set(self, **kwargs: Any) -> None:
        with self._lock: self._data.update(kwargs)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._serial = serial.Serial(self.port, self.baudrate, timeout=1)
                # ESP32 may reset when the serial port opens; allow it to boot.
                time.sleep(2)
                self._set(connected=True, connection_status="CONNECTED", last_error=None)
                while not self._stop.is_set():
                    raw = self._serial.readline()
                    if not raw: continue
                    line = raw.decode("utf-8", errors="ignore").strip()
                    match = LINE_PATTERN.search(line)
                    if not match: continue
                    fluid, bpm, spo2 = map(float, match.group(1, 2, 3))
                    device_status = match.group(4).upper()
                    self._set(
                        connected=True, connection_status="CONNECTED", port=self.port,
                        fluid_ml=fluid, bpm=bpm, spo2=spo2, device_status=device_status,
                        last_update=datetime.now(timezone.utc).isoformat(), last_error=None,
                        validity={"fluid": fluid >= 0, "bpm": 40 <= bpm <= 200, "spo2": 70 <= spo2 <= 100}
                    )
            except (SerialException, OSError) as exc:
                self._set(connected=False, connection_status="DISCONNECTED", last_error=str(exc))
                time.sleep(3)
            finally:
                if self._serial:
                    try:
                        if self._serial.is_open: self._serial.close()
                    except Exception: pass
                    self._serial = None

serial_monitor = SerialMonitor()
