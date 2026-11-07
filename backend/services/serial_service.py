"""Non-blocking ESP32 serial reader for TBP_final.ino."""
from __future__ import annotations
import logging
import os, re, threading, time
from datetime import datetime, timezone
from typing import Any
import serial
from serial import SerialException

logger = logging.getLogger(__name__)

_NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
LINE_PATTERN = re.compile(
    rf"Fluid:\s*({_NUMBER_PATTERN})\s*mL\s*\|\s*"
    rf"BPM:\s*({_NUMBER_PATTERN})\s*\|\s*"
    rf"SpO2:\s*({_NUMBER_PATTERN})%\s*\|\s*"
    r"STATUS:\s*(\w+)"
)

class SerialMonitor:
    def __init__(self, port: str | None = None, baudrate: int = 115200, device_id: str | None = None):
        self.port = port or os.getenv("SMART_ICU_SERIAL_PORT", "COM9")
        self.baudrate = int(os.getenv("SMART_ICU_BAUDRATE", baudrate))
        self.device_id = device_id or os.getenv("SMART_ICU_DEVICE_ID", "ESP32-001")
        self._on_reading_callback: Any = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._serial: serial.Serial | None = None
        self._data: dict[str, Any] = {
            "connected": False, "port": self.port, "baudrate": self.baudrate,
            "device_id": self.device_id,
            "fluid_ml": None, "bpm": None, "spo2": None,
            "device_status": None, "connection_status": "DISCONNECTED",
            "last_update": None, "last_error": None,
            "validity": {"fluid": False, "bpm": False, "spo2": False}
        }

    def set_on_reading(self, callback: Any) -> None:
        self._on_reading_callback = callback

    def simulate_reading(
        self,
        fluid_ml: float,
        bpm: float,
        spo2: float,
        device_status: str = "NORMAL",
        reading_time: str | None = None,
    ) -> dict[str, Any]:
        now_iso = reading_time or datetime.now(timezone.utc).isoformat()
        validity = {"fluid": fluid_ml >= 0, "bpm": 40 <= bpm <= 200, "spo2": 70 <= spo2 <= 100}
        reading_payload = {
            "fluid_ml": fluid_ml,
            "bpm": bpm,
            "spo2": spo2,
            "device_status": device_status.upper(),
            "connection_status": "CONNECTED",
            "reading_time": now_iso,
            "validity": validity,
        }
        self._set(
            connected=True, connection_status="CONNECTED", port=self.port,
            device_id=self.device_id,
            fluid_ml=fluid_ml, bpm=bpm, spo2=spo2, device_status=device_status.upper(),
            last_update=now_iso, last_error=None,
            validity=validity,
        )
        if self._on_reading_callback:
            try:
                return self._on_reading_callback(self.device_id, reading_payload)
            except Exception as exc:
                logger.exception(
                    "Error in simulated reading callback for device '%s': %s",
                    self.device_id,
                    exc,
                )
                return reading_payload
        return reading_payload

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
                self._set(
                    connected=False,
                    connection_status="WAITING_FOR_DATA",
                    fluid_ml=None,
                    bpm=None,
                    spo2=None,
                    device_status=None,
                    last_update=None,
                    last_error=None,
                    validity={"fluid": False, "bpm": False, "spo2": False},
                )
                while not self._stop.is_set():
                    raw = self._serial.readline()
                    if not raw: continue
                    line = raw.decode("utf-8", errors="ignore").strip()
                    match = LINE_PATTERN.search(line)
                    if not match: continue
                    fluid, bpm, spo2 = map(float, match.group(1, 2, 3))
                    device_status = match.group(4).upper()
                    now_iso = datetime.now(timezone.utc).isoformat()
                    validity = {"fluid": fluid >= 0, "bpm": 40 <= bpm <= 200, "spo2": 70 <= spo2 <= 100}
                    self._set(
                        connected=True, connection_status="CONNECTED", port=self.port,
                        device_id=self.device_id,
                        fluid_ml=fluid, bpm=bpm, spo2=spo2, device_status=device_status,
                        last_update=now_iso, last_error=None,
                        validity=validity
                    )
                    if self._on_reading_callback:
                        try:
                            self._on_reading_callback(self.device_id, {
                                "fluid_ml": fluid,
                                "bpm": bpm,
                                "spo2": spo2,
                                "device_status": device_status,
                                "connection_status": "CONNECTED",
                                "reading_time": now_iso,
                                "validity": validity,
                            })
                        except Exception as exc:
                            logger.exception(
                                "Error in serial reading callback for device '%s': %s",
                                self.device_id,
                                exc,
                            )
            except (SerialException, OSError) as exc:
                self._set(connected=False, connection_status="DISCONNECTED", last_error=str(exc))
                if self._stop.wait(3.0):
                    break
                if self._stop.is_set():
                    break
            finally:
                if self._serial:
                    try:
                        if self._serial.is_open: self._serial.close()
                    except Exception: pass
                    self._serial = None

serial_monitor = SerialMonitor()
