import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from services.serial_service import SerialMonitor


class FakeSerial:
    is_open = True

    def close(self) -> None:
        self.is_open = False


class MalformedFrameSerial(FakeSerial):
    def __init__(self, monitor: SerialMonitor) -> None:
        self.monitor = monitor

    def readline(self) -> bytes:
        self.monitor._stop.set()
        return (
            b"Fluid: 1.2.3 mL | BPM: 80.0 | "
            b"SpO2: 97.0% | STATUS: NORMAL"
        )


class SerialMonitorTests(unittest.TestCase):
    def test_open_port_without_data_is_not_connected(self) -> None:
        monitor = SerialMonitor(port="test")
        fake_serial = FakeSerial()

        def finish_boot_wait(_seconds: float) -> None:
            monitor._stop.set()

        with (
            patch("services.serial_service.serial.Serial", return_value=fake_serial),
            patch("services.serial_service.time.sleep", side_effect=finish_boot_wait),
        ):
            monitor._run()

        snapshot = monitor.snapshot()
        self.assertFalse(snapshot["connected"])
        self.assertEqual(snapshot["connection_status"], "WAITING_FOR_DATA")
        self.assertIsNone(snapshot["fluid_ml"])
        self.assertIsNone(snapshot["last_update"])
        self.assertEqual(
            snapshot["validity"],
            {"fluid": False, "bpm": False, "spo2": False},
        )

    def test_malformed_frame_does_not_mark_stream_connected(self) -> None:
        monitor = SerialMonitor(port="test")
        fake_serial = MalformedFrameSerial(monitor)

        with (
            patch("services.serial_service.serial.Serial", return_value=fake_serial),
            patch("services.serial_service.time.sleep"),
        ):
            monitor._run()

        snapshot = monitor.snapshot()
        self.assertFalse(snapshot["connected"])
        self.assertEqual(snapshot["connection_status"], "WAITING_FOR_DATA")
        self.assertIsNone(snapshot["last_update"])

    def test_old_telemetry_is_reported_as_stale(self) -> None:
        monitor = SerialMonitor(port="test")
        old_update = (
            datetime.now(timezone.utc) - timedelta(seconds=10)
        ).isoformat()
        monitor._set(
            connected=True,
            connection_status="CONNECTED",
            last_update=old_update,
        )

        snapshot = monitor.snapshot()

        self.assertFalse(snapshot["connected"])
        self.assertEqual(snapshot["connection_status"], "STALE")

    def test_fresh_valid_telemetry_remains_connected(self) -> None:
        monitor = SerialMonitor(port="test")
        monitor._set(
            connected=True,
            connection_status="CONNECTED",
            fluid_ml=42.5,
            device_status="NORMAL",
            last_update=datetime.now(timezone.utc).isoformat(),
            validity={"fluid": True, "bpm": False, "spo2": False},
        )

        snapshot = monitor.snapshot()

        self.assertTrue(snapshot["connected"])
        self.assertEqual(snapshot["connection_status"], "CONNECTED")
        self.assertEqual(snapshot["fluid_ml"], 42.5)
        self.assertTrue(snapshot["validity"]["fluid"])


if __name__ == "__main__":
    unittest.main()
