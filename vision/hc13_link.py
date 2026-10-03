"""Safe PC-side HC-13 command output for the real-time vision program.

The MCU accepts one ASCII line per vision command::

    V,<servo_g_angle_deg>,<speed>,<confidence>\r\n

``servo_g_angle_deg`` is the calibrated HX8 ``g`` command, not the raw camera
optical-axis offset.  The link deliberately requires a run of stable frames
before transmitting.  If the target disappears or becomes invalid, it stops
transmitting; the MCU's vision timeout then keeps the last safe position.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import median
import time
from typing import Iterable, Optional


@dataclass(frozen=True)
class HC13GateResult:
    """Observable result of one target-gate update."""

    status: str
    confirmed_frames: int = 0
    required_frames: int = 0
    angle_deg: Optional[int] = None
    confidence_percent: int = 0
    should_send: bool = False


def select_control_target(targets: Iterable[dict], min_confidence: float) -> Optional[dict]:
    """Select the best target that is safe to convert into a servo command."""

    valid = [
        target
        for target in targets
        if target.get("target_valid")
        and target.get("angle_mode") == "servo_calibrated"
        and target.get("servo_calibration_valid")
        and target.get("lateral_valid")
        and target.get("missed_frames", 0) == 0
        and float(target.get("score", 0.0)) >= min_confidence
        and "servo_command_deg" in target
    ]
    if not valid:
        return None
    return max(valid, key=lambda item: float(item.get("score", 0.0)))


class StableTargetGate:
    """Require consecutive, same-track, low-spread angle measurements."""

    def __init__(
        self,
        confirmation_frames: int = 5,
        max_spread_deg: float = 2.0,
        send_hz: float = 20.0,
    ) -> None:
        if confirmation_frames < 1:
            raise ValueError("confirmation_frames must be >= 1")
        if max_spread_deg < 0:
            raise ValueError("max_spread_deg must be >= 0")
        if send_hz <= 0:
            raise ValueError("send_hz must be > 0")

        self.confirmation_frames = int(confirmation_frames)
        self.max_spread_deg = float(max_spread_deg)
        self.send_interval_s = 1.0 / float(send_hz)
        self._track_id: Optional[int] = None
        self._angles: deque[float] = deque(maxlen=self.confirmation_frames)
        self._last_send_s: Optional[float] = None

    def reset(self) -> None:
        self._track_id = None
        self._angles.clear()
        self._last_send_s = None

    def update(self, target: Optional[dict], now_s: float) -> HC13GateResult:
        if target is None:
            self.reset()
            return HC13GateResult(
                status="NO VALID TARGET",
                required_frames=self.confirmation_frames,
            )

        track_id = int(target["track_id"])
        if self._track_id != track_id:
            self._track_id = track_id
            self._angles.clear()
            self._last_send_s = None

        self._angles.append(float(target["servo_command_deg"]))
        count = len(self._angles)
        confidence = max(0, min(100, int(round(float(target["score"]) * 100.0))))

        if count < self.confirmation_frames:
            return HC13GateResult(
                status="CONFIRMING",
                confirmed_frames=count,
                required_frames=self.confirmation_frames,
                confidence_percent=confidence,
            )

        spread = max(self._angles) - min(self._angles)
        if spread > self.max_spread_deg:
            return HC13GateResult(
                status=f"UNSTABLE {spread:.1f}deg",
                confirmed_frames=count,
                required_frames=self.confirmation_frames,
                confidence_percent=confidence,
            )

        angle_deg = int(round(median(self._angles)))
        due = (
            self._last_send_s is None
            or now_s - self._last_send_s >= self.send_interval_s
        )
        if due:
            self._last_send_s = now_s
        return HC13GateResult(
            status="LOCKED",
            confirmed_frames=count,
            required_frames=self.confirmation_frames,
            angle_deg=angle_deg,
            confidence_percent=confidence,
            should_send=due,
        )


class HC13SerialLink:
    """Small pyserial wrapper kept separate from detection and gating logic."""

    def __init__(self, port: str, baudrate: int = 230400) -> None:
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError(
                "HC-13 output requires pyserial; run: python -m pip install -r requirements.txt"
            ) from exc

        self.port = str(port)
        self.baudrate = int(baudrate)
        # The HC-USB-T/CH340 fixture may route DTR/RTS to the HC-13 red
        # configuration key.  Keep both inactive before opening; otherwise
        # PySerial can force AT mode and every V packet receives "ERROR".
        self._serial = serial.Serial(
            port=None,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0,
            write_timeout=0.05,
            rtscts=False,
            dsrdtr=False,
        )
        self._serial.dtr = False
        self._serial.rts = False
        self._serial.port = self.port
        self._serial.open()
        time.sleep(0.10)
        self._serial.reset_input_buffer()

    @staticmethod
    def encode_target(angle_deg: int, speed: int, confidence_percent: int) -> bytes:
        angle = max(-180, min(180, int(angle_deg)))
        speed_value = max(0, min(65535, int(speed)))
        confidence = max(0, min(100, int(confidence_percent)))
        return f"V,{angle},{speed_value},{confidence}\r\n".encode("ascii")

    def send_target(self, angle_deg: int, speed: int, confidence_percent: int) -> bytes:
        # Drain optional legacy bring-up echoes so they cannot accumulate in
        # the Windows receive buffer during a long real-time run.
        waiting = int(getattr(self._serial, "in_waiting", 0))
        if waiting > 0:
            self._serial.read(waiting)
        packet = self.encode_target(angle_deg, speed, confidence_percent)
        self._serial.write(packet)
        return packet

    def close(self) -> None:
        if self._serial is not None and self._serial.is_open:
            self._serial.close()
