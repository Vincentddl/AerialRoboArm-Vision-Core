import argparse
import json
import queue
import re
import shutil
import socket
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RTT_VALUE_PATTERN = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)=(-?\d+(?:\.\d+)?)")


# Preview-only collection script for improving detector generalization.  The
# recorder always writes the original camera JPEG packet, never this overlay.
GENERALIZATION_GUIDES = {
    "generalization": [
        {
            "name": "STILL / CENTER",
            "seconds": 8.0,
            "instruction": "Hold target still at center; then make small natural movements.",
            "overlay": "center",
        },
        {
            "name": "SLOW / ALL AREAS",
            "seconds": 20.0,
            "instruction": "Move slowly through the 3x3 areas; exact alignment is not needed.",
            "overlay": "grid",
        },
        {
            "name": "DISTANCE / FAR-MID-NEAR",
            "seconds": 18.0,
            "instruction": "Move far -> middle -> near; pause briefly at each distance.",
            "overlay": "center",
        },
        {
            "name": "EDGE ENTRY / EXIT",
            "seconds": 18.0,
            "instruction": "Enter and leave from left, right, top and bottom edges.",
            "overlay": "edges",
        },
        {
            "name": "FAST CROSSING",
            "seconds": 18.0,
            "instruction": "Cross left-right quickly 5 times at realistic arm speed.",
            "overlay": "horizontal",
        },
        {
            "name": "FAST APPROACH / STOP",
            "seconds": 18.0,
            "instruction": "Approach quickly, stop, hold 1 second, retreat; repeat 5 times.",
            "overlay": "center",
        },
        {
            "name": "SMALL TILT",
            "seconds": 15.0,
            "instruction": "Use front view plus small tilts only; do not make a full rotation.",
            "overlay": "full",
        },
        {
            "name": "PARTIAL OCCLUSION",
            "seconds": 15.0,
            "instruction": "Cover 10-30 percent with hand/gripper, then reveal the target.",
            "overlay": "full",
        },
        {
            "name": "NO TARGET / HARD NEGATIVE",
            "seconds": 25.0,
            "instruction": "Remove foam. Move hand, gripper and similar objects in view.",
            "overlay": "full",
        },
    ],
    "hard-negative": [
        {
            "name": "EMPTY BACKGROUND",
            "seconds": 12.0,
            "instruction": "No foam target. Keep an empty scene and vary room lighting.",
            "overlay": "full",
        },
        {
            "name": "HAND / CLOTHING",
            "seconds": 18.0,
            "instruction": "No foam target. Move hands, sleeves and clothing through view.",
            "overlay": "full",
        },
        {
            "name": "GRIPPER / ARM",
            "seconds": 18.0,
            "instruction": "No foam target. Move the white gripper and robotic arm.",
            "overlay": "full",
        },
        {
            "name": "SIMILAR OBJECTS",
            "seconds": 20.0,
            "instruction": "No foam target. Show paper, tissue, blocks and similar objects.",
            "overlay": "full",
        },
    ],
    # Held-out difficult-scene test. Keep the foam target fixed on the
    # gripper for tasks 1-6; only task 7 removes it for a false-positive check.
    "gripper-hard": [
        {
            "name": "DIM / THREE POSITIONS",
            "seconds": 12.0,
            "instruction": "Target on gripper. Hold at lower, middle and upper points on the center axis.",
            "overlay": "axis",
        },
        {
            "name": "SLOW / CENTER AXIS",
            "seconds": 12.0,
            "instruction": "Move slowly up and down along the camera center axis; no left-right motion.",
            "overlay": "axis",
        },
        {
            "name": "TOP-BOTTOM ENTRY / EXIT",
            "seconds": 12.0,
            "instruction": "Enter/leave only from reachable top or bottom; stay on the center axis.",
            "overlay": "axis",
        },
        {
            "name": "FAST / CENTER AXIS",
            "seconds": 12.0,
            "instruction": "Move quickly along the center axis at realistic arm speed; never move sideways.",
            "overlay": "axis",
        },
        {
            "name": "FAST MOVE / STOP",
            "seconds": 12.0,
            "instruction": "Move quickly on-axis, stop and hold 1 second, then return; repeat safely.",
            "overlay": "axis",
        },
        {
            "name": "PARTIAL OCCLUSION",
            "seconds": 12.0,
            "instruction": "Stay on-axis. Let the gripper hide 10-30 percent; do not fully cover the target.",
            "overlay": "axis",
        },
        {
            "name": "NO TARGET / HARD NEGATIVE",
            "seconds": 20.0,
            "instruction": "Remove foam. Move the empty gripper only along the same center axis.",
            "overlay": "axis",
        },
    ],
}


def safe_name(value):
    cleaned = "".join(char if char.isalnum() or char in "-_" else "_" for char in value.strip())
    return cleaned.strip("_") or "capture"


def resolve_output_dir(value, object_name):
    if value:
        path = Path(value)
        return path if path.is_absolute() else PROJECT_DIR / path
    return PROJECT_DIR / "data" / "raw" / object_name


def parse_angle_sequence(value):
    if not value.strip():
        return []
    angles = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        angles.append(float(item))
    if not angles:
        raise ValueError("--angles must contain at least one numeric angle")
    return angles


def parse_point_sequence(value):
    """Parse ``x:y,x:y`` preview-guide points in source-image pixels."""
    if not value.strip():
        return []
    points = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        components = item.split(":")
        if len(components) != 2:
            raise ValueError(
                "--guide-points must use x:y pairs separated by commas"
            )
        points.append((int(components[0]), int(components[1])))
    if not points:
        raise ValueError("--guide-points must contain at least one x:y point")
    return points


class JpegPipeReader:
    def __init__(self, stream):
        self.stream = stream
        self.buffer = bytearray()

    def read(self):
        while True:
            start = self.buffer.find(b"\xff\xd8")
            if start >= 0:
                end = self.buffer.find(b"\xff\xd9", start + 2)
                if end >= 0:
                    packet = bytes(self.buffer[start : end + 2])
                    del self.buffer[: end + 2]
                    return packet
                if start > 0:
                    del self.buffer[:start]
            elif len(self.buffer) > 2:
                del self.buffer[:-2]

            chunk = self.stream.read(65536)
            if not chunk:
                return None
            self.buffer.extend(chunk)


class JpegFramePump:
    """Read the FFmpeg pipe off the UI thread so the preview stays responsive."""

    def __init__(self, reader, max_queue_size=250):
        self.reader = reader
        self.frames = queue.Queue(maxsize=max_queue_size)
        self.error = None
        self.finished = False
        self.thread = threading.Thread(target=self._run, name="jpeg-frame-pump", daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        try:
            while True:
                packet = self.reader.read()
                if packet is None:
                    break
                self.frames.put(packet)
        except BaseException as exc:
            self.error = exc
        finally:
            self.finished = True


class RttTelemetryLogger:
    """Record servo feedback from the OpenOCD RTT TCP bridge with host timestamps."""

    def __init__(self, host, port, output_path, started_monotonic):
        self.host = host
        self.port = port
        self.output_path = output_path
        self.started_monotonic = started_monotonic
        self.socket = None
        self.output_file = None
        self.stop_event = threading.Event()
        self.thread = None
        self.error = None

    def start(self):
        self.socket = socket.create_connection((self.host, self.port), timeout=2.0)
        self.socket.settimeout(0.5)
        self.output_file = self.output_path.open("w", encoding="utf-8", buffering=1)
        self.thread = threading.Thread(target=self._run, name="rtt-telemetry", daemon=True)
        self.thread.start()

    @staticmethod
    def _parse_value(value):
        return float(value) if "." in value else int(value)

    def _write_line(self, raw_line):
        line = raw_line.decode("utf-8", errors="replace").replace("\x00", "").strip()
        if not line:
            return
        now = time.monotonic()
        record = {
            "captured_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "monotonic_seconds": now,
            "elapsed_seconds": now - self.started_monotonic,
            "raw": line,
        }
        for key, value in RTT_VALUE_PATTERN.findall(line):
            record[key] = self._parse_value(value)
        self.output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _run(self):
        pending = bytearray()
        try:
            while not self.stop_event.is_set():
                try:
                    chunk = self.socket.recv(4096)
                except socket.timeout:
                    continue
                if not chunk:
                    break
                pending.extend(chunk)
                # SEGGER RTT commonly pads each terminal message with NUL
                # bytes. Treat both NUL and LF as record boundaries so a
                # capture does not wait for the next console command before
                # writing accumulated servo snapshots.
                while True:
                    boundaries = [
                        index
                        for index in (pending.find(b"\x00"), pending.find(b"\n"))
                        if index >= 0
                    ]
                    if not boundaries:
                        break
                    boundary = min(boundaries)
                    raw_line = bytes(pending[:boundary]).rstrip(b"\r")
                    del pending[: boundary + 1]
                    self._write_line(raw_line)
            if pending:
                self._write_line(bytes(pending))
        except OSError as exc:
            if not self.stop_event.is_set():
                self.error = exc

    def stop(self):
        self.stop_event.set()
        if self.socket is not None:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self.output_file is not None:
            self.output_file.close()
            self.output_file = None

    def send_command(self, command):
        if self.socket is None:
            raise RuntimeError("RTT telemetry is not connected")
        payload = command.rstrip("\r\n") + "\n"
        self.socket.sendall(payload.encode("ascii"))


class ManualRecorder:
    def __init__(
        self,
        ffmpeg,
        output_dir,
        prefix,
        fps,
        events_suffix="angles",
        rtt_host="",
        rtt_port=9090,
    ):
        self.ffmpeg = ffmpeg
        self.output_dir = output_dir
        self.prefix = prefix
        self.fps = fps
        self.events_suffix = events_suffix
        self.raw_file = None
        self.raw_path = None
        self.video_path = None
        self.timestamps_path = None
        self.events_path = None
        self.timestamps_file = None
        self.events_file = None
        self.log_file = None
        self.frame_count = 0
        self.started_monotonic = None
        self.rtt_host = rtt_host
        self.rtt_port = rtt_port
        self.rtt_path = None
        self.rtt_logger = None

    @property
    def active(self):
        return self.raw_file is not None

    def start(self):
        if self.active:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        stem = f"{self.prefix}_{stamp}"
        self.video_path = self.output_dir / f"{stem}.mkv"
        self.raw_path = self.output_dir / f"{stem}.mjpeg.partial"
        self.timestamps_path = self.output_dir / f"{stem}.timestamps.jsonl"
        self.events_path = self.output_dir / f"{stem}.{self.events_suffix}.jsonl"
        self.rtt_path = self.output_dir / f"{stem}.rtt.jsonl"
        log_path = self.output_dir / f"{stem}.ffmpeg.log"
        self.frame_count = 0
        self.started_monotonic = time.monotonic()
        if self.rtt_host:
            self.rtt_logger = RttTelemetryLogger(
                self.rtt_host,
                self.rtt_port,
                self.rtt_path,
                self.started_monotonic,
            )
            try:
                self.rtt_logger.start()
            except OSError as exc:
                self.rtt_logger = None
                raise RuntimeError(
                    f"RTT telemetry connection failed: {self.rtt_host}:{self.rtt_port}: {exc}"
                ) from exc
        self.timestamps_file = self.timestamps_path.open("w", encoding="utf-8", buffering=1)
        self.events_file = self.events_path.open("w", encoding="utf-8", buffering=1)
        self.log_file = log_path.open("w", encoding="utf-8")
        # Write validated camera JPEG packets directly while recording. Feeding a
        # second FFmpeg process synchronously can block the UI while that process
        # probes a damaged first analog-link frame. The raw stream is remuxed on S.
        self.raw_file = self.raw_path.open("wb", buffering=1024 * 1024)

    def write(self, jpeg_packet):
        if not self.active:
            return
        self.raw_file.write(jpeg_packet)
        now = time.monotonic()
        record = {
            "frame_index": self.frame_count,
            "captured_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "monotonic_seconds": now,
            "elapsed_seconds": now - self.started_monotonic,
        }
        self.timestamps_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.frame_count += 1

    def record_event(self, event, **values):
        if not self.active or self.events_file is None:
            return
        now = time.monotonic()
        record = {
            "event": event,
            "frame_index": self.frame_count,
            "captured_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "elapsed_seconds": now - self.started_monotonic,
            **values,
        }
        self.events_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def send_rtt_command(self, command):
        if self.rtt_logger is None:
            raise RuntimeError("RTT angle control requires --rtt-host")
        self.rtt_logger.send_command(command)

    def stop(self):
        if not self.active:
            return None
        raw_file = self.raw_file
        self.raw_file = None
        try:
            if self.rtt_logger is not None:
                self.rtt_logger.stop()
                self.rtt_logger = None
            raw_file.close()
            self.timestamps_file.close()
            self.timestamps_file = None
            self.events_file.close()
            self.events_file = None

            if self.frame_count <= 0:
                raise RuntimeError("recording stopped without receiving a valid frame")

            command = [
                self.ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-probesize",
                "10M",
                "-analyzeduration",
                "10M",
                "-f",
                "mjpeg",
                "-framerate",
                f"{self.fps:g}",
                "-i",
                str(self.raw_path),
                "-an",
                "-c:v",
                "copy",
                "-f",
                "matroska",
                str(self.video_path),
            ]
            creationflags = (
                subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            )
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=self.log_file,
                creationflags=creationflags,
                check=False,
            )
            if result.returncode:
                raise RuntimeError(f"recording FFmpeg exited with code {result.returncode}")
            self.raw_path.unlink()
        finally:
            if not raw_file.closed:
                raw_file.close()
            if self.timestamps_file is not None:
                self.timestamps_file.close()
                self.timestamps_file = None
            if self.events_file is not None:
                self.events_file.close()
                self.events_file = None
            self.log_file.close()
            self.log_file = None
        return self.video_path


def measure_center_axis_light(frame):
    """Measure the working corridor instead of bright clothing at the edges."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    x1, x2 = round(width * 0.30), round(width * 0.70)
    y1, y2 = round(height * 0.10), round(height * 0.85)
    brightness = float(gray[y1:y2, x1:x2].mean())
    if brightness < 35.0:
        status = "TOO DARK - ADD LIGHT"
        color = (0, 0, 255)
    elif brightness < 55.0:
        status = "DARK - ADD A LITTLE LIGHT"
        color = (0, 165, 255)
    elif brightness <= 85.0:
        status = "DIM OK"
        color = (0, 255, 0)
    else:
        status = "NORMAL / BRIGHT"
        color = (0, 255, 255)
    return {
        "brightness": brightness,
        "status": status,
        "color": color,
        "dim_ok": 55.0 <= brightness <= 85.0,
        "roi": (x1, y1, x2, y2),
    }


def draw_status(
    frame,
    recording,
    fps,
    recorded_frames,
    elapsed,
    guide=None,
    light_info=None,
    start_warning="",
    require_dim_light=False,
):
    display = frame.copy()
    status = "REC" if recording else "PREVIEW - NOT RECORDING"
    color = (0, 0, 255) if recording else (0, 255, 255)
    light_info = light_info or measure_center_axis_light(frame)
    brightness = light_info["brightness"]
    light_status = light_info["status"]
    light_color = light_info["color"]
    x1, y1, x2, y2 = light_info["roi"]
    if require_dim_light:
        cv2.rectangle(display, (x1, y1), (x2, y2), light_color, 1, cv2.LINE_AA)
    lines = [
        status,
        f"USB Video | 640x480 MJPEG | requested {fps:g} FPS",
        f"R: start   S: stop/save   Q/Esc: exit",
        (
            f"AXIS LIGHT: {brightness:.1f} | {light_status}"
            if require_dim_light
            else f"AXIS LIGHT: {brightness:.1f} | REFERENCE ONLY"
        ),
    ]
    if recording:
        lines.append(f"recorded {recorded_frames} frames | {elapsed:.1f} s")
    for index, line in enumerate(lines):
        y = 28 + index * 27
        line_color = light_color if line.startswith("AXIS LIGHT:") else color
        cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4)
        cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, line_color, 1)
    if recording:
        cv2.circle(display, (display.shape[1] - 24, 24), 9, color, -1)
    if start_warning:
        cv2.putText(
            display,
            start_warning,
            (10, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            start_warning,
            (10, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    if guide:
        if guide.get("mode") == "task":
            state_color = (0, 255, 0) if guide["state"] == "finished" else (0, 255, 255)
            height, width = display.shape[:2]
            overlay_kind = guide["task"]["overlay"]

            # Spatial hints are deliberately approximate: training benefits
            # from natural variation, not from repeatedly hitting one pixel.
            if overlay_kind == "grid":
                for x in (width // 3, 2 * width // 3):
                    cv2.line(display, (x, 0), (x, height - 1), (255, 220, 0), 1, cv2.LINE_AA)
                for y in (height // 3, 2 * height // 3):
                    cv2.line(display, (0, y), (width - 1, y), (255, 220, 0), 1, cv2.LINE_AA)
                for row in range(3):
                    for column in range(3):
                        cv2.circle(
                            display,
                            ((2 * column + 1) * width // 6, (2 * row + 1) * height // 6),
                            12,
                            (255, 220, 0),
                            1,
                            cv2.LINE_AA,
                        )
            elif overlay_kind == "center":
                cv2.circle(display, (width // 2, height // 2), 48, (255, 220, 0), 2, cv2.LINE_AA)
                cv2.line(display, (width // 2 - 18, height // 2), (width // 2 + 18, height // 2), (255, 220, 0), 2, cv2.LINE_AA)
                cv2.line(display, (width // 2, height // 2 - 18), (width // 2, height // 2 + 18), (255, 220, 0), 2, cv2.LINE_AA)
            elif overlay_kind == "edges":
                margin = 42
                cv2.rectangle(display, (margin, margin), (width - margin, height - margin), (255, 220, 0), 2, cv2.LINE_AA)
                for start, end in (
                    ((0, height // 2), (margin + 25, height // 2)),
                    ((width - 1, height // 2), (width - margin - 25, height // 2)),
                    ((width // 2, 0), (width // 2, margin + 25)),
                    ((width // 2, height - 1), (width // 2, height - margin - 25)),
                ):
                    cv2.arrowedLine(display, start, end, (255, 220, 0), 2, cv2.LINE_AA, tipLength=0.25)
            elif overlay_kind == "horizontal":
                cv2.arrowedLine(display, (35, height // 2), (width - 35, height // 2), (255, 220, 0), 2, cv2.LINE_AA, tipLength=0.06)
                cv2.arrowedLine(display, (width - 35, height // 2 + 35), (35, height // 2 + 35), (255, 220, 0), 2, cv2.LINE_AA, tipLength=0.06)
            elif overlay_kind == "axis":
                axis_x = width // 2
                cv2.line(
                    display,
                    (axis_x, 0),
                    (axis_x, height - 1),
                    (255, 220, 0),
                    2,
                    cv2.LINE_AA,
                )
                for y in (height // 4, height // 2, 3 * height // 4):
                    cv2.circle(display, (axis_x, y), 22, (255, 220, 0), 2, cv2.LINE_AA)
                cv2.arrowedLine(
                    display,
                    (axis_x - 35, 3 * height // 4),
                    (axis_x - 35, height // 4),
                    (255, 220, 0),
                    2,
                    cv2.LINE_AA,
                    tipLength=0.08,
                )
                cv2.arrowedLine(
                    display,
                    (axis_x + 35, height // 4),
                    (axis_x + 35, 3 * height // 4),
                    (255, 220, 0),
                    2,
                    cv2.LINE_AA,
                    tipLength=0.08,
                )

            top_overlay = display.copy()
            cv2.rectangle(top_overlay, (0, 132), (width, 241), (0, 0, 0), -1)
            cv2.addWeighted(top_overlay, 0.78, display, 0.22, 0, display)
            header = (
                f"TASK {guide['step']:02d}/{guide['count']:02d} | "
                f"{guide['task']['name']}"
            )
            if guide["state"] == "finished":
                timer = "ALL TASKS COMPLETE | press S to stop/save"
            else:
                timer = (
                    f"THIS TASK {guide['task_elapsed']:.1f}s / "
                    f"recommended {guide['task']['seconds']:.0f}s"
                )
            controls = "SPACE: next task    B: previous task"
            for line, y, scale, color in (
                (header, 164, 0.62, state_color),
                (guide["task"]["instruction"], 195, 0.43, state_color),
                (timer + " | " + controls, 226, 0.42, state_color),
            ):
                cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
            return display

        if guide.get("mode") == "position":
            state = guide["state"]
            state_color = {
                "idle": (0, 255, 255),
                "move": (0, 255, 255),
                "hold": (0, 165, 255),
                "done": (0, 255, 0),
                "finished": (0, 255, 0),
            }[state]
            target_x, target_y = guide["point"]
            # Position guides belong only to the preview. The recorder writes
            # the original JPEG packet, so these lines never enter the video.
            cv2.line(
                display,
                (target_x, 0),
                (target_x, display.shape[0] - 1),
                (255, 220, 0),
                1,
                cv2.LINE_AA,
            )
            unique_points = []
            for point in guide["all_points"]:
                if point not in unique_points:
                    unique_points.append(point)
            for point_index, (point_x, point_y) in enumerate(unique_points, start=1):
                if (point_x, point_y) == (target_x, target_y):
                    continue
                cv2.circle(
                    display,
                    (point_x, point_y),
                    12,
                    (170, 170, 170),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display,
                    f"P{point_index}",
                    (point_x + 16, point_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (210, 210, 210),
                    1,
                    cv2.LINE_AA,
                )
            cv2.circle(display, (target_x, target_y), 30, state_color, 2, cv2.LINE_AA)
            cv2.line(
                display,
                (target_x - 13, target_y),
                (target_x + 13, target_y),
                state_color,
                2,
                cv2.LINE_AA,
            )
            cv2.line(
                display,
                (target_x, target_y - 13),
                (target_x, target_y + 13),
                state_color,
                2,
                cv2.LINE_AA,
            )

            if state == "idle":
                message = "PRESS R | keep broad face toward camera | do not rotate"
            elif state == "move":
                message = "ALIGN CENTER | FRONT FACE | NO ROTATION | press SPACE"
            elif state == "hold":
                message = (
                    f"HOLD STILL {guide['hold_elapsed']:.1f} / "
                    f"{guide['hold_seconds']:.1f} s | DO NOT ROTATE"
                )
            elif state == "done" and guide["next_point"] is not None:
                next_x, next_y = guide["next_point"]
                message = f"CAPTURED | SPACE for next target ({next_x}, {next_y})"
            elif state == "done":
                message = "LAST POSITION CAPTURED | press SPACE to finish"
            else:
                message = "SEQUENCE COMPLETE | press S to save"

            overlay = display.copy()
            panel_top = display.shape[0] - 82
            cv2.rectangle(
                overlay,
                (0, panel_top),
                (display.shape[1], display.shape[0]),
                (0, 0, 0),
                -1,
            )
            cv2.addWeighted(overlay, 0.80, display, 0.20, 0, display)
            header = (
                f"STEP {guide['step']:02d}/{guide['count']:02d} | "
                f"TARGET CENTER x={target_x} y={target_y}"
            )
            for line, y, scale in (
                (header, panel_top + 28, 0.60),
                (message, panel_top + 61, 0.48),
            ):
                cv2.putText(
                    display,
                    line,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    scale,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display,
                    line,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    scale,
                    state_color,
                    1,
                    cv2.LINE_AA,
                )
            return display

        overlay = display.copy()
        cv2.rectangle(overlay, (0, 128), (display.shape[1], 266), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.78, display, 0.22, 0, display)
        state = guide["state"]
        state_color = {
            "idle": (0, 255, 255),
            "move": (0, 255, 255),
            "hold": (0, 165, 255),
            "done": (0, 255, 0),
            "finished": (0, 255, 0),
        }[state]
        angle_text = f"g {guide['angle']:g} deg"
        header = f"STEP {guide['step']:02d}/{guide['count']:02d}    COMMAND: {angle_text}"
        if state == "idle":
            message = "Press R to record, then move the servo to this angle"
        elif state == "move":
            message = f"MOVE TO {angle_text} | when stable, press SPACE"
        elif state == "hold":
            message = f"HOLD STILL  {guide['hold_elapsed']:.1f} / {guide['hold_seconds']:.1f} s"
        elif state == "done" and guide["next_angle"] is not None:
            message = f"CAPTURED | press SPACE for next: g {guide['next_angle']:g} deg"
        elif state == "done":
            message = "LAST ANGLE CAPTURED | press SPACE to finish sequence"
        else:
            message = "SEQUENCE COMPLETE | press S to save"
        controls = "SPACE: confirm/next    B: previous angle"
        if guide.get("rtt_angle_control"):
            controls = "G: send shown angle    SPACE: confirm/next    B: previous"
        guide_lines = [header, message, controls]
        for index, line in enumerate(guide_lines):
            y = 160 + index * 43
            scale = 0.82 if index == 0 else 0.66
            cv2.putText(display, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
            cv2.putText(display, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, state_color, 2)
    return display


def main():
    parser = argparse.ArgumentParser(
        description="Manual 50 FPS DirectShow preview and lossless MJPEG recording."
    )
    parser.add_argument("--device", default="USB Video", help="DirectShow video device name.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--object", default="arm_motion_slow_50fps")
    parser.add_argument("--out", default="data/raw/new_angle_recordings")
    parser.add_argument(
        "--angles",
        default="",
        help="Optional comma-separated guided servo angles, for example -30,-35,-40.",
    )
    parser.add_argument(
        "--guide-points",
        default="",
        help="Optional preview-only target-centre points as x:y pairs, for "
        "example 320:350,320:295,320:240.",
    )
    parser.add_argument(
        "--guide-preset",
        choices=sorted(GENERALIZATION_GUIDES),
        default="",
        help="Optional preview-only task guide for generalization or hard-negative recording.",
    )
    parser.add_argument(
        "--require-dim-light",
        action="store_true",
        help="Block R until the center-axis brightness is in the DIM OK range.",
    )
    parser.add_argument("--hold-seconds", type=float, default=3.0)
    parser.add_argument(
        "--rtt-host",
        default="",
        help="Optional OpenOCD RTT TCP host; records synchronized servo telemetry.",
    )
    parser.add_argument("--rtt-port", type=int, default=9090)
    parser.add_argument(
        "--rtt-angle-control",
        action="store_true",
        help="Allow G in an angle guide to send the displayed g command over RTT.",
    )
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0 or args.fps <= 0 or args.hold_seconds <= 0:
        raise ValueError("width, height, fps, and hold-seconds must be greater than zero")
    guide_angles = parse_angle_sequence(args.angles)
    guide_points = parse_point_sequence(args.guide_points)
    enabled_guides = sum(bool(value) for value in (guide_angles, guide_points, args.guide_preset))
    if enabled_guides > 1:
        raise ValueError("--angles, --guide-points, and --guide-preset are mutually exclusive")
    if args.rtt_angle_control and (not guide_angles or not args.rtt_host):
        raise ValueError("--rtt-angle-control requires --angles and --rtt-host")
    for point_x, point_y in guide_points:
        if not (0 <= point_x < args.width and 0 <= point_y < args.height):
            raise ValueError(
                f"guide point ({point_x}, {point_y}) is outside "
                f"{args.width}x{args.height}"
            )
    guide_tasks = GENERALIZATION_GUIDES.get(args.guide_preset, [])
    guide_mode = (
        "angle" if guide_angles else "position" if guide_points else "task" if guide_tasks else ""
    )
    guide_values = guide_angles or guide_points or guide_tasks

    def guide_event_fields(index):
        if guide_mode == "angle":
            return {
                "guide_mode": "angle",
                "angle_deg": guide_angles[index],
            }
        if guide_mode == "position":
            point_x, point_y = guide_points[index]
            return {
                "guide_mode": "position",
                "target_pixel": [point_x, point_y],
                "target_x_px": point_x,
                "target_y_px": point_y,
            }
        task = guide_tasks[index]
        return {
            "guide_mode": "task",
            "guide_preset": args.guide_preset,
            "task_name": task["name"],
            "recommended_seconds": task["seconds"],
        }

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found on PATH")

    object_name = safe_name(args.object)
    output_dir = resolve_output_dir(args.out, object_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    session_path = output_dir / "manual_capture_session.json"
    session_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "device": args.device,
                "width": args.width,
                "height": args.height,
                "fps": args.fps,
                "format": "MJPEG",
                "control": {
                    "start": "R",
                    "confirm_or_next_step": "Space",
                    "previous_step": "B",
                    "stop_save": "S",
                    "exit": ["Q", "Esc"],
                },
                "angle_guide": {
                    "angles_deg": guide_angles,
                    "hold_seconds": args.hold_seconds,
                },
                "position_guide": {
                    "points_px": [list(point) for point in guide_points],
                    "hold_seconds": args.hold_seconds,
                    "preview_only": True,
                    "instruction": "Keep the broad foam face toward the camera; do not rotate.",
                },
                "task_guide": {
                    "preset": args.guide_preset,
                    "tasks": guide_tasks,
                    "preview_only": True,
                    "require_dim_light": args.require_dim_light,
                    "dim_axis_brightness_range": [55.0, 85.0],
                    "instruction": "Press Space manually after completing each task.",
                },
                "rtt_telemetry": {
                    "enabled": bool(args.rtt_host),
                    "host": args.rtt_host,
                    "port": args.rtt_port,
                    "angle_control_enabled": args.rtt_angle_control,
                    "fields_expected": ["pos", "tgt", "load", "wr", "rd"],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    producer_command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "dshow",
        "-video_size",
        f"{args.width}x{args.height}",
        "-framerate",
        f"{args.fps:g}",
        "-vcodec",
        "mjpeg",
        "-i",
        f"video={args.device}",
        "-an",
        "-c:v",
        "copy",
        "-f",
        "image2pipe",
        "pipe:1",
    ]
    creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
    producer = subprocess.Popen(
        producer_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        creationflags=creationflags,
    )
    reader = JpegPipeReader(producer.stdout)
    frame_pump = JpegFramePump(reader)
    frame_pump.start()
    recorder = ManualRecorder(
        ffmpeg,
        output_dir,
        object_name,
        args.fps,
        events_suffix=(
            "positions" if guide_points else "tasks" if guide_tasks else "angles"
        ),
        rtt_host=args.rtt_host,
        rtt_port=args.rtt_port,
    )
    window_name = "Manual 50 FPS Capture"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    last_frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    guide_index = 0
    guide_state = "idle"
    guide_hold_started = None
    guide_task_started = None
    start_warning = ""
    start_warning_until = 0.0

    try:
        while True:
            try:
                packet = frame_pump.frames.get(timeout=0.02)
            except queue.Empty:
                packet = None

            if packet is None and frame_pump.finished:
                error = producer.stderr.read().decode("utf-8", errors="replace").strip()
                if frame_pump.error:
                    raise RuntimeError(str(frame_pump.error)) from frame_pump.error
                raise RuntimeError(error or "capture FFmpeg stopped producing frames")

            packet_is_valid = False
            if packet is not None:
                frame = cv2.imdecode(np.frombuffer(packet, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    last_frame = frame
                    packet_is_valid = True

            if recorder.active and packet_is_valid:
                recorder.write(packet)
            elapsed = (
                time.monotonic() - recorder.started_monotonic if recorder.active else 0.0
            )
            if guide_values and recorder.active and guide_state == "hold":
                hold_elapsed = time.monotonic() - guide_hold_started
                if hold_elapsed >= args.hold_seconds:
                    guide_state = "done"
                    recorder.record_event(
                        "hold_completed",
                        step=guide_index + 1,
                        **guide_event_fields(guide_index),
                    )
            else:
                hold_elapsed = 0.0

            guide = None
            if guide_values:
                guide = {
                    "mode": guide_mode,
                    "state": guide_state,
                    "step": guide_index + 1,
                    "count": len(guide_values),
                    "hold_elapsed": min(hold_elapsed, args.hold_seconds),
                    "hold_seconds": args.hold_seconds,
                    "rtt_angle_control": args.rtt_angle_control,
                }
                if guide_mode == "angle":
                    guide.update(
                        {
                            "angle": guide_angles[guide_index],
                            "next_angle": (
                                guide_angles[guide_index + 1]
                                if guide_index + 1 < len(guide_angles)
                                else None
                            ),
                        }
                    )
                elif guide_mode == "position":
                    guide.update(
                        {
                            "point": guide_points[guide_index],
                            "all_points": guide_points,
                            "next_point": (
                                guide_points[guide_index + 1]
                                if guide_index + 1 < len(guide_points)
                                else None
                            ),
                        }
                    )
                else:
                    guide.update(
                        {
                            "task": guide_tasks[guide_index],
                            "task_elapsed": (
                                time.monotonic() - guide_task_started
                                if recorder.active and guide_task_started is not None
                                else 0.0
                            ),
                        }
                    )
            light_info = measure_center_axis_light(last_frame)
            if start_warning and time.monotonic() >= start_warning_until:
                start_warning = ""
            display = draw_status(
                last_frame,
                recorder.active,
                args.fps,
                recorder.frame_count,
                elapsed,
                guide,
                light_info,
                start_warning,
                args.require_dim_light,
            )
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("r"), ord("R")) and not recorder.active:
                if args.require_dim_light and not light_info["dim_ok"]:
                    start_warning = "START BLOCKED: adjust AXIS LIGHT until DIM OK"
                    start_warning_until = time.monotonic() + 3.0
                else:
                    recorder.start()
                    start_warning = ""
                    if guide_values:
                        guide_index = 0
                        guide_state = "active" if guide_mode == "task" else "move"
                        guide_hold_started = None
                        guide_task_started = time.monotonic() if guide_mode == "task" else None
                        recorder.record_event(
                            "guide_started",
                            step=guide_index + 1,
                            **guide_event_fields(guide_index),
                        )
            elif key == ord(" ") and recorder.active and guide_values:
                if guide_mode == "task" and guide_state == "active":
                    recorder.record_event(
                        "task_completed",
                        step=guide_index + 1,
                        actual_seconds=time.monotonic() - guide_task_started,
                        **guide_event_fields(guide_index),
                    )
                    if guide_index + 1 < len(guide_values):
                        guide_index += 1
                        guide_task_started = time.monotonic()
                        recorder.record_event(
                            "task_started",
                            step=guide_index + 1,
                            **guide_event_fields(guide_index),
                        )
                    else:
                        guide_state = "finished"
                        recorder.record_event(
                            "sequence_completed",
                            steps=len(guide_values),
                            guide_mode=guide_mode,
                        )
                elif guide_state == "move":
                    guide_state = "hold"
                    guide_hold_started = time.monotonic()
                    recorder.record_event(
                        "hold_started",
                        step=guide_index + 1,
                        **guide_event_fields(guide_index),
                    )
                elif guide_state == "done":
                    if guide_index + 1 < len(guide_values):
                        guide_index += 1
                        guide_state = "move"
                        guide_hold_started = None
                        recorder.record_event(
                            "move_prompt",
                            step=guide_index + 1,
                            **guide_event_fields(guide_index),
                        )
                    else:
                        guide_state = "finished"
                        recorder.record_event(
                            "sequence_completed",
                            steps=len(guide_values),
                            guide_mode=guide_mode,
                        )
            elif key in (ord("b"), ord("B")) and recorder.active and guide_values:
                guide_index = max(0, guide_index - 1)
                guide_state = "active" if guide_mode == "task" else "move"
                guide_hold_started = None
                guide_task_started = time.monotonic() if guide_mode == "task" else None
                recorder.record_event(
                    "step_back",
                    step=guide_index + 1,
                    **guide_event_fields(guide_index),
                )
            elif (
                key in (ord("g"), ord("G"))
                and recorder.active
                and guide_mode == "angle"
                and args.rtt_angle_control
            ):
                command = f"g {guide_angles[guide_index]:g}"
                recorder.send_rtt_command(command)
                recorder.record_event(
                    "rtt_angle_command_sent",
                    step=guide_index + 1,
                    command=command,
                    **guide_event_fields(guide_index),
                )
            elif key in (ord("s"), ord("S")) and recorder.active:
                if guide_values:
                    recorder.record_event(
                        "recording_stopped",
                        guide_state=guide_state,
                        guide_mode=guide_mode,
                        completed_steps=(guide_index + 1 if guide_state in {"done", "finished"} else guide_index),
                    )
                saved_path = recorder.stop()
                print(f"Saved: {saved_path}", flush=True)
            elif key in (ord("q"), ord("Q"), 27):
                break
    finally:
        if recorder.active:
            saved_path = recorder.stop()
            print(f"Saved: {saved_path}", flush=True)
        producer.terminate()
        try:
            producer.wait(timeout=5)
        except subprocess.TimeoutExpired:
            producer.kill()
            producer.wait(timeout=5)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
