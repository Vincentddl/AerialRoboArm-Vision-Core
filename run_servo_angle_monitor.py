"""Infer the calibrated servo ``g`` angle from the live red marker image.

This program does not read the HX8 encoder. ``inferred_servo_angle_deg`` is
the vision-inferred g-equivalent angle obtained from the camera ray and the
2026-08-15 post-slip red-marker lookup table. Keep the marker, camera,
servo zero and mechanism in the same physical configuration used during calibration.
"""

from __future__ import annotations

import argparse
import json
import queue
import shutil
import subprocess
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from tools.capture_ffmpeg_manual import JpegFramePump, JpegPipeReader
from vision.red_marker import RedMarkerDetector
from vision.servo_angle_calibration import ServoOpticalAngleCalibration
from vision.tracker import PixelToWorldMapper


ROOT = Path(__file__).resolve().parent
CAMERA_CALIBRATION = ROOT / "configs" / "camera_2p1mm_640x480_fisheye.json"
SERVO_CALIBRATION = (
    ROOT / "configs" / "servo_to_optical_angle_red_marker_lut_20260815_v3.json"
)
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm", ".wmv"}


class VideoSource:
    def __init__(self, source: str) -> None:
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            raise RuntimeError(f"failed to open video source: {source}")

    def read(self):
        ok, frame = self.capture.read()
        return ok, time.perf_counter(), frame

    def release(self) -> None:
        self.capture.release()


class DirectShowFfmpegSource:
    """Read every live MJPEG frame through the proven 50 FPS FFmpeg path."""

    def __init__(self, device: str, width: int, height: int, fps: float) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg was not found on PATH")
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "dshow",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            f"{fps:g}",
            "-vcodec",
            "mjpeg",
            "-i",
            f"video={device}",
            "-an",
            "-c:v",
            "copy",
            "-f",
            "image2pipe",
            "pipe:1",
        ]
        creationflags = (
            subprocess.CREATE_NO_WINDOW
            if hasattr(subprocess, "CREATE_NO_WINDOW")
            else 0
        )
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            creationflags=creationflags,
        )
        assert self.process.stdout is not None
        self.pump = JpegFramePump(JpegPipeReader(self.process.stdout), max_queue_size=100)
        self.pump.start()

    def read(self):
        try:
            packet = self.pump.frames.get(timeout=0.05)
        except queue.Empty:
            if self.pump.finished:
                assert self.process.stderr is not None
                error = self.process.stderr.read().decode("utf-8", errors="replace").strip()
                raise RuntimeError(error or "FFmpeg camera stream ended")
            return False, time.perf_counter(), None
        # Drop buffered frames so the displayed angle follows the current arm.
        while True:
            try:
                packet = self.pump.frames.get_nowait()
            except queue.Empty:
                break
        array = np.frombuffer(packet, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
        return frame is not None, time.perf_counter(), frame

    def release(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2.0)


def draw_angle_scale(
    frame: np.ndarray,
    mapper: PixelToWorldMapper,
    calibration: ServoOpticalAngleCalibration,
) -> None:
    principal = mapper.principal_point
    if principal is None:
        return
    center_x, center_y = (int(round(value)) for value in principal)
    cv2.line(frame, (center_x, 0), (center_x, frame.shape[0] - 1), (255, 220, 0), 1)
    cv2.line(frame, (0, center_y), (frame.shape[1] - 1, center_y), (180, 180, 0), 1)
    if calibration.model_kind != "piecewise_linear_lut":
        return
    for servo_angle, optical_angle in zip(
        calibration.servo_nodes_deg,
        calibration.optical_nodes_deg,
    ):
        pixel_x, pixel_y = mapper.to_pixel((0.0, optical_angle))
        y = int(round(pixel_y))
        if not 0 <= y < frame.shape[0]:
            continue
        cv2.line(frame, (center_x - 9, y), (center_x + 9, y), (0, 230, 230), 1)
        cv2.putText(
            frame,
            f"{servo_angle:.0f}",
            (center_x + 13, max(12, y + 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 230, 230),
            1,
            cv2.LINE_AA,
        )


def draw_state(frame: np.ndarray, state: dict, fps: float) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 104), (0, 0, 0), -1)
    if state.get("detected"):
        valid = state["valid"]
        color = (40, 230, 40) if valid else (0, 180, 255)
        title = f"VISION SERVO g: {state['smoothed_servo_angle_deg']:+7.2f} deg"
        details = (
            f"raw {state['inferred_servo_angle_deg']:+.2f} | "
            f"optical beta {state['optical_offset_deg']:+.2f} | "
            f"lateral {state['lateral_angle_deg']:+.2f}"
        )
        status = "VALID - red-marker LUT" if valid else "OUT OF CALIBRATED RANGE / AXIS"
    else:
        color = (0, 0, 255)
        title = "VISION SERVO g: ---"
        details = "Red marker not detected"
        status = "Keep the red circle visible on the camera centre plane"
    cv2.putText(frame, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2, cv2.LINE_AA)
    cv2.putText(frame, details, (12, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(frame, status, (12, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"display {fps:.1f} FPS | Q/Esc exit", (410, 101), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer servo g angle from the calibrated red marker."
    )
    parser.add_argument("--source", default="", help="Video path; empty uses the live DirectShow camera.")
    parser.add_argument("--device", default="USB Video", help="DirectShow device name for live mode.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--camera-calibration", default=str(CAMERA_CALIBRATION))
    parser.add_argument("--servo-calibration", default=str(SERVO_CALIBRATION))
    parser.add_argument("--median-window", type=int, default=5)
    parser.add_argument("--lateral-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--save-jsonl", default="")
    parser.add_argument("--print-values", action="store_true")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    if args.median_window < 1:
        raise ValueError("--median-window must be at least 1")

    mapper = PixelToWorldMapper(args.camera_calibration)
    calibration = ServoOpticalAngleCalibration.from_json(args.servo_calibration)
    detector = RedMarkerDetector()
    print(calibration.summary)
    print("Output is a vision-inferred g-equivalent angle, not HX8 encoder telemetry.")

    source_path = Path(args.source) if args.source else None
    if source_path and source_path.suffix.lower() in VIDEO_SUFFIXES:
        source = VideoSource(str(source_path))
        live = False
    elif args.source:
        source = DirectShowFfmpegSource(args.source, args.width, args.height, args.fps)
        live = True
    else:
        source = DirectShowFfmpegSource(args.device, args.width, args.height, args.fps)
        live = True

    history: deque[float] = deque(maxlen=args.median_window)
    output_file = (
        open(args.save_jsonl, "w", encoding="utf-8")
        if args.save_jsonl
        else None
    )
    frame_count = 0
    detected_count = 0
    valid_count = 0
    last_print = 0.0
    fps_value = 0.0
    fps_frames = 0
    fps_start = time.perf_counter()
    window_name = "Red Marker - Vision Inferred Servo Angle"
    if not args.no_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, timestamp, frame = source.read()
            if not ok:
                if not live:
                    break
                continue
            frame_count += 1
            detection = detector.detect(frame)
            state = {
                "frame_index": frame_count - 1,
                "time": timestamp,
                "detected": detection is not None,
            }
            if detection is not None:
                detected_count += 1
                camera_angles = mapper.to_world(detection.center)
                optical_offset = mapper.camera_plane_angle_deg(camera_angles)
                lateral_angle = mapper.lateral_angle_deg(camera_angles)
                inferred_servo = calibration.servo_from_optical_offset(optical_offset)
                history.append(inferred_servo)
                smoothed_servo = float(np.median(np.asarray(history)))
                valid = (
                    calibration.is_optical_offset_in_range(optical_offset)
                    and calibration.is_servo_in_range(inferred_servo)
                    and abs(lateral_angle) <= args.lateral_tolerance_deg
                )
                valid_count += int(valid)
                state.update(
                    {
                        "valid": valid,
                        "pixel": [round(detection.center[0], 3), round(detection.center[1], 3)],
                        "marker_area_px2": round(detection.area_px2, 3),
                        "optical_offset_deg": round(optical_offset, 4),
                        "lateral_angle_deg": round(lateral_angle, 4),
                        "inferred_servo_angle_deg": round(inferred_servo, 4),
                        "smoothed_servo_angle_deg": round(smoothed_servo, 4),
                    }
                )
                x, y, width, height = detection.bbox
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.circle(
                    frame,
                    (int(round(detection.center[0])), int(round(detection.center[1]))),
                    4,
                    (255, 255, 255),
                    -1,
                )
            else:
                history.clear()

            if output_file is not None:
                output_file.write(json.dumps(state, ensure_ascii=False) + "\n")
            now = time.perf_counter()
            if args.print_values and now - last_print >= 0.1:
                print(json.dumps(state, ensure_ascii=False), flush=True)
                last_print = now

            fps_frames += 1
            elapsed = now - fps_start
            if elapsed >= 0.5:
                fps_value = fps_frames / elapsed
                fps_frames = 0
                fps_start = now
            if not args.no_window:
                draw_angle_scale(frame, mapper, calibration)
                draw_state(frame, state, fps_value)
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
    finally:
        source.release()
        if output_file is not None:
            output_file.close()
        if not args.no_window:
            cv2.destroyAllWindows()
    detection_rate = detected_count / frame_count if frame_count else 0.0
    valid_rate = valid_count / frame_count if frame_count else 0.0
    print(
        f"frames={frame_count}, marker_detected={detected_count} "
        f"({detection_rate:.1%}), valid={valid_count} ({valid_rate:.1%})"
    )


if __name__ == "__main__":
    main()
