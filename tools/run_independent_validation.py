"""Launch a separate midpoint capture, then audit and evaluate its saved data."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ANGLES = list(range(-22, -78, -5)) + list(range(-72, -21, 5))
DEFAULT_CANDIDATE = ROOT / "outputs/foam_center_servo_camera_lut_20261006_candidate.json"
MODEL = ROOT / "models/foam_center_v9_gripper_axis_normal_bg03_20260816_candidate.pt"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_devices() -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is missing from PATH.")
    devices = subprocess.run(
        [ffmpeg, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
        capture_output=True, timeout=15,
    )
    text = (devices.stdout + devices.stderr).decode("utf-8", errors="replace")
    if '"USB Video" (video)' not in text:
        raise RuntimeError("USB Video camera is not connected. Connect the calibrated fisheye camera.")
    with socket.create_connection(("127.0.0.1", 9090), timeout=2) as link:
        link.settimeout(0.5)
        received = bytearray()
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            try:
                chunk = link.recv(4096)
            except socket.timeout:
                continue
            if not chunk:
                break
            received.extend(chunk)
    lines = received.decode("utf-8", errors="replace").replace("\x00", "\n").splitlines()
    snapshots = {}
    for tag in ("[DBG]", "[SAFE]", "[PWR]"):
        matches = [line for line in lines if tag in line]
        if not matches:
            raise RuntimeError(f"No {tag} telemetry from RTT. Check the MCU and OpenOCD connection.")
        snapshots[tag] = matches[-1]
        print(matches[-1])
    voltage = re.search(r"voltage=(\d+)", snapshots["[PWR]"])
    age = re.search(r"fb_age=(\d+)", snapshots["[SAFE]"])
    if not voltage or not 9000 <= int(voltage[1]) <= 12600:
        raise RuntimeError("HX8 supply telemetry is invalid or outside 9.0..12.6 V.")
    if not age or int(age[1]) >= 140 or "stall=0" not in snapshots["[SAFE]"]:
        raise RuntimeError("HX8 feedback is stale or stall protection is active.")
    print("Ready. Preposition near -22 deg in small steps before pressing R.")


def audit(video: Path, candidate: dict) -> dict:
    import cv2

    events = read_jsonl(video.with_suffix(".angles.jsonl"))
    telemetry = read_jsonl(video.with_suffix(".rtt.jsonl"))
    timestamps = read_jsonl(video.with_suffix(".timestamps.jsonl"))
    started = [e for e in events if e.get("event") == "hold_started"]
    completed = [e for e in events if e.get("event") == "hold_completed"]
    if len(started) != 23 or len(completed) != 23:
        raise RuntimeError("Expected exactly 23 completed holds. Save this take and repeat the sequence.")
    if [e["angle_deg"] for e in started] != ANGLES:
        raise RuntimeError("Recorded guide angles differ from the independent validation sequence.")
    for index, (first, last) in enumerate(zip(started, completed), 1):
        if first["step"] != last["step"] or not 2.9 <= last["elapsed_seconds"] - first["elapsed_seconds"] <= 3.3:
            raise RuntimeError(f"Invalid hold timing at step {index}.")
        rows = [r for r in telemetry if first["elapsed_seconds"] <= r["elapsed_seconds"] <= last["elapsed_seconds"]]
        positions = [r["pos"] for r in rows if "pos" in r]
        voltages = [r["voltage"] for r in rows if "voltage" in r]
        targets = [r["tgt"] for r in rows if "tgt" in r]
        if len(positions) < 2 or max(positions) - min(positions) > 0.25:
            raise RuntimeError(f"Missing or unstable encoder position at step {index}.")
        if not voltages or min(voltages) < 9000 or max(voltages) > 12600:
            raise RuntimeError(f"Invalid supply voltage at step {index}.")
        if not targets or abs(statistics.median(targets) - first["angle_deg"]) > 0.1:
            raise RuntimeError(f"MCU target differs from the guide at step {index}.")
        bus_rows = [r for r in rows if "rxe" in r]
        bus_changed = any(
            bus_rows and bus_rows[-1].get(key, 0) != bus_rows[0].get(key, 0)
            for key in ("txe", "rxe", "rearm", "overflow")
        )
        if any(r.get("stall", 0) or r.get("rd", 0) for r in rows) or bus_changed:
            raise RuntimeError(f"Servo or bus error recorded at step {index}.")
    capture = cv2.VideoCapture(str(video))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = capture.get(cv2.CAP_PROP_FPS)
    decoded = []
    for frame in (0, frames // 2, frames - 1):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        decoded.append(capture.read()[0])
    capture.release()
    if (size != (640, 480) or abs(fps - 50) > 0.1 or frames != len(timestamps)
            or not all(decoded) or not all(t["frame_index"] == i for i, t in enumerate(timestamps))):
        raise RuntimeError("Video decode or frame/timestamp integrity check failed.")
    video_hash = sha256(video)
    training_video = Path(candidate.get("source", {}).get("video", ""))
    if training_video.is_file() and video_hash == sha256(training_video):
        raise RuntimeError("Validation video is identical to the training video.")
    return {"holds": 23, "frames": frames, "video_sha256": video_hash, "material_pass": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--video", type=Path, help="Audit/evaluate an already saved independent take")
    parser.add_argument("--check-only", action="store_true", help="Check files, camera and telemetry without capturing")
    args = parser.parse_args()
    candidate_path = args.candidate.resolve()
    for path in (candidate_path, MODEL):
        if not path.is_file():
            raise RuntimeError(f"Required local asset missing: {path}")
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if candidate.get("type") != "servo_optical_angle_lut":
        raise RuntimeError("Select the command-angle LUT, not an encoder-position LUT.")
    if not args.video:
        check_devices()
        if args.check_only:
            return 0
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = ROOT / "outputs/independent_validation" / stamp
    run_dir.mkdir(parents=True)
    frozen = run_dir / "candidate_frozen.json"
    shutil.copy2(candidate_path, frozen)
    if args.video:
        video = args.video.resolve()
    else:
        capture_dir = ROOT / "data/raw/independent_validation" / stamp
        print("R: record | G: send shown angle | Space: stable hold/next | S: save | Q: exit and evaluate", flush=True)
        subprocess.run([sys.executable, str(ROOT / "tools/capture_ffmpeg_manual.py"),
                        "--device", "USB Video", "--width", "640", "--height", "480", "--fps", "50",
                        "--object", "foam_angle_independent", "--out", str(capture_dir),
                        "--angles=" + ",".join(map(str, ANGLES)), "--hold-seconds", "3",
                        "--rtt-host", "127.0.0.1", "--rtt-port", "9090", "--rtt-angle-control"], check=True, cwd=ROOT)
        videos = list(capture_dir.glob("*.mkv"))
        if not videos:
            raise RuntimeError("No saved video. Press R to record and S to save next time.")
        video = max(videos, key=lambda p: p.stat().st_mtime)
    result = audit(video, candidate)
    (run_dir / "material_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_path = run_dir / "evaluation.json"
    subprocess.run([sys.executable, str(ROOT / "tools/evaluate_no_red_servo_relation.py"),
                    "--video", str(video), "--events", str(video.with_suffix(".angles.jsonl")),
                    "--rtt", str(video.with_suffix(".rtt.jsonl")), "--model", str(MODEL),
                    "--servo-calibration", str(frozen), "--device", "cpu", "--output", str(report_path)], check=True, cwd=ROOT)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    passed = report["acceptance"]["overall_pass"] and report["inside_calibration_range_rate"] >= 0.95
    report["workflow_acceptance"] = {"material_pass": True, "range_coverage_min": 0.95,
                                      "passed": passed, "active_calibration_changed": False}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'PASS' if passed else 'FAIL'}: independent validation. Report: {report_path}")
    print("The active calibration has not been changed.")
    return 0 if passed else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
