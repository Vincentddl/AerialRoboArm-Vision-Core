import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"

from .ballistic_predictor import BallisticPredictor
from .hc13_link import (
    HC13GateResult,
    HC13SerialLink,
    StableTargetGate,
    select_control_target,
)
from .servo_angle_calibration import ServoOpticalAngleCalibration  # noqa: E402
from .tracker import (  # noqa: E402
    Detection,
    PixelToWorldMapper,
    TrajectoryEstimator,
    draw_arm_angle_overlay,
    draw_tracks,
)
from .trajectory_predictor import EnsembleTrajectoryPredictor  # noqa: E402
from ultralytics import YOLO  # noqa: E402


DEFAULT_MODEL = MODELS_DIR / "foam_center_v9_gripper_axis_normal_bg03_20260816_candidate.pt"


def resolve_path(path):
    if not path:
        return ""
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return str(candidate)
    return str(PROJECT_ROOT / candidate)


def load_model(model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    print(f"Using model: {model_path}")
    return YOLO(str(model_path))


def resolve_device(device):
    if not device:
        return None
    normalized = device.lower()
    wants_cuda = normalized.isdigit() or normalized.startswith("cuda")
    if wants_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "You requested GPU inference, but this Python environment cannot see CUDA.\n"
            f"torch version: {torch.__version__}\n"
            f"torch.version.cuda: {torch.version.cuda}\n"
            "Install a CUDA-enabled PyTorch build, then retry with --device 0.\n"
            "For now, run with --device cpu or omit --device."
        )
    return device


def parse_class_filter(class_filter, names):
    if not class_filter:
        return None

    allowed = set()
    class_items = [item.strip().lower() for item in class_filter.split(",") if item.strip()]
    normalized_names = {idx: str(name).lower() for idx, name in names.items()}

    for item in class_items:
        if item.isdigit():
            allowed.add(int(item))
            continue

        exact_matches = [idx for idx, name in normalized_names.items() if name == item]
        partial_matches = [idx for idx, name in normalized_names.items() if item in name]
        matches = exact_matches or partial_matches
        if not matches:
            examples = ", ".join(str(name) for _, name in list(names.items())[:12])
            raise ValueError(f"unknown class '{item}'. Example classes: {examples} ...")
        allowed.update(matches)

    print("Class filter:", ", ".join(f"{idx}:{names[idx]}" for idx in sorted(allowed)))
    return allowed


CAMERA_BACKENDS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "any": cv2.CAP_ANY,
}


def _configure_camera(cap, exposure=None, gain=None, brightness=None):
    """Configure exposure without forcing an overexposed high-gain image."""
    if exposure is None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
    if gain is not None:
        cap.set(cv2.CAP_PROP_GAIN, float(gain))
    if brightness is not None:
        cap.set(cv2.CAP_PROP_BRIGHTNESS, float(brightness))

    print(
        "Camera image controls: "
        f"exposure={cap.get(cv2.CAP_PROP_EXPOSURE):.2f}, "
        f"gain={cap.get(cv2.CAP_PROP_GAIN):.2f}, "
        f"brightness={cap.get(cv2.CAP_PROP_BRIGHTNESS):.2f}"
    )


def _warmup_camera(cap, frames=10):
    """Read and discard *frames* to let auto-exposure settle."""
    for _ in range(frames):
        cap.read()


def open_source(
    source,
    backend="dshow",
    lock_camera=False,
    width=0,
    height=0,
    exposure=None,
    gain=None,
    brightness=None,
):
    if source.isdigit():
        candidates = (
            [("dshow", cv2.CAP_DSHOW), ("msmf", cv2.CAP_MSMF), ("any", cv2.CAP_ANY)]
            if backend == "auto"
            else [(backend, CAMERA_BACKENDS[backend])]
        )
        for backend_name, backend_id in candidates:
            cap = cv2.VideoCapture(int(source), backend_id)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                print(f"Camera source {source} opened with {backend_name}")
                if width:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                if height:
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                _configure_camera(
                    cap,
                    exposure=exposure,
                    gain=gain,
                    brightness=brightness,
                )
                _warmup_camera(cap, frames=3 if exposure is not None else 10)
                if lock_camera and exposure is None:
                    _lock_camera_settings(cap)
                return cap
            cap.release()
        return cap

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _lock_camera_settings(cap):
    """Disable auto-exposure and auto-white-balance for stable detection.

    Auto-exposure changes frame brightness continuously, which makes the
    object's appearance inconsistent from frame to frame.  This is the #1
    reason why real-time detection often looks worse than recorded video.

    We disable the auto modes AFTER auto-exposure has settled, so the
    camera retains a usable brightness level rather than a near-black frame.
    """
    settings = [
        (cv2.CAP_PROP_AUTO_EXPOSURE, 0.25, "auto_exposure=locked"),
        (cv2.CAP_PROP_AUTO_WB, 0.0, "auto_white_balance=locked"),
    ]
    applied = []
    for prop, value, name in settings:
        success = cap.set(prop, value)
        if success:
            applied.append(name)
        else:
            applied.append(f"{name}(unsupported)")
    print(f"Camera lock: {', '.join(applied)}")


class LatestFrameCapture:
    """Read camera frames in the background and always expose the newest frame."""

    def __init__(
        self,
        source,
        width=0,
        height=0,
        backend="dshow",
        lock_camera=False,
        exposure=None,
        gain=None,
        brightness=None,
    ):
        self.cap = open_source(
            source,
            backend=backend,
            lock_camera=lock_camera,
            width=width,
            height=height,
            exposure=exposure,
            gain=gain,
            brightness=brightness,
        )
        if not self.cap.isOpened():
            raise RuntimeError(f"failed to open source: {source}")

        self.lock = threading.Lock()
        self.running = True
        self.frame = None
        self.timestamp = 0.0
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self.lock:
                self.frame = frame
                self.timestamp = time.perf_counter()

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, 0.0, None
            return True, self.timestamp, self.frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


class SequentialVideoCapture:
    """Read every frame from a finite video and use video time instead of inference time."""

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"failed to open video: {source}")
        reported_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.fps = reported_fps if reported_fps > 1 else 25.0
        self.frame_index = 0
        self.timestamps = None

        timestamp_path = Path(source).with_suffix(".timestamps.jsonl")
        if timestamp_path.exists():
            values = []
            for line in timestamp_path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                values.append(float(record["monotonic_seconds"]))
            if len(values) >= 2:
                start = values[0]
                self.timestamps = [value - start for value in values]
                duration = self.timestamps[-1]
                if duration > 0:
                    self.fps = (len(self.timestamps) - 1) / duration
                print(f"Using frame timestamps: {timestamp_path} ({self.fps:.3f} FPS effective)")

    def read(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return False, 0.0, None
        if self.timestamps is not None and self.frame_index < len(self.timestamps):
            timestamp = self.timestamps[self.frame_index]
        else:
            timestamp = self.frame_index / self.fps
        self.frame_index += 1
        return True, timestamp, frame

    def release(self):
        self.cap.release()


def result_to_detections(result, allowed_classes=None, max_box_area=0.0, min_box_area=0.0, image_shape=None):
    detections = []
    names = result.names
    boxes = result.boxes
    if boxes is None:
        return detections

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    mask_polygons = result.masks.xy if result.masks is not None else None

    effective_shape = image_shape or getattr(result, "orig_shape", None)
    image_area = float(effective_shape[0] * effective_shape[1]) if effective_shape else 1.0

    for index, (bbox, score, class_id) in enumerate(zip(xyxy, conf, cls)):
        if allowed_classes is not None and class_id not in allowed_classes:
            continue
        # Filter by bounding-box area (fraction of image)
        if max_box_area > 0 or min_box_area > 0:
            box_w = float(bbox[2] - bbox[0])
            box_h = float(bbox[3] - bbox[1])
            box_area = box_w * box_h
            area_frac = box_area / image_area
            if max_box_area > 0 and area_frac > max_box_area:
                continue
            if min_box_area > 0 and area_frac < min_box_area:
                continue
        else:
            box_w = float(bbox[2] - bbox[0])
            box_h = float(bbox[3] - bbox[1])
            box_area = box_w * box_h
            area_frac = box_area / image_area

        center_override = None
        center_source = "box_center"
        if mask_polygons is not None and index < len(mask_polygons):
            polygon = np.asarray(mask_polygons[index], dtype=np.float32).reshape(-1, 1, 2)
            moments = cv2.moments(polygon)
            # Center definition for the V9 segmentation model:
            # - normal target sizes: centroid of the visible foam contour;
            # - very small masks (<1% image): box center avoids mask quantization;
            # - very large close-ups (>13% image): box center is more robust when
            #   one side face is under-segmented.
            if abs(moments["m00"]) >= 1e-9 and 0.01 <= area_frac <= 0.13:
                center_override = (
                    float(moments["m10"] / moments["m00"]),
                    float(moments["m01"] / moments["m00"]),
                )
                center_source = "mask_centroid"
            else:
                center_source = "hybrid_box_center"
        detections.append(
            Detection(
                bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                score=float(score),
                class_id=int(class_id),
                label=str(names.get(int(class_id), class_id)),
                center_override=center_override,
                center_source=center_source,
            )
        )
    return detections


def draw_detection_centers(image, detections):
    for detection in detections:
        cx, cy = [int(v) for v in detection.center]
        cv2.drawMarker(
            image,
            (cx, cy),
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=16,
            thickness=2,
        )
        cv2.putText(
            image,
            f"C({cx},{cy})",
            (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
    return image


def main():
    parser = argparse.ArgumentParser(description="YOLO trash detection with trajectory prediction")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to trained YOLO .pt model")
    parser.add_argument("--source", default="0", help="Camera index, video file, image file, or stream URL")
    parser.add_argument("--classes", default="", help="Class filter, e.g. bottle, Clear plastic bottle, 7, 21")
    parser.add_argument("--list-classes", action="store_true", help="Print model classes and exit")
    parser.add_argument("--conf", type=float, default=0.2, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=512, help="YOLO inference image size")
    parser.add_argument("--device", default="", help="Device, e.g. cpu, 0, cuda:0. Empty means auto")
    parser.add_argument(
        "--target-mode",
        choices=["current", "predictive"],
        default="current",
        help="current outputs the latest confirmed detection without future "
             "extrapolation; predictive enables the optional trajectory predictors",
    )
    parser.add_argument("--predict-seconds", type=float, default=0.4, help="Future trajectory horizon in seconds")
    parser.add_argument(
        "--trajectory-steps",
        type=int,
        default=8,
        help="Number of samples in the predicted trajectory, excluding the current point",
    )
    parser.add_argument("--min-age", type=int, default=1, help="Frames required before a track is sent as an arm target")
    parser.add_argument("--max-match-distance", type=float, default=120.0, help="Max pixel distance for ID matching")
    parser.add_argument("--max-missed", type=int, default=15, help="Drop a track after this many missed frames")
    parser.add_argument("--process-noise", type=float, default=250.0, help="Kalman process noise for horizontal (x) direction")
    parser.add_argument(
        "--process-noise-y",
        type=float,
        default=None,
        help="Kalman process noise for vertical (y) direction. Higher = faster vertical response. "
             "Defaults to --process-noise if not set.",
    )
    parser.add_argument("--measurement-noise", type=float, default=25.0, help="Kalman measurement noise; higher smooths detector jitter more")
    parser.add_argument(
        "--new-track-conf",
        type=float,
        default=0.0,
        help="Minimum confidence required to create a new track; lower-confidence detections may still continue one",
    )
    parser.add_argument(
        "--new-track-confirm-frames",
        type=int,
        default=1,
        help="Matched frames required before a detection candidate becomes a track",
    )
    parser.add_argument(
        "--new-track-min-motion",
        type=float,
        default=0.0,
        help="Minimum pixel displacement required before a detection candidate becomes a track",
    )
    parser.add_argument(
        "--new-track-candidate-max-age",
        type=int,
        default=4,
        help="Maximum frames allowed for a candidate to satisfy track confirmation",
    )
    parser.add_argument(
        "--min-target-speed",
        type=float,
        default=0.0,
        help="Minimum track speed in pixels/second before output to the arm",
    )
    parser.add_argument(
        "--stationary-max-frames",
        type=int,
        default=8,
        help="Drop a track after this many consecutive frames below --min-target-speed",
    )
    parser.add_argument(
        "--coast-frames",
        type=int,
        default=0,
        help="Continue output from the motion model for this many consecutive detector misses",
    )
    parser.add_argument("--class-agnostic-tracking", action="store_true", help="Keep the same ID even if YOLO class changes between similar trash classes")
    parser.add_argument("--camera-width", type=int, default=640, help="Requested camera width")
    parser.add_argument("--camera-height", type=int, default=480, help="Requested camera height")
    parser.add_argument(
        "--camera-backend",
        choices=["auto", "dshow", "msmf", "any"],
        default="dshow",
        help="Windows camera backend for numeric sources",
    )
    parser.add_argument(
        "--camera-exposure",
        type=float,
        default=None,
        help="Manual camera exposure value. Lower DirectShow values shorten "
             "exposure and reduce motion blur; omit to use auto exposure.",
    )
    parser.add_argument(
        "--camera-gain",
        type=float,
        default=None,
        help="Optional manual camera gain. Raise only enough to compensate for short exposure.",
    )
    parser.add_argument(
        "--camera-brightness",
        type=float,
        default=None,
        help="Optional manual camera brightness. Omit to keep the device default.",
    )
    parser.add_argument("--skip-unchanged", action="store_true", help="Skip duplicate camera frames from the latest-frame reader")
    parser.add_argument("--lock-camera", action="store_true", help="Disable camera auto-exposure/auto-white-balance for stable real-time detection")
    center_display = parser.add_mutually_exclusive_group()
    center_display.add_argument(
        "--hide-centers",
        dest="hide_centers",
        action="store_true",
        help="Hide immediate detection centroids.",
    )
    center_display.add_argument(
        "--show-centers",
        dest="hide_centers",
        action="store_false",
        help="Show the measured target centre as a white C(x,y) cross.",
    )
    parser.set_defaults(hide_centers=False)
    parser.add_argument("--print-targets", action="store_true", help="Print target JSON to terminal")
    parser.add_argument("--calibration", default="", help="Optional pixel-to-robot-plane calibration JSON")
    parser.add_argument(
        "--servo-angle-calibration",
        default="",
        help="Optional empirical servo-command to optical-axis angle calibration JSON. "
             "When set, live output keeps optical, servo and legacy mechanical angles separate.",
    )
    parser.add_argument(
        "--camera-down-tilt-deg",
        "--camera-elevation-deg",
        dest="camera_down_tilt_deg",
        type=float,
        default=0.0,
        help="Camera optical axis downward tilt from the mechanical horizontal "
             "x-axis. Positive means the camera points downward.",
    )
    parser.add_argument(
        "--lateral-tolerance-deg",
        type=float,
        default=5.0,
        help="Maximum absolute left/right angle for a target to be considered "
             "inside the single-axis arm plane.",
    )
    parser.add_argument(
        "--angle-tick-step-deg",
        type=float,
        default=5.0,
        help="Mechanical-angle spacing for the calibrated video overlay.",
    )
    parser.add_argument(
        "--hide-angle-overlay",
        action="store_true",
        help="Hide the camera crosshair and single-axis angle scale.",
    )
    parser.add_argument(
        "--max-prediction-uncertainty-deg",
        type=float,
        default=8.0,
        help="Mark a future target invalid when estimated angular uncertainty exceeds this value.",
    )
    parser.add_argument(
        "--max-box-area",
        type=float,
        default=0.0,
        help="Ignore detections whose bounding box exceeds this fraction of the image (0=off). "
             "Useful for filtering out large false positives like faces/hands near the camera.",
    )
    parser.add_argument(
        "--min-box-area",
        type=float,
        default=0.0,
        help="Ignore detections whose bounding box is smaller than this fraction of the image (0=off).",
    )
    parser.add_argument(
        "--predictor",
        choices=["kalman", "ensemble", "angle_kalman"],
        default="kalman",
        help="Prediction method for 0.4s trajectory: kalman (pixel-space, default), "
             "ensemble (fuses pixel + angle Kalman + polynomial + ballistic), angle_kalman (pure angle-space)",
    )
    parser.add_argument(
        "--angle-process-noise",
        type=float,
        default=150.0,
        help="Angle-space Kalman process noise (higher = faster response, default 150)",
    )
    parser.add_argument(
        "--angle-measurement-noise",
        type=float,
        default=0.5,
        help="Angle-space Kalman measurement noise deg^2 (higher = smoother, default 0.5)",
    )
    parser.add_argument(
        "--ballistic-gravity",
        type=float,
        default=400.0,
        help="Initial g_eff prior for ballistic predictor (deg/s^2). "
             "Pitch is positive downward, so gravity is normally positive. Default 400.",
    )
    parser.add_argument(
        "--ballistic-ema-alpha",
        type=float,
        default=0.30,
        help="EMA smoothing factor for online g_eff estimation (0-1). Default 0.30.",
    )
    parser.add_argument(
        "--no-ballistic",
        action="store_true",
        help="Disable the ballistic predictor even when using ensemble mode.",
    )
    parser.add_argument("--save-jsonl", default="", help="Optional path to append arm target JSON lines")
    parser.add_argument("--output", default="", help="Optional output image/video path")
    parser.add_argument("--no-window", action="store_true", help="Run without cv2.imshow")
    parser.add_argument(
        "--window-title",
        default="YOLO Object Trajectory",
        help="Title for the interactive preview window.",
    )
    parser.add_argument(
        "--hc13-port",
        default="",
        help="PC-side HC-13 serial port, for example COM5. Empty disables output.",
    )
    parser.add_argument("--hc13-baud", type=int, default=230400)
    parser.add_argument(
        "--no-hc13",
        action="store_true",
        help="Disable HC-13 output even when the launcher supplies a default port.",
    )
    parser.add_argument(
        "--hc13-confirm-frames",
        type=int,
        default=5,
        help="Consecutive valid frames required before a servo command is sent.",
    )
    parser.add_argument(
        "--hc13-max-angle-spread-deg",
        type=float,
        default=2.0,
        help="Maximum servo-angle spread across confirmation frames.",
    )
    parser.add_argument(
        "--hc13-min-confidence",
        type=float,
        default=0.60,
        help="Minimum YOLO confidence allowed to control the arm (0..1).",
    )
    parser.add_argument(
        "--hc13-send-hz",
        type=float,
        default=20.0,
        help="Heartbeat rate while a target remains confirmed.",
    )
    parser.add_argument(
        "--hc13-target-speed",
        type=int,
        default=0,
        help="Protocol speed field. Zero keeps the MCU's configured default speed.",
    )
    parser.add_argument(
        "--hc13-arm-on-start",
        action="store_true",
        help="Start transmitting immediately. Normally press A in the preview to arm.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.hc13_min_confidence <= 1.0:
        parser.error("--hc13-min-confidence must be in [0,1]")

    device = resolve_device(args.device)
    model = load_model(resolve_path(args.model))
    if args.list_classes:
        for class_id, name in model.names.items():
            print(f"{class_id}: {name}")
        return

    allowed_classes = parse_class_filter(args.classes, model.names)
    mapper = PixelToWorldMapper(resolve_path(args.calibration) if args.calibration else None)
    servo_angle_calibration = (
        ServoOpticalAngleCalibration.from_json(
            resolve_path(args.servo_angle_calibration)
        )
        if args.servo_angle_calibration
        else None
    )
    if servo_angle_calibration is not None:
        if mapper.unit != "deg":
            raise RuntimeError(
                "--servo-angle-calibration requires a pinhole/fisheye camera calibration"
            )
        print(servo_angle_calibration.summary)

    # Prediction is optional. The default current-target mode avoids starting
    # any future predictor and is intended for hand-held slow/fast approach.
    ballistic_predictor = None
    if args.target_mode == "predictive" and not args.no_ballistic:
        ballistic_predictor = BallisticPredictor(
            ema_alpha=args.ballistic_ema_alpha,
            initial_gravity_deg_s2=args.ballistic_gravity,
        )
        print(f"Ballistic predictor enabled (initial g_eff={args.ballistic_gravity} deg/s²)")

    # Build ensemble predictor when requested and calibration is available
    ensemble_predictor = None
    if (
        args.target_mode == "predictive"
        and args.predictor in ("ensemble", "angle_kalman")
        and mapper.unit == "deg"
    ):
        ensemble_predictor = EnsembleTrajectoryPredictor(
            mapper=mapper,
            angle_process_noise=args.angle_process_noise,
            angle_measurement_noise=args.angle_measurement_noise,
            ballistic_predictor=ballistic_predictor,
            mode=args.predictor,
        )
        print(f"Prediction method: {args.predictor}")
        if args.predictor == "angle_kalman":
            print("  (pure angle-space Kalman — ensemble fusion disabled)")
    elif (
        args.target_mode == "predictive"
        and args.predictor in ("ensemble", "angle_kalman")
    ):
        print(
            f"Warning: --predictor={args.predictor} requires a camera calibration "
            f"(pinhole/fisheye). Falling back to pixel kalman."
        )
    if args.target_mode == "current":
        print("Target mode: current detection (future prediction disabled)")

    estimator = TrajectoryEstimator(
        max_match_distance=args.max_match_distance,
        max_missed=args.max_missed,
        process_noise=args.process_noise,
        process_noise_y=args.process_noise_y,
        measurement_noise=args.measurement_noise,
        new_track_min_score=args.new_track_conf,
        new_track_confirmation_frames=args.new_track_confirm_frames,
        new_track_min_motion=args.new_track_min_motion,
        new_track_candidate_max_age=args.new_track_candidate_max_age,
        stationary_speed_threshold=args.min_target_speed,
        stationary_max_frames=args.stationary_max_frames,
        match_classes=not args.class_agnostic_tracking,
        mapper=mapper,
        ensemble_predictor=ensemble_predictor,
        servo_angle_calibration=servo_angle_calibration,
    )

    source = args.source
    source_path = Path(source)
    is_image = source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    is_video = source_path.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv", ".m4v", ".wmv", ".webm"}
    capture = None
    writer = None
    calibration_size_checked = False
    inference_device_reported = False
    jsonl_file = open(args.save_jsonl, "a", encoding="utf-8") if args.save_jsonl else None
    hc13_gate = StableTargetGate(
        confirmation_frames=args.hc13_confirm_frames,
        max_spread_deg=args.hc13_max_angle_spread_deg,
        send_hz=args.hc13_send_hz,
    )
    hc13_link = None
    hc13_armed = bool(args.hc13_arm_on_start)
    hc13_error = ""
    hc13_result = HC13GateResult(
        status="DISABLED",
        required_frames=args.hc13_confirm_frames,
    )
    if args.hc13_port and not args.no_hc13:
        try:
            hc13_link = HC13SerialLink(args.hc13_port, args.hc13_baud)
            print(
                f"HC-13 connected: {args.hc13_port} @ {args.hc13_baud} 8N1. "
                f"{'ARMED by command line' if hc13_armed else 'Press A in the preview to arm output.'}"
            )
        except Exception as exc:
            hc13_error = str(exc)
            hc13_armed = False
            print(f"HC-13 unavailable: {hc13_error}")

    try:
        if is_image:
            frame = cv2.imread(source)
            if frame is None:
                raise RuntimeError(f"failed to read image: {source}")
            frames = [(time.perf_counter(), frame)]
        elif is_video:
            capture = SequentialVideoCapture(source)
            frames = None
        else:
            capture = LatestFrameCapture(
                source,
                width=args.camera_width,
                height=args.camera_height,
                backend=args.camera_backend,
                lock_camera=args.lock_camera,
                exposure=args.camera_exposure,
                gain=args.camera_gain,
                brightness=args.camera_brightness,
            )
            frames = None

        last_timestamp = None
        while True:
            if frames is None:
                ok, timestamp, frame = capture.read()
                if not ok:
                    if is_video:
                        break
                    time.sleep(0.005)
                    continue
                if args.skip_unchanged and last_timestamp is not None and timestamp == last_timestamp:
                    time.sleep(0.001)
                    continue
                last_timestamp = timestamp
                if frame is None:
                    break
            else:
                if not frames:
                    break
                timestamp, frame = frames.pop(0)

            if not calibration_size_checked and mapper.image_size is not None:
                actual_size = (frame.shape[1], frame.shape[0])
                if actual_size != mapper.image_size:
                    raise RuntimeError(
                        f"calibration expects {mapper.image_size[0]}x{mapper.image_size[1]}, "
                        f"but source is {actual_size[0]}x{actual_size[1]}"
                    )
                print(mapper.calibration_info)
                calibration_size_checked = True

            start = time.perf_counter()
            result = model.predict(
                source=frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=device,
                verbose=False,
            )[0]
            if not inference_device_reported:
                print(f"Inference device: {next(model.model.parameters()).device}")
                inference_device_reported = True
            detections = result_to_detections(
                result,
                allowed_classes,
                max_box_area=args.max_box_area,
                min_box_area=args.min_box_area,
                image_shape=frame.shape[:2],
            )
            tracks = estimator.update(detections, timestamp)
            if args.target_mode == "current":
                arm_targets = estimator.current_targets(
                    min_age=args.min_age,
                    max_missed=args.coast_frames,
                    camera_down_tilt_deg=args.camera_down_tilt_deg,
                    lateral_tolerance_deg=args.lateral_tolerance_deg,
                )
            else:
                arm_targets = estimator.arm_targets(
                    predict_seconds=args.predict_seconds,
                    min_age=args.min_age,
                    max_missed=args.coast_frames,
                    min_speed=args.min_target_speed,
                    trajectory_steps=args.trajectory_steps,
                    camera_down_tilt_deg=args.camera_down_tilt_deg,
                    lateral_tolerance_deg=args.lateral_tolerance_deg,
                    max_prediction_uncertainty_deg=args.max_prediction_uncertainty_deg,
                )
            active_tracks = [track for track in tracks if track.missed <= args.coast_frames]
            infer_ms = (time.perf_counter() - start) * 1000.0

            # Control output is intentionally separate from detection output.
            # Only a calibrated, in-plane target that remains stable for the
            # configured number of frames is allowed to reach the MCU.
            if hc13_link is not None and hc13_armed:
                control_target = select_control_target(
                    arm_targets,
                    min_confidence=args.hc13_min_confidence,
                )
                hc13_result = hc13_gate.update(control_target, time.perf_counter())
                if hc13_result.should_send:
                    try:
                        hc13_link.send_target(
                            hc13_result.angle_deg,
                            args.hc13_target_speed,
                            hc13_result.confidence_percent,
                        )
                    except Exception as exc:
                        hc13_error = str(exc)
                        hc13_armed = False
                        hc13_gate.reset()
                        print(f"HC-13 send failed; output disarmed: {hc13_error}")
            elif hc13_link is not None:
                hc13_gate.reset()
                hc13_result = HC13GateResult(
                    status="DISARMED - PRESS A",
                    required_frames=args.hc13_confirm_frames,
                )
            elif hc13_error:
                hc13_result = HC13GateResult(
                    status="LINK ERROR",
                    required_frames=args.hc13_confirm_frames,
                )

            if jsonl_file and arm_targets:
                jsonl_file.write(json.dumps({"time": timestamp, "targets": arm_targets}, ensure_ascii=False) + "\n")
                jsonl_file.flush()

            if args.print_targets and arm_targets:
                print(json.dumps({"time": round(timestamp, 3), "targets": arm_targets}, ensure_ascii=False))

            vis = draw_tracks(
                frame.copy(),
                active_tracks,
                mapper,
                predict_seconds=args.predict_seconds,
                min_age=args.min_age,
                trajectory_steps=args.trajectory_steps,
                targets=arm_targets,
            )
            if not args.hide_angle_overlay:
                vis = draw_arm_angle_overlay(
                    vis,
                    mapper,
                    camera_down_tilt_deg=args.camera_down_tilt_deg,
                    tick_step_deg=args.angle_tick_step_deg,
                    servo_angle_calibration=servo_angle_calibration,
                )
            if not args.hide_centers:
                vis = draw_detection_centers(vis, detections)
            cv2.putText(
                vis,
                f"YOLO {args.target_mode} "
                f"{'SERVO_CAL' if servo_angle_calibration else 'LEGACY_GEOMETRIC'} "
                f"{infer_ms:.1f}ms | "
                f"det {len(detections)} | tracks {len(tracks)}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            if args.no_hc13 or not args.hc13_port:
                hc13_text = "HC13 DISABLED"
                hc13_color = (128, 128, 128)
            elif hc13_error:
                hc13_text = f"HC13 ERROR: {hc13_error[:48]}"
                hc13_color = (0, 0, 255)
            else:
                count_text = (
                    f" {hc13_result.confirmed_frames}/{hc13_result.required_frames}"
                    if hc13_result.required_frames
                    else ""
                )
                angle_text = (
                    f" g={hc13_result.angle_deg:+d}deg"
                    if hc13_result.angle_deg is not None
                    else ""
                )
                hc13_text = (
                    f"HC13 {'ARMED' if hc13_armed else 'SAFE'} "
                    f"{hc13_result.status}{count_text}{angle_text}"
                )
                hc13_color = (0, 255, 0) if hc13_armed else (0, 255, 255)
            cv2.putText(
                vis,
                hc13_text,
                (10, 96),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                4,
            )
            cv2.putText(
                vis,
                hc13_text,
                (10, 96),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                hc13_color,
                2,
            )

            if args.output and not is_image:
                if writer is None:
                    h, w = vis.shape[:2]
                    fps = getattr(capture, "fps", capture.cap.get(cv2.CAP_PROP_FPS)) if capture is not None else 25
                    fallback_fps = 30.0 if args.source.isdigit() else 25.0
                    fps = fps if fps and fps > 1 else fallback_fps
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
                writer.write(vis)

            if not args.no_window:
                window_name = args.window_title
                cv2.imshow(window_name, vis)
                key = cv2.waitKey(0 if is_image else 1) & 0xFF
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in (27, ord("q")):
                    break
                if key in (ord("a"), ord("A")):
                    if hc13_link is None:
                        print("HC-13 cannot be armed: serial link is not connected.")
                    else:
                        hc13_armed = not hc13_armed
                        hc13_gate.reset()
                        if hc13_armed:
                            hc13_error = ""
                        print(
                            "HC-13 output ARMED"
                            if hc13_armed
                            else "HC-13 output DISARMED; MCU will hold after timeout"
                        )

            if is_image:
                if args.output:
                    cv2.imwrite(args.output, vis)
                break
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if jsonl_file is not None:
            jsonl_file.close()
        if hc13_link is not None:
            hc13_link.close()
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
