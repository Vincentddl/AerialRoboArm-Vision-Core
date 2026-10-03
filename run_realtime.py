"""Recommended real-time foam-board detection entry point.

The displayed ``optical_axis_offset_deg`` is the angle between the ray from
the camera optical center to the segmented foam-board center and the camera
optical axis, projected into the arm's single motion plane. The image center
axis is therefore 0 degrees. The camera is mounted 30 degrees downward.

This launcher uses the independently validated marker-free foam-centre V1
relationship. It does not output a 0.4-second future prediction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from vision import runtime


PROJECT_ROOT = Path(__file__).resolve().parent
# This exact segmentation checkpoint was used to construct and independently
# validate the marker-free V1 lookup table. Changing the detector can move the
# geometric centre, so a different checkpoint requires a new validation.
MODEL = (
    PROJECT_ROOT
    / "models"
    / "foam_center_v9_gripper_axis_normal_bg03_20260816_candidate.pt"
)
CAMERA_CALIBRATION = PROJECT_ROOT / "configs" / "camera_2p1mm_640x480_fisheye.json"
# Marker-free V1 definition: YOLO finds the visible foam geometric centre,
# fisheye calibration converts that pixel to an optical-axis offset, and this
# empirical table returns the corresponding nominal g command. The table is
# valid for g=-97..-47 degrees and already includes the 40:48 transmission.
SERVO_CALIBRATION = (
    PROJECT_ROOT / "configs" / "servo_to_optical_angle_foam_center_lut_20260816_v1.json"
)


def main() -> None:
    if not MODEL.exists():
        raise FileNotFoundError(f"Missing runtime model: {MODEL}")

    cuda_available = torch.cuda.is_available()
    defaults = [
        "--model",
        str(MODEL),
        "--classes",
        "foam_board",
        "--conf",
        "0.50",
        "--new-track-conf",
        "0.50",
        "--new-track-confirm-frames",
        "2",
        "--new-track-min-motion",
        "0",
        "--new-track-candidate-max-age",
        "4",
        "--min-target-speed",
        "0",
        "--stationary-max-frames",
        "8",
        "--coast-frames",
        "0",
        "--iou",
        "0.5",
        "--imgsz",
        "640" if cuda_available else "512",
        "--device",
        "0" if cuda_available else "cpu",
        "--target-mode",
        "current",
        "--min-age",
        "2",
        "--max-match-distance",
        "140" if cuda_available else "220",
        "--max-missed",
        "12",
        "--measurement-noise",
        "30",
        "--process-noise",
        "250",
        "--camera-width",
        "640",
        "--camera-height",
        "480",
        "--camera-backend",
        "dshow",
        "--camera-exposure",
        "-7",
        "--camera-gain",
        "20",
        "--calibration",
        str(CAMERA_CALIBRATION),
        "--servo-angle-calibration",
        str(SERVO_CALIBRATION),
        "--camera-down-tilt-deg",
        "30",
        "--lateral-tolerance-deg",
        "5",
        "--min-box-area",
        "0.0005",
        "--skip-unchanged",
        "--window-title",
        "Foam Board - Marker-Free V1 Realtime",
        # PC-side HC-13. Opening COM5 does not move the arm by itself: output
        # starts only after the operator presses A in the preview window.
        "--hc13-port",
        "COM5",
        "--hc13-confirm-frames",
        "5",
        "--hc13-max-angle-spread-deg",
        "2.0",
        "--hc13-min-confidence",
        "0.60",
        "--hc13-send-hz",
        "20",
        "--hc13-target-speed",
        "0",
    ]
    # Keep the daily display uncluttered, while allowing the dedicated centre
    # validation command to request the white C(x,y) marker explicitly.
    user_args = sys.argv[1:]
    if "--show-centers" not in user_args and "--hide-centers" not in user_args:
        defaults.append("--hide-centers")
    # User arguments come last so explicit options override ordinary defaults.
    sys.argv = [sys.argv[0], *defaults, *user_args]
    runtime.main()


if __name__ == "__main__":
    main()
