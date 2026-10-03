"""Quick, read-only integrity check for the migrated Core project."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "run_realtime.py",
    "models/foam_center_v9_seg.pt",
    "models/foam_center_v9_seg.json",
    "models/foam_center_v9_gripper_axis_normal_bg03_20260816_candidate.pt",
    "models/foam_board_2p1mm_v8.pt",
    "configs/camera_2p1mm_640x480_fisheye.json",
    "configs/servo_to_optical_angle_yolo_v8_20260804_v1.json",
    "configs/servo_to_optical_angle_red_marker_lut_20260806_v1.json",
    "configs/servo_to_optical_angle_red_marker_lut_20260813_v2.json",
    "configs/servo_to_optical_angle_red_marker_lut_20260815_v3.json",
    "configs/servo_to_optical_angle_foam_center_lut_20260816_v1.json",
    "data/raw/v8_source_20260803/servo_camera_angle_calibration_50fps_20260803_000945_262004.mkv",
    "data/raw/servo_angle_20260804/arm_motion_slow_50fps_20260804_222641_587720.mkv",
    "data/training/foam_board_v8/foam_board_2p1mm.yaml",
    "data/training/foam_center_v9/foam_center_v9.yaml",
)


def main() -> None:
    missing = [relative for relative in REQUIRED_FILES if not (ROOT / relative).is_file()]
    if missing:
        for relative in missing:
            print(f"MISSING: {relative}")
        raise SystemExit(1)

    camera = json.loads(
        (ROOT / "configs/camera_2p1mm_640x480_fisheye.json").read_text(encoding="utf-8")
    )
    servo = json.loads(
        (ROOT / "configs/servo_to_optical_angle_yolo_v8_20260804_v1.json").read_text(
            encoding="utf-8"
        )
    )
    red_marker_servo = json.loads(
        (ROOT / "configs/servo_to_optical_angle_red_marker_lut_20260815_v3.json").read_text(
            encoding="utf-8"
        )
    )
    foam_center_servo = json.loads(
        (ROOT / "configs/servo_to_optical_angle_foam_center_lut_20260816_v1.json").read_text(
            encoding="utf-8"
        )
    )
    data_files = [path for path in (ROOT / "data").rglob("*") if path.is_file()]
    data_bytes = sum(path.stat().st_size for path in data_files)

    print("AerialRoboArm Vision Core: OK")
    print(f"data: {len(data_files)} files, {data_bytes / 1024 / 1024:.2f} MiB")
    print(
        "camera: "
        f"{camera['model']}, {camera['image_size'][0]}x{camera['image_size'][1]}, "
        f"reprojection={camera['reprojection_error_px']:.3f} px"
    )
    print(
        "legacy linear servo fit: "
        f"beta={servo['model']['slope']:.6f}*g+{servo['model']['intercept_deg']:.6f} deg"
    )
    print(
        "active marker-free foam-centre LUT: "
        f"{foam_center_servo['name']}, {len(foam_center_servo['model']['points'])} nodes, "
        f"g={foam_center_servo['valid_range']['servo_command_deg']} deg"
    )
    print(
        "retained red-marker gripper LUT: "
        f"{red_marker_servo['name']}, {len(red_marker_servo['model']['points'])} nodes"
    )
    print(
        "active target model: "
        "foam_center_v9_gripper_axis_normal_bg03_20260816_candidate.pt"
    )
    print("retained previous target model: foam_center_v9_seg.pt")
    print("legacy target model: foam_board_2p1mm_v8.pt (box center, retained)")


if __name__ == "__main__":
    main()
