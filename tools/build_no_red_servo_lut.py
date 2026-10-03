"""Build a marker-free foam-centre servo/optical LUT from one calibration report."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boundary-tolerance-deg", type=float, default=2.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing LUT: {args.output}")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    grouped = {}
    for hold in report["holds"]:
        optical = hold.get("visual_optical_offset_deg_median")
        encoder = hold.get("encoder_pos_deg")
        if optical is None or encoder is None:
            continue
        grouped.setdefault(float(hold["guide_angle_deg"]), []).append(hold)
    if len(grouped) < 2:
        raise RuntimeError("calibration report does not contain enough angle nodes")

    points = []
    for command in sorted(grouped):
        holds = grouped[command]
        optical_values = [float(hold["visual_optical_offset_deg_median"]) for hold in holds]
        encoder_values = [float(hold["encoder_pos_deg"]) for hold in holds]
        points.append(
            {
                "servo_command_deg": command,
                "optical_offset_deg": round(statistics.median(optical_values), 6),
                "encoder_position_median_deg": round(statistics.median(encoder_values), 4),
                "hold_count": len(holds),
                "sampled_frame_count": sum(int(hold["sampled_frames"]) for hold in holds),
                "detection_rate": round(
                    sum(int(hold["detected_frames"]) for hold in holds)
                    / sum(int(hold["sampled_frames"]) for hold in holds),
                    6,
                ),
                "direction_optical_repeat_difference_deg": (
                    round(abs(optical_values[0] - optical_values[1]), 6)
                    if len(optical_values) == 2
                    else None
                ),
            }
        )
    optical_values = [point["optical_offset_deg"] for point in points]
    if not all(right > left for left, right in zip(optical_values, optical_values[1:])):
        raise RuntimeError("foam-centre optical nodes are not strictly monotonic")

    servo_min = points[0]["servo_command_deg"]
    servo_max = points[-1]["servo_command_deg"]
    encoder_errors = [
        abs(point["encoder_position_median_deg"] - point["servo_command_deg"])
        for point in points
    ]
    repeat_values = [
        point["direction_optical_repeat_difference_deg"]
        for point in points
        if point["direction_optical_repeat_difference_deg"] is not None
    ]
    data = {
        "type": "servo_optical_angle_lut",
        "name": "foam_center_servo_camera_lut_20260816_v1",
        "version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "calibration_complete_independent_validation_pending",
        "definitions": {
            "servo_command_deg": "Nominal MCU g target used during calibration.",
            "encoder_position_median_deg": "Median synchronized HX8 encoder pos during the stable hold; retained as validation metadata.",
            "optical_offset_deg": "Signed fisheye-corrected camera-ray angle of the marker-free segmented foam geometric centre in the arm motion plane; optical axis is zero and image-down is positive.",
            "inferred_servo_angle_deg": "The g-equivalent target command inferred by inverse interpolation of this marker-free foam-centre lookup table."
        },
        "runtime_anchor": {
            "detector": "foam_center_v9_gripper_axis_normal_bg03_20260816_candidate.pt",
            "point": "marker-free visible-foam segmentation centroid with the runtime contour/box hybrid rule",
            "warning": "This table uses the foam geometric centre, not the old red gripper-centre marker. Do not mix the two anchor definitions."
        },
        "mechanical_transmission": {
            "servo_pinion_teeth": 40,
            "driven_gear_teeth": 48,
            "included_in_empirical_fit": True,
            "note": "Do not apply the 40:48 ratio again at runtime."
        },
        "model": {
            "kind": "piecewise_linear_lut",
            "interpolation": "linear",
            "inverse_interpolation": "linear",
            "points": points,
        },
        "valid_range": {
            "servo_command_deg": [servo_min, servo_max],
            "optical_offset_deg": [min(optical_values), max(optical_values)],
        },
        "boundary_tolerance_deg": float(args.boundary_tolerance_deg),
        "fit": {
            "method": "Per-command median of stable marker-free foam-centre holds; outbound and return medians combined; -97 degree turnaround has one hold.",
            "node_count": len(points),
            "calibration_detection_rate": report["detection_rate"],
            "encoder_command_absolute_error_deg": {
                "mean": round(statistics.mean(encoder_errors), 6),
                "max": round(max(encoder_errors), 6),
            },
            "direction_optical_repeat_difference_deg": {
                "mean": round(statistics.mean(repeat_values), 6),
                "max": round(max(repeat_values), 6),
            },
            "absolute_accuracy_note": "This is an operational camera/encoder correspondence calibration, not externally traceable protractor metrology."
        },
        "source": {
            "calibration_report": str(args.report.resolve()),
            "calibration_video": report["video"],
            "calibration_angle_events": report["events"],
            "calibration_rtt_telemetry": report["rtt"],
            "camera_calibration": report["camera_calibration"],
            "old_reference_only": report["servo_calibration"],
            "data_split_policy": "Only session1 was used to construct nodes. Session2 is reserved exclusively for independent validation."
        },
        "independent_validation": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "node_count": len(points),
        "servo_range_deg": [servo_min, servo_max],
        "optical_range_deg": [min(optical_values), max(optical_values)],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
