"""Compare marker-free visual servo inference with synchronized encoder telemetry."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from vision.servo_angle_calibration import ServoOpticalAngleCalibration
from vision.tracker import PixelToWorldMapper


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--rtt", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_DIR
        / "models"
        / "foam_center_v9_gripper_axis_normal_bg03_20260816_candidate.pt",
    )
    parser.add_argument(
        "--camera-calibration",
        type=Path,
        default=PROJECT_DIR / "configs" / "camera_2p1mm_640x480_fisheye.json",
    )
    parser.add_argument(
        "--servo-calibration",
        type=Path,
        default=PROJECT_DIR
        / "configs"
        / "servo_to_optical_angle_red_marker_lut_20260815_v3.json",
    )
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--reject-clipped-targets", action="store_true",
                        help="Reject target boxes touching the image border (new merged-fit protocol)")
    parser.add_argument("--device", default="0")
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--trim-start-seconds", type=float, default=0.8)
    parser.add_argument("--trim-end-seconds", type=float, default=0.2)
    parser.add_argument("--detection-rate-min", type=float, default=0.95)
    parser.add_argument("--mae-deg-max", type=float, default=1.0)
    parser.add_argument("--p95-deg-max", type=float, default=2.0)
    parser.add_argument("--max-deg-max", type=float, default=3.0)
    parser.add_argument("--direction-repeat-deg-max", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path):
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return None
    return {
        "mean": round(float(array.mean()), 4),
        "median": round(float(np.median(array)), 4),
        "rmse": round(float(math.sqrt(np.mean(array * array))), 4),
        "p95": round(float(np.percentile(array, 95)), 4),
        "max": round(float(array.max()), 4),
    }


def predicted_centroid(result, reject_clipped=False):
    if not len(result.boxes):
        return None, None, "none"
    best = int(result.boxes.conf.argmax().cpu())
    confidence = float(result.boxes.conf[best].cpu())
    box = result.boxes.xyxy[best].cpu().numpy()
    if reject_clipped and (box[0] <= 2 or box[1] <= 2
                           or box[2] >= result.orig_shape[1] - 2
                           or box[3] >= result.orig_shape[0] - 2):
        return None, confidence, "clipped_target"
    box_center = np.asarray([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    box_area_fraction = float((box[2] - box[0]) * (box[3] - box[1])) / float(
        result.orig_shape[0] * result.orig_shape[1]
    )
    if result.masks is not None and best < len(result.masks.xy):
        polygon = np.asarray(result.masks.xy[best], dtype=np.float32).reshape(-1, 1, 2)
        moments = cv2.moments(polygon)
        if abs(moments["m00"]) >= 1e-9:
            center = np.asarray(
                [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]
            )
            if 0.01 <= box_area_fraction <= 0.13:
                return center, confidence, "mask_centroid"
            return box_center, confidence, "hybrid_box_center"
    return box_center, confidence, "box_center"


def build_holds(events):
    started = {}
    holds = []
    occurrence = 0
    for record in events:
        event = record.get("event")
        if event == "hold_started":
            occurrence += 1
            started[occurrence] = record
        elif event == "hold_completed" and occurrence in started:
            first = started.pop(occurrence)
            holds.append(
                {
                    "sequence_index": occurrence,
                    "step": int(first["step"]),
                    "guide_angle_deg": float(first["angle_deg"]),
                    "start_frame": int(first["frame_index"]),
                    "end_frame": int(record["frame_index"]),
                    "start_elapsed": float(first["elapsed_seconds"]),
                    "end_elapsed": float(record["elapsed_seconds"]),
                }
            )
    return holds


def main():
    args = parse_args()
    for path in (
        args.video,
        args.events,
        args.rtt,
        args.model,
        args.camera_calibration,
        args.servo_calibration,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    events = read_jsonl(args.events)
    telemetry = [record for record in read_jsonl(args.rtt) if record.get("pos") is not None]
    holds = build_holds(events)
    if not holds:
        raise RuntimeError("no completed angle holds found")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {args.video}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 50.0
    frame_step = max(1, int(round(source_fps / args.sample_fps)))
    selected = []
    for hold_index, hold in enumerate(holds):
        first = hold["start_frame"] + int(round(args.trim_start_seconds * source_fps))
        last = hold["end_frame"] - int(round(args.trim_end_seconds * source_fps))
        for frame_index in range(first, max(first, last) + 1, frame_step):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if ok:
                selected.append((hold_index, frame_index, frame))
    capture.release()

    model = YOLO(str(args.model))
    mapper = PixelToWorldMapper(str(args.camera_calibration))
    servo_calibration = ServoOpticalAngleCalibration.from_json(args.servo_calibration)
    predictions = {index: [] for index in range(len(holds))}
    batch_size = 16
    for start in range(0, len(selected), batch_size):
        batch = selected[start : start + batch_size]
        results = model.predict(
            [item[2] for item in batch],
            imgsz=640,
            conf=args.confidence,
            max_det=5,
            batch=len(batch),
            device=args.device,
            verbose=False,
        )
        for (hold_index, frame_index, _), result in zip(batch, results):
            center, confidence, mode = predicted_centroid(result, args.reject_clipped_targets)
            item = {
                "frame_index": frame_index,
                "confidence": round(float(confidence), 6) if confidence is not None else None,
                "mode": mode,
                "center_xy": None,
                "optical_offset_deg": None,
                "inferred_servo_deg": None,
            }
            if center is not None:
                camera_angles = mapper.to_world((float(center[0]), float(center[1])))
                optical_offset = mapper.camera_plane_angle_deg(camera_angles)
                inferred_servo = servo_calibration.servo_from_optical_offset(optical_offset)
                item.update(
                    {
                        "center_xy": [round(float(center[0]), 3), round(float(center[1]), 3)],
                        "optical_offset_deg": round(float(optical_offset), 5),
                        "inferred_servo_deg": round(float(inferred_servo), 5),
                        "inside_calibration_range": servo_calibration.is_optical_offset_in_range(
                            optical_offset
                        ),
                    }
                )
            predictions[hold_index].append(item)

    all_absolute_errors = []
    all_signed_errors = []
    hold_reports = []
    for hold_index, hold in enumerate(holds):
        telemetry_start = hold["start_elapsed"] + args.trim_start_seconds
        telemetry_end = hold["end_elapsed"] - args.trim_end_seconds
        hold_telemetry = [
            record
            for record in telemetry
            if telemetry_start <= float(record["elapsed_seconds"]) <= telemetry_end
        ]
        if not hold_telemetry:
            hold_telemetry = [
                record
                for record in telemetry
                if hold["start_elapsed"]
                <= float(record["elapsed_seconds"])
                <= hold["end_elapsed"]
            ]
        encoder_positions = [float(record["pos"]) for record in hold_telemetry]
        targets = [float(record["tgt"]) for record in hold_telemetry if record.get("tgt") is not None]
        encoder_position = statistics.median(encoder_positions) if encoder_positions else None
        target = statistics.median(targets) if targets else None
        samples = predictions[hold_index]
        detected_samples = [
            sample for sample in samples if sample.get("inferred_servo_deg") is not None
        ]
        valid_samples = [
            sample
            for sample in detected_samples
            if sample.get("inside_calibration_range")
        ]
        inferred_values = [float(sample["inferred_servo_deg"]) for sample in valid_samples]
        optical_values = [
            float(sample["optical_offset_deg"])
            for sample in detected_samples
            if sample.get("optical_offset_deg") is not None
        ]
        signed_errors = (
            [value - encoder_position for value in inferred_values]
            if encoder_position is not None
            else []
        )
        absolute_errors = [abs(value) for value in signed_errors]
        all_signed_errors.extend(signed_errors)
        all_absolute_errors.extend(absolute_errors)
        hold_reports.append(
            {
                **hold,
                "telemetry_samples": len(hold_telemetry),
                "encoder_pos_deg": round(float(encoder_position), 4)
                if encoder_position is not None
                else None,
                "encoder_tgt_deg": round(float(target), 4) if target is not None else None,
                "guide_target_difference_deg": round(float(target - hold["guide_angle_deg"]), 4)
                if target is not None
                else None,
                "sampled_frames": len(samples),
                "detected_frames": len(detected_samples),
                "detection_rate": round(len(detected_samples) / len(samples), 6)
                if samples
                else None,
                "frames_inside_calibration_range": len(valid_samples),
                "inside_calibration_range_rate": round(
                    len(valid_samples) / len(samples), 6
                )
                if samples
                else None,
                "visual_inferred_servo_deg_median": round(float(np.median(inferred_values)), 4)
                if inferred_values
                else None,
                "visual_optical_offset_deg_median": round(float(np.median(optical_values)), 5)
                if optical_values
                else None,
                "signed_error_deg_median": round(float(np.median(signed_errors)), 4)
                if signed_errors
                else None,
                "absolute_error_deg": summarize(absolute_errors),
            }
        )

    direction_pairs = []
    by_angle = {}
    for report in hold_reports:
        by_angle.setdefault(report["guide_angle_deg"], []).append(report)
    for angle, reports in sorted(by_angle.items()):
        if len(reports) != 2:
            continue
        first, second = reports
        if (
            first["visual_inferred_servo_deg_median"] is not None
            and second["visual_inferred_servo_deg_median"] is not None
        ):
            visual_repeat = abs(
                first["visual_inferred_servo_deg_median"]
                - second["visual_inferred_servo_deg_median"]
            )
        else:
            visual_repeat = None
        direction_pairs.append(
            {
                "guide_angle_deg": angle,
                "outbound_error_deg": first["signed_error_deg_median"],
                "return_error_deg": second["signed_error_deg_median"],
                "visual_inferred_repeat_difference_deg": round(float(visual_repeat), 4)
                if visual_repeat is not None
                else None,
            }
        )

    repeat_values = [
        pair["visual_inferred_repeat_difference_deg"]
        for pair in direction_pairs
        if pair["visual_inferred_repeat_difference_deg"] is not None
    ]
    total_sampled = sum(report["sampled_frames"] for report in hold_reports)
    total_detected = sum(report["detected_frames"] for report in hold_reports)
    total_inside_calibration = sum(
        report["frames_inside_calibration_range"] for report in hold_reports
    )
    acceptance = {
        "criteria": {
            "detection_rate_min": args.detection_rate_min,
            "mae_deg_max": args.mae_deg_max,
            "p95_deg_max": args.p95_deg_max,
            "max_deg_max": args.max_deg_max,
            "direction_repeat_mean_deg_max": args.direction_repeat_deg_max,
        },
        "detection_rate_pass": bool(
            total_detected / total_sampled >= args.detection_rate_min
        ),
        "mae_pass": bool(all_absolute_errors)
        and bool(np.mean(all_absolute_errors) <= args.mae_deg_max),
        "p95_pass": bool(all_absolute_errors)
        and bool(np.percentile(all_absolute_errors, 95) <= args.p95_deg_max),
        "max_pass": bool(all_absolute_errors)
        and bool(max(all_absolute_errors) <= args.max_deg_max),
        "direction_repeat_pass": bool(repeat_values)
        and bool(np.mean(repeat_values) <= args.direction_repeat_deg_max),
    }
    acceptance["overall_pass"] = all(
        value for key, value in acceptance.items() if key.endswith("_pass")
    )

    report = {
        "video": str(args.video.resolve()),
        "events": str(args.events.resolve()),
        "rtt": str(args.rtt.resolve()),
        "model": str(args.model.resolve()),
        "camera_calibration": str(args.camera_calibration.resolve()),
        "servo_calibration": str(args.servo_calibration.resolve()),
        "confidence_threshold": args.confidence,
        "reject_clipped_targets": args.reject_clipped_targets,
        "source_fps": source_fps,
        "completed_holds": len(holds),
        "total_sampled_frames": total_sampled,
        "total_detected_frames": total_detected,
        "detection_rate": round(total_detected / total_sampled, 6),
        "total_frames_inside_calibration_range": total_inside_calibration,
        "inside_calibration_range_rate": round(
            total_inside_calibration / total_sampled, 6
        ),
        "signed_error_deg": summarize(all_signed_errors),
        "absolute_error_deg": summarize(all_absolute_errors),
        "direction_repeat_difference_deg": summarize(repeat_values),
        "acceptance": acceptance,
        "holds": hold_reports,
        "direction_pairs": direction_pairs,
        "definition": "error = marker-free visual inferred g-equivalent angle - synchronized HX8 encoder pos",
        "metrology_note": "This is an operational camera/servo correspondence test, not external protractor metrology.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "completed_holds",
        "total_sampled_frames",
        "total_detected_frames",
        "detection_rate",
        "total_frames_inside_calibration_range",
        "inside_calibration_range_rate",
        "signed_error_deg",
        "absolute_error_deg",
        "direction_repeat_difference_deg",
        "acceptance",
    )}, ensure_ascii=False, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
