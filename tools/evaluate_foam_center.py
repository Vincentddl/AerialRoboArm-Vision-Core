"""Evaluate target-center localization on positive masks and reviewed negatives."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from vision.tracker import PixelToWorldMapper
from vision.servo_angle_calibration import ServoOpticalAngleCalibration


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_DIR / "data" / "training" / "foam_center_v9"
    )
    parser.add_argument(
        "--calibration",
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
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def ground_truth_centroid(label_path: Path, width: int, height: int):
    values = label_path.read_text(encoding="utf-8").split()
    if not values:
        return None
    polygon = np.asarray(values[1:], dtype=np.float64).reshape(-1, 2)
    polygon *= np.asarray([width, height], dtype=np.float64)
    contour = polygon.astype(np.float32).reshape(-1, 1, 2)
    moments = cv2.moments(contour)
    if abs(moments["m00"]) < 1e-9:
        return None
    return np.asarray([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])


def predicted_centroid(result):
    if not len(result.boxes):
        return None, None, "none"
    best = int(result.boxes.conf.argmax().cpu())
    confidence = float(result.boxes.conf[best].cpu())
    box = result.boxes.xyxy[best].cpu().numpy()
    box_center = np.asarray([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    box_area = float((box[2] - box[0]) * (box[3] - box[1]))
    image_area = float(result.orig_shape[0] * result.orig_shape[1])
    box_area_fraction = box_area / image_area
    if result.masks is not None and best < len(result.masks.xy):
        polygon = np.asarray(result.masks.xy[best], dtype=np.float32).reshape(-1, 1, 2)
        moments = cv2.moments(polygon)
        if abs(moments["m00"]) >= 1e-9:
            center = np.asarray(
                [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]
            )
            # Very small masks are quantized; very large close-up masks can
            # miss a side face. In both cases the independently predicted box
            # center is more robust. Use the mask centroid in the normal
            # operating-size range where it gives sub-pixel median accuracy.
            if 0.01 <= box_area_fraction <= 0.13:
                return center, confidence, "mask_centroid"
            return box_center, confidence, "hybrid_box_center"
    return box_center, confidence, "box_center"


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return None
    return {
        "mean": round(float(array.mean()), 4),
        "median": round(float(np.median(array)), 4),
        "p95": round(float(np.percentile(array, 95)), 4),
        "max": round(float(array.max()), 4),
    }


def main():
    args = parse_args()
    image_dir = args.dataset / "images" / args.split
    label_dir = args.dataset / "labels" / args.split
    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        raise RuntimeError(f"no images found in {image_dir}")
    mapper = PixelToWorldMapper(str(args.calibration))
    servo_calibration = ServoOpticalAngleCalibration.from_json(args.servo_calibration)
    model = YOLO(str(args.model))

    positive_count = 0
    positive_detected = 0
    negative_count = 0
    false_positive_count = 0
    pixel_errors = []
    x_errors = []
    y_errors = []
    arm_plane_angle_errors = []
    inferred_servo_angle_errors = []
    servo_range_matched = 0
    prediction_modes = {}
    matched_samples = []
    false_positive_samples = []

    for batch_start in range(0, len(image_paths), args.batch):
        batch_paths = image_paths[batch_start : batch_start + args.batch]
        images = [cv2.imread(str(path)) for path in batch_paths]
        results = model.predict(
            images,
            imgsz=640,
            conf=args.conf,
            max_det=5,
            batch=len(images),
            device=args.device,
            verbose=False,
        )
        for image_path, image, result in zip(batch_paths, images, results):
            height, width = image.shape[:2]
            truth = ground_truth_centroid(label_dir / f"{image_path.stem}.txt", width, height)
            prediction, confidence, mode = predicted_centroid(result)
            prediction_modes[mode] = prediction_modes.get(mode, 0) + 1
            if truth is None:
                negative_count += 1
                false_positive_count += prediction is not None
                if prediction is not None:
                    false_positive_samples.append(
                        {"image": image_path.name, "confidence": round(float(confidence), 6)}
                    )
                continue
            positive_count += 1
            if prediction is None:
                continue
            positive_detected += 1
            difference = prediction - truth
            pixel_errors.append(float(np.linalg.norm(difference)))
            x_errors.append(abs(float(difference[0])))
            y_errors.append(abs(float(difference[1])))
            truth_angles = mapper.to_world((float(truth[0]), float(truth[1])))
            prediction_angles = mapper.to_world((float(prediction[0]), float(prediction[1])))
            truth_arm_plane = mapper.camera_plane_angle_deg(truth_angles)
            prediction_arm_plane = mapper.camera_plane_angle_deg(prediction_angles)
            angle_error = abs(prediction_arm_plane - truth_arm_plane)
            arm_plane_angle_errors.append(angle_error)
            inferred_servo_error = None
            if servo_calibration.is_optical_offset_in_range(
                truth_arm_plane
            ) and servo_calibration.is_optical_offset_in_range(prediction_arm_plane):
                truth_servo = servo_calibration.servo_from_optical_offset(truth_arm_plane)
                prediction_servo = servo_calibration.servo_from_optical_offset(
                    prediction_arm_plane
                )
                inferred_servo_error = abs(prediction_servo - truth_servo)
                inferred_servo_angle_errors.append(inferred_servo_error)
                servo_range_matched += 1
            matched_samples.append(
                {
                    "image": image_path.name,
                    "confidence": round(float(confidence), 6),
                    "mode": mode,
                    "truth_xy": [round(float(value), 3) for value in truth],
                    "prediction_xy": [round(float(value), 3) for value in prediction],
                    "center_error_px": round(float(np.linalg.norm(difference)), 4),
                    "arm_plane_angle_error_deg": round(float(angle_error), 4),
                    "inferred_servo_angle_error_deg": (
                        round(float(inferred_servo_error), 4)
                        if inferred_servo_error is not None
                        else None
                    ),
                }
            )

    report = {
        "model": str(args.model),
        "dataset": str(args.dataset),
        "split": args.split,
        "confidence_threshold": args.conf,
        "positive_images": positive_count,
        "positive_detected": positive_detected,
        "positive_recall": round(positive_detected / positive_count, 6) if positive_count else None,
        "negative_images": negative_count,
        "negative_false_positives": false_positive_count,
        "negative_false_positive_rate": round(false_positive_count / negative_count, 6)
        if negative_count
        else None,
        "prediction_modes": prediction_modes,
        "center_error_px": summarize(pixel_errors),
        "center_x_absolute_error_px": summarize(x_errors),
        "center_y_absolute_error_px": summarize(y_errors),
        "arm_plane_angle_absolute_error_deg": summarize(arm_plane_angle_errors),
        "inferred_servo_angle_absolute_error_deg": summarize(
            inferred_servo_angle_errors
        ),
        "inferred_servo_angle_samples_in_v3_range": servo_range_matched,
        "servo_calibration": str(args.servo_calibration),
        "largest_center_errors": sorted(
            matched_samples, key=lambda sample: sample["center_error_px"], reverse=True
        )[:20],
        "negative_false_positive_samples": false_positive_samples,
        "note": (
            "Ground truth is reviewed, GrabCut-refined visible-foam contour centroid; "
            "not external metrology. Inferred servo error includes only samples whose "
            "truth and predicted optical offsets are inside the V3 LUT range."
        ),
    }
    output = args.output or PROJECT_DIR / "outputs" / f"center_eval_{args.model.stem}_{args.split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
