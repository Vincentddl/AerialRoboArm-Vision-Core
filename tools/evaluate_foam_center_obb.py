"""Evaluate a model center against V10 rotated-box geometric-center labels.

OBB models use their rotated-box center. Segmentation/detection baselines use
the same hybrid mask/axis-aligned-box center policy as the current runtime, so
V9 and V10 can be compared against exactly the same target definition.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from vision.tracker import PixelToWorldMapper


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_DIR / "data" / "training" / "foam_center_v10_obb",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=PROJECT_DIR / "configs" / "camera_2p1mm_640x480_fisheye.json",
    )
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def truth_center(label_path: Path, width: int, height: int):
    values = label_path.read_text(encoding="utf-8").split()
    if not values:
        return None
    corners = np.asarray(values[1:], dtype=np.float64).reshape(4, 2)
    corners *= np.asarray([width, height], dtype=np.float64)
    return corners.mean(axis=0)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    return {
        "mean": round(float(values.mean()), 4),
        "median": round(float(np.median(values)), 4),
        "p95": round(float(np.percentile(values, 95)), 4),
        "max": round(float(values.max()), 4),
    }


def prediction_center(result):
    if result.obb is not None and len(result.obb):
        best = int(result.obb.conf.argmax().cpu())
        return (
            result.obb.xywhr[best, :2].cpu().numpy().astype(float),
            float(result.obb.conf[best].cpu()),
            "obb_center",
        )
    if result.boxes is None or not len(result.boxes):
        return None, None, "none"
    best = int(result.boxes.conf.argmax().cpu())
    confidence = float(result.boxes.conf[best].cpu())
    box = result.boxes.xyxy[best].cpu().numpy().astype(float)
    box_center = np.asarray([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    area_fraction = ((box[2] - box[0]) * (box[3] - box[1])) / float(
        result.orig_shape[0] * result.orig_shape[1]
    )
    if result.masks is not None and best < len(result.masks.xy):
        polygon = np.asarray(result.masks.xy[best], dtype=np.float32).reshape(-1, 1, 2)
        moments = cv2.moments(polygon)
        if abs(moments["m00"]) >= 1e-9 and 0.01 <= area_fraction <= 0.13:
            center = np.asarray(
                [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]
            )
            return center, confidence, "mask_centroid"
        return box_center, confidence, "hybrid_box_center"
    return box_center, confidence, "box_center"


def main():
    args = parse_args()
    image_paths = sorted((args.dataset / "images" / args.split).glob("*.jpg"))
    label_dir = args.dataset / "labels" / args.split
    model = YOLO(str(args.model))
    mapper = PixelToWorldMapper(str(args.calibration))
    positives = detections = negatives = false_positives = 0
    pixel_errors = []
    angle_errors = []
    largest = []
    prediction_modes = {}

    for start in range(0, len(image_paths), args.batch):
        paths = image_paths[start : start + args.batch]
        images = [cv2.imread(str(path)) for path in paths]
        results = model.predict(
            images,
            imgsz=640,
            conf=args.conf,
            max_det=5,
            batch=len(images),
            device=args.device,
            verbose=False,
        )
        for path, image, result in zip(paths, images, results):
            height, width = image.shape[:2]
            truth = truth_center(label_dir / f"{path.stem}.txt", width, height)
            prediction, confidence, mode = prediction_center(result)
            prediction_modes[mode] = prediction_modes.get(mode, 0) + 1
            if truth is None:
                negatives += 1
                if prediction is not None:
                    false_positives += 1
                continue
            positives += 1
            if prediction is None:
                continue
            detections += 1
            error = float(np.linalg.norm(prediction - truth))
            truth_angles = mapper.to_world(tuple(truth))
            pred_angles = mapper.to_world(tuple(prediction))
            angle_error = abs(
                mapper.camera_plane_angle_deg(pred_angles)
                - mapper.camera_plane_angle_deg(truth_angles)
            )
            pixel_errors.append(error)
            angle_errors.append(angle_error)
            largest.append(
                {
                    "image": path.name,
                    "confidence": round(confidence, 6),
                    "center_error_px": round(error, 4),
                    "arm_plane_angle_error_deg": round(float(angle_error), 4),
                }
            )

    report = {
        "model": str(args.model),
        "dataset": str(args.dataset),
        "split": args.split,
        "confidence_threshold": args.conf,
        "positive_images": positives,
        "positive_detected": detections,
        "positive_recall": round(detections / positives, 6) if positives else None,
        "negative_images": negatives,
        "negative_false_positives": false_positives,
        "prediction_modes": prediction_modes,
        "center_error_px": summarize(pixel_errors),
        "arm_plane_angle_absolute_error_deg": summarize(angle_errors),
        "largest_center_errors": sorted(
            largest, key=lambda item: item["center_error_px"], reverse=True
        )[:20],
        "note": (
            "Ground truth is the center of the minimum-area rectangle fitted to each "
            "reviewed V9 visible-foam contour; it is not external metrology."
        ),
    }
    output = args.output or PROJECT_DIR / "outputs" / f"center_eval_{args.model.stem}_{args.split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
