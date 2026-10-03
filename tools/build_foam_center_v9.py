"""Build a reviewed-candidate segmentation dataset from the no-marker videos.

The V8 detector is used only to propose a coarse region. OpenCV GrabCut then
separates the visible black foam target from its surroundings. The exported
label is the refined contour, so the training target center is the contour
centroid rather than the center of the V8 bounding box.

This script deliberately skips uncertain frames instead of turning detector
misses into negative examples. It also keeps complete recording sessions in a
single split to prevent adjacent video frames leaking across train/validation.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_DIR / "models" / "foam_board_2p1mm_v8.pt"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "training" / "foam_center_v9"
OLD_DATASET = PROJECT_DIR / "data" / "training" / "foam_board_v8"

VIDEO_SESSIONS = (
    (
        "normal_slow",
        "train",
        PROJECT_DIR / "data" / "raw" / "target_detection_no_red_normal_slow_20260806",
    ),
    (
        "normal_medium",
        "train",
        PROJECT_DIR / "data" / "raw" / "target_detection_no_red_normal_medium_20260806",
    ),
    (
        "normal_fast",
        "train",
        PROJECT_DIR / "data" / "raw" / "target_detection_no_red_normal_fast_20260806",
    ),
    (
        "dim_light",
        "train",
        PROJECT_DIR / "data" / "raw" / "target_detection_no_red_dim_light_20260806",
    ),
    (
        "alternate_background",
        "val",
        PROJECT_DIR / "data" / "raw" / "target_detection_no_red_alt_background_20260806",
    ),
)


@dataclass
class SampleRecord:
    session: str
    split: str
    frame: int
    time_seconds: float
    detector_confidence: float | None
    detector_box_xyxy: list[float] | None
    mask_box_xyxy: list[int] | None
    mask_area: int | None
    mask_centroid_xy: list[float] | None
    contrast: float | None
    status: str
    reason: str
    image: str | None = None
    label: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--det-conf", type=float, default=0.10)
    parser.add_argument("--min-accepted-conf", type=float, default=0.18)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def find_single_video(directory: Path) -> Path:
    videos = sorted(directory.glob("*.mkv"))
    if len(videos) != 1:
        raise RuntimeError(f"expected exactly one MKV in {directory}, found {len(videos)}")
    return videos[0]


def prepare_output(output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    for relative in (
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
        "debug/train",
        "debug/val",
        "review/accepted",
        "review/rejected",
    ):
        (output / relative).mkdir(parents=True, exist_ok=True)


def read_sampled_frames(video: Path, sample_fps: float):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    stride = max(1, round(source_fps / sample_fps))
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_index % stride == 0:
                yield frame_index, frame_index / source_fps, frame
            frame_index += 1
    finally:
        cap.release()


def expanded_rectangle(box: np.ndarray, width: int, height: int, margin: float = 0.12):
    x1, y1, x2, y2 = map(float, box)
    box_width = x2 - x1
    box_height = y2 - y1
    left = max(0, math.floor(x1 - margin * box_width))
    top = max(0, math.floor(y1 - margin * box_height))
    right = min(width - 1, math.ceil(x2 + margin * box_width))
    bottom = min(height - 1, math.ceil(y2 + margin * box_height))
    return left, top, right, bottom


def refine_mask(frame: np.ndarray, detector_box: np.ndarray):
    height, width = frame.shape[:2]
    left, top, right, bottom = expanded_rectangle(detector_box, width, height)
    if right - left < 8 or bottom - top < 8:
        return None, "proposal_too_small", None

    # GrabCut runtime scales with pixel count. Run it only around the detector
    # proposal and translate the accepted contour back to full-frame pixels.
    roi = frame[top : bottom + 1, left : right + 1]
    roi_height, roi_width = roi.shape[:2]
    if roi_width < 8 or roi_height < 8:
        return None, "proposal_too_small", None
    grabcut_mask = np.zeros((roi_height, roi_width), np.uint8)
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(
            roi,
            grabcut_mask,
            (1, 1, roi_width - 2, roi_height - 2),
            background_model,
            foreground_model,
            5,
            cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error:
        return None, "grabcut_failed", None

    foreground = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    component_count, components, stats, centroids = cv2.connectedComponentsWithStats(foreground, 8)
    proposal_center_global = np.array(
        [(detector_box[0] + detector_box[2]) / 2, (detector_box[1] + detector_box[3]) / 2]
    )
    proposal_center = proposal_center_global - np.array([left, top])
    proposal_area = max(1.0, float((detector_box[2] - detector_box[0]) * (detector_box[3] - detector_box[1])))

    candidates = []
    for component_id in range(1, component_count):
        x, y, w, h, area = map(int, stats[component_id])
        if area < 80 or w < 6 or h < 6:
            continue
        center = centroids[component_id]
        contains_proposal_center = x <= proposal_center[0] < x + w and y <= proposal_center[1] < y + h
        distance = float(np.linalg.norm(center - proposal_center))
        score = (2.0 if contains_proposal_center else 0.0) + area / proposal_area - 0.002 * distance
        candidates.append((score, component_id))
    if not candidates:
        return None, "no_foreground_component", None

    component_id = max(candidates)[1]
    mask = (components == component_id).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "no_contour", None
    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    if contour_area < 80:
        return None, "contour_too_small", None

    x, y, w, h = cv2.boundingRect(contour)
    if x <= 0 or y <= 0 or x + w >= roi_width or y + h >= roi_height:
        return None, "target_touches_proposal_edge", None
    area_ratio = contour_area / proposal_area
    if not 0.16 <= area_ratio <= 1.35:
        return None, "implausible_mask_area", None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    object_pixels = gray[mask > 0]
    ring = np.full_like(mask, 255)
    ring[mask > 0] = 0
    ring_pixels = gray[ring > 0]
    if not len(object_pixels) or not len(ring_pixels):
        return None, "missing_contrast_region", None
    contrast = float(np.median(ring_pixels) - np.median(object_pixels))
    if contrast < 8.0:
        return None, "low_object_contrast", {"contrast": contrast}

    moments = cv2.moments(contour)
    if abs(moments["m00"]) < 1e-6:
        return None, "zero_area_moment", None
    centroid = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
    centroid_global = centroid + np.array([left, top])
    if not (
        detector_box[0] <= centroid_global[0] <= detector_box[2]
        and detector_box[1] <= centroid_global[1] <= detector_box[3]
    ):
        return None, "centroid_outside_proposal", {"contrast": contrast}

    perimeter = cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, max(1.0, 0.004 * perimeter), True).reshape(-1, 2)
    if len(polygon) < 4:
        return None, "polygon_too_short", {"contrast": contrast}
    polygon_global = polygon + np.array([left, top])
    contour_global = contour + np.array([[[left, top]]], dtype=contour.dtype)
    mask_global = np.zeros((height, width), np.uint8)
    mask_global[top : bottom + 1, left : right + 1] = mask
    return {
        "mask": mask_global,
        "contour": contour_global,
        "polygon": polygon_global,
        "centroid": centroid_global,
        "box": np.array([x + left, y + top, x + w + left, y + h + top]),
        "area": int(np.count_nonzero(mask)),
        "contrast": contrast,
    }, "accepted", None


def segmentation_label(polygon: np.ndarray, width: int, height: int) -> str:
    points = []
    for x, y in polygon:
        points.extend((np.clip(x / width, 0, 1), np.clip(y / height, 0, 1)))
    return "0 " + " ".join(f"{value:.8f}" for value in points) + "\n"


def draw_debug(frame, detector_box, confidence, refinement, status, reason):
    debug = frame.copy()
    if detector_box is not None:
        x1, y1, x2, y2 = map(int, detector_box)
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 255), 2)
    if refinement is not None:
        mask = refinement["mask"] > 0
        debug[mask] = (0.55 * debug[mask] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
        center = tuple(map(round, refinement["centroid"]))
        cv2.circle(debug, center, 6, (0, 0, 255), -1)
    color = (0, 255, 0) if status == "accepted" else (0, 0, 255)
    message = f"{status} {reason}"
    if confidence is not None:
        message += f" conf={confidence:.2f}"
    cv2.putText(debug, message, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return debug


def write_contact_sheets(debug_paths, destination: Path, prefix: str) -> None:
    cell_width, cell_height = 160, 120
    columns, rows = 8, 6
    page_size = columns * rows
    destination.mkdir(parents=True, exist_ok=True)
    for page_index in range(0, len(debug_paths), page_size):
        canvas = np.zeros((rows * cell_height, columns * cell_width, 3), np.uint8)
        for offset, image_path in enumerate(debug_paths[page_index : page_index + page_size]):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            image = cv2.resize(image, (cell_width, cell_height), interpolation=cv2.INTER_AREA)
            row, column = divmod(offset, columns)
            canvas[row * cell_height : (row + 1) * cell_height, column * cell_width : (column + 1) * cell_width] = image
        output_path = destination / f"{prefix}_{page_index // page_size + 1:02d}.jpg"
        cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])


def copy_reviewed_negatives(output: Path):
    copied = {"train": 0, "val": 0}
    for split in ("train", "val"):
        label_dir = OLD_DATASET / "labels" / split
        for label_path in sorted(label_dir.glob("*.txt")):
            if label_path.read_text(encoding="utf-8").strip():
                continue
            source_image = OLD_DATASET / "images" / split / f"{label_path.stem}.jpg"
            if not source_image.exists():
                continue
            stem = f"legacy_negative_{label_path.stem}"
            shutil.copy2(source_image, output / "images" / split / f"{stem}.jpg")
            (output / "labels" / split / f"{stem}.txt").write_text("", encoding="utf-8")
            copied[split] += 1
    return copied


def process_session(model, name, split, video, args, output):
    accepted_paths = []
    rejected_paths = []
    records = []
    samples = list(read_sampled_frames(video, args.sample_fps))
    print(f"{name}: sampled {len(samples)} frames from {video.name}", flush=True)

    for batch_start in range(0, len(samples), args.batch):
        batch = samples[batch_start : batch_start + args.batch]
        frames = [item[2] for item in batch]
        predictions = model.predict(
            frames,
            imgsz=640,
            conf=args.det_conf,
            max_det=5,
            batch=min(args.batch, len(frames)),
            device=args.device,
            verbose=False,
        )
        for (frame_index, time_seconds, frame), prediction in zip(batch, predictions):
            detector_box = None
            confidence = None
            refinement = None
            status = "rejected"
            reason = "no_detector_proposal"
            extra = None
            if len(prediction.boxes):
                best_index = int(prediction.boxes.conf.argmax().cpu())
                confidence = float(prediction.boxes.conf[best_index].cpu())
                detector_box = prediction.boxes.xyxy[best_index].cpu().numpy()
                if confidence < args.min_accepted_conf:
                    reason = "detector_confidence_too_low"
                else:
                    refinement, reason, extra = refine_mask(frame, detector_box)
                    if refinement is not None:
                        # Full-sheet review showed that the dim-light proposal
                        # model occasionally locks onto a large clothing/arm
                        # region. The real foam target never exceeded this
                        # conservative area in that recording.
                        if name == "dim_light" and refinement["area"] > 40_000:
                            reason = "review_rule_dim_large_mask"
                            refinement = None
                        else:
                            status = "accepted"

            stem = f"{name}_{frame_index:06d}"
            debug = draw_debug(frame, detector_box, confidence, refinement, status, reason)
            debug_path = output / "debug" / split / f"{stem}.jpg"
            cv2.imwrite(str(debug_path), debug, [cv2.IMWRITE_JPEG_QUALITY, 92])

            image_relative = None
            label_relative = None
            if status == "accepted":
                image_relative = Path("images") / split / f"{stem}.jpg"
                label_relative = Path("labels") / split / f"{stem}.txt"
                cv2.imwrite(str(output / image_relative), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                height, width = frame.shape[:2]
                (output / label_relative).write_text(
                    segmentation_label(refinement["polygon"], width, height), encoding="utf-8"
                )
                accepted_paths.append(debug_path)
            else:
                rejected_paths.append(debug_path)

            record = SampleRecord(
                session=name,
                split=split,
                frame=frame_index,
                time_seconds=round(time_seconds, 6),
                detector_confidence=round(confidence, 6) if confidence is not None else None,
                detector_box_xyxy=[round(float(value), 3) for value in detector_box] if detector_box is not None else None,
                mask_box_xyxy=[int(value) for value in refinement["box"]] if refinement is not None else None,
                mask_area=refinement["area"] if refinement is not None else None,
                mask_centroid_xy=[round(float(value), 3) for value in refinement["centroid"]]
                if refinement is not None
                else None,
                contrast=round(float(refinement["contrast"]), 3)
                if refinement is not None
                else round(float(extra["contrast"]), 3)
                if extra and "contrast" in extra
                else None,
                status=status,
                reason=reason,
                image=image_relative.as_posix() if image_relative else None,
                label=label_relative.as_posix() if label_relative else None,
            )
            records.append(record)
        print(
            f"{name}: processed {min(batch_start + len(batch), len(samples))}/{len(samples)}",
            flush=True,
        )
    return records, accepted_paths, rejected_paths


def main() -> None:
    args = parse_args()
    if args.sample_fps <= 0:
        raise ValueError("--sample-fps must be positive")
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    sessions = [(name, split, find_single_video(directory)) for name, split, directory in VIDEO_SESSIONS]
    prepare_output(args.output)
    model = YOLO(str(args.model))

    all_records = []
    session_summary = {}
    for name, split, video in sessions:
        records, accepted_paths, rejected_paths = process_session(model, name, split, video, args, args.output)
        all_records.extend(records)
        write_contact_sheets(accepted_paths, args.output / "review" / "accepted", name)
        write_contact_sheets(rejected_paths, args.output / "review" / "rejected", name)
        session_summary[name] = {
            "split": split,
            "video": str(video),
            "sampled": len(records),
            "accepted": sum(record.status == "accepted" for record in records),
            "rejected": sum(record.status != "accepted" for record in records),
        }

    negative_counts = copy_reviewed_negatives(args.output)
    yaml_path = args.output / "foam_center_v9.yaml"
    yaml_path.write_text(
        f"path: {args.output.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: foam_board\n",
        encoding="utf-8",
    )
    manifest = {
        "annotation_definition": "visible foam contour; center is the contour area centroid",
        "proposal_model": str(args.model),
        "proposal_role": "coarse ROI only",
        "refinement": "OpenCV GrabCut plus connected-component and contrast checks",
        "sample_fps": args.sample_fps,
        "detector_confidence": args.det_conf,
        "minimum_accepted_confidence": args.min_accepted_conf,
        "split_policy": "complete recording sessions; alternate_background is validation only",
        "sessions": session_summary,
        "reviewed_legacy_negatives": negative_counts,
        "records": [asdict(record) for record in all_records],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    accepted_train = sum(record.status == "accepted" and record.split == "train" for record in all_records)
    accepted_val = sum(record.status == "accepted" and record.split == "val" for record in all_records)
    print(f"dataset: {args.output}")
    print(f"accepted positive train/val: {accepted_train}/{accepted_val}")
    print(f"reviewed negative train/val: {negative_counts['train']}/{negative_counts['val']}")
    print(f"yaml: {yaml_path}")
    print("Review every sheet in review/accepted before training.")


if __name__ == "__main__":
    main()
