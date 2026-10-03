"""Convert reviewed V9 segmentation contours into a V10 oriented-box dataset.

The train/validation split is copied exactly from V9. Positive segmentation
polygons become minimum-area rotated rectangles; empty negative labels stay
empty. The V9 dataset is read-only and is never modified by this tool.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_DIR / "data" / "training" / "foam_center_v9",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "data" / "training" / "foam_center_v10_obb",
    )
    parser.add_argument("--review-images", type=int, default=80)
    return parser.parse_args()


def ordered_corners(points: np.ndarray) -> np.ndarray:
    """Return four corners clockwise, beginning at the top-most/left-most point."""
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    # In image coordinates, increasing atan2 order follows clockwise motion.
    start = min(range(4), key=lambda i: (ordered[i, 1], ordered[i, 0]))
    return np.roll(ordered, -start, axis=0)


def convert_label(label_path: Path, width: int, height: int):
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return "", None, None
    values = [float(value) for value in text.split()]
    if int(values[0]) != 0 or (len(values) - 1) < 6 or (len(values) - 1) % 2:
        raise ValueError(f"invalid V9 segmentation label: {label_path}")
    polygon = np.asarray(values[1:], dtype=np.float32).reshape(-1, 2)
    polygon_px = polygon * np.asarray([width, height], dtype=np.float32)
    rect = cv2.minAreaRect(polygon_px.reshape(-1, 1, 2))
    corners_px = ordered_corners(cv2.boxPoints(rect))
    corners = corners_px / np.asarray([width, height], dtype=np.float32)
    corners = np.clip(corners, 0.0, 1.0)
    label = "0 " + " ".join(f"{value:.8f}" for value in corners.reshape(-1)) + "\n"
    return label, polygon_px, corners_px


def draw_review(image, polygon, corners, name):
    canvas = image.copy()
    if polygon is None:
        cv2.putText(canvas, "NEGATIVE", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return canvas
    poly_i = np.rint(polygon).astype(np.int32).reshape(-1, 1, 2)
    corners_i = np.rint(corners).astype(np.int32).reshape(-1, 1, 2)
    center = corners.mean(axis=0)
    cv2.polylines(canvas, [poly_i], True, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.polylines(canvas, [corners_i], True, (0, 200, 255), 2, cv2.LINE_AA)
    cv2.drawMarker(
        canvas,
        tuple(np.rint(center).astype(int)),
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=15,
        thickness=2,
    )
    cv2.putText(canvas, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
    return canvas


def make_contact_sheet(images, output_path: Path, columns=5, cell_size=(320, 240)):
    if not images:
        return
    rows = (len(images) + columns - 1) // columns
    sheet = np.zeros((rows * cell_size[1], columns * cell_size[0], 3), dtype=np.uint8)
    for index, image in enumerate(images):
        resized = cv2.resize(image, cell_size, interpolation=cv2.INTER_AREA)
        row, column = divmod(index, columns)
        y1, x1 = row * cell_size[1], column * cell_size[0]
        sheet[y1 : y1 + cell_size[1], x1 : x1 + cell_size[0]] = resized
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists; refusing to overwrite: {args.output}")

    manifest = {
        "name": "foam_center_v10_obb",
        "source_dataset": str(args.source),
        "conversion": "minimum-area rotated rectangle around each reviewed V9 visible-foam polygon",
        "split_policy": "identical to V9; alternate_background remains validation-only",
        "label_format": "class x1 y1 x2 y2 x3 y3 x4 y4; normalized coordinates",
        "splits": {},
    }

    for split in ("train", "val"):
        source_images = args.source / "images" / split
        source_labels = args.source / "labels" / split
        output_images = args.output / "images" / split
        output_labels = args.output / "labels" / split
        output_images.mkdir(parents=True)
        output_labels.mkdir(parents=True)
        image_paths = sorted(source_images.glob("*.jpg"))
        review_stride = max(1, len(image_paths) // max(args.review_images, 1))
        review = []
        positive = 0
        negative = 0

        for index, image_path in enumerate(image_paths):
            label_path = source_labels / f"{image_path.stem}.txt"
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"failed to read {image_path}")
            height, width = image.shape[:2]
            converted, polygon, corners = convert_label(label_path, width, height)
            shutil.copy2(image_path, output_images / image_path.name)
            (output_labels / label_path.name).write_text(converted, encoding="utf-8")
            if converted:
                positive += 1
            else:
                negative += 1
            if index % review_stride == 0 and len(review) < args.review_images:
                review.append(draw_review(image, polygon, corners, image_path.name))

        make_contact_sheet(review, args.output / "review" / f"{split}_obb_contact_sheet.jpg")
        manifest["splits"][split] = {
            "images": len(image_paths),
            "positive": positive,
            "negative": negative,
        }

    yaml_text = (
        f"path: {args.output.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: foam_board\n"
    )
    (args.output / "foam_center_v10_obb.yaml").write_text(yaml_text, encoding="utf-8")
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"created: {args.output}")


if __name__ == "__main__":
    main()
