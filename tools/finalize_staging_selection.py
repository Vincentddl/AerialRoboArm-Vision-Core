"""Finalize reviewed staged images without modifying the source staging set."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reject-frames", default="")
    parser.add_argument("--selection-note", required=True)
    return parser.parse_args()


def mask_metrics(image_path: Path, label_path: Path):
    image = cv2.imread(str(image_path))
    text = label_path.read_text(encoding="utf-8").strip()
    if image is None or not text:
        return None
    values = [float(value) for value in text.split()]
    height, width = image.shape[:2]
    polygon = np.asarray(values[1:], dtype=np.float32).reshape(-1, 2)
    polygon[:, 0] *= width
    polygon[:, 1] *= height
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 255)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixels = gray[mask > 0]
    if not pixels.size:
        return None
    return {
        "dark80": round(float(np.mean(pixels <= 80)), 6),
        "median_gray": round(float(np.median(pixels)), 3),
        "label_pixel_area": int(pixels.size),
    }


def main():
    args = parse_args()
    staging = args.staging.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    reject_frames = {
        int(value.strip())
        for value in args.reject_frames.split(",")
        if value.strip()
    }
    (output / "images").mkdir(parents=True)
    (output / "labels").mkdir(parents=True)
    (output / "manual_rejected").mkdir(parents=True)

    source = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
    selected = []
    rejected = []
    for record in source["records"]:
        if not record["status"].startswith("accepted"):
            continue
        frame = int(record["frame"])
        image_path = staging / "images" / record["image"]
        label_path = staging / "labels" / record["label"]
        if frame in reject_frames and not record.get("is_negative"):
            rejected.append({**record, "manual_reject_reason": "wrong gripper-part mask"})
            debug_path = staging / "debug" / "accepted" / f"{Path(record['image']).stem}.jpg"
            if debug_path.exists():
                shutil.copy2(debug_path, output / "manual_rejected" / debug_path.name)
            continue
        shutil.copy2(image_path, output / "images" / image_path.name)
        shutil.copy2(label_path, output / "labels" / label_path.name)
        metrics = None if record.get("is_negative") else mask_metrics(image_path, label_path)
        selected.append({**record, **(metrics or {})})

    positives = [record for record in selected if not record.get("is_negative")]
    negatives = [record for record in selected if record.get("is_negative")]
    by_task = Counter(str(record["step"]) for record in selected)
    manifest = {
        "source_manifest": str((staging / "manifest.json").resolve()),
        "selection_rule": args.selection_note,
        "manual_rejected_frames": sorted(reject_frames),
        "summary": {
            "positive": len(positives),
            "negative": len(negatives),
            "manual_rejected": len(rejected),
            "by_task": dict(sorted(by_task.items(), key=lambda item: int(item[0]))),
        },
        "records": selected,
        "rejected_records": rejected,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
    print(f"final selection: {output}")


if __name__ == "__main__":
    main()
