"""Merge one reviewed train-only addition into the foam-center dataset safely."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--session-name", required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--task-log", type=Path, required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--source-fps", type=float, default=50.0)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = args.dataset.resolve()
    selection = args.selection.resolve()
    manifest_path = dataset / "manifest.json"
    selection_manifest_path = selection / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    sessions = manifest.setdefault("augmentation_sessions", {})
    if args.session_name in sessions:
        raise RuntimeError(f"session already merged: {args.session_name}")

    image_target = dataset / "images" / "train"
    label_target = dataset / "labels" / "train"
    records = selected["records"]
    collisions = []
    for record in records:
        for source_dir, target_dir, key in (
            (selection / "images", image_target, "image"),
            (selection / "labels", label_target, "label"),
        ):
            source = source_dir / record[key]
            target = target_dir / record[key]
            if not source.is_file():
                raise FileNotFoundError(source)
            if target.exists():
                collisions.append(str(target))
    if collisions:
        raise FileExistsError(f"refusing to overwrite {len(collisions)} files")

    backup = dataset / f"manifest.before_{args.session_name}.json"
    if backup.exists():
        raise FileExistsError(f"backup already exists: {backup}")
    shutil.copy2(manifest_path, backup)
    for record in records:
        shutil.copy2(selection / "images" / record["image"], image_target / record["image"])
        shutil.copy2(selection / "labels" / record["label"], label_target / record["label"])

    for record in records:
        negative = bool(record.get("is_negative"))
        manifest["records"].append(
            {
                "session": args.session_name,
                "split": "train",
                "frame": int(record["frame"]),
                "time_seconds": round(float(record["frame"]) / args.source_fps, 6),
                "detector_confidence": record.get("confidence"),
                "detector_box_xyxy": None,
                "mask_box_xyxy": None,
                "mask_area": record.get("area"),
                "mask_centroid_xy": record.get("centroid"),
                "contrast": None,
                "status": "accepted_negative" if negative else "accepted",
                "reason": (
                    "guided hard-negative; foam absent; reviewed empty gripper"
                    if negative
                    else "reviewed visible foam contour on single-axis gripper"
                ),
                "image": f"images/train/{record['image']}",
                "label": f"labels/train/{record['label']}",
                "guide_task": int(record["step"]),
                "guide_task_name": record["name"],
            }
        )

    summary = selected["summary"]
    sessions[args.session_name] = {
        "split": "train",
        "source_video": str(args.source_video.resolve()),
        "task_log": str(args.task_log.resolve()),
        "staging_manifest": str(selection_manifest_path),
        "purpose": args.purpose,
        "positive_added": int(summary["positive"]),
        "negative_added": int(summary["negative"]),
        "manual_rejected": int(summary["manual_rejected"]),
        "manual_rejected_frames": selected["manual_rejected_frames"],
        "by_task": summary["by_task"],
        "validation_policy": "train only; existing validation sessions unchanged",
        "selection_rule": selected["selection_rule"],
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)

    train_cache = dataset / "labels" / "train.cache"
    cache_backup = dataset / "labels" / f"train.cache.before_{args.session_name}"
    if train_cache.exists():
        if cache_backup.exists():
            raise FileExistsError(f"cache backup already exists: {cache_backup}")
        train_cache.replace(cache_backup)

    print(json.dumps(sessions[args.session_name], ensure_ascii=False, indent=2))
    print(f"manifest backup: {backup}")


if __name__ == "__main__":
    main()
