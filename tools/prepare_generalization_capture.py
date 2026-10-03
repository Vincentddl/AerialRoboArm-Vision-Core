"""Prepare reviewed-candidate segmentation samples from one guided recording.

The task log produced by ``capture_ffmpeg_manual.py --guide-preset
generalization`` defines positive tasks 1-8 and the hard-negative task 9.
Positive proposals are collected from both the deployed V9 segmenter and the
older V8 detector.  A dark, mainly elongated proposal is refined with GrabCut.
The script only creates a staging directory and contact sheets; it never edits
the official training set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from build_foam_center_v9 import draw_debug, refine_mask, segmentation_label


PROJECT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_PERIOD_SECONDS = {
    1: 2.0,
    2: 2.5,
    3: 2.0,
    4: 1.5,
    5: 0.5,
    6: 0.5,
    7: 1.0,
    8: 0.5,
    9: 1.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prefix",
        default="generalization_gripper",
        help="Safe filename prefix for staged images and labels.",
    )
    parser.add_argument(
        "--v9-model",
        type=Path,
        default=PROJECT_DIR / "models" / "foam_center_v9_seg.pt",
    )
    parser.add_argument(
        "--v8-model",
        type=Path,
        default=PROJECT_DIR / "models" / "foam_board_2p1mm_v8.pt",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=32)
    return parser.parse_args()


def read_tasks(path: Path):
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    active = {}
    completed = {}
    for event in events:
        step = event.get("step")
        if step is None:
            continue
        if event["event"] in {"guide_started", "task_started", "step_back"}:
            active[step] = event
        elif event["event"] == "task_completed" and step in active:
            name = event["task_name"]
            completed[step] = {
                "step": step,
                "name": name,
                "is_negative": any(
                    marker in name.upper()
                    for marker in ("NO TARGET", "HARD NEGATIVE", "EMPTY BACKGROUND")
                ),
                "start": active[step]["frame_index"],
                "end": event["frame_index"],
            }
    if not completed:
        raise RuntimeError("the task log contains no completed tasks")
    expected = set(range(1, max(completed) + 1))
    missing = sorted(expected - set(completed))
    if missing:
        raise RuntimeError(f"guide tasks were not completed: {missing}")
    return [completed[step] for step in sorted(completed)]


def sampled_indices(tasks, fps):
    result = []
    for task in tasks:
        period = max(1, round(SAMPLE_PERIOD_SECONDS[task["step"]] * fps))
        start = task["start"]
        # The operator needs a moment to remove the positive target at the
        # beginning of task 9.  Never label that transition as negative.
        if task["is_negative"]:
            start += round(2.0 * fps)
        for frame_index in range(start, task["end"], period):
            result.append((frame_index, task))
    return result


def read_frames(video: Path, requested):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    requested_by_index = {index: task for index, task in requested}
    frames, metadata = [], []
    frame_index = 0
    try:
        while requested_by_index:
            ok, frame = cap.read()
            if not ok:
                break
            task = requested_by_index.pop(frame_index, None)
            if task is not None:
                frames.append(frame)
                metadata.append({"frame": frame_index, **task})
            frame_index += 1
    finally:
        cap.release()
    if requested_by_index:
        raise RuntimeError(f"video ended before {len(requested_by_index)} requested frames")
    return frames, metadata


def proposal_quality(frame, box, confidence, step):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = np.asarray(box, dtype=float)
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
    box_width, box_height = x2 - x1, y2 - y1
    if box_width < 7 or box_height < 14:
        return None
    aspect = box_height / max(1, box_width)
    minimum_aspect = 0.70 if step in {7, 8} else 1.10
    if aspect < minimum_aspect:
        return None
    area_fraction = box_width * box_height / float(width * height)
    if not 0.0003 <= area_fraction <= 0.28:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inside = gray[y1:y2, x1:x2]
    if not inside.size:
        return None
    margin_x, margin_y = max(5, box_width // 3), max(5, box_height // 3)
    rx1, ry1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
    rx2, ry2 = min(width, x2 + margin_x), min(height, y2 + margin_y)
    ring = gray[ry1:ry2, rx1:rx2].copy()
    mask = np.ones(ring.shape, dtype=bool)
    mask[y1 - ry1 : y2 - ry1, x1 - rx1 : x2 - rx1] = False
    outside = ring[mask]
    if not outside.size:
        return None
    inside_median = float(np.median(inside))
    dark_gain = float(np.median(outside) - inside_median)
    if inside_median > 125 or dark_gain < 8:
        return None
    score = float(confidence) + 0.12 * min(aspect, 3.0) + 0.004 * min(dark_gain, 60.0)
    return score


def choose_proposal(frame, predictions, step):
    candidates = []
    for source, prediction in predictions:
        for box, confidence in zip(prediction.boxes.xyxy.cpu().numpy(), prediction.boxes.conf.cpu().numpy()):
            quality = proposal_quality(frame, box, float(confidence), step)
            if quality is not None:
                candidates.append((quality, source, float(confidence), box))
    return max(candidates, default=None, key=lambda item: item[0])


def contact_sheets(paths, output_dir, prefix, columns=6, rows=5):
    output_dir.mkdir(parents=True, exist_ok=True)
    page_size = columns * rows
    for page_start in range(0, len(paths), page_size):
        canvas = np.zeros((rows * 240, columns * 320, 3), np.uint8)
        for offset, path in enumerate(paths[page_start : page_start + page_size]):
            image = cv2.imread(str(path))
            if image is None:
                continue
            image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
            row, column = divmod(offset, columns)
            canvas[row * 240 : (row + 1) * 240, column * 320 : (column + 1) * 320] = image
        cv2.imwrite(
            str(output_dir / f"{prefix}_{page_start // page_size + 1:02d}.jpg"),
            canvas,
            [cv2.IMWRITE_JPEG_QUALITY, 94],
        )


def main():
    args = parse_args()
    args.video = args.video.resolve()
    tasks_path = (args.tasks or args.video.with_suffix(".tasks.jsonl")).resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite staging directory: {output}")
    for relative in ("images", "labels", "debug/accepted", "debug/rejected", "review"):
        (output / relative).mkdir(parents=True, exist_ok=True)

    tasks = read_tasks(tasks_path)
    cap = cv2.VideoCapture(str(args.video))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    requested = sampled_indices(tasks, fps)
    frames, metadata = read_frames(args.video, requested)
    print(f"sampled {len(frames)} frames", flush=True)

    models = {
        "v9": YOLO(str(args.v9_model)),
        "v8": YOLO(str(args.v8_model)),
    }
    predictions = {name: [] for name in models}
    for name, model in models.items():
        for start in range(0, len(frames), args.batch):
            batch = frames[start : start + args.batch]
            predictions[name].extend(
                model.predict(
                    batch,
                    imgsz=640,
                    conf=0.04,
                    max_det=5,
                    batch=len(batch),
                    device=args.device,
                    verbose=False,
                )
            )
            print(f"{name}: {min(start + len(batch), len(frames))}/{len(frames)}", flush=True)

    records, accepted_debug, rejected_debug = [], [], []
    for index, (frame, meta) in enumerate(zip(frames, metadata)):
        step = meta["step"]
        prefix = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in args.prefix
        ).strip("_")
        stem = f"{prefix}_s{step:02d}_f{meta['frame']:06d}"
        image_path = output / "images" / f"{stem}.jpg"
        label_path = output / "labels" / f"{stem}.txt"
        if meta["is_negative"]:
            cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            label_path.write_text("", encoding="utf-8")
            debug = frame.copy()
            cv2.putText(debug, "NEGATIVE: no foam target", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            debug_path = output / "debug" / "accepted" / f"{stem}.jpg"
            cv2.imwrite(str(debug_path), debug, [cv2.IMWRITE_JPEG_QUALITY, 92])
            accepted_debug.append(debug_path)
            records.append({**meta, "status": "accepted_negative", "image": image_path.name, "label": label_path.name})
            continue

        proposal = choose_proposal(
            frame,
            [(name, predictions[name][index]) for name in models],
            step,
        )
        refinement = None
        if proposal is None:
            status, reason = "rejected", "no_dark_elongated_proposal"
            source, confidence, box = None, None, None
        else:
            _, source, confidence, box = proposal
            refinement, reason, _ = refine_mask(frame, box)
            status = "accepted" if refinement is not None else "rejected"

        debug = draw_debug(frame, box, confidence, refinement, status, reason)
        cv2.putText(debug, f"task={step} source={source or 'none'}", (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 0), 1)
        debug_path = output / "debug" / status / f"{stem}.jpg"
        cv2.imwrite(str(debug_path), debug, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if status == "accepted":
            cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            label_path.write_text(segmentation_label(refinement["polygon"], frame.shape[1], frame.shape[0]), encoding="utf-8")
            accepted_debug.append(debug_path)
        else:
            rejected_debug.append(debug_path)
        records.append(
            {
                **meta,
                "status": status,
                "reason": reason,
                "proposal_source": source,
                "confidence": round(confidence, 6) if confidence is not None else None,
                "centroid": [round(float(value), 3) for value in refinement["centroid"]] if refinement is not None else None,
                "area": refinement["area"] if refinement is not None else None,
                "image": image_path.name if status == "accepted" else None,
                "label": label_path.name if status == "accepted" else None,
            }
        )

    contact_sheets(accepted_debug, output / "review", "accepted")
    contact_sheets(rejected_debug, output / "review", "rejected")
    summary = {}
    for step in sorted({task["step"] for task in tasks}):
        rows = [record for record in records if record["step"] == step]
        summary[str(step)] = {
            "task": next(task["name"] for task in tasks if task["step"] == step),
            "sampled": len(rows),
            "accepted": sum(record["status"].startswith("accepted") for record in rows),
            "rejected": sum(record["status"] == "rejected" for record in rows),
        }
    manifest = {
        "source_video": str(args.video),
        "task_log": str(tasks_path),
        "annotation_definition": "visible black foam contour; area centroid is image center",
        "selection": "V9/V8 dark elongated proposal + GrabCut; task 9 is reviewed hard negative after 2 s transition exclusion",
        "sample_period_seconds": SAMPLE_PERIOD_SECONDS,
        "summary": summary,
        "records": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"staging: {output}")
    print("Review every accepted contact sheet before merging into the official dataset.")


if __name__ == "__main__":
    main()
