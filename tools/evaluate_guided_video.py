"""Evaluate detector continuity on a guided recording without training on it.

The guide event log defines the time range for every task.  If the operator
uses ``B`` and repeats a task, the last completed attempt is evaluated.  Tasks
1-8 are target-present operational checks; task 9 is a no-target false-positive
check.  This script never modifies a dataset or model.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--tasks", type=Path)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Model label and path; may be supplied more than once.",
    )
    parser.add_argument("--thresholds", default="0.30,0.50")
    parser.add_argument("--sample-fps", type=float, default=3.0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument(
        "--environment-label",
        default="",
        help="Experiment-defined environment label stored in the report.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_models(values):
    models = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"model must use NAME=PATH: {value}")
        name, path = value.split("=", 1)
        path = Path(path)
        path = path if path.is_absolute() else PROJECT_DIR / path
        if not path.is_file():
            raise FileNotFoundError(path)
        models[name.strip()] = path.resolve()
    return models


def last_completed_task_ranges(path):
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
            start = active[step]
            completed[step] = {
                "step": step,
                "task": event["task_name"],
                "is_negative": any(
                    marker in event["task_name"].upper()
                    for marker in ("NO TARGET", "HARD NEGATIVE", "EMPTY BACKGROUND")
                ),
                "start_frame": int(start["frame_index"]),
                "end_frame": int(event["frame_index"]),
                "actual_seconds": float(event.get("actual_seconds", 0.0)),
            }
    if not completed:
        raise RuntimeError("the guide log contains no completed tasks")
    expected = set(range(1, max(completed) + 1))
    missing = sorted(expected - set(completed))
    if missing:
        raise RuntimeError(f"guide tasks were not completed: {missing}")
    return [completed[step] for step in sorted(completed)]


def sample_frames(video, tasks, sample_fps):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    period = max(1, round(source_fps / sample_fps))
    requested = {}
    for task in tasks:
        start = task["start_frame"]
        # Allow time to remove the target before evaluating hard negatives.
        if task["is_negative"]:
            start += round(2.0 * source_fps)
        for frame_index in range(start, task["end_frame"], period):
            requested[frame_index] = task

    frames = []
    metadata = []
    frame_index = 0
    try:
        while requested:
            ok, frame = cap.read()
            if not ok:
                break
            task = requested.pop(frame_index, None)
            if task is not None:
                frames.append(frame)
                metadata.append({"frame": frame_index, **task})
            frame_index += 1
    finally:
        cap.release()
    if requested:
        raise RuntimeError(f"video ended before {len(requested)} sampled frames")
    return source_fps, frames, metadata


def longest_run(values, wanted):
    best = current = 0
    for value in values:
        if value == wanted:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def summarize(confidences, metadata, thresholds, sample_fps):
    rows = []
    for step in sorted({item["step"] for item in metadata}):
        indices = [index for index, item in enumerate(metadata) if item["step"] == step]
        scores = [confidences[index] for index in indices]
        task = metadata[indices[0]]["task"]
        is_negative = metadata[indices[0]]["is_negative"]
        row = {
            "step": step,
            "task": task,
            "is_negative": is_negative,
            "sampled": len(scores),
            "median_top_conf": round(statistics.median(scores), 6),
            "mean_top_conf": round(statistics.fmean(scores), 6),
        }
        for threshold in thresholds:
            detected = [score >= threshold for score in scores]
            suffix = f"{threshold:.2f}".replace(".", "p")
            row[f"detected_at_{suffix}"] = sum(detected)
            row[f"rate_at_{suffix}"] = round(sum(detected) / len(detected), 6)
            if is_negative:
                run = longest_run(detected, True)
                row[f"longest_false_positive_run_s_at_{suffix}"] = round(
                    run / sample_fps, 3
                )
            else:
                run = longest_run(detected, False)
                row[f"longest_missing_run_s_at_{suffix}"] = round(
                    run / sample_fps, 3
                )
        rows.append(row)
    return rows


def main():
    args = parse_args()
    video = args.video.resolve()
    tasks_path = (args.tasks or video.with_suffix(".tasks.jsonl")).resolve()
    models = parse_models(args.model)
    thresholds = sorted({float(value) for value in args.thresholds.split(",")})
    tasks = last_completed_task_ranges(tasks_path)
    source_fps, frames, metadata = sample_frames(video, tasks, args.sample_fps)
    print(f"sampled {len(frames)} frames at {args.sample_fps:g} FPS", flush=True)

    report = {
        "video": str(video),
        "task_log": str(tasks_path),
        "source_fps": source_fps,
        "sample_fps": args.sample_fps,
        "environment_label": args.environment_label,
        "thresholds": thresholds,
        "task_ranges": tasks,
        "models": {},
    }
    for name, path in models.items():
        model = YOLO(str(path))
        confidences = []
        sample_detections = []
        for start in range(0, len(frames), args.batch):
            batch = frames[start : start + args.batch]
            predictions = model.predict(
                batch,
                imgsz=640,
                conf=0.01,
                max_det=5,
                batch=len(batch),
                device=args.device,
                verbose=False,
            )
            for prediction, meta in zip(predictions, metadata[start : start + len(batch)]):
                scores = prediction.boxes.conf.cpu().tolist()
                boxes = prediction.boxes.xyxy.cpu().tolist()
                if scores:
                    best = max(range(len(scores)), key=scores.__getitem__)
                    top_confidence = float(scores[best])
                    top_box = [round(float(value), 2) for value in boxes[best]]
                else:
                    top_confidence = 0.0
                    top_box = None
                confidences.append(top_confidence)
                sample_detections.append(
                    {
                        "frame": meta["frame"],
                        "step": meta["step"],
                        "top_conf": round(top_confidence, 6),
                        "top_box_xyxy": top_box,
                    }
                )
            print(f"{name}: {min(start + len(batch), len(frames))}/{len(frames)}", flush=True)
        report["models"][name] = {
            "path": str(path),
            "summary": summarize(confidences, metadata, thresholds, args.sample_fps),
            "samples": sample_detections,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
