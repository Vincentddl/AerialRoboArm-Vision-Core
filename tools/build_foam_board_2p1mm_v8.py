import json
import shutil
from pathlib import Path

import cv2


LAB_DIR = Path(__file__).resolve().parents[1]
SOURCE_DATASET = LAB_DIR / "data" / "training" / "foam_board_v8"
RAW_DIR = LAB_DIR / "data" / "raw" / "v8_source_20260803"
VIDEO_PATH = RAW_DIR / "servo_camera_angle_calibration_50fps_20260803_000945_262004.mkv"
CANDIDATE_PATH = RAW_DIR / "v8_candidate_selection.json"
OUTPUT_DIR = LAB_DIR / "outputs" / "foam_board_v8_rebuild"


# Full-resolution contact-sheet review confirmed that the black foam target is
# outside the image in these frames. Frames 4600, 6100 and 6200 were rejected
# because the target is still partially visible at the top edge.
TRAIN_NEGATIVE_FRAME_IDS = {
    0,
    100,
    300,
    400,
    4700,
    4800,
    4900,
    5000,
    5100,
    5200,
    5300,
    5400,
    5500,
}
VAL_NEGATIVE_FRAME_IDS = {5600, 5700, 5800, 5900, 6000, 9000, 9100, 9200, 9300}


def prepare_directories():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for relative in (
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
        "debug/train",
        "debug/val",
    ):
        (OUTPUT_DIR / relative).mkdir(parents=True, exist_ok=True)


def copy_source_dataset():
    counts = {}
    negative_counts = {}
    for split in ("train", "val"):
        counts[split] = 0
        negative_counts[split] = 0
        for image_path in sorted((SOURCE_DATASET / "images" / split).glob("*.jpg")):
            label_path = SOURCE_DATASET / "labels" / split / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(label_path)
            shutil.copy2(image_path, OUTPUT_DIR / "images" / split / image_path.name)
            shutil.copy2(label_path, OUTPUT_DIR / "labels" / split / label_path.name)
            counts[split] += 1
            negative_counts[split] += not label_path.read_text(encoding="utf-8").strip()
    return counts, negative_counts


def load_candidates():
    data = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))
    samples = data["samples"]
    frame_ids = [item["frame"] for item in samples]
    if len(frame_ids) != len(set(frame_ids)):
        raise RuntimeError("duplicate frame ids in V8 candidate selection")
    for item in samples:
        if item["split"] not in {"train", "val"}:
            raise ValueError(f"invalid split for frame {item['frame']}: {item['split']}")
        if len(item["box"]) != 4:
            raise ValueError(f"invalid box for frame {item['frame']}: {item['box']}")
    return data, samples


def load_frames(frame_ids):
    wanted = set(frame_ids)
    frames = {}
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {VIDEO_PATH}")
    try:
        index = 0
        while wanted:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if index in wanted:
                frames[index] = frame.copy()
                wanted.remove(index)
            index += 1
    finally:
        cap.release()
    if wanted:
        raise RuntimeError(f"failed to read frames: {sorted(wanted)}")
    return frames


def yolo_label(box, width, height):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / (2 * width)
    center_y = (y1 + y2) / (2 * height)
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return f"0 {center_x:.8f} {center_y:.8f} {box_width:.8f} {box_height:.8f}\n"


def save_sample(frame, frame_id, split, box, annotation_source):
    sample_type = "positive" if box is not None else "negative"
    stem = f"servo_angle_20260803_{sample_type}_{frame_id:06d}"
    image_path = OUTPUT_DIR / "images" / split / f"{stem}.jpg"
    label_path = OUTPUT_DIR / "labels" / split / f"{stem}.txt"
    height, width = frame.shape[:2]
    cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    debug = frame.copy()
    if box is None:
        label_path.write_text("", encoding="utf-8")
        cv2.putText(debug, "NEGATIVE", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    else:
        x1, y1, x2, y2 = (round(value) for value in box)
        clipped = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
        if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
            raise ValueError(f"invalid box for frame {frame_id}: {box}")
        label_path.write_text(yolo_label(clipped, width, height), encoding="utf-8")
        cv2.rectangle(debug, (clipped[0], clipped[1]), (clipped[2], clipped[3]), (0, 255, 0), 2)
        cv2.putText(
            debug,
            annotation_source,
            (clipped[0], max(24, clipped[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )
    cv2.imwrite(str(OUTPUT_DIR / "debug" / split / f"{stem}.jpg"), debug)
    return {
        "frame": frame_id,
        "split": split,
        "type": sample_type,
        "box": box,
        "annotation_source": annotation_source,
    }


def main():
    for required in (SOURCE_DATASET, VIDEO_PATH, CANDIDATE_PATH):
        if not required.exists():
            raise FileNotFoundError(required)

    candidate_data, candidates = load_candidates()
    positive_ids = {item["frame"] for item in candidates}
    negative_ids = TRAIN_NEGATIVE_FRAME_IDS | VAL_NEGATIVE_FRAME_IDS
    overlap = positive_ids & negative_ids
    if overlap:
        raise RuntimeError(f"frames marked both positive and negative: {sorted(overlap)}")

    prepare_directories()
    copied_counts, copied_negative_counts = copy_source_dataset()
    frames = load_frames(positive_ids | negative_ids)
    samples = []

    for item in candidates:
        samples.append(
            save_sample(
                frames[item["frame"]],
                item["frame"],
                item["split"],
                item["box"],
                item["source"],
            )
        )
    for frame_id in sorted(TRAIN_NEGATIVE_FRAME_IDS):
        samples.append(save_sample(frames[frame_id], frame_id, "train", None, "reviewed_negative"))
    for frame_id in sorted(VAL_NEGATIVE_FRAME_IDS):
        samples.append(save_sample(frames[frame_id], frame_id, "val", None, "reviewed_negative"))

    yaml_path = OUTPUT_DIR / "foam_board_2p1mm.yaml"
    yaml_path.write_text(
        f"path: {OUTPUT_DIR.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: foam_board\n",
        encoding="utf-8",
    )
    new_val_images = sorted((OUTPUT_DIR / "images" / "val").glob("servo_angle_20260803_*.jpg"))
    new_val_list_path = OUTPUT_DIR / "new_val.txt"
    new_val_list_path.write_text(
        "".join(f"{image_path.as_posix()}\n" for image_path in new_val_images),
        encoding="utf-8",
    )
    new_val_yaml_path = OUTPUT_DIR / "foam_board_2p1mm_new_val.yaml"
    new_val_yaml_path.write_text(
        f"path: {OUTPUT_DIR.as_posix()}\ntrain: images/train\nval: {new_val_list_path.as_posix()}\nnames:\n  0: foam_board\n",
        encoding="utf-8",
    )
    positive_train = sum(item["split"] == "train" for item in candidates)
    positive_val = sum(item["split"] == "val" for item in candidates)
    new_counts = {
        "positive_train": positive_train,
        "positive_val": positive_val,
        "negative_train": len(TRAIN_NEGATIVE_FRAME_IDS),
        "negative_val": len(VAL_NEGATIVE_FRAME_IDS),
        "reviewed_v7_boxes": sum(item["source"] == "reviewed_v7_box" for item in candidates),
        "marker_position_transfers": sum(item["source"] == "marker_position_transfer" for item in candidates),
    }
    manifest = {
        "source_dataset": str(SOURCE_DATASET),
        "source_video": str(VIDEO_PATH),
        "candidate_file": str(CANDIDATE_PATH),
        "candidate_policy": candidate_data["annotation_policy"],
        "split_policy": candidate_data["sampling"],
        "copied_counts": copied_counts,
        "copied_negative_counts": copied_negative_counts,
        "new_counts": new_counts,
        "samples": samples,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total_train = copied_counts["train"] + positive_train + len(TRAIN_NEGATIVE_FRAME_IDS)
    total_val = copied_counts["val"] + positive_val + len(VAL_NEGATIVE_FRAME_IDS)
    print(f"dataset: {OUTPUT_DIR}")
    print(f"copied train/val: {copied_counts['train']}/{copied_counts['val']}")
    print(f"new positives train/val: {positive_train}/{positive_val}")
    print(f"new negatives train/val: {len(TRAIN_NEGATIVE_FRAME_IDS)}/{len(VAL_NEGATIVE_FRAME_IDS)}")
    print(f"total train/val: {total_train}/{total_val}")
    print(f"yaml: {yaml_path}")
    print(f"new-only val yaml: {new_val_yaml_path} ({len(new_val_images)} images)")


if __name__ == "__main__":
    main()
