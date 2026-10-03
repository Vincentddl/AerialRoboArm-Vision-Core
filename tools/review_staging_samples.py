"""Create labelled contact sheets for a subset of staged segmentation samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-confidence", type=float, default=1.0)
    parser.add_argument("--positive-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    staging = args.staging.resolve()
    manifest = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
    records = []
    for record in manifest["records"]:
        if record["status"] != "accepted":
            continue
        if args.positive_only and record.get("is_negative"):
            continue
        confidence = record.get("confidence")
        if confidence is not None and confidence > args.max_confidence:
            continue
        records.append(record)

    args.output.mkdir(parents=True, exist_ok=True)
    columns, rows = 5, 4
    tile_width, tile_height = 320, 240
    page_size = columns * rows
    for page_start in range(0, len(records), page_size):
        canvas = np.zeros((rows * tile_height, columns * tile_width, 3), np.uint8)
        for offset, record in enumerate(records[page_start : page_start + page_size]):
            stem = Path(record["image"]).stem
            path = staging / "debug" / "accepted" / f"{stem}.jpg"
            image = cv2.imread(str(path))
            if image is None:
                continue
            image = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
            label = (
                f"#{page_start + offset + 1} frame={record['frame']} "
                f"s={record['step']} conf={record.get('confidence', 0) or 0:.2f}"
            )
            cv2.rectangle(image, (0, tile_height - 24), (tile_width, tile_height), (0, 0, 0), -1)
            cv2.putText(
                image,
                label,
                (5, tile_height - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            row, column = divmod(offset, columns)
            canvas[
                row * tile_height : (row + 1) * tile_height,
                column * tile_width : (column + 1) * tile_width,
            ] = image
        output = args.output / f"review_{page_start // page_size + 1:02d}.jpg"
        cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"selected={len(records)} pages={(len(records) + page_size - 1) // page_size}")


if __name__ == "__main__":
    main()
