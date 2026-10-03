"""Train the first oriented-box model for foam geometric-center localization."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_DIR
        / "data"
        / "training"
        / "foam_center_v10_obb"
        / "foam_center_v10_obb.yaml",
    )
    parser.add_argument("--initial-weights", default="yolo11s-obb.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="foam_center_v10_obb")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(args.data)
    model = YOLO(args.initial_weights)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        optimizer="AdamW",
        lr0=2e-4,
        lrf=0.05,
        weight_decay=5e-4,
        warmup_epochs=2.0,
        patience=12,
        close_mosaic=5,
        degrees=15.0,
        translate=0.12,
        scale=0.35,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.01,
        hsv_s=0.25,
        hsv_v=0.25,
        project=str(PROJECT_DIR / "outputs" / "training"),
        name=args.name,
        exist_ok=False,
        deterministic=True,
        seed=807,
        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
