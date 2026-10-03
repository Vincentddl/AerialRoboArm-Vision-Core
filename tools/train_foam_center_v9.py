"""Train a segmentation model whose output centroid is the foam target center."""

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
        default=PROJECT_DIR / "data" / "training" / "foam_center_v9" / "foam_center_v9.yaml",
    )
    parser.add_argument("--initial-weights", type=Path, default=PROJECT_DIR / "models" / "foam_board_2p1mm_v8.pt")
    parser.add_argument(
        "--initial-seg-weights",
        type=Path,
        default=None,
        help="Continue training an existing segmentation checkpoint instead of rebuilding the segmentation head.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="foam_center_v9_seg")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(args.data)
    if args.initial_seg_weights is not None and not args.initial_seg_weights.exists():
        raise FileNotFoundError(args.initial_seg_weights)
    if args.initial_seg_weights is None and not args.initial_weights.exists():
        raise FileNotFoundError(args.initial_weights)

    if args.initial_seg_weights is not None:
        # Preserve the already trained segmentation head when adding reviewed
        # generalization samples. This is the normal path for later V9 updates.
        model = YOLO(str(args.initial_seg_weights))
    else:
        # Bootstrap the first segmentation version from the V8 detector's
        # compatible backbone. The segmentation head starts new in this path.
        model = YOLO("yolo11s-seg.yaml").load(str(args.initial_weights))
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
        patience=10,
        close_mosaic=5,
        degrees=12.0,
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
        pretrained=False,
        deterministic=True,
        seed=806,
        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
