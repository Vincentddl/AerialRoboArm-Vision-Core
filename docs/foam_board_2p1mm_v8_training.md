# Foam Board 2.1 mm V8 Training

## Source

- Video: `data/raw/v8_source_20260803/servo_camera_angle_calibration_50fps_20260803_000945_262004.mkv`
- Resolution and rate: 640x480 MJPEG at 50 FPS
- Frames: 9368
- Encoded duration: 187.36 seconds
- Timestamp duration: 187.687 seconds (49.908 FPS effective)

The target moves slowly along the camera center axis in a forward and reverse servo sweep. The useful part of the
recording is static/slow object-detection data, not trajectory-dynamics ground truth: servo commands were not logged
with frame timestamps, and the target leaves the image near both sweep endpoints.

## Annotation Policy

The V7 detector was evaluated on one frame per second. Boxes that matched the yellow marker and the visible black
foam target were reviewed at full contact-sheet resolution. Dark frames missed by V7 inherit box geometry from the
nearest reviewed forward-pass frame at the same marker Y position. The complete forward and reverse contact sheets
were reviewed before dataset construction.

Train positives use the forward sweep and validation positives use the temporally separated reverse sweep. Reviewed
target-free intervals supply hard negatives containing the arm, clothing and the same dark background. Frames with
analog-link blackouts and frames where the target is clipped at the image edge are excluded.

## Dataset

- Prepared dataset: `data/training/foam_board_v8/`
- Builder/review tool: `tools/build_foam_board_2p1mm_v8.py`
- Historical starting weights: V7 (not copied into the Core runtime)
- Existing V7 train/validation: 616/167
- New positive train/validation: 32/31
- New negative train/validation: 13/9
- Final train/validation: 661/207

## Training

- Final weights: `models/foam_board_2p1mm_v8.pt`
- Run: `runs/foam_board_2p1mm_v8`
- Best epoch: 19
- Device: NVIDIA GeForce RTX 4070 Laptop GPU
- Image size: 640; batch: 4; epochs: 20
- Optimizer: AdamW; initial learning rate: 0.0002

## Results

On the mixed V8 validation set, V7 and V8 compare as follows:

| Model | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| V7 baseline | 0.886 | 0.667 | 0.742 | 0.424 |
| V8 | 0.843 | 0.881 | 0.867 | 0.558 |

On the 40 new-only validation images (31 reverse-pass positives and 9 reviewed negatives):

| Model | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| V7 | 0.243 | 0.226 | 0.196 | 0.157 |
| V8 | 0.844 | 0.871 | 0.914 | 0.595 |

At confidence 0.20, V7 detected 8/31 positives and produced detections on 3/9 negative images. V8 detected 27/31
positives and produced zero detections on all 9 negative images.

On the original V7 validation set, V8 obtained P=0.871, R=0.856, mAP50=0.861 and mAP50-95=0.558. The V7
baseline in the same runtime was P=0.911, R=0.846, mAP50=0.884 and mAP50-95=0.504. V8 therefore improves recall
and stricter localization while trading away some aggregate precision and mAP50 on older scenes.

The new-only score measures a temporally separate reverse pass, but it is still the same camera session and marker-
transferred annotations. It should not be interpreted as independent real-world accuracy. A second recording with
synchronized servo angles and independently reviewed boxes remains necessary.
