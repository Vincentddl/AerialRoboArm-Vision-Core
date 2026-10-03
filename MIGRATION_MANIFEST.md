# Migration Manifest

Migration date: 2026-08-05

Source project (left unchanged):

```text
D:\Study_data\AerialRoboArm-Vision-new\AerialRoboArm-Vision
```

Target project:

```text
D:\Study_data\AerialRoboArm-Vision-Core
```

## Copied material

| Target | Files | Bytes | Purpose |
| --- | ---: | ---: | --- |
| `data/raw/v8_source_20260803/` | 21 | 270,542,457 | Complete V8 source recording and review metadata |
| `data/raw/servo_angle_20260804/` | 5 | 270,414,719 | Latest guided servo-angle recording |
| `data/training/foam_board_v8/` | 1,825 | 61,813,877 | Prepared V8 training/validation data; stale generated cache removed |
| `data/camera_calibration/chessboard/` | 53 | 25,862,418 | Final 2.1 mm fisheye checkerboard images |
| `models/foam_board_2p1mm_v8.pt` | 1 | 19,159,770 | Current runtime model |

The retained data and model total 647,793,241 bytes (about 617.8 MiB). Dataset YAML and `new_val.txt` paths were intentionally rewritten for the new absolute project location.

## Integrity checks

```text
V8 source video SHA-256:
4A26FDD20B3175C7514833E326EF85AAAE8173962B0E39562D15404CF4CB237B

2026-08-04 servo-angle video SHA-256:
DE0BB183C4B851603FF8C4FF575731CD321D39C1E52280BAA7225CB07BBD5902

V8 model SHA-256:
C59932F07945F945F8AD7213295852BBAE8E75F062B974101158E8C5DDA9F2A5
```

All three hashes match their source files. The V8 raw tree and checkerboard tree also match source file counts, names and byte sizes exactly.

## Verification results

- 44 unit tests passed.
- The V8 model loaded and ran inference on a migrated positive validation image.
- The output contained `optical_axis_offset_deg`, `servo_command_deg`, and the separated legacy/lateral angle fields.
- All 53 migrated checkerboard images were accepted.
- Recomputed fisheye RMS and reprojection error: `0.228637 px`, identical to the selected runtime calibration.
- Both migrated dataset YAML files resolve to the new project paths.

## Deliberately excluded

- Virtual environments and editor caches.
- `outputs/` from the old project (over 5 GB).
- Old V3–V7 models and duplicated datasets.
- Training run artifacts and calibration debug images from the old project.
- Old packaging scripts that still launched the V7 entry point.
