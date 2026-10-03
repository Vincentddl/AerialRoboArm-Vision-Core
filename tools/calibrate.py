import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")


def collect_images(images_dir):
    paths = []
    for pattern in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(str(Path(images_dir) / pattern)))
    return sorted(paths)


def make_object_points(cols, rows, square_size):
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size)
    return objp


def detect_corners(image_path, pattern_size):
    image = cv2.imread(image_path)
    if image is None:
        return None, None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found:
        return image.shape[1::-1], None, image

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return image.shape[1::-1], corners, image


def calibrate_pinhole(objpoints, imgpoints, image_size):
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )
    return ret, camera_matrix, dist_coeffs, rvecs, tvecs


def calibrate_fisheye(objpoints, imgpoints, image_size):
    fish_objpoints = [points.reshape(1, -1, 3).astype(np.float64) for points in objpoints]
    fish_imgpoints = [points.reshape(1, -1, 2).astype(np.float64) for points in imgpoints]
    # A pinhole estimate gives the fisheye solver a stable starting point for
    # strongly distorted wide-angle images.
    _, camera_matrix, _, _, _ = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )
    camera_matrix = camera_matrix.astype(np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in fish_objpoints]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in fish_objpoints]
    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        + cv2.fisheye.CALIB_CHECK_COND
        + cv2.fisheye.CALIB_FIX_SKEW
        + cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
    )
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-6,
    )
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.fisheye.calibrate(
        fish_objpoints,
        fish_imgpoints,
        image_size,
        camera_matrix,
        dist_coeffs,
        rvecs,
        tvecs,
        flags,
        criteria,
    )
    return ret, camera_matrix, dist_coeffs, rvecs, tvecs


def reprojection_error(model, objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs):
    total_error = 0.0
    total_points = 0
    per_image_errors = []
    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        if model == "fisheye":
            projected, _ = cv2.fisheye.projectPoints(
                objp.reshape(1, -1, 3).astype(np.float64),
                rvec,
                tvec,
                camera_matrix,
                dist_coeffs,
            )
            projected = projected.reshape(-1, 2)
        else:
            projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
            projected = projected.reshape(-1, 2)

        observed = imgp.reshape(-1, 2).astype(np.float64)
        projected = projected.astype(np.float64)
        error = cv2.norm(observed, projected, cv2.NORM_L2)
        total_error += error * error
        total_points += len(objp)
        per_image_errors.append(float(np.sqrt((error * error) / max(len(objp), 1))))

    return float(np.sqrt(total_error / max(total_points, 1))), per_image_errors


def save_debug_images(debug_dir, valid_records, pattern_size):
    debug_path = Path(debug_dir)
    debug_path.mkdir(parents=True, exist_ok=True)
    for index, (image_path, image, corners) in enumerate(valid_records, start=1):
        vis = image.copy()
        cv2.drawChessboardCorners(vis, pattern_size, corners, True)
        cv2.imwrite(str(debug_path / f"corners_{index:04d}.png"), vis)


def main():
    parser = argparse.ArgumentParser(description="Calibrate camera with checkerboard images.")
    parser.add_argument("--images", default="data/camera_calibration/chessboard", help="Directory containing checkerboard images.")
    parser.add_argument("--model", choices=["pinhole", "fisheye"], default="pinhole", help="Calibration model.")
    parser.add_argument("--cols", type=int, default=9, help="Checkerboard inner corners along width.")
    parser.add_argument("--rows", type=int, default=6, help="Checkerboard inner corners along height.")
    parser.add_argument("--square-size", type=float, default=20.0, help="Actual square size, usually in millimeters.")
    parser.add_argument("--out", default="configs/camera_calibration_new.json", help="Output calibration JSON.")
    parser.add_argument("--debug-dir", default="outputs/camera_calibration_debug", help="Directory for corner debug images.")
    args = parser.parse_args()

    image_paths = collect_images(args.images)
    if not image_paths:
        raise RuntimeError(f"no images found in {args.images}")

    pattern_size = (args.cols, args.rows)
    base_objp = make_object_points(args.cols, args.rows, args.square_size)
    objpoints = []
    imgpoints = []
    valid_records = []
    image_size = None

    print(f"images: {len(image_paths)}")
    print(f"model: {args.model}")
    print(f"inner corners: {pattern_size}")
    print(f"square size: {args.square_size}")

    for path in image_paths:
        detected_size, corners, image = detect_corners(path, pattern_size)
        if image_size is None and detected_size is not None:
            image_size = detected_size
        if corners is None:
            print(f"skip: {path}")
            continue
        if detected_size != image_size:
            print(f"skip different size: {path} {detected_size} != {image_size}")
            continue
        objpoints.append(base_objp.copy())
        imgpoints.append(corners)
        valid_records.append((path, image, corners))
        print(f"ok: {path}")

    if len(objpoints) < 8:
        raise RuntimeError(f"only {len(objpoints)} valid images; capture at least 15-25, preferably 25-40 for wide angle")

    if args.model == "fisheye":
        rms, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_fisheye(objpoints, imgpoints, image_size)
    else:
        rms, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_pinhole(objpoints, imgpoints, image_size)

    error, per_image_errors = reprojection_error(
        args.model,
        objpoints,
        imgpoints,
        rvecs,
        tvecs,
        camera_matrix,
        dist_coeffs,
    )
    save_debug_images(args.debug_dir, valid_records, pattern_size)

    image_errors = [
        {"image": record[0], "error_px": image_error}
        for record, image_error in zip(valid_records, per_image_errors)
    ]
    image_errors.sort(key=lambda item: item["error_px"], reverse=True)

    output = {
        "model": args.model,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "checkerboard_inner_corners": [args.cols, args.rows],
        "square_size": float(args.square_size),
        "square_size_unit": "mm",
        "valid_image_count": len(objpoints),
        "rms": float(rms),
        "reprojection_error_px": error,
        "per_image_reprojection_error_px": image_errors,
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
        "notes": [
            "Use the same resolution during runtime.",
            "Recalibrate if camera position, lens, focus, or capture resolution changes.",
            "For grab angle, undistort the detected target center before computing angle.",
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"valid images: {len(objpoints)}")
    print(f"rms: {rms:.6f}")
    print(f"reprojection error: {error:.3f} px")
    print("worst per-image errors:")
    for item in image_errors[:10]:
        print(f"  {item['error_px']:.3f} px  {item['image']}")
    print(f"saved: {out_path}")
    print(f"debug corners: {args.debug_dir}")


if __name__ == "__main__":
    main()
