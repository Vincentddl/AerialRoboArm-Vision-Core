import json
from pathlib import Path

import cv2
import numpy as np


def load_calibration(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    image_size = tuple(int(v) for v in data["image_size"])
    return data, camera_matrix, dist_coeffs, image_size


def undistort_frame(frame, calibration_data, camera_matrix, dist_coeffs, balance=0.0, fov_scale=1.0):
    model = calibration_data["model"]
    h, w = frame.shape[:2]
    image_size = (w, h)

    if model == "fisheye":
        new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            camera_matrix,
            dist_coeffs,
            image_size,
            np.eye(3),
            balance=balance,
            fov_scale=fov_scale,
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix,
            dist_coeffs,
            np.eye(3),
            new_camera_matrix,
            image_size,
            cv2.CV_16SC2,
        )
        undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undistorted, new_camera_matrix

    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        image_size,
        alpha=balance,
        newImgSize=image_size,
    )
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
    return undistorted, new_camera_matrix


def undistort_points(points, calibration_data, camera_matrix, dist_coeffs):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    model = calibration_data["model"]
    if model == "fisheye":
        corrected = cv2.fisheye.undistortPoints(points, camera_matrix, dist_coeffs, P=camera_matrix)
    else:
        corrected = cv2.undistortPoints(points, camera_matrix, dist_coeffs, P=camera_matrix)
    return corrected.reshape(-1, 2)


def point_to_angles(point, camera_matrix):
    u, v = float(point[0]), float(point[1])
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    angle_x = np.degrees(np.arctan((u - cx) / fx))
    angle_y = np.degrees(np.arctan((v - cy) / fy))
    return float(angle_x), float(angle_y)

