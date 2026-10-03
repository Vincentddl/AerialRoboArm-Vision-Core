import argparse
import time
from pathlib import Path

import cv2

from calibration_utils import load_calibration, undistort_frame


def parse_source(source):
    return int(source) if str(source).isdigit() else source


def is_image_path(source):
    return Path(str(source)).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser(description="Preview or save undistorted images/video.")
    parser.add_argument("--calibration", required=True, help="Calibration JSON.")
    parser.add_argument("--source", default="0", help="Camera index, image path, or video path.")
    parser.add_argument("--output", default="", help="Optional output image/video path.")
    parser.add_argument("--balance", type=float, default=0.0, help="0 crops more, 1 keeps more FOV/black borders.")
    parser.add_argument("--fov-scale", type=float, default=1.0, help="Fisheye output FOV scale.")
    parser.add_argument("--width", type=int, default=0, help="Optional camera width.")
    parser.add_argument("--height", type=int, default=0, help="Optional camera height.")
    args = parser.parse_args()

    calibration_data, camera_matrix, dist_coeffs, _ = load_calibration(args.calibration)

    if is_image_path(args.source):
        frame = cv2.imread(args.source)
        if frame is None:
            raise RuntimeError(f"failed to read image: {args.source}")
        undistorted, _ = undistort_frame(frame, calibration_data, camera_matrix, dist_coeffs, args.balance, args.fov_scale)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(args.output, undistorted)
            print(f"saved {args.output}")
        else:
            cv2.imshow("original", frame)
            cv2.imshow("undistorted", undistorted)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    cap = cv2.VideoCapture(parse_source(args.source), cv2.CAP_DSHOW)
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open source: {args.source}")

    writer = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            undistorted, _ = undistort_frame(
                frame,
                calibration_data,
                camera_matrix,
                dist_coeffs,
                args.balance,
                args.fov_scale,
            )

            if args.output:
                if writer is None:
                    h, w = undistorted.shape[:2]
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    fps = fps if fps and fps > 1 else 25
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
                writer.write(undistorted)

            cv2.imshow("original", frame)
            cv2.imshow("undistorted", undistorted)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

