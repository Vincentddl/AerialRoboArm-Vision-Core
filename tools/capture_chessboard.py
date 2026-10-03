import argparse
import time
from pathlib import Path

import cv2


def parse_source(source):
    return int(source) if str(source).isdigit() else source


def main():
    parser = argparse.ArgumentParser(description="Capture checkerboard images for camera calibration.")
    parser.add_argument("--source", default="0", help="Camera index or video source.")
    parser.add_argument("--out", default="data/camera_calibration/new_chessboard", help="Output image directory.")
    parser.add_argument("--cols", type=int, default=9, help="Checkerboard inner corners along width.")
    parser.add_argument("--rows", type=int, default=6, help="Checkerboard inner corners along height.")
    parser.add_argument("--width", type=int, default=0, help="Optional camera width.")
    parser.add_argument("--height", type=int, default=0, help="Optional camera height.")
    parser.add_argument("--prefix", default="frame", help="Saved image filename prefix.")
    parser.add_argument(
        "--max-images",
        type=int,
        default=60,
        help="Stop after this many saved images in the output directory; use 0 for unlimited.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(parse_source(args.source), cv2.CAP_DSHOW)
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(f"failed to open source: {args.source}")

    pattern_size = (args.cols, args.rows)
    saved = len(list(out_dir.glob(f"{args.prefix}_*.png")))
    if args.max_images > 0 and saved >= args.max_images:
        print(f"Already have {saved} images (limit: {args.max_images}); nothing to capture.")
        cap.release()
        return

    print("Capture checkerboard images")
    print(f"inner corners: {pattern_size}")
    print("keys: s=save, q/Esc=quit")
    print("Move the board through center, edges, corners, different tilts, and different distances.")

    window_name = "checkerboard capture"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, pattern_size)

            vis = frame.copy()
            if found:
                cv2.drawChessboardCorners(vis, pattern_size, corners, found)

            status = "FOUND" if found else "not found"
            cv2.putText(
                vis,
                f"{status} | saved {saved} | s save | q quit",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0) if found else (0, 0, 255),
                2,
            )
            cv2.imshow(window_name, vis)

            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Capture window closed.")
                break
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                if not found:
                    print("not saved: all checkerboard corners were not detected")
                    continue
                filename = out_dir / f"{args.prefix}_{saved + 1:04d}.png"
                cv2.imwrite(str(filename), frame)
                saved += 1
                print(f"saved {filename} ({status})")
                if args.max_images > 0 and saved >= args.max_images:
                    print(f"Reached image limit ({args.max_images}); stopping capture.")
                    break
    except KeyboardInterrupt:
        print("Capture interrupted from the terminal.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
