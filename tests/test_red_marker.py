from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from vision.red_marker import RedMarkerDetector  # noqa: E402


class RedMarkerDetectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = RedMarkerDetector()

    def test_detects_saturated_red_circle_near_camera_axis(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(image, (330, 210), 18, (0, 0, 255), -1)

        detection = self.detector.detect(image)

        self.assertIsNotNone(detection)
        self.assertAlmostEqual(detection.center[0], 330.5, delta=1.0)
        self.assertAlmostEqual(detection.center[1], 210.5, delta=1.0)

    def test_rejects_red_distractor_outside_centre_band(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(image, (560, 210), 18, (0, 0, 255), -1)

        self.assertIsNone(self.detector.detect(image))


if __name__ == "__main__":
    unittest.main()

