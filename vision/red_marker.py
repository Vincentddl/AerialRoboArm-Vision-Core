"""High-contrast red calibration marker detection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RedMarkerDetection:
    center: tuple[float, float]
    area_px2: float
    circularity: float
    bbox: tuple[int, int, int, int]


class RedMarkerDetector:
    """Detect the saturated red circle fixed to the calibrated arm point."""

    def __init__(
        self,
        minimum_area_px2: float = 40.0,
        maximum_area_px2: float = 6000.0,
        minimum_circularity: float = 0.35,
        center_band: tuple[float, float] = (0.30, 0.70),
    ) -> None:
        self.minimum_area_px2 = float(minimum_area_px2)
        self.maximum_area_px2 = float(maximum_area_px2)
        self.minimum_circularity = float(minimum_circularity)
        self.center_band = center_band

    def detect(self, frame: np.ndarray) -> RedMarkerDetection | None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        low_red = cv2.inRange(
            hsv,
            np.asarray([0, 80, 80], dtype=np.uint8),
            np.asarray([16, 255, 255], dtype=np.uint8),
        )
        high_red = cv2.inRange(
            hsv,
            np.asarray([164, 80, 80], dtype=np.uint8),
            np.asarray([179, 255, 255], dtype=np.uint8),
        )
        mask = cv2.bitwise_or(low_red, high_red)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        width = frame.shape[1]
        minimum_x = self.center_band[0] * width
        maximum_x = self.center_band[1] * width
        best_score = -1.0
        best_detection = None
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not self.minimum_area_px2 <= area <= self.maximum_area_px2:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0:
                continue
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            x, y, box_width, box_height = cv2.boundingRect(contour)
            center_x = x + box_width / 2.0
            center_y = y + box_height / 2.0
            aspect_ratio = box_width / max(box_height, 1)
            if not minimum_x <= center_x <= maximum_x:
                continue
            if not 0.48 <= aspect_ratio <= 2.0:
                continue
            if circularity < self.minimum_circularity:
                continue
            score = area * circularity
            if score > best_score:
                best_score = score
                best_detection = RedMarkerDetection(
                    center=(center_x, center_y),
                    area_px2=area,
                    circularity=circularity,
                    bbox=(x, y, box_width, box_height),
                )
        return best_detection

