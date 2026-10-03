from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from vision.bearing import BearingMapper, BearingObservation, RobustBearingPredictor  # noqa: E402


CALIBRATION = PROJECT_DIR / "configs" / "camera_2p1mm_640x480_fisheye.json"


class BearingMapperTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = BearingMapper(CALIBRATION)

    def test_pixel_angle_round_trip(self) -> None:
        for pixel in ((320.0, 240.0), (80.0, 80.0), (560.0, 400.0)):
            angles = self.mapper.pixel_to_angles(pixel)
            projected = self.mapper.angles_to_pixel(angles)
            self.assertAlmostEqual(pixel[0], projected[0], places=5)
            self.assertAlmostEqual(pixel[1], projected[1], places=5)

    def test_angular_error_is_symmetric(self) -> None:
        first = (-12.0, 8.0)
        second = (15.0, -4.0)
        self.assertAlmostEqual(
            self.mapper.angular_error_deg(first, second),
            self.mapper.angular_error_deg(second, first),
            places=10,
        )
        self.assertAlmostEqual(self.mapper.angular_error_deg(first, first), 0.0, places=8)


class RobustBearingPredictorTest(unittest.TestCase):
    def test_hold_uses_latest_real_observation(self) -> None:
        observations = [
            BearingObservation(timestamp=index * 0.04, yaw_deg=index, pitch_deg=-index)
            for index in range(6)
        ]
        prediction = RobustBearingPredictor().predict(observations, 0.4, "hold")
        self.assertEqual(prediction.yaw_deg, 5.0)
        self.assertEqual(prediction.pitch_deg, -5.0)

    def test_robust_quadratic_fit_rejects_low_confidence_outlier(self) -> None:
        observations = []
        for index, timestamp in enumerate((-0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.0)):
            yaw = 5.0 + 20.0 * timestamp + 0.5 * 40.0 * timestamp * timestamp
            pitch = -3.0 + 8.0 * timestamp - 0.5 * 20.0 * timestamp * timestamp
            score = 0.9
            if index == 2:
                yaw += 15.0
                pitch -= 10.0
                score = 0.05
            observations.append(
                BearingObservation(
                    timestamp=timestamp,
                    yaw_deg=yaw,
                    pitch_deg=pitch,
                    score=score,
                )
            )

        prediction = RobustBearingPredictor(huber_delta_deg=0.25).predict(
            observations, 0.20, "angular_acceleration"
        )
        expected_yaw = 5.0 + 20.0 * 0.20 + 0.5 * 40.0 * 0.20 * 0.20
        expected_pitch = -3.0 + 8.0 * 0.20 - 0.5 * 20.0 * 0.20 * 0.20
        self.assertAlmostEqual(prediction.yaw_deg, expected_yaw, delta=0.8)
        self.assertAlmostEqual(prediction.pitch_deg, expected_pitch, delta=0.8)


if __name__ == "__main__":
    unittest.main()
