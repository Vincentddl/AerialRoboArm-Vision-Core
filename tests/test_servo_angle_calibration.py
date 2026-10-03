from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from vision.servo_angle_calibration import ServoOpticalAngleCalibration  # noqa: E402
from vision.tracker import (  # noqa: E402
    Detection,
    PixelToWorldMapper,
    TrajectoryEstimator,
    draw_arm_angle_overlay,
)


CAMERA_CALIBRATION = (
    PROJECT_DIR / "configs" / "camera_2p1mm_640x480_fisheye.json"
)
SERVO_CALIBRATION = (
    PROJECT_DIR
    / "configs"
    / "servo_to_optical_angle_yolo_v8_20260804_v1.json"
)
RED_MARKER_LUT_CALIBRATION = (
    PROJECT_DIR
    / "configs"
    / "servo_to_optical_angle_red_marker_lut_20260815_v3.json"
)


class ServoAngleCalibrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.servo_calibration = ServoOpticalAngleCalibration.from_json(
            SERVO_CALIBRATION
        )

    def test_linear_mapping_and_inverse_are_consistent(self) -> None:
        optical_offset = self.servo_calibration.optical_offset_from_servo(-80.0)
        recovered_servo = self.servo_calibration.servo_from_optical_offset(
            optical_offset
        )

        self.assertAlmostEqual(optical_offset, -9.059796361878045, places=9)
        self.assertAlmostEqual(recovered_servo, -80.0, places=9)
        self.assertAlmostEqual(
            self.servo_calibration.servo_from_optical_offset(0.0),
            -66.44393056759483,
            places=9,
        )

    def test_valid_range_does_not_silently_extrapolate(self) -> None:
        self.assertTrue(self.servo_calibration.is_servo_in_range(-45.0))
        self.assertTrue(self.servo_calibration.is_servo_in_range(-95.0))
        self.assertTrue(self.servo_calibration.is_servo_in_range(-44.9))
        self.assertTrue(self.servo_calibration.is_servo_in_range(-95.1))
        self.assertFalse(self.servo_calibration.is_servo_in_range(-44.2))
        self.assertFalse(self.servo_calibration.is_servo_in_range(-95.8))

    def test_current_target_keeps_three_angle_definitions_separate(self) -> None:
        mapper = PixelToWorldMapper(str(CAMERA_CALIBRATION))
        principal_x, principal_y = mapper.principal_point
        detection = Detection(
            (
                principal_x - 20.0,
                principal_y - 20.0,
                principal_x + 20.0,
                principal_y + 20.0,
            ),
            0.95,
            0,
            "foam_board",
        )
        estimator = TrajectoryEstimator(
            mapper=mapper,
            servo_angle_calibration=self.servo_calibration,
        )
        estimator.update([detection], timestamp=0.0)

        target = estimator.current_targets(
            min_age=1,
            camera_down_tilt_deg=30.0,
        )[0]

        self.assertEqual(target["angle_mode"], "servo_calibrated")
        self.assertAlmostEqual(target["optical_axis_offset_deg"], 0.0, places=3)
        self.assertAlmostEqual(target["servo_command_deg"], -66.444, places=3)
        self.assertAlmostEqual(
            target["legacy_mechanical_angle_deg"], 30.0, places=3
        )
        self.assertEqual(
            target["legacy_mechanical_angle_deg"], target["offset_angle_deg"]
        )
        self.assertTrue(target["servo_calibration_valid"])
        self.assertTrue(target["target_valid"])

    def test_target_outside_servo_fit_range_is_not_control_valid(self) -> None:
        mapper = PixelToWorldMapper(str(CAMERA_CALIBRATION))
        principal_x, _ = mapper.principal_point
        detection = Detection(
            (principal_x - 20.0, 439.0, principal_x + 20.0, 479.0),
            0.95,
            0,
            "foam_board",
        )
        estimator = TrajectoryEstimator(
            mapper=mapper,
            servo_angle_calibration=self.servo_calibration,
        )
        estimator.update([detection], timestamp=0.0)

        target = estimator.current_targets(min_age=1)[0]

        self.assertFalse(target["servo_calibration_valid"])
        self.assertFalse(target["target_valid"])

    def test_legacy_mode_remains_available_without_servo_calibration(self) -> None:
        mapper = PixelToWorldMapper(str(CAMERA_CALIBRATION))
        principal_x, principal_y = mapper.principal_point
        detection = Detection(
            (
                principal_x - 10.0,
                principal_y - 10.0,
                principal_x + 10.0,
                principal_y + 10.0,
            ),
            0.95,
            0,
            "foam_board",
        )
        estimator = TrajectoryEstimator(mapper=mapper)
        estimator.update([detection], timestamp=0.0)

        target = estimator.current_targets(
            min_age=1,
            camera_down_tilt_deg=30.0,
        )[0]

        self.assertEqual(target["angle_mode"], "legacy_geometric")
        self.assertNotIn("servo_command_deg", target)
        self.assertAlmostEqual(target["optical_axis_offset_deg"], 0.0, places=3)
        self.assertAlmostEqual(
            target["legacy_mechanical_angle_deg"], 30.0, places=3
        )

    def test_servo_overlay_draws_calibrated_axis_and_ticks(self) -> None:
        mapper = PixelToWorldMapper(str(CAMERA_CALIBRATION))
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        result = draw_arm_angle_overlay(
            image,
            mapper,
            camera_down_tilt_deg=30.0,
            tick_step_deg=5.0,
            servo_angle_calibration=self.servo_calibration,
        )

        center_x = int(round(mapper.principal_point[0]))
        center_y = int(round(mapper.principal_point[1]))
        self.assertIs(result, image)
        self.assertTrue(np.any(result[:, center_x] != 0))
        self.assertTrue(np.any(result[center_y, :] != 0))


class ServoAngleLookupCalibrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.calibration = ServoOpticalAngleCalibration.from_json(
            RED_MARKER_LUT_CALIBRATION
        )

    def test_lookup_nodes_and_inverse_are_consistent(self) -> None:
        for servo_angle in (-100.0, -95.0, -70.0, -45.0):
            optical_angle = self.calibration.optical_offset_from_servo(
                servo_angle
            )
            recovered = self.calibration.servo_from_optical_offset(
                optical_angle
            )
            self.assertAlmostEqual(recovered, servo_angle, places=9)

    def test_lookup_interpolates_between_five_degree_nodes(self) -> None:
        left = self.calibration.optical_offset_from_servo(-80.0)
        right = self.calibration.optical_offset_from_servo(-75.0)
        midpoint = self.calibration.optical_offset_from_servo(-77.5)

        self.assertAlmostEqual(midpoint, (left + right) / 2.0, places=9)
        self.assertAlmostEqual(
            self.calibration.servo_from_optical_offset(midpoint),
            -77.5,
            places=9,
        )

    def test_lookup_range_includes_endpoint_but_rejects_unobserved_edge(self) -> None:
        self.assertTrue(self.calibration.is_servo_in_range(-100.0))
        self.assertTrue(self.calibration.is_servo_in_range(-45.0))
        self.assertFalse(self.calibration.is_servo_in_range(-40.0))
        self.assertAlmostEqual(
            self.calibration.optical_offset_from_servo(-100.0),
            -14.6998,
            places=9,
        )



if __name__ == "__main__":
    unittest.main()
