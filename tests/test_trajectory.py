from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from vision.bearing import BearingMapper  # noqa: E402
from vision.tracker import (  # noqa: E402
    Detection,
    PixelToWorldMapper,
    TrajectoryEstimator,
    draw_arm_angle_overlay,
)


CALIBRATION = PROJECT_DIR / "configs" / "camera_2p1mm_640x480_fisheye.json"


class TrajectoryPredictionTest(unittest.TestCase):
    def test_calibrated_mapper_uses_the_same_undistorted_angles_as_bearing_mapper(self) -> None:
        point = (80.0, 80.0)
        trajectory_mapper = PixelToWorldMapper(str(CALIBRATION))
        bearing_mapper = BearingMapper(CALIBRATION)

        self.assertEqual(trajectory_mapper.unit, "deg")
        actual = trajectory_mapper.to_world(point)
        expected = bearing_mapper.pixel_to_angles(point)
        self.assertAlmostEqual(actual[0], expected[0], places=9)
        self.assertAlmostEqual(actual[1], expected[1], places=9)

    def test_arm_target_contains_a_0p4_second_path_and_endpoint(self) -> None:
        mapper = PixelToWorldMapper(str(CALIBRATION))
        estimator = TrajectoryEstimator(mapper=mapper)
        detection = Detection((300.0, 200.0, 340.0, 240.0), 0.95, 0, "foam_board")
        estimator.update([detection], timestamp=0.0)
        track = estimator.tracks[0]
        track.age = 8
        track.state = np.asarray([320.0, 220.0, 10.0, 100.0])
        track._sync_from_state(shift_bbox=False)

        target = estimator.arm_targets(predict_seconds=0.4, min_age=1, trajectory_steps=4)[0]

        self.assertEqual(target["predict_seconds"], 0.4)
        self.assertEqual(len(target["trajectory_pixel"]), 5)
        self.assertEqual(len(target["trajectory_world"]), 5)
        self.assertEqual(target["trajectory_pixel"][-1], target["predicted_pixel"])
        self.assertEqual(target["trajectory_world"][-1], target["predicted_world"])
        # CV: 320 + 10*0.4 = 324, 220 + 100*0.4 = 260
        self.assertEqual(target["predicted_pixel"], [324.0, 260.0])
        self.assertEqual(target["world_unit"], "deg")
        self.assertEqual(
            target["predicted_angular_displacement_deg"],
            [
                round(target["predicted_angle_deg"][0] - target["angle_deg"][0], 3),
                round(target["predicted_angle_deg"][1] - target["angle_deg"][1], 3),
            ],
        )
        self.assertAlmostEqual(
            target["offset_angle_deg"],
            mapper.arm_angle_deg(mapper.to_world(track.center)),
            places=3,
        )
        self.assertAlmostEqual(
            target["predicted_offset_angle_deg"],
            mapper.arm_angle_deg(
                mapper.to_world(tuple(target["predicted_pixel"]))
            ),
            places=3,
        )
        self.assertEqual(
            target["offset_angle_convention"],
            "signed angle from mechanical horizontal x-axis; positive downward",
        )

    def test_camera_down_tilt_shifts_single_axis_arm_angle(self) -> None:
        mapper = PixelToWorldMapper(str(CALIBRATION))
        estimator = TrajectoryEstimator(mapper=mapper)
        detection = Detection((300.0, 200.0, 340.0, 240.0), 0.95, 0, "foam_board")
        estimator.update([detection], timestamp=0.0)

        level_target = estimator.arm_targets(
            predict_seconds=0.4,
            min_age=1,
            camera_down_tilt_deg=0.0,
        )[0]
        tilted_target = estimator.arm_targets(
            predict_seconds=0.4,
            min_age=1,
            camera_down_tilt_deg=30.0,
        )[0]

        self.assertAlmostEqual(
            tilted_target["offset_angle_deg"]
            - level_target["offset_angle_deg"],
            30.0,
            places=3,
        )

    def test_principal_point_has_zero_optical_axis_offset(self) -> None:
        mapper = PixelToWorldMapper(str(CALIBRATION))
        self.assertIsNotNone(mapper.principal_point)
        angles = mapper.to_world(mapper.principal_point)

        self.assertAlmostEqual(mapper.optical_axis_offset_deg(angles), 0.0, places=7)

    def test_current_target_uses_detection_center_without_prediction(self) -> None:
        mapper = PixelToWorldMapper(str(CALIBRATION))
        estimator = TrajectoryEstimator(mapper=mapper)
        detection = Detection(
            (300.0, 200.0, 340.0, 240.0), 0.95, 0, "foam_board"
        )
        estimator.update([detection], timestamp=0.0)
        track = estimator.tracks[0]
        track.state = np.asarray([310.0, 210.0, 500.0, 500.0])
        track._sync_from_state(shift_bbox=False)

        target = estimator.current_targets(
            min_age=1,
            camera_down_tilt_deg=30.0,
            lateral_tolerance_deg=5.0,
        )[0]

        self.assertEqual(target["target_mode"], "current")
        self.assertTrue(target["target_valid"])
        self.assertEqual(target["pixel"], [320.0, 220.0])
        self.assertEqual(target["bbox_size_px"], [40.0, 40.0])
        self.assertAlmostEqual(
            target["bbox_area_ratio"], 1600.0 / (640.0 * 480.0), places=6
        )
        self.assertIn("angle_deg", target)
        self.assertIn("offset_angle_deg", target)
        self.assertEqual(target["offset_angle_deg"], target["arm_angle_deg"])
        self.assertEqual(target["camera_down_tilt_deg"], 30.0)
        self.assertTrue(target["lateral_valid"])
        self.assertNotIn("predicted_pixel", target)
        self.assertNotIn("predicted_angle_deg", target)
        self.assertNotIn("trajectory_pixel", target)
        self.assertNotIn("prediction_valid", target)

    def test_stationary_hand_held_target_can_be_confirmed(self) -> None:
        estimator = TrajectoryEstimator(
            new_track_confirmation_frames=2,
            new_track_min_motion=0.0,
        )
        detection = Detection(
            (300.0, 200.0, 340.0, 240.0), 0.95, 0, "foam_board"
        )

        estimator.update([detection], timestamp=0.0)
        estimator.update([detection], timestamp=0.1)

        targets = estimator.current_targets(min_age=2)
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0]["pixel"], [320.0, 220.0])

    def test_mechanical_horizontal_is_zero_after_30_degree_compensation(self) -> None:
        mapper = PixelToWorldMapper(str(CALIBRATION))

        arm_angle = mapper.arm_angle_deg(
            (0.0, -30.0),
            camera_down_tilt_deg=30.0,
        )

        self.assertAlmostEqual(arm_angle, 0.0, places=9)

    def test_off_axis_target_is_invalid_for_single_axis_arm(self) -> None:
        mapper = PixelToWorldMapper(str(CALIBRATION))
        estimator = TrajectoryEstimator(mapper=mapper)
        detection = Detection(
            (40.0, 200.0, 80.0, 240.0), 0.95, 0, "foam_board"
        )
        estimator.update([detection], timestamp=0.0)

        target = estimator.current_targets(
            min_age=1,
            camera_down_tilt_deg=30.0,
            lateral_tolerance_deg=5.0,
        )[0]

        self.assertFalse(target["lateral_valid"])
        self.assertFalse(target["target_valid"])

    def test_arm_angle_scale_uses_calibrated_projection(self) -> None:
        mapper = PixelToWorldMapper(str(CALIBRATION))
        pixel = mapper.arm_angle_to_pixel(
            30.0,
            camera_down_tilt_deg=30.0,
        )

        self.assertAlmostEqual(pixel[0], mapper.principal_point[0], places=7)
        self.assertAlmostEqual(pixel[1], mapper.principal_point[1], places=7)

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_arm_angle_overlay(
            image,
            mapper,
            camera_down_tilt_deg=30.0,
            tick_step_deg=5.0,
        )

        center_x = int(round(mapper.principal_point[0]))
        center_y = int(round(mapper.principal_point[1]))
        self.assertIs(result, image)
        self.assertTrue(np.any(result[:, center_x] != 0))
        self.assertTrue(np.any(result[center_y, :] != 0))


if __name__ == "__main__":
    unittest.main()
