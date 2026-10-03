"""Tests for angle-space Kalman filter and ensemble trajectory predictor."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from vision.bearing import BearingObservation  # noqa: E402
from vision.tracker import Detection, PixelToWorldMapper, TrajectoryEstimator  # noqa: E402
from vision.trajectory_predictor import (  # noqa: E402
    AngleKalmanFilter,
    EnsembleTrajectoryPredictor,
    FoamMotionPrior,
    bearing_history_from_pixel_track,
)

CALIBRATION = PROJECT_DIR / "configs" / "camera_2p1mm_640x480_fisheye.json"


# ---------------------------------------------------------------------------
# AngleKalmanFilter
# ---------------------------------------------------------------------------


class AngleKalmanFilterTest(unittest.TestCase):
    def test_initial_state_is_zero(self) -> None:
        akf = AngleKalmanFilter()
        self.assertFalse(akf.initialized)
        self.assertEqual(akf.current_angles, (0.0, 0.0))
        self.assertEqual(akf.velocity, (0.0, 0.0))

    def test_first_update_sets_position(self) -> None:
        akf = AngleKalmanFilter()
        akf.update((10.0, -5.0), timestamp=1.0)
        self.assertTrue(akf.initialized)
        self.assertEqual(akf.current_angles, (10.0, -5.0))
        self.assertEqual(akf.age, 1)

    def test_predict_zero_horizon_returns_current(self) -> None:
        akf = AngleKalmanFilter()
        akf.update((30.0, 15.0), timestamp=0.0)
        predicted = akf.predict(0.0)
        self.assertAlmostEqual(predicted[0], 30.0, places=8)
        self.assertAlmostEqual(predicted[1], 15.0, places=8)

    def test_constant_velocity_prediction(self) -> None:
        akf = AngleKalmanFilter(process_noise=1e-6, measurement_noise=1e-6)
        for i, t in enumerate((0.0, 0.1, 0.2, 0.3, 0.4)):
            yaw = 20.0 + 50.0 * t
            pitch = -10.0 + 30.0 * t
            akf.update((yaw, pitch), timestamp=t)
        # After converging, predict 0.4 s ahead
        predicted = akf.predict(0.4)
        self.assertAlmostEqual(predicted[0], 20.0 + 50.0 * 0.8, delta=2.0)
        self.assertAlmostEqual(predicted[1], -10.0 + 30.0 * 0.8, delta=2.0)

    def test_predict_trajectory_includes_current_and_endpoint(self) -> None:
        akf = AngleKalmanFilter()
        akf.update((5.0, 3.0), timestamp=0.0)
        akf.state[2] = 10.0  # yaw vel
        akf.state[3] = -5.0  # pitch vel
        trajectory = akf.predict_trajectory(horizon_seconds=0.4, steps=4)
        self.assertEqual(len(trajectory), 5)
        self.assertEqual(trajectory[0], (5.0, 3.0))
        self.assertAlmostEqual(trajectory[-1][0], 5.0 + 10.0 * 0.4, places=6)
        self.assertAlmostEqual(trajectory[-1][1], 3.0 - 5.0 * 0.4, places=6)

    def test_prediction_uncertainty_grows_with_horizon(self) -> None:
        akf = AngleKalmanFilter()
        akf.update((0.0, 0.0), timestamp=0.0)
        unc_short = akf.prediction_uncertainty(0.1)
        unc_long = akf.prediction_uncertainty(1.0)
        self.assertGreater(unc_long[0], unc_short[0])
        self.assertGreater(unc_long[1], unc_short[1])

    def test_uncertainty_is_non_negative(self) -> None:
        akf = AngleKalmanFilter()
        akf.update((0.0, 0.0), timestamp=0.0)
        unc = akf.prediction_uncertainty(0.5)
        self.assertGreaterEqual(unc[0], 0.0)
        self.assertGreaterEqual(unc[1], 0.0)


# ---------------------------------------------------------------------------
# FoamMotionPrior
# ---------------------------------------------------------------------------


class FoamMotionPriorTest(unittest.TestCase):
    def test_zero_smooth_factor_is_identity(self) -> None:
        prior = FoamMotionPrior(pitch_smooth_factor=0.0)
        out = prior.apply(10.0, 20.0, 5.0, 10.0)
        self.assertEqual(out, (10.0, 20.0))

    def test_full_smooth_factor_zeros_pitch_change(self) -> None:
        prior = FoamMotionPrior(pitch_smooth_factor=1.0)
        out = prior.apply(10.0, 50.0, 10.0, 20.0)
        self.assertAlmostEqual(out[0], 10.0)
        self.assertAlmostEqual(out[1], 20.0)  # pitch change fully damped

    def test_yaw_is_never_affected(self) -> None:
        prior = FoamMotionPrior(pitch_smooth_factor=0.5)
        out = prior.apply(30.0, 15.0, 0.0, 0.0)
        self.assertAlmostEqual(out[0], 30.0)
        self.assertLess(out[1], 15.0)  # pitch change damped


# ---------------------------------------------------------------------------
# EnsembleTrajectoryPredictor
# ---------------------------------------------------------------------------


class EnsembleTrajectoryPredictorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mapper = PixelToWorldMapper(str(CALIBRATION))
        cls.center = (320.0, 240.0)

    def test_ensemble_requires_deg_calibration(self) -> None:
        plain = PixelToWorldMapper()  # no calib → px
        with self.assertRaises(ValueError):
            EnsembleTrajectoryPredictor(plain)

    def test_predict_returns_expected_keys(self) -> None:
        ensemble = EnsembleTrajectoryPredictor(self.mapper)
        estimator = TrajectoryEstimator(mapper=self.mapper)
        detection = Detection((300.0, 200.0, 340.0, 240.0), 0.95, 0, "foam_board")
        estimator.update([detection], timestamp=0.0)
        track = estimator.tracks[0]
        track.age = 8
        track.state = np.asarray([320.0, 220.0, 20.0, 80.0])
        track._sync_from_state(shift_bbox=False)

        # Build up some angle history for the ensemble
        for t in np.linspace(0.0, 0.3, 10):
            px = (
                320.0 + 20.0 * t,
                220.0 + 80.0 * t,
            )
            angles = self.mapper.to_world(px)
            ensemble.update_with_track(track, timestamp=t)

        result = ensemble.predict(track, horizon_seconds=0.4, trajectory_steps=4)
        self.assertIn("predicted_angle", result)
        self.assertIn("trajectory", result)
        self.assertIn("method", result)
        self.assertIn("sources", result)
        self.assertEqual(result["method"], "ensemble")
        self.assertEqual(len(result["trajectory"]), 5)

    def test_predict_trajectory_is_monotonic_from_current(self) -> None:
        ensemble = EnsembleTrajectoryPredictor(self.mapper)
        estimator = TrajectoryEstimator(mapper=self.mapper)
        detection = Detection((300.0, 200.0, 340.0, 240.0), 0.95, 0, "foam_board")
        estimator.update([detection], timestamp=0.0)
        track = estimator.tracks[0]
        track.age = 8
        track.state = np.asarray([320.0, 220.0, 30.0, -60.0])
        track._sync_from_state(shift_bbox=False)

        for t in np.linspace(0.0, 0.3, 10):
            px = (
                320.0 + 30.0 * t,
                220.0 - 60.0 * t,
            )
            angles = self.mapper.to_world(px)
            ensemble.update_with_track(track, timestamp=t)

        result = ensemble.predict(track, horizon_seconds=0.4, trajectory_steps=8)
        traj = result["trajectory"]
        current = self.mapper.to_world(track.center)
        self.assertAlmostEqual(traj[0][0], round(current[0], 3), places=1)
        self.assertAlmostEqual(traj[0][1], round(current[1], 3), places=1)

    def test_cleanup_removes_per_track_state(self) -> None:
        ensemble = EnsembleTrajectoryPredictor(self.mapper)
        estimator = TrajectoryEstimator(mapper=self.mapper)
        detection = Detection((300.0, 200.0, 340.0, 240.0), 0.95, 0, "foam_board")
        estimator.update([detection], timestamp=0.0)
        track = estimator.tracks[0]

        ensemble.update_with_track(track, timestamp=0.0)
        self.assertIn(track.track_id, ensemble._angle_kalman_filters)
        self.assertIn(track.track_id, ensemble._angle_histories)

        ensemble.cleanup_track(track.track_id)
        self.assertNotIn(track.track_id, ensemble._angle_kalman_filters)
        self.assertNotIn(track.track_id, ensemble._angle_histories)

    def test_missed_kalman_state_is_not_fed_back_as_an_observation(self) -> None:
        ensemble = EnsembleTrajectoryPredictor(self.mapper)
        estimator = TrajectoryEstimator(
            mapper=self.mapper,
            ensemble_predictor=ensemble,
        )
        detection = Detection((300.0, 200.0, 340.0, 240.0), 0.95, 0, "foam_board")
        estimator.update([detection], timestamp=0.0)
        estimator.update([detection], timestamp=0.033)
        track_id = estimator.tracks[0].track_id
        observed_count = len(ensemble._angle_histories[track_id])

        estimator.update([], timestamp=0.066)

        self.assertEqual(len(ensemble._angle_histories[track_id]), observed_count)

    def test_angle_kalman_mode_reports_the_actual_method(self) -> None:
        ensemble = EnsembleTrajectoryPredictor(self.mapper, mode="angle_kalman")
        estimator = TrajectoryEstimator(
            mapper=self.mapper,
            ensemble_predictor=ensemble,
        )
        for index in range(8):
            cy = 220.0 + index * 2.0
            detection = Detection(
                (300.0, cy - 20.0, 340.0, cy + 20.0),
                0.95,
                0,
                "foam_board",
            )
            estimator.update([detection], timestamp=index / 30.0)

        target = estimator.arm_targets(
            predict_seconds=0.4,
            min_age=1,
            trajectory_steps=4,
        )[0]

        self.assertEqual(target["prediction_method"], "angle_kalman")
        self.assertEqual(len(target["trajectory_pixel"]), 5)
        if target["predicted_pixel"] is not None:
            expected = self.mapper.to_pixel(tuple(target["predicted_angle_deg"]))
            self.assertAlmostEqual(target["predicted_pixel"][0], expected[0], delta=0.01)
            self.assertAlmostEqual(target["predicted_pixel"][1], expected[1], delta=0.01)


# ---------------------------------------------------------------------------
# bearing_history_from_pixel_track
# ---------------------------------------------------------------------------


class BearingHistoryTest(unittest.TestCase):
    def test_empty_history_returns_empty(self) -> None:
        mapper = PixelToWorldMapper(str(CALIBRATION))
        estimator = TrajectoryEstimator(mapper=mapper)
        detection = Detection((300.0, 200.0, 340.0, 240.0), 0.95, 0, "foam_board")
        estimator.update([detection], timestamp=0.0)
        track = estimator.tracks[0]
        # Only one point in history — needs 2+
        observations = bearing_history_from_pixel_track(track, mapper)
        self.assertEqual(len(observations), 0)

    def test_two_point_history_produces_two_observations(self) -> None:
        mapper = PixelToWorldMapper(str(CALIBRATION))
        estimator = TrajectoryEstimator(mapper=mapper)
        detection = Detection((300.0, 200.0, 340.0, 240.0), 0.95, 0, "foam_board")
        estimator.update([detection], timestamp=0.0)
        estimator.update([detection], timestamp=0.033)
        track = estimator.tracks[0]
        observations = bearing_history_from_pixel_track(track, mapper)
        self.assertEqual(len(observations), 2)
        self.assertIsInstance(observations[0], BearingObservation)
        self.assertTrue(observations[0].timestamp < observations[1].timestamp)


# ---------------------------------------------------------------------------
# PixelToWorldMapper additions
# ---------------------------------------------------------------------------


class PixelToWorldMapperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mapper = PixelToWorldMapper(str(CALIBRATION))

    def test_to_pixel_round_trips_to_world(self) -> None:
        center = (320.0, 240.0)
        angles = self.mapper.to_world(center)
        pixel = self.mapper.to_pixel(angles)
        self.assertAlmostEqual(center[0], pixel[0], places=2)
        self.assertAlmostEqual(center[1], pixel[1], places=2)

    def test_to_pixel_near_corners(self) -> None:
        for pixel in ((80.0, 80.0), (560.0, 400.0), (100.0, 400.0)):
            angles = self.mapper.to_world(pixel)
            projected = self.mapper.to_pixel(angles)
            self.assertAlmostEqual(pixel[0], projected[0], places=2)
            self.assertAlmostEqual(pixel[1], projected[1], places=2)

    def test_calibration_info_contains_fisheye(self) -> None:
        info = self.mapper.calibration_info
        self.assertIn("fisheye", info.lower())
        self.assertIn("640", info)

    def test_plain_mapper_to_pixel_raises(self) -> None:
        plain = PixelToWorldMapper()
        with self.assertRaises(RuntimeError):
            plain.to_pixel((10.0, 5.0))

    def test_plain_mapper_calibration_info(self) -> None:
        plain = PixelToWorldMapper()
        self.assertEqual(plain.calibration_info, "no camera calibration loaded")


if __name__ == "__main__":
    unittest.main()
