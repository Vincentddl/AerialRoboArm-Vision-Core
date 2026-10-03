"""Angle-space Kalman filter and ensemble trajectory predictor.

This module complements the pixel-space Kalman tracker in ``trajectory.py``
by operating directly in bearing-angle space (yaw, pitch in degrees).  Because
the robotic arm ultimately needs angular targets, predicting in angle space
avoids the noise amplification that occurs when pixel predictions are
converted to angles.

Key components
--------------
* ``AngleKalmanFilter`` – 6-state Kalman filter in (yaw, pitch) space.
* ``FoamMotionPrior`` – lightweight prior for predominantly vertical motion.
* ``EnsembleTrajectoryPredictor`` – fuses pixel Kalman, angle Kalman and
  robust polynomial predictions into one output.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ballistic_predictor import BallisticPredictor
from .bearing import BearingMapper, BearingObservation, RobustBearingPredictor
from .tracker import PixelToWorldMapper, TrackState

# ---------------------------------------------------------------------------
# Angle-space Kalman filter
# ---------------------------------------------------------------------------


class AngleKalmanFilter:
    """4-state Kalman filter operating directly on bearing angles.

    State vector: ``[yaw, pitch, yaw_vel, pitch_vel]``
    where angles are in degrees and velocities in deg/s.

    Uses a constant-velocity (CV) model — well-suited for short-horizon
    prediction (≤ 0.4 s) of predominantly vertical foam-block motion.

    Parameters
    ----------
    process_noise:
        Continuous process-noise spectral density.  Higher values make the
        filter follow rapid angle changes more quickly.
    measurement_noise:
        Measurement noise variance per angle channel (deg²).  Higher values
        smooth detector jitter more aggressively.
    """

    def __init__(
        self,
        process_noise: float = 150.0,
        measurement_noise: float = 0.5,
    ) -> None:
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.state: np.ndarray = np.zeros(4, dtype=np.float64)
        self.covariance: np.ndarray = np.diag(
            [25.0, 25.0, 2500.0, 2500.0]
        ).astype(np.float64)
        self.last_time: float = 0.0
        self.initialized: bool = False
        self._age: int = 0

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def predict_to(self, timestamp: float) -> None:
        """Advance the state to *timestamp* without a measurement."""
        if not self.initialized:
            return
        dt = max(timestamp - self.last_time, 1e-6)

        # Constant-velocity transition
        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        dt2 = dt * dt
        G = np.array(
            [
                [0.5 * dt2, 0.0],
                [0.0, 0.5 * dt2],
                [dt, 0.0],
                [0.0, dt],
            ],
            dtype=np.float64,
        )
        Q = self.process_noise * G @ G.T

        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + Q
        self.last_time = timestamp

    def update(
        self, angles: Tuple[float, float], timestamp: float
    ) -> None:
        """Incorporate a new bearing measurement.

        Parameters
        ----------
        angles:
            ``(yaw_deg, pitch_deg)`` measurement.
        timestamp:
            Measurement time in seconds (must be monotonic).
        """
        if not self.initialized:
            self.state[0] = float(angles[0])
            self.state[1] = float(angles[1])
            self.last_time = timestamp
            self.initialized = True
            self._age = 1
            return

        self.predict_to(timestamp)

        H = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        R = np.eye(2, dtype=np.float64) * self.measurement_noise
        z = np.asarray(angles, dtype=np.float64)

        innovation = z - H @ self.state
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ innovation
        identity = np.eye(4, dtype=np.float64)
        self.covariance = (identity - K @ H) @ self.covariance
        self._age += 1

    def predict(self, horizon_seconds: float) -> Tuple[float, float]:
        """Return predicted ``(yaw, pitch)`` at *horizon_seconds* from now."""
        t = float(horizon_seconds)
        return (
            float(self.state[0] + self.state[2] * t),
            float(self.state[1] + self.state[3] * t),
        )

    def predict_trajectory(
        self, horizon_seconds: float, steps: int = 8
    ) -> List[Tuple[float, float]]:
        """Return a sampled future angle path.

        The first element is the current estimated position; the last is the
        full *horizon_seconds* ahead.
        """
        path: List[Tuple[float, float]] = [
            (float(self.state[0]), float(self.state[1]))
        ]
        for idx in range(1, steps + 1):
            t = float(horizon_seconds) * idx / steps
            path.append(self.predict(t))
        return path

    def prediction_uncertainty(
        self, horizon_seconds: float
    ) -> Tuple[float, float]:
        """Approximate 1-σ uncertainty of the predicted angles (deg)."""
        t = float(horizon_seconds)
        J = np.array(
            [
                [1.0, 0.0, t, 0.0],
                [0.0, 1.0, 0.0, t],
            ],
            dtype=np.float64,
        )
        pred_cov = J @ self.covariance @ J.T
        return (
            float(math.sqrt(max(pred_cov[0, 0], 0.0))),
            float(math.sqrt(max(pred_cov[1, 1], 0.0))),
        )

    @property
    def age(self) -> int:
        return self._age

    @property
    def current_angles(self) -> Tuple[float, float]:
        return (float(self.state[0]), float(self.state[1]))

    @property
    def velocity(self) -> Tuple[float, float]:
        return (float(self.state[2]), float(self.state[3]))


# ---------------------------------------------------------------------------
# Foam motion prior
# ---------------------------------------------------------------------------


class FoamMotionPrior:
    """Lightweight prior for foam-block vertical (pitch-dominant) motion.

    Foam blocks tossed upward/downward primarily move in the pitch direction
    from the camera's perspective.  Without a prior, Kalman filters sometimes
    produce implausibly aggressive pitch extrapolation on noisy data.

    This prior gently damps the pitch-change component of a prediction,
    leaving the yaw component untouched.
    """

    def __init__(self, pitch_smooth_factor: float = 0.12) -> None:
        self.pitch_smooth_factor = max(0.0, min(1.0, float(pitch_smooth_factor)))

    def apply(
        self,
        predicted_yaw: float,
        predicted_pitch: float,
        current_yaw: float,
        current_pitch: float,
    ) -> Tuple[float, float]:
        pitch_delta = predicted_pitch - current_pitch
        damped_delta = pitch_delta * (1.0 - self.pitch_smooth_factor)
        return (predicted_yaw, current_pitch + damped_delta)


# ---------------------------------------------------------------------------
# Ensemble trajectory predictor
# ---------------------------------------------------------------------------


class EnsembleTrajectoryPredictor:
    """Combine pixel Kalman, angle Kalman and polynomial predictions.

    The ensemble maintains per-track angle-Kalman filters and observation
    histories.  On each frame it feeds the current pixel-track state to every
    sub-predictor, then fuses their outputs with an uncertainty-aware
    weighted average.

    Usage sketch with an existing ``TrajectoryEstimator``::

        mapper = PixelToWorldMapper("calib.json")
        estimator = TrajectoryEstimator(mapper=mapper)
        ensemble = EnsembleTrajectoryPredictor(mapper)

        for frame in video:
            detections = model(frame)
            tracks = estimator.update(detections, timestamp)

            for track in tracks:
                ensemble.update_with_track(track, timestamp)

            for track in tracks:
                result = ensemble.predict(track, horizon_seconds=0.4)
                # result["predicted_angle"]  -> fused (yaw, pitch)
                # result["trajectory"]       -> sampled angle path
    """

    def __init__(
        self,
        mapper: PixelToWorldMapper,
        angle_process_noise: float = 150.0,
        angle_measurement_noise: float = 0.5,
        history_seconds: float = 0.35,
        pitch_smooth: float = 0.0,
        ballistic_predictor: Optional[BallisticPredictor] = None,
        mode: str = "ensemble",
    ) -> None:
        if mapper.unit != "deg":
            raise ValueError(
                "EnsembleTrajectoryPredictor requires a calibration that maps "
                f"to degrees, got unit={mapper.unit!r}"
            )
        self.mapper = mapper
        if mode not in {"ensemble", "angle_kalman"}:
            raise ValueError(f"unsupported predictor mode: {mode}")
        self.mode = mode
        self._angle_kalman_filters: Dict[int, AngleKalmanFilter] = {}
        self._angle_histories: Dict[int, List[Tuple[float, float, float]]] = {}
        self._last_pixel_states: Dict[
            int, Tuple[float, Tuple[float, float], Tuple[float, float]]
        ] = {}
        self._bearing_predictor = RobustBearingPredictor(
            history_seconds=history_seconds,
        )
        self._motion_prior = FoamMotionPrior(pitch_smooth_factor=pitch_smooth)
        self._history_seconds = float(history_seconds)
        self._angle_process_noise = float(angle_process_noise)
        self._angle_measurement_noise = float(angle_measurement_noise)
        self._ballistic = ballistic_predictor
        # Track per-method recent errors for dynamic weighting
        self._prediction_errors: Dict[int, Dict[str, float]] = defaultdict(
            lambda: {
                "pixel_kalman": 1.0,
                "angle_kalman": 1.0,
                "polynomial": 1.0,
                "ballistic": 1.0,
            }
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def update_with_track(self, track: TrackState, timestamp: float) -> None:
        """Feed the latest track state into all ensemble sub-predictors."""
        track_id = track.track_id
        current_angles = self.mapper.to_world(track.center)

        history = self._angle_histories.get(track_id, [])
        if history:
            dt = timestamp - history[-1][0]
            if dt > 1e-6:
                # Score every method against the new real observation before
                # incorporating it. Scoring a post-update filter against the
                # same measurement would make its error artificially zero.
                self._update_running_errors(
                    track_id, current_angles, dt
                )

        akf = self._get_angle_kalman(track_id)
        akf.update(current_angles, timestamp)

        if track_id not in self._angle_histories:
            self._angle_histories[track_id] = []
        self._angle_histories[track_id].append(
            (timestamp, current_angles[0], current_angles[1])
        )
        cutoff = timestamp - self._history_seconds
        self._angle_histories[track_id] = [
            entry
            for entry in self._angle_histories[track_id]
            if entry[0] >= cutoff
        ]

        # Feed ballistic predictor
        if self._ballistic is not None:
            self._ballistic.update(track_id, current_angles, timestamp)
        self._last_pixel_states[track_id] = (
            timestamp,
            track.center,
            track.velocity,
        )

    def predict(
        self,
        track: TrackState,
        horizon_seconds: float = 0.4,
        trajectory_steps: int = 8,
    ) -> dict:
        """Generate a fused angle-trajectory prediction.

        Returns a dict with keys ``predicted_angle``, ``trajectory``,
        ``method`` and ``sources``.
        """
        trajectory_steps = max(1, int(trajectory_steps))
        track_id = track.track_id
        current_angles = self.mapper.to_world(track.center)

        # ---- source 1: pixel Kalman (existing tracker) -------------------
        pixel_predicted_px = track.predict_pixel(horizon_seconds)
        pixel_angles = self.mapper.to_world(pixel_predicted_px)

        # ---- source 2: angle Kalman --------------------------------------
        akf = self._get_angle_kalman(track_id)
        if akf.initialized:
            angle_kf_angles = akf.predict(horizon_seconds)
            angle_kf_unc = akf.prediction_uncertainty(horizon_seconds)
        else:
            angle_kf_angles = current_angles
            angle_kf_unc = (99.0, 99.0)

        # ---- source 3: robust polynomial fit -----------------------------
        poly_angles: Optional[Tuple[float, float]] = None
        history = self._angle_histories.get(track_id, [])
        if len(history) >= self._bearing_predictor.min_samples:
            try:
                observations = [
                    BearingObservation(
                        timestamp=ts,
                        yaw_deg=ya,
                        pitch_deg=pi,
                        score=0.9,
                    )
                    for ts, ya, pi in history
                ]
                poly_pred = self._bearing_predictor.predict(
                    observations, horizon_seconds, "angular_acceleration"
                )
                poly_angles = (poly_pred.yaw_deg, poly_pred.pitch_deg)
            except (ValueError, np.linalg.LinAlgError):
                poly_angles = None

        # ---- source 4: ballistic predictor -------------------------------
        ballistic_angles: Optional[Tuple[float, float]] = None
        if self._ballistic is not None:
            ballistic_angles = self._ballistic.predict(track_id, horizon_seconds)

        # ---- fuse --------------------------------------------------------
        if self.mode == "angle_kalman":
            fused_yaw, fused_pitch = angle_kf_angles
        else:
            fused_yaw, fused_pitch = self._fuse(
                track_id,
                horizon_seconds,
                current_angles,
                pixel_angles,
                angle_kf_angles,
                angle_kf_unc,
                poly_angles,
                ballistic_angles,
            )

        # Apply foam motion prior
        fused_yaw, fused_pitch = self._motion_prior.apply(
            fused_yaw, fused_pitch, current_angles[0], current_angles[1]
        )

        # ---- build trajectory --------------------------------------------
        if self.mode == "angle_kalman":
            raw_trajectory = akf.predict_trajectory(
                horizon_seconds, steps=trajectory_steps
            )
        elif self._ballistic is not None:
            ballistic_path = self._ballistic.predict_trajectory(
                track_id, horizon_seconds, steps=trajectory_steps
            )
            if ballistic_path:
                end_correction = (
                    fused_yaw - ballistic_path[-1][0],
                    fused_pitch - ballistic_path[-1][1],
                )
                raw_trajectory = [
                    (
                        point[0] + (index / trajectory_steps) * end_correction[0],
                        point[1] + (index / trajectory_steps) * end_correction[1],
                    )
                    for index, point in enumerate(ballistic_path)
                ]
            else:
                raw_trajectory = []
        else:
            raw_trajectory = []

        if not raw_trajectory:
            raw_trajectory = [
                (
                    current_angles[0]
                    + (index / trajectory_steps) * (fused_yaw - current_angles[0]),
                    current_angles[1]
                    + (index / trajectory_steps) * (fused_pitch - current_angles[1]),
                )
                for index in range(trajectory_steps + 1)
            ]
        trajectory = [
            (round(float(yaw), 3), round(float(pitch), 3))
            for yaw, pitch in raw_trajectory
        ]

        available_sources = [pixel_angles, angle_kf_angles]
        if poly_angles is not None:
            available_sources.append(poly_angles)
        if ballistic_angles is not None:
            available_sources.append(ballistic_angles)
        disagreement = math.sqrt(
            sum(
                (source[0] - fused_yaw) ** 2 + (source[1] - fused_pitch) ** 2
                for source in available_sources
            )
            / max(len(available_sources), 1)
        )
        kalman_uncertainty = math.hypot(*angle_kf_unc)
        uncertainty_deg = max(disagreement, kalman_uncertainty)

        return {
            "predicted_angle": (
                round(float(fused_yaw), 3),
                round(float(fused_pitch), 3),
            ),
            "trajectory": trajectory,
            "method": self.mode,
            "uncertainty_deg": round(float(uncertainty_deg), 3),
            "sources": {
                "pixel_kalman": (
                    round(pixel_angles[0], 3),
                    round(pixel_angles[1], 3),
                ),
                "angle_kalman": (
                    round(angle_kf_angles[0], 3),
                    round(angle_kf_angles[1], 3),
                ),
                "polynomial": (
                    (round(poly_angles[0], 3), round(poly_angles[1], 3))
                    if poly_angles is not None
                    else None
                ),
                "ballistic": (
                    (round(ballistic_angles[0], 3), round(ballistic_angles[1], 3))
                    if ballistic_angles is not None
                    else None
                ),
            },
        }

    def cleanup_track(self, track_id: int) -> None:
        """Release per-track state when the tracker drops a track."""
        self._angle_kalman_filters.pop(track_id, None)
        self._angle_histories.pop(track_id, None)
        self._last_pixel_states.pop(track_id, None)
        self._prediction_errors.pop(track_id, None)
        if self._ballistic is not None:
            self._ballistic.cleanup_track(track_id)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _get_angle_kalman(self, track_id: int) -> AngleKalmanFilter:
        if track_id not in self._angle_kalman_filters:
            self._angle_kalman_filters[track_id] = AngleKalmanFilter(
                process_noise=self._angle_process_noise,
                measurement_noise=self._angle_measurement_noise,
            )
        return self._angle_kalman_filters[track_id]

    def _update_running_errors(
        self,
        track_id: int,
        current_angles: Tuple[float, float],
        dt: float,
    ) -> None:
        """Score predictions made from the previous real observation."""
        errors = self._prediction_errors[track_id]
        previous_pixel = self._last_pixel_states.get(track_id)
        if previous_pixel is not None:
            _, prev_center, vel_pixel = previous_pixel
            predicted_px = (
                prev_center[0] + vel_pixel[0] * dt,
                prev_center[1] + vel_pixel[1] * dt,
            )
            predicted = self.mapper.to_world(predicted_px)
            pixel_err = math.hypot(
                predicted[0] - current_angles[0],
                predicted[1] - current_angles[1],
            )
            errors["pixel_kalman"] = 0.7 * errors["pixel_kalman"] + 0.3 * pixel_err

        # Angle Kalman state is still at the previous observation here.
        akf = self._angle_kalman_filters.get(track_id)
        if akf is not None and akf.initialized:
            akf_pred = akf.predict(dt)
            akf_err = math.hypot(
                akf_pred[0] - current_angles[0],
                akf_pred[1] - current_angles[1],
            )
            errors["angle_kalman"] = 0.7 * errors["angle_kalman"] + 0.3 * akf_err

        # Polynomial history does not yet contain the current observation.
        history = self._angle_histories.get(track_id, [])
        if len(history) >= self._bearing_predictor.min_samples:
            try:
                observations = [
                    BearingObservation(
                        timestamp=ts,
                        yaw_deg=ya,
                        pitch_deg=pi,
                        score=0.9,
                    )
                    for ts, ya, pi in history
                ]
                if len(observations) >= self._bearing_predictor.min_samples:
                    poly_pred = self._bearing_predictor.predict(
                        observations, dt, "angular_acceleration"
                    )
                    poly_err = math.hypot(
                        poly_pred.yaw_deg - current_angles[0],
                        poly_pred.pitch_deg - current_angles[1],
                    )
                    errors["polynomial"] = 0.7 * errors["polynomial"] + 0.3 * poly_err
            except (ValueError, np.linalg.LinAlgError):
                pass

        # Ballistic state also still ends at the previous observation.
        if self._ballistic is not None:
            bal_pred = self._ballistic.predict(track_id, dt)
            if bal_pred is not None:
                bal_err = math.hypot(
                    bal_pred[0] - current_angles[0],
                    bal_pred[1] - current_angles[1],
                )
                errors["ballistic"] = 0.7 * errors.get("ballistic", 1.0) + 0.3 * bal_err

    def _fuse(
        self,
        track_id: int,
        horizon_seconds: float,
        current: Tuple[float, float],
        pixel: Tuple[float, float],
        angle_kf: Tuple[float, float],
        kf_unc: Tuple[float, float],
        poly: Optional[Tuple[float, float]],
        ballistic: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float]:
        """Weighted fusion of prediction sources with horizon-dependent weights.

        At short horizons (&lt;100 ms) the pixel Kalman dominates because its
        constant-velocity assumption holds well.  At long horizons (&gt;250 ms)
        the ballistic predictor is up-weighted because it is the only source
        that models gravity — CV/CA models diverge catastrophically beyond
        ~250 ms on thrown objects.
        """
        errors = self._prediction_errors.get(
            track_id,
            {
                "pixel_kalman": 1.0,
                "angle_kalman": 1.0,
                "polynomial": 1.0,
                "ballistic": 1.0,
            },
        )

        # Convert errors to weights (inverse-error)
        def _inv(e: float) -> float:
            return 1.0 / max(e, 1e-6)

        w_pixel = _inv(errors["pixel_kalman"])
        w_angle = _inv(errors["angle_kalman"])

        # Reduce angle-Kalman weight when uncertainty is high
        unc_total = kf_unc[0] + kf_unc[1]
        if unc_total > 10.0:
            w_angle *= 0.3
        elif unc_total > 5.0:
            w_angle *= 0.6

        if poly is not None:
            w_poly = _inv(errors["polynomial"])
        else:
            w_poly = 0.0

        # Ballistic weight — available only with a ballistic predictor
        if ballistic is not None and self._ballistic is not None:
            w_ballistic = _inv(errors["ballistic"])
            # Horizon-dependent scaling:
            #   < 100 ms: down-weight (CV is good enough)
            #   > 250 ms: up-weight (CV/CA diverge, ballistic needed)
            h = float(horizon_seconds)
            if h < 0.10:
                w_ballistic *= 0.40
            elif h < 0.20:
                w_ballistic *= 0.75
            elif h > 0.30:
                w_ballistic *= 1.80
            elif h > 0.25:
                w_ballistic *= 1.40
        else:
            w_ballistic = 0.0

        total = w_pixel + w_angle + w_poly + w_ballistic
        if total < 1e-9:
            # All methods have near-zero weight — fall back to equal blend
            w_pixel = 1.0
            w_angle = 1.0
            w_poly = 1.0 if poly is not None else 0.0
            w_ballistic = 1.0 if ballistic is not None else 0.0
            total = w_pixel + w_angle + w_poly + w_ballistic

        w_pixel /= total
        w_angle /= total
        w_poly /= total
        w_ballistic /= total

        # Weighted blend of displacement-from-current
        fused_yaw = current[0]
        fused_pitch = current[1]
        fused_yaw += w_pixel * (pixel[0] - current[0])
        fused_pitch += w_pixel * (pixel[1] - current[1])
        fused_yaw += w_angle * (angle_kf[0] - current[0])
        fused_pitch += w_angle * (angle_kf[1] - current[1])
        if poly is not None:
            fused_yaw += w_poly * (poly[0] - current[0])
            fused_pitch += w_poly * (poly[1] - current[1])
        if ballistic is not None:
            fused_yaw += w_ballistic * (ballistic[0] - current[0])
            fused_pitch += w_ballistic * (ballistic[1] - current[1])

        return (fused_yaw, fused_pitch)


# ---------------------------------------------------------------------------
# Utility: generate bearing observations from a pixel track history
# ---------------------------------------------------------------------------


def bearing_history_from_pixel_track(
    track: TrackState,
    mapper: PixelToWorldMapper,
    nominal_fps: float = 30.0,
) -> List[BearingObservation]:
    """Convert a pixel-track's ``.history`` to ``BearingObservation`` list.

    Timestamps are reconstructed assuming a constant frame rate, which is
    adequate for short histories up to ~0.5 s.  Use recorded timestamps for
    offline evaluation.
    """
    observations: List[BearingObservation] = []
    if len(track.history) < 2:
        return observations
    for idx, pixel in enumerate(track.history):
        t = track.last_time - (len(track.history) - 1 - idx) / nominal_fps
        angles = mapper.to_world(pixel)
        observations.append(
            BearingObservation(
                timestamp=t,
                yaw_deg=angles[0],
                pitch_deg=angles[1],
                score=track.score,
                pixel=pixel,
            )
        )
    return observations
