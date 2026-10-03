"""Gravity-aware ballistic trajectory predictor.

This module adds a physics-informed predictor that models the foam board's
motion as a ballistic (parabolic) trajectory in bearing-angle space. Unlike
constant-velocity or polynomial extrapolation, it explicitly represents an
apparent gravity term. It is still an approximation and must be validated on
the actual throw geometry before control use.

Key components
--------------
* ``BallisticPredictor`` — per-track online g_eff estimator + ballistic
  extrapolation.

Theory
------
In the camera's bearing-angle frame, a tossed foam board follows:

    yaw(t)   = yaw_0   + yaw_vel * t                         (no gravity in yaw)
    pitch(t) = pitch_0 + pitch_vel * t + 0.5 * g_eff * t²    (gravity in pitch)

where *g_eff* is the *apparent* gravity in deg/s² — the projection of
real-world gravity onto the camera's pitch axis.  Because the camera is
fixed and the throw direction is roughly consistent, g_eff is approximately
constant during a single throw.

The predictor learns g_eff online via an exponential moving average of
consecutive pitch-velocity deltas, then uses that estimate for extrapolation.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Ballistic predictor
# ---------------------------------------------------------------------------


class BallisticPredictor:
    """Per-track ballistic trajectory predictor with online g_eff estimation.

    The horizontal (yaw) component uses a constant-velocity model.
    The vertical (pitch) component uses a constant-acceleration model
    where the acceleration *g_eff* is estimated online from recent
    pitch-velocity changes.

    Parameters
    ----------
    ema_alpha:
        Smoothing factor for the online g_eff estimate.  0.3 means the new
        measurement contributes 30 % and the old estimate keeps 70 %.
    min_history:
        Minimum pitch-velocity deltas needed before the predictor activates.
    max_speed_deg_s:
        Clamp for yaw/pitch velocity magnitudes.
    max_gravity_deg_s2:
        Clamp for |g_eff| magnitude.
    initial_gravity_deg_s2:
        Initial g_eff prior (deg/s²).  A negative value means the pitch
        decelerates over time (object rising, slowing down), which is the
        typical case when foam boards are tossed upward in front of the camera.
    """

    def __init__(
        self,
        ema_alpha: float = 0.30,
        min_history: int = 3,
        max_speed_deg_s: float = 600.0,
        max_gravity_deg_s2: float = 3000.0,
        initial_gravity_deg_s2: float = 400.0,
    ) -> None:
        self.ema_alpha = float(ema_alpha)
        self.min_history = int(min_history)
        self.max_speed = float(max_speed_deg_s)
        self.max_gravity = float(max_gravity_deg_s2)
        self.initial_gravity = float(initial_gravity_deg_s2)

        # Per-track state: track_id -> {yaw_vel, pitch_vel, g_eff, last_angles, last_time, initialized}
        self._state: Dict[int, dict] = {}
        # Per-track velocity history for g_eff estimation
        self._vel_history: Dict[int, list] = {}

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def update(
        self,
        track_id: int,
        angles: Tuple[float, float],
        timestamp: float,
    ) -> None:
        """Feed a new bearing observation into the predictor.

        Parameters
        ----------
        track_id:
            Unique identifier for the track.
        angles:
            ``(yaw_deg, pitch_deg)`` at the current frame.
        timestamp:
            Monotonic timestamp in seconds.
        """
        if track_id not in self._state:
            self._state[track_id] = {
                "yaw_vel": 0.0,
                "pitch_vel": 0.0,
                "g_eff": self.initial_gravity,
                "last_angles": angles,
                "last_time": timestamp,
                "initialized": False,
            }
            self._vel_history[track_id] = []
            return

        st = self._state[track_id]
        dt = max(timestamp - st["last_time"], 1e-6)

        yaw, pitch = angles
        prev_yaw, prev_pitch = st["last_angles"]

        # Compute instantaneous velocities (deg/s)
        raw_yaw_vel = (yaw - prev_yaw) / dt
        raw_pitch_vel = (pitch - prev_pitch) / dt

        # Clamp
        yaw_vel = max(-self.max_speed, min(self.max_speed, raw_yaw_vel))
        pitch_vel = max(-self.max_speed, min(self.max_speed, raw_pitch_vel))

        # EMA-smooth velocities
        if st["initialized"]:
            alpha_v = 0.35  # velocity smoothing
            st["yaw_vel"] = alpha_v * yaw_vel + (1.0 - alpha_v) * st["yaw_vel"]
            st["pitch_vel"] = alpha_v * pitch_vel + (1.0 - alpha_v) * st["pitch_vel"]
        else:
            st["yaw_vel"] = yaw_vel
            st["pitch_vel"] = pitch_vel
            st["initialized"] = True

        # Track pitch-velocity deltas for g_eff estimation
        vel_h = self._vel_history[track_id]
        vel_h.append((timestamp, st["pitch_vel"]))
        # Keep at most 2x min_history entries
        max_keep = self.min_history * 2
        if len(vel_h) > max_keep:
            vel_h = vel_h[-max_keep:]
            self._vel_history[track_id] = vel_h

        # Update g_eff from pitch-velocity deltas
        if len(vel_h) >= self.min_history:
            g_samples = []
            for i in range(1, len(vel_h)):
                t1, v1 = vel_h[i - 1]
                t2, v2 = vel_h[i]
                dv = v2 - v1
                dt_v = max(t2 - t1, 1e-6)
                g_measured = dv / dt_v
                g_samples.append(g_measured)

            if g_samples:
                g_instant = float(np.median(g_samples))
                g_instant = max(-self.max_gravity, min(self.max_gravity, g_instant))
                st["g_eff"] = (
                    self.ema_alpha * g_instant
                    + (1.0 - self.ema_alpha) * st["g_eff"]
                )

        st["last_angles"] = angles
        st["last_time"] = timestamp

    def predict(
        self,
        track_id: int,
        horizon_seconds: float,
    ) -> Optional[Tuple[float, float]]:
        """Predict ``(yaw, pitch)`` at *horizon_seconds* from now.

        Returns ``None`` if the predictor has not been initialized yet.
        """
        st = self._state.get(track_id)
        if st is None or not st["initialized"]:
            return None

        t = float(horizon_seconds)
        yaw, pitch = st["last_angles"]
        pred_yaw = yaw + st["yaw_vel"] * t
        pred_pitch = pitch + st["pitch_vel"] * t + 0.5 * st["g_eff"] * t * t
        return (float(pred_yaw), float(pred_pitch))

    def predict_trajectory(
        self,
        track_id: int,
        horizon_seconds: float,
        steps: int = 8,
    ) -> Optional[list]:
        """Return a sampled future angle path or ``None``.

        The first element is the current position; the last is the full
        *horizon_seconds* ahead.
        """
        st = self._state.get(track_id)
        if st is None or not st["initialized"]:
            return None

        yaw, pitch = st["last_angles"]
        path = [(float(yaw), float(pitch))]
        for idx in range(1, steps + 1):
            t = horizon_seconds * idx / steps
            path.append(
                (
                    float(yaw + st["yaw_vel"] * t),
                    float(pitch + st["pitch_vel"] * t + 0.5 * st["g_eff"] * t * t),
                )
            )
        return path

    def get_state(self, track_id: int) -> Optional[dict]:
        """Return a copy of the per-track state dict, or ``None``."""
        st = self._state.get(track_id)
        if st is None:
            return None
        return {
            "yaw_vel": st["yaw_vel"],
            "pitch_vel": st["pitch_vel"],
            "g_eff": st["g_eff"],
            "last_angles": st["last_angles"],
            "initialized": st["initialized"],
        }

    def cleanup_track(self, track_id: int) -> None:
        """Release per-track state when the tracker drops a track."""
        self._state.pop(track_id, None)
        self._vel_history.pop(track_id, None)

    @property
    def active_track_count(self) -> int:
        return len(self._state)
