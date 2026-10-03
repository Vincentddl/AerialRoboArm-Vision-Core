from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


AnglePair = Tuple[float, float]
Point = Tuple[float, float]


@dataclass(frozen=True)
class BearingObservation:
    timestamp: float
    yaw_deg: float
    pitch_deg: float
    score: float = 1.0
    frame_index: int = -1
    pixel: Point = (math.nan, math.nan)


@dataclass(frozen=True)
class BearingPrediction:
    method: str
    yaw_deg: float
    pitch_deg: float
    residual_deg: float
    sample_count: int
    history_span_s: float


class BearingMapper:
    """Convert distorted image pixels to camera-frame bearing angles and rays."""

    def __init__(self, calibration_path: str | Path):
        calibration_path = Path(calibration_path)
        data = json.loads(calibration_path.read_text(encoding="utf-8"))
        self.model = str(data["model"])
        if self.model not in {"pinhole", "fisheye"}:
            raise ValueError(f"unsupported camera model: {self.model}")

        self.camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
        self.dist_coeffs = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
        self.image_size = tuple(int(value) for value in data["image_size"])
        self.angle_offset_deg = np.asarray(data.get("angle_offset_deg", [0.0, 0.0]), dtype=np.float64)
        if self.camera_matrix.shape != (3, 3):
            raise ValueError("camera_matrix must be 3x3")

    def pixel_to_ray(self, point: Point) -> np.ndarray:
        source = np.asarray([[point]], dtype=np.float64)
        if self.model == "fisheye":
            normalized = cv2.fisheye.undistortPoints(
                source, self.camera_matrix, self.dist_coeffs
            )[0, 0]
        else:
            normalized = cv2.undistortPoints(
                source, self.camera_matrix, self.dist_coeffs
            )[0, 0]
        ray = np.asarray([normalized[0], normalized[1], 1.0], dtype=np.float64)
        return ray / np.linalg.norm(ray)

    def pixel_to_angles(self, point: Point) -> AnglePair:
        ray = self.pixel_to_ray(point)
        yaw = math.degrees(math.atan2(float(ray[0]), float(ray[2])))
        pitch = math.degrees(
            math.atan2(float(ray[1]), math.hypot(float(ray[0]), float(ray[2])))
        )
        return (
            yaw + float(self.angle_offset_deg[0]),
            pitch + float(self.angle_offset_deg[1]),
        )

    def angles_to_ray(self, angles: AnglePair) -> np.ndarray:
        yaw = math.radians(angles[0] - float(self.angle_offset_deg[0]))
        pitch = math.radians(angles[1] - float(self.angle_offset_deg[1]))
        ray = np.asarray(
            [
                math.sin(yaw) * math.cos(pitch),
                math.sin(pitch),
                math.cos(yaw) * math.cos(pitch),
            ],
            dtype=np.float64,
        )
        return ray / np.linalg.norm(ray)

    def angles_to_pixel(self, angles: AnglePair) -> Point:
        point = self.angles_to_ray(angles).reshape(1, 1, 3)
        zero = np.zeros((3, 1), dtype=np.float64)
        if self.model == "fisheye":
            projected, _ = cv2.fisheye.projectPoints(
                point, zero, zero, self.camera_matrix, self.dist_coeffs
            )
        else:
            projected, _ = cv2.projectPoints(
                point, zero, zero, self.camera_matrix, self.dist_coeffs
            )
        pixel = projected[0, 0]
        return float(pixel[0]), float(pixel[1])

    def is_visible(self, angles: AnglePair, margin_px: float = 0.0) -> bool:
        if self.angles_to_ray(angles)[2] <= 0.0:
            return False
        u, v = self.angles_to_pixel(angles)
        width, height = self.image_size
        return (
            math.isfinite(u)
            and math.isfinite(v)
            and -margin_px <= u < width + margin_px
            and -margin_px <= v < height + margin_px
        )

    def angular_error_deg(self, first: AnglePair, second: AnglePair) -> float:
        dot = float(np.dot(self.angles_to_ray(first), self.angles_to_ray(second)))
        return math.degrees(math.acos(float(np.clip(dot, -1.0, 1.0))))


class RobustBearingPredictor:
    """Fit short bearing histories without feeding extrapolated points back as data."""

    METHODS: Dict[str, int] = {
        "hold": 0,
        "angular_velocity": 1,
        "angular_acceleration": 2,
    }

    def __init__(
        self,
        history_seconds: float = 0.35,
        min_samples: int = 6,
        recency_tau_seconds: float = 0.20,
        huber_delta_deg: float = 0.75,
        max_angular_speed_deg_s: float = 600.0,
        max_angular_acceleration_deg_s2: float = 2500.0,
    ):
        self.history_seconds = float(history_seconds)
        self.min_samples = int(min_samples)
        self.recency_tau_seconds = float(recency_tau_seconds)
        self.huber_delta_deg = float(huber_delta_deg)
        self.max_angular_speed_deg_s = float(max_angular_speed_deg_s)
        self.max_angular_acceleration_deg_s2 = float(max_angular_acceleration_deg_s2)

    def select_history(
        self, observations: Iterable[BearingObservation], now: float | None = None
    ) -> List[BearingObservation]:
        ordered = sorted(observations, key=lambda item: item.timestamp)
        if not ordered:
            return []
        end = ordered[-1].timestamp if now is None else float(now)
        start = end - self.history_seconds
        return [item for item in ordered if start <= item.timestamp <= end]

    def predict(
        self,
        observations: Sequence[BearingObservation],
        horizon_seconds: float,
        method: str,
    ) -> BearingPrediction:
        if method not in self.METHODS:
            raise ValueError(f"unknown prediction method: {method}")
        history = self.select_history(observations)
        degree = self.METHODS[method]
        required = 1 if degree == 0 else max(self.min_samples, degree + 2)
        if len(history) < required:
            raise ValueError(f"{method} requires at least {required} observations")

        latest_time = history[-1].timestamp
        times = np.asarray([item.timestamp - latest_time for item in history], dtype=np.float64)
        scores = np.asarray([item.score for item in history], dtype=np.float64)
        yaw = np.rad2deg(np.unwrap(np.deg2rad([item.yaw_deg for item in history])))
        pitch = np.asarray([item.pitch_deg for item in history], dtype=np.float64)

        if degree == 0:
            predicted = np.asarray([yaw[-1], pitch[-1]], dtype=np.float64)
            residual = 0.0
        else:
            yaw_coeffs, yaw_residual = self._robust_fit(times, yaw, scores, degree)
            pitch_coeffs, pitch_residual = self._robust_fit(times, pitch, scores, degree)
            yaw_coeffs = self._constrain(yaw_coeffs)
            pitch_coeffs = self._constrain(pitch_coeffs)
            target_time = float(horizon_seconds)
            basis = np.asarray([target_time**power for power in range(degree + 1)])
            predicted = np.asarray(
                [float(basis @ yaw_coeffs), float(basis @ pitch_coeffs)], dtype=np.float64
            )
            residual = math.hypot(yaw_residual, pitch_residual)

        return BearingPrediction(
            method=method,
            yaw_deg=float(predicted[0]),
            pitch_deg=float(predicted[1]),
            residual_deg=float(residual),
            sample_count=len(history),
            history_span_s=float(history[-1].timestamp - history[0].timestamp),
        )

    def predict_all(
        self, observations: Sequence[BearingObservation], horizon_seconds: float
    ) -> Dict[str, BearingPrediction]:
        return {
            method: self.predict(observations, horizon_seconds, method)
            for method in self.METHODS
        }

    def _robust_fit(
        self,
        times: np.ndarray,
        values: np.ndarray,
        scores: np.ndarray,
        degree: int,
    ) -> Tuple[np.ndarray, float]:
        design = np.column_stack([times**power for power in range(degree + 1)])
        confidence = np.clip(scores, 0.05, 1.0)
        recency = np.exp(times / max(self.recency_tau_seconds, 1e-6))
        base_weights = confidence * recency
        weights = base_weights.copy()
        coefficients = np.zeros(degree + 1, dtype=np.float64)

        for _ in range(8):
            weighted_design = design * np.sqrt(weights)[:, None]
            weighted_values = values * np.sqrt(weights)
            coefficients = np.linalg.lstsq(weighted_design, weighted_values, rcond=None)[0]
            residuals = values - design @ coefficients
            median = float(np.median(residuals))
            mad = float(np.median(np.abs(residuals - median)))
            robust_scale = max(1.4826 * mad, self.huber_delta_deg)
            normalized = np.abs(residuals) / robust_scale
            huber = np.ones_like(normalized)
            mask = normalized > 1.0
            huber[mask] = 1.0 / normalized[mask]
            next_weights = base_weights * huber
            if np.allclose(weights, next_weights, rtol=1e-4, atol=1e-7):
                weights = next_weights
                break
            weights = next_weights

        residuals = values - design @ coefficients
        weighted_rms = math.sqrt(float(np.average(residuals * residuals, weights=weights)))
        return coefficients, weighted_rms

    def _constrain(self, coefficients: np.ndarray) -> np.ndarray:
        constrained = coefficients.copy()
        if len(constrained) >= 2:
            constrained[1] = float(
                np.clip(
                    constrained[1],
                    -self.max_angular_speed_deg_s,
                    self.max_angular_speed_deg_s,
                )
            )
        if len(constrained) >= 3:
            max_quadratic = 0.5 * self.max_angular_acceleration_deg_s2
            constrained[2] = float(np.clip(constrained[2], -max_quadratic, max_quadratic))
        return constrained


def summarize(values: Sequence[float]) -> Dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "p90": None, "mean": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "median": round(float(np.median(array)), 4),
        "p90": round(float(np.percentile(array, 90)), 4),
        "mean": round(float(np.mean(array)), 4),
    }
