"""Empirical mappings between servo commands and camera-axis target angles."""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


def _interpolate_or_extrapolate(
    value: float,
    inputs: Tuple[float, ...],
    outputs: Tuple[float, ...],
) -> float:
    """Piecewise-linear interpolation with endpoint-segment extrapolation.

    Callers must use ``is_servo_in_range`` when an extrapolated result is not
    safe for control. Extrapolation is retained for compatibility with the old
    linear calibration, which returned a diagnostic value outside its fit range.
    """

    x = float(value)
    if x <= inputs[0]:
        index = 0
    elif x >= inputs[-1]:
        index = len(inputs) - 2
    else:
        index = bisect.bisect_right(inputs, x) - 1
    x0, x1 = inputs[index], inputs[index + 1]
    y0, y1 = outputs[index], outputs[index + 1]
    ratio = (x - x0) / (x1 - x0)
    return y0 + ratio * (y1 - y0)


@dataclass(frozen=True)
class ServoOpticalAngleCalibration:
    """Map labelled servo angle ``g`` to camera optical-axis offset ``beta``.

    ``optical_offset_deg`` is the signed target bearing in the camera's
    single-axis plane. Positive values point downward in the image. Supported
    models are the original linear fit and a monotonic piecewise-linear lookup
    table. The lookup model is preferred for the red-marker validation because
    the mechanism flattens near ``g=-100`` and a global polynomial extrapolates
    poorly there.
    """

    name: str
    version: int
    servo_range_deg: Tuple[float, float]
    model_kind: str = "linear"
    boundary_tolerance_deg: float = 0.0
    fit_rmse_deg: float | None = None
    fit_r_squared: float | None = None
    slope: float | None = None
    intercept_deg: float | None = None
    servo_nodes_deg: Tuple[float, ...] = ()
    optical_nodes_deg: Tuple[float, ...] = ()

    @classmethod
    def from_json(cls, path: str | Path) -> "ServoOpticalAngleCalibration":
        calibration_path = Path(path)
        data = json.loads(calibration_path.read_text(encoding="utf-8"))
        calibration_type = data.get("type")
        if calibration_type not in {
            "servo_optical_angle_linear",
            "servo_optical_angle_lut",
        }:
            raise ValueError(
                "servo calibration type must be 'servo_optical_angle_linear' "
                "or 'servo_optical_angle_lut'"
            )

        model = data.get("model", {})
        valid_range = data.get("valid_range", {}).get("servo_command_deg")
        if not isinstance(valid_range, list) or len(valid_range) != 2:
            raise ValueError(
                "valid_range.servo_command_deg must contain two numbers"
            )
        limits = sorted(float(value) for value in valid_range)
        fit = data.get("fit", {})
        common = dict(
            name=str(data.get("name", calibration_path.stem)),
            version=int(data.get("version", 1)),
            servo_range_deg=(limits[0], limits[1]),
            boundary_tolerance_deg=max(
                0.0, float(data.get("boundary_tolerance_deg", 0.0))
            ),
            fit_rmse_deg=(
                float(fit["rmse_deg"])
                if fit.get("rmse_deg") is not None
                else None
            ),
            fit_r_squared=(
                float(fit["r_squared"])
                if fit.get("r_squared") is not None
                else None
            ),
        )

        if calibration_type == "servo_optical_angle_linear":
            slope = float(model["slope"])
            if abs(slope) < 1e-12:
                raise ValueError("servo calibration slope must be non-zero")
            return cls(
                **common,
                model_kind="linear",
                slope=slope,
                intercept_deg=float(model["intercept_deg"]),
            )

        points = model.get("points")
        if not isinstance(points, list) or len(points) < 2:
            raise ValueError("lookup calibration model.points needs at least two points")
        pairs = sorted(
            (
                float(point["servo_command_deg"]),
                float(point["optical_offset_deg"]),
            )
            for point in points
        )
        servo_nodes = tuple(pair[0] for pair in pairs)
        optical_nodes = tuple(pair[1] for pair in pairs)
        if any(
            right <= left
            for left, right in zip(servo_nodes, servo_nodes[1:])
        ):
            raise ValueError("lookup servo nodes must be unique")
        optical_deltas = [
            right - left
            for left, right in zip(optical_nodes, optical_nodes[1:])
        ]
        if not (all(delta > 0 for delta in optical_deltas) or all(delta < 0 for delta in optical_deltas)):
            raise ValueError("lookup optical nodes must be strictly monotonic")
        if servo_nodes[0] > limits[0] or servo_nodes[-1] < limits[1]:
            raise ValueError("lookup nodes must cover the declared valid servo range")
        return cls(
            **common,
            model_kind="piecewise_linear_lut",
            servo_nodes_deg=servo_nodes,
            optical_nodes_deg=optical_nodes,
        )

    def optical_offset_from_servo(self, servo_command_deg: float) -> float:
        """Return signed optical-axis offset for a labelled servo angle."""
        if self.model_kind == "linear":
            assert self.slope is not None and self.intercept_deg is not None
            return self.slope * float(servo_command_deg) + self.intercept_deg
        return _interpolate_or_extrapolate(
            servo_command_deg,
            self.servo_nodes_deg,
            self.optical_nodes_deg,
        )

    def servo_from_optical_offset(self, optical_offset_deg: float) -> float:
        """Return the vision-inferred servo angle for a measured bearing."""
        if self.model_kind == "linear":
            assert self.slope is not None and self.intercept_deg is not None
            return (float(optical_offset_deg) - self.intercept_deg) / self.slope
        if self.optical_nodes_deg[0] < self.optical_nodes_deg[-1]:
            optical_nodes = self.optical_nodes_deg
            servo_nodes = self.servo_nodes_deg
        else:
            optical_nodes = tuple(reversed(self.optical_nodes_deg))
            servo_nodes = tuple(reversed(self.servo_nodes_deg))
        return _interpolate_or_extrapolate(
            optical_offset_deg,
            optical_nodes,
            servo_nodes,
        )

    def is_servo_in_range(self, servo_command_deg: float) -> bool:
        minimum, maximum = self.servo_range_deg
        tolerance = self.boundary_tolerance_deg
        return minimum - tolerance <= float(servo_command_deg) <= maximum + tolerance

    def is_optical_offset_in_range(self, optical_offset_deg: float) -> bool:
        minimum, maximum = self.optical_range_deg
        if self.model_kind == "linear":
            assert self.slope is not None
            tolerance = abs(self.slope) * self.boundary_tolerance_deg
        else:
            endpoint_slopes = [
                abs(
                    (self.optical_nodes_deg[index + 1] - self.optical_nodes_deg[index])
                    / (self.servo_nodes_deg[index + 1] - self.servo_nodes_deg[index])
                )
                for index in (0, len(self.servo_nodes_deg) - 2)
            ]
            tolerance = max(endpoint_slopes) * self.boundary_tolerance_deg
        return minimum - tolerance <= float(optical_offset_deg) <= maximum + tolerance

    @property
    def optical_range_deg(self) -> Tuple[float, float]:
        if self.model_kind == "piecewise_linear_lut":
            return (min(self.optical_nodes_deg), max(self.optical_nodes_deg))
        values = [
            self.optical_offset_from_servo(limit)
            for limit in self.servo_range_deg
        ]
        return (min(values), max(values))

    @property
    def summary(self) -> str:
        minimum, maximum = self.servo_range_deg
        if self.model_kind == "linear":
            assert self.slope is not None and self.intercept_deg is not None
            model_text = f"beta={self.slope:.6f}*g{self.intercept_deg:+.6f} deg"
        else:
            model_text = f"piecewise-linear LUT ({len(self.servo_nodes_deg)} nodes)"
        return (
            f"servo calibration: {self.name} v{self.version}, {model_text}, "
            f"valid g=[{minimum:.1f}, {maximum:.1f}] deg "
            f"(boundary tolerance {self.boundary_tolerance_deg:.2f} deg)"
        )
