from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.servo_angle_calibration import ServoOpticalAngleCalibration
from vision.hc13_link import select_control_target


CONFIG = ROOT / "configs" / "servo_to_optical_angle_foam_center_lut_20260816_v1.json"


class NoRedServoLookupTest(unittest.TestCase):
    def setUp(self):
        self.data = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.calibration = ServoOpticalAngleCalibration.from_json(CONFIG)

    def test_range_and_nodes(self):
        self.assertEqual(self.calibration.servo_range_deg, (-97.0, -47.0))
        self.assertEqual(len(self.calibration.servo_nodes_deg), 11)

    def test_every_node_round_trips(self):
        for servo in self.calibration.servo_nodes_deg:
            optical = self.calibration.optical_offset_from_servo(servo)
            recovered = self.calibration.servo_from_optical_offset(optical)
            self.assertAlmostEqual(recovered, servo, places=9)

    def test_independent_validation_is_recorded_and_passed(self):
        validation = self.data["independent_validation"]
        self.assertFalse(validation["used_for_fit"])
        self.assertTrue(validation["acceptance"]["overall_pass"])
        self.assertEqual(validation["sampled_frames"], 230)


class RestrictedVisualRangeTest(unittest.TestCase):
    def test_boundaries_reject_extrapolated_targets_without_tolerance(self):
        cal = ServoOpticalAngleCalibration.from_json(
            ROOT / "configs/servo_to_optical_angle_foam_center_lut_20261006_restricted_v1.json"
        )
        self.assertEqual(cal.servo_range_deg, (-77.0, -27.0))
        for angle in (-77.0, -27.0):
            self.assertTrue(cal.is_optical_offset_in_range(cal.optical_offset_from_servo(angle)))
        for angle in (-77.01, -26.99):
            optical = cal.optical_offset_from_servo(angle)
            self.assertFalse(cal.is_optical_offset_in_range(optical))
            target = {"target_valid": True, "angle_mode": "servo_calibrated",
                      "servo_calibration_valid": cal.is_optical_offset_in_range(optical),
                      "lateral_valid": True, "score": 0.95, "servo_command_deg": angle}
            self.assertIsNone(select_control_target([target], 0.6))


if __name__ == "__main__":
    unittest.main()
