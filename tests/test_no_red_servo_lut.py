from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.servo_angle_calibration import ServoOpticalAngleCalibration


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


if __name__ == "__main__":
    unittest.main()
