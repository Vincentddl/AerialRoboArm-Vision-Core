import unittest

from vision.hc13_link import HC13SerialLink, StableTargetGate, select_control_target


def target(track_id=1, angle=-70.0, score=0.9, valid=True):
    return {
        "track_id": track_id,
        "score": score,
        "target_valid": valid,
        "angle_mode": "servo_calibrated",
        "servo_calibration_valid": valid,
        "lateral_valid": valid,
        "missed_frames": 0,
        "servo_command_deg": angle,
    }


class StableTargetGateTests(unittest.TestCase):
    def test_requires_consecutive_stable_frames(self):
        gate = StableTargetGate(confirmation_frames=3, max_spread_deg=1.0, send_hz=20)
        self.assertFalse(gate.update(target(angle=-70.2), 0.00).should_send)
        self.assertFalse(gate.update(target(angle=-69.8), 0.02).should_send)
        result = gate.update(target(angle=-70.0), 0.04)
        self.assertTrue(result.should_send)
        self.assertEqual(result.angle_deg, -70)

    def test_track_change_restarts_confirmation(self):
        gate = StableTargetGate(confirmation_frames=2)
        gate.update(target(track_id=1), 0.0)
        result = gate.update(target(track_id=2), 0.1)
        self.assertEqual(result.status, "CONFIRMING")
        self.assertEqual(result.confirmed_frames, 1)

    def test_unstable_angles_do_not_send(self):
        gate = StableTargetGate(confirmation_frames=3, max_spread_deg=1.0)
        gate.update(target(angle=-70), 0.0)
        gate.update(target(angle=-72), 0.1)
        result = gate.update(target(angle=-69), 0.2)
        self.assertFalse(result.should_send)
        self.assertTrue(result.status.startswith("UNSTABLE"))

    def test_invalid_target_resets_gate(self):
        gate = StableTargetGate(confirmation_frames=2)
        gate.update(target(), 0.0)
        gate.update(None, 0.1)
        result = gate.update(target(), 0.2)
        self.assertEqual(result.confirmed_frames, 1)


class HC13ProtocolTests(unittest.TestCase):
    def test_packet_is_integer_ascii_expected_by_mcu(self):
        self.assertEqual(
            HC13SerialLink.encode_target(-70, 0, 91),
            b"V,-70,0,91\r\n",
        )

    def test_selects_only_calibrated_valid_target(self):
        low = target(track_id=1, score=0.4)
        high = target(track_id=2, score=0.85)
        self.assertIs(select_control_target([low, high], 0.6), high)
        self.assertIsNone(select_control_target([low], 0.6))


if __name__ == "__main__":
    unittest.main()
