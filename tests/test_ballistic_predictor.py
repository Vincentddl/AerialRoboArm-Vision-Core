from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from vision.ballistic_predictor import BallisticPredictor  # noqa: E402


class BallisticPredictorTest(unittest.TestCase):
    def test_default_gravity_uses_down_positive_pitch_convention(self) -> None:
        predictor = BallisticPredictor()
        predictor.update(1, (0.0, 0.0), timestamp=0.0)
        predictor.update(1, (0.0, -10.0), timestamp=0.1)

        state = predictor.get_state(1)

        self.assertIsNotNone(state)
        self.assertGreater(state["g_eff"], 0.0)

    def test_positive_gravity_bends_rising_path_back_down(self) -> None:
        predictor = BallisticPredictor(initial_gravity_deg_s2=100.0)
        predictor.update(1, (0.0, 0.0), timestamp=0.0)
        predictor.update(1, (0.0, -2.0), timestamp=0.1)

        path = predictor.predict_trajectory(1, horizon_seconds=0.4, steps=4)

        self.assertIsNotNone(path)
        # Initial pitch velocity is -20 deg/s. With +100 deg/s^2 apparent
        # gravity, the path reaches an apex and then pitch increases downward.
        self.assertLess(path[2][1], path[1][1])
        self.assertGreater(path[-1][1], path[2][1])


if __name__ == "__main__":
    unittest.main()
