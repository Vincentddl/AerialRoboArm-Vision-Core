import unittest
from types import SimpleNamespace

import numpy as np
import torch

from tools.refit_merged_servo_relation import fit, inverse
from tools.evaluate_no_red_servo_relation import predicted_centroid


class Boxes(SimpleNamespace):
    def __len__(self):
        return len(self.conf)


class MergedRelationFitTests(unittest.TestCase):
    def test_recovers_known_relation_with_an_isolated_bad_hold(self):
        knots = np.arange(-77., -26., 5.)
        rows = [{"session": session, "command_deg": float(angle),
                 "optical": {"runtime_hybrid": [0.7 * angle + 30.]}}
                for session in range(3) for angle in knots]
        rows[7]["optical"]["runtime_hybrid"] = [50.]
        optical = fit(rows, "runtime_hybrid", knots)
        self.assertTrue(np.all(np.diff(optical) > 0))
        recovered = inverse(0.7 * knots + 30., optical, knots)
        self.assertLess(float(np.max(np.abs(recovered - knots))), 0.5)

    def test_complete_target_required_by_merged_protocol(self):
        result = SimpleNamespace(
            boxes=Boxes(conf=torch.tensor([0.99]),
                                  xyxy=torch.tensor([[300., 380., 370., 480.]])),
            masks=None, orig_shape=(480, 640),
        )
        center, _, mode = predicted_centroid(result, reject_clipped=True)
        self.assertIsNone(center)
        self.assertEqual(mode, "clipped_target")
        self.assertIsNotNone(predicted_centroid(result)[0])
