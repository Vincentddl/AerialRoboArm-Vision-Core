import unittest
from types import SimpleNamespace

import numpy as np

from vision.runtime import result_to_detections
from vision.tracker import TrajectoryEstimator


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def fake_result(box, polygon=None):
    boxes = SimpleNamespace(
        xyxy=FakeTensor([box]),
        conf=FakeTensor([0.9]),
        cls=FakeTensor([0]),
    )
    masks = SimpleNamespace(xy=[np.asarray(polygon, dtype=np.float32)]) if polygon is not None else None
    return SimpleNamespace(boxes=boxes, masks=masks, names={0: "foam_board"}, orig_shape=(480, 640))


class SegmentationCenterTest(unittest.TestCase):
    def test_normal_size_uses_mask_centroid(self):
        result = fake_result(
            [100, 100, 200, 200],
            [[110, 120], [180, 120], [180, 180], [110, 180]],
        )
        detection = result_to_detections(result, image_shape=(480, 640))[0]
        self.assertEqual(detection.center_source, "mask_centroid")
        self.assertAlmostEqual(detection.center[0], 145.0, places=3)
        self.assertAlmostEqual(detection.center[1], 150.0, places=3)

    def test_large_close_up_uses_box_center(self):
        result = fake_result(
            [100, 80, 400, 300],
            [[110, 90], [250, 90], [250, 290], [110, 290]],
        )
        detection = result_to_detections(result, image_shape=(480, 640))[0]
        self.assertEqual(detection.center_source, "hybrid_box_center")
        self.assertEqual(detection.center, (250.0, 190.0))

    def test_detection_model_still_uses_box_center(self):
        result = fake_result([20, 40, 60, 100])
        detection = result_to_detections(result, image_shape=(480, 640))[0]
        self.assertEqual(detection.center_source, "box_center")
        self.assertEqual(detection.center, (40.0, 70.0))

    def test_current_target_preserves_latest_segmentation_center(self):
        result = fake_result(
            [100, 100, 200, 200],
            [[110, 120], [180, 120], [180, 180], [110, 180]],
        )
        detection = result_to_detections(result, image_shape=(480, 640))[0]
        estimator = TrajectoryEstimator(new_track_confirmation_frames=1)
        estimator.update([detection], timestamp=1.0)

        current = estimator.current_targets(min_age=1)[0]

        self.assertEqual(current["pixel"], [145.0, 150.0])
        self.assertEqual(current["center_source"], "mask_centroid")


if __name__ == "__main__":
    unittest.main()
