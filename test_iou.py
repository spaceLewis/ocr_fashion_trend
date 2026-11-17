import unittest

import numpy as np

from iou import calculate_iou


class CalculateIoUTest(unittest.TestCase):
    def test_intersection_union_and_missing_classes(self):
        prediction = np.array([[0, 1, 1], [2, 0, 0]])
        reference = np.array([[0, 1, 0], [0, 3, 0]])

        self.assertEqual(calculate_iou(prediction, reference), {
            0: 0.4,
            1: 0.5,
            2: 0.0,
            3: 0.0,
        })

    def test_rejects_mismatched_shapes(self):
        with self.assertRaisesRegex(ValueError, "mêmes dimensions"):
            calculate_iou(np.zeros((2, 2)), np.zeros((1, 2)))

    def test_rejects_invalid_class_and_color_masks(self):
        with self.assertRaisesRegex(ValueError, "identifiants de classe"):
            calculate_iou(np.array([[18]]), np.array([[0]]))
        with self.assertRaisesRegex(ValueError, "un identifiant par pixel"):
            calculate_iou(np.zeros((1, 1, 3)), np.zeros((1, 1)))


if __name__ == "__main__":
    unittest.main()
