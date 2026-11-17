import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from iou import calculate_iou, compare_directories, load_mask


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


class CompareDirectoriesTest(unittest.TestCase):
    def test_compares_all_png_pairs_by_filename(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            predictions = root / "Mask"
            references = root / "Reference"
            predictions.mkdir()
            references.mkdir()

            Image.fromarray(np.array([[0, 1]], dtype=np.uint8)).save(predictions / "b.png")
            Image.fromarray(np.array([[0, 1]], dtype=np.uint8)).save(references / "b.png")
            Image.fromarray(np.array([[0, 2]], dtype=np.uint8)).save(predictions / "a.png")
            Image.fromarray(np.array([[0, 3]], dtype=np.uint8)).save(references / "a.png")

            results = compare_directories(predictions, references)

        self.assertEqual(list(results), ["a.png", "b.png"])
        self.assertEqual(results["a.png"], {0: 1.0, 2: 0.0, 3: 0.0})
        self.assertEqual(results["b.png"], {0: 1.0, 1: 1.0})

    def test_compares_rgb_and_opaque_rgba_class_ids(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            predictions = root / "Mask"
            references = root / "Reference"
            predictions.mkdir()
            references.mkdir()
            Image.fromarray(np.array([[[0, 0, 0], [1, 1, 1]]], dtype=np.uint8)).save(predictions / "mask.png")
            Image.fromarray(np.array([[[0, 0, 0, 255], [2, 2, 2, 255]]], dtype=np.uint8)).save(references / "mask.png")

            results = compare_directories(predictions, references)

        self.assertEqual(results["mask.png"], {0: 1.0, 1: 0.0, 2: 0.0})

    def test_rejects_colored_or_transparent_images(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "mask.png"
            Image.fromarray(np.array([[[0, 1, 0]]], dtype=np.uint8)).save(path)
            with self.assertRaisesRegex(ValueError, "couleurs ambiguës"):
                load_mask(path)
            Image.fromarray(np.array([[[1, 1, 1, 0]]], dtype=np.uint8)).save(path)
            with self.assertRaisesRegex(ValueError, "transparents"):
                load_mask(path)

    def test_rejects_unpaired_png_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            predictions = root / "Mask"
            references = root / "Reference"
            predictions.mkdir()
            references.mkdir()
            Image.fromarray(np.zeros((1, 1), dtype=np.uint8)).save(predictions / "mask.png")

            with self.assertRaisesRegex(ValueError, "appariés par nom"):
                compare_directories(predictions, references)


if __name__ == "__main__":
    unittest.main()
