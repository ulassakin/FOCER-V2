import json
import tempfile
import unittest
from pathlib import Path

from hdr4rtt_experiment import EXPECTED_OBJECTS, EXPECTED_SIZES, split_stats, subset_coco
from split_manifest import load_manifest


class HDR4RTTProtocolTests(unittest.TestCase):
    def test_manifest_is_exact_partition(self):
        manifest = load_manifest()
        assignments = {name: set(values) for name, values in manifest["assignments"].items()}
        seen = set()
        for name in ("train", "val", "test", "guard"):
            self.assertEqual(len(assignments[name]), EXPECTED_SIZES[name])
            self.assertFalse(seen & assignments[name])
            seen |= assignments[name]
        self.assertEqual(len(seen), 1596)
        self.assertEqual(assignments["guard"], {
            287, 288, 289, 491, 492, 493, 494, 495, 496, 561, 562, 563, 1054, 1055,
        })

    def test_protocol_metadata(self):
        manifest = load_manifest()
        self.assertEqual(manifest["protocol_version"], "hdr4rtt_source_temporal_guard_v1")
        self.assertEqual(manifest["cross_split_guard_radius_frames"], 3)
        self.assertEqual(manifest["seed"], 42)

    def test_subset_and_statistics(self):
        master = {
            "images": [{"id": 1, "file_name": "a.exr"}, {"id": 2, "file_name": "b.exr"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1},
                {"id": 2, "image_id": 1, "category_id": 3},
                {"id": 3, "image_id": 2, "category_id": 8},
            ],
            "categories": [{"id": x, "name": str(x)} for x in (1, 2, 3, 8)],
        }
        result = subset_coco(master, [1], "unit")
        self.assertEqual([item["id"] for item in result["images"]], [1])
        self.assertEqual(len(result["annotations"]), 2)
        stats = split_stats(result)
        self.assertEqual(stats["objects_by_paper_category"], {"1": 1, "2": 0, "3": 1, "8": 0})


if __name__ == "__main__":
    unittest.main()
