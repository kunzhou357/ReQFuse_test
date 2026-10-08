"""Offline corruption invariants and real entry-point IO; artifacts stay here."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

import generate_degradation as entry
from utils.dataset import PairedFusionDataset
from utils.synthetic_degradation import (INFRARED_TYPES, LEVELS, VISIBLE_TYPES,
                                            SyntheticDegrader, local_envelope)


TESTS = Path(__file__).resolve().parent


class CorruptionTests(unittest.TestCase):
    def setUp(self):
        yy, xx = np.meshgrid(np.linspace(0, 1, 96), np.linspace(0, 1, 128), indexing="ij")
        self.vi = np.stack((0.1 + 0.7*xx, 0.1 + 0.7*yy, 0.2 + 0.4*xx*yy), -1).astype(np.float32)
        texture = 0.04 * np.sin(np.arange(128)[None, :] * np.pi / 4) * np.cos(np.arange(96)[:, None] * np.pi / 4)
        self.vi += texture[..., None].astype(np.float32)
        self.ir = (0.15 + 0.5*xx + 0.2*yy).astype(np.float32)
        self.degrader = SyntheticDegrader()

    def test_all_types_levels_scopes_are_finite_and_preserve_intact_pixels(self):
        for modality, image, types in (("visible", self.vi, VISIBLE_TYPES),
                                      ("infrared", self.ir, INFRARED_TYPES)):
            for kind in types:
                for level in LEVELS:
                    for global_scope in (True, False):
                        with self.subTest(modality=modality, kind=kind, level=level, global_scope=global_scope):
                            result = self.degrader(image, kind, level, np.random.default_rng(101),
                                modality=modality, global_degradation=global_scope)
                            self.assertEqual(result.image.shape, image.shape)
                            self.assertTrue(np.isfinite(result.image).all())
                            self.assertGreaterEqual(result.image.min(), 0)
                            self.assertLessEqual(result.image.max(), 1)
                            self.assertGreater(np.abs(result.image-image).mean(), 1e-5)
                            outside = result.mask == 0
                            np.testing.assert_array_equal(result.image[outside], image[outside])
                            self.assertTrue(np.all(result.mask[result.envelope == 0] == 0))
                            if not global_scope:
                                self.assertTrue((result.envelope == 0).any())
                                self.assertAlmostEqual(float((result.envelope >= 0.5).mean()), 0.4, delta=0.03)

    def test_repeatability_clean_identity_and_column_noise(self):
        clean = self.degrader(self.vi, "clean", "medium", np.random.default_rng(1))
        np.testing.assert_array_equal(clean.image, self.vi)
        self.assertEqual(clean.mask.sum(), 0)
        for kind in VISIBLE_TYPES:
            a = self.degrader(self.vi, kind, "medium", np.random.default_rng(123), global_degradation=False)
            b = self.degrader(self.vi, kind, "medium", np.random.default_rng(123), global_degradation=False)
            np.testing.assert_array_equal(a.image, b.image)
            np.testing.assert_array_equal(a.mask, b.mask)
            self.assertEqual(a.parameters, b.parameters)
        gray = np.full((96, 128), 0.5, dtype=np.float32)
        result = self.degrader(gray, "stripe", "heavy", np.random.default_rng(8), modality="infrared")
        np.testing.assert_array_equal(result.image, np.repeat(result.image[:1], len(gray), axis=0))

    def test_severity_has_expected_effect_and_haze_depth_direction(self):
        for kind in ("noise", "low_light", "haze"):
            errors = [np.abs(self.degrader(self.vi, kind, level, np.random.default_rng(9)).image-self.vi).mean()
                      for level in LEVELS]
            self.assertLess(errors[0], errors[1])
            self.assertLess(errors[1], errors[2])
        result = self.degrader(self.ir, "low_contrast", "medium", np.random.default_rng(9), modality="infrared")
        self.assertAlmostEqual(float(result.image.mean()), float(self.ir.mean()), places=6)
        self.assertAlmostEqual(float(result.image.std()/self.ir.std()), 0.5, places=5)
        depth = np.tile(np.linspace(1, 0, 128, dtype=np.float32), (96, 1))
        image = np.full((96, 128, 3), 0.2, dtype=np.float32)
        result = self.degrader(image, "haze", "medium", np.random.default_rng(9), depth=depth)
        self.assertLess(result.image[:, :16].mean(), result.image[:, -16:].mean())
        self.assertEqual(result.parameters["depth_mode"], "inverse_depth_map")


class GenerationIOTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="degradation_", dir=TESTS)
        self.root = Path(self.temporary.name)
        self.vi, self.ir = self.root/"visible_camera", self.root/"thermal_sensor"
        (self.vi/"scene").mkdir(parents=True)
        (self.ir/"scene").mkdir(parents=True)
        y, x = np.meshgrid(np.arange(32), np.arange(40), indexing="ij")
        rgb = np.stack((x*5, y*7, (x+y)*3), -1).astype(np.uint8)
        gray = (x*3+y*2).astype(np.uint8)
        Image.fromarray(rgb).save(self.vi/"scene/frame.png")
        Image.fromarray(gray).save(self.ir/"scene/frame.bmp")
        self.settings = entry._build_settings()
        self.settings.update(visible_dir=self.vi, infrared_dir=self.ir, output=self.root/"generated",
                             levels=["medium"], preview_count=1, progress=False)

    def tearDown(self):
        self.temporary.cleanup()

    def run_silently(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return entry.run(self.settings)

    def test_all_nine_conditions_output_pairs_masks_records_and_reproducible_rerun(self):
        self.settings["conditions"] = [(kind, "clean") for kind in VISIBLE_TYPES] + [("clean", kind) for kind in INFRARED_TYPES]
        roots = self.run_silently()
        self.assertEqual(len(roots), 9)
        before = {root: (root/"vi/scene/frame.png").read_bytes() for root in roots}
        for root in roots:
            info = json.loads((root/"run_info.json").read_text())
            self.assertTrue(info["complete"])
            self.assertEqual(info["pairs"], 1)
            with (root/"records.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertTrue((root/"previews/scene/frame.png").is_file())
            ds = PairedFusionDataset(root/"vi", root/"ir", mode="paired", hq_visible_dir=self.vi,
                hq_infrared_dir=self.ir, mask_visible_dir=root/"mask_vi", mask_infrared_dir=root/"mask_ir")
            self.assertEqual(ds[0]["visible"].shape, (3, 32, 40))
        self.run_silently()
        for root in roots:
            self.assertEqual(before[root], (root/"vi/scene/frame.png").read_bytes())
        self.settings["seed"] += 1
        with self.assertRaisesRegex(ValueError, "different settings"):
            self.run_silently()

    def test_npy_only_depth_and_local_two_modality_pair(self):
        depth_dir = self.root/"depth"
        (depth_dir/"scene").mkdir(parents=True)
        np.save(depth_dir/"scene/frame.npy", np.tile(np.linspace(1, 0, 40), (32, 1)))
        self.settings.update(conditions=[("haze", "stripe")], depth_dir=depth_dir, global_degradation=False)
        root = self.run_silently()[0]
        with (root/"records.csv").open(newline="") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(json.loads(row["vi_parameters"])["depth_mode"], "inverse_depth_map")
        with Image.open(root/"envelope_vi/scene/frame.png") as image:
            envelope = np.asarray(image)
        np.testing.assert_array_equal(envelope, np.repeat(envelope[:1], 32, axis=0))
        self.assertTrue((envelope == 0).any())

    def test_pair_mismatch_output_overlap_and_invalid_depth_fail_clearly(self):
        self.settings["output"] = self.vi/"output"
        with self.assertRaisesRegex(ValueError, "separate"):
            self.run_silently()
        self.settings["output"] = self.root/"generated"
        extra = self.ir/"extra.png"
        Image.new("L", (40, 32)).save(extra)
        with self.assertRaisesRegex(ValueError, "Unpaired"):
            self.run_silently()
        extra.unlink()
        depth_dir = self.root/"depth"
        (depth_dir/"scene").mkdir(parents=True)
        np.save(depth_dir/"scene/frame.npy", np.ones((3, 3)))
        self.settings.update(conditions=[("haze", "clean")], depth_dir=depth_dir)
        with self.assertRaisesRegex(ValueError, "Depth must"):
            self.run_silently()


if __name__ == "__main__":
    unittest.main()
