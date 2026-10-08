"""Regression checks for the diagnosed saturation and region dilution failures."""
from pathlib import Path
import tempfile
import unittest

import torch

from utils.auxiliary_loss import AuxiliaryLoss, balanced_mean, masked_mean
from utils.experiment import dataset_options, read_checkpoint, training_signature
import train


class RestorationV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_extremely_negative_state_outputs_have_recovery_gradients(self):
        outputs = {name:{} for name in ("restored", "demand_logits", "error_raw")}
        batch = {}
        for m, c in (("visible", 3), ("infrared", 1)):
            batch[m] = torch.full((1,c,5,7), .4)
            batch[f"hq_{m}"] = torch.full((1,c,5,7), .5)
            batch[f"mask_{m}"] = torch.ones(1,1,5,7)
            outputs["restored"][m] = batch[m].clone().requires_grad_()
            outputs["demand_logits"][m] = torch.full((1,1,5,7), -280., requires_grad=True)
            outputs["error_raw"][m] = torch.full((1,1,5,7), -230., requires_grad=True)
        criterion = AuxiliaryLoss(reconstruction_weight=0, keep_weight=0)
        loss, _ = criterion(outputs, batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for m in ("visible", "infrared"):
            # Gradient descent increases both negative raw quantities.
            for key in ("demand_logits", "error_raw"):
                gradient = outputs[key][m].grad
                self.assertTrue(torch.all(gradient < 0))
                self.assertGreater(gradient.abs().sum().item(), .4)
            # The error/demand truth must not optimize the restoration upstream.
            self.assertEqual(outputs["restored"][m].grad.abs().sum().item(), 0)

    def test_small_damaged_region_is_not_diluted_and_empty_regions_are_finite(self):
        loss_map = torch.zeros(3,1,10,10, requires_grad=True)
        mask = torch.zeros_like(loss_map)
        mask[0,0,0,0] = 1
        mask[1].fill_(1)
        with torch.no_grad():
            loss_map[0,0,0,0] = 10
            loss_map[1].fill_(4)
            loss_map[2].fill_(2)
        # Image 0: (10+0)/2; image 1: 4; image 2: 2.
        balanced = balanced_mean(loss_map, mask)
        self.assertAlmostEqual(balanced.item(), 11/3, places=6)
        self.assertAlmostEqual(masked_mean(loss_map, mask).item(), 7, places=6)
        balanced.backward()
        self.assertGreater(loss_map.grad[0,0,0,0].item(), .1)
        self.assertEqual(masked_mean(loss_map, torch.zeros_like(mask)).item(), 0)

    def test_old_checkpoint_and_changed_semantics_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            path = Path(temp)/"old.pth"
            torch.save({"format_version": 1, "model": {}, "config": {}}, path)
            with self.assertRaisesRegex(ValueError, "fresh training"):
                read_checkpoint(path)
        config = train._build_settings()
        old_signature = training_signature(config)
        config["implementation"]["demand_curriculum"] = "different_semantics"
        self.assertNotEqual(old_signature, training_signature(config))

    def test_default_recipe_is_the_complete_v2_candidate(self):
        config = train._build_settings()
        self.assertEqual(config["fusion_loss"]["gradient_weight"], 2)
        self.assertEqual(config["training"]["restoration_warmup_steps"], 2142)
        self.assertEqual(config["model"]["error_scale"], config["auxiliary_loss"]["error_scale"])
        self.assertTrue(config["model"]["use_demand"] and config["model"]["use_quality"])

    def test_paired_generator_masks_are_discovered_without_manual_options(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            for name in ("vi", "ir", "mask_vi", "mask_ir"):
                (root/name).mkdir()
            options = {"mode": "paired", "visible_dir": str(root/"vi"), "infrared_dir": str(root/"ir"),
                       "mask_visible_dir": None, "mask_infrared_dir": None}
            settings = dataset_options(options, 1, True, None)
            self.assertEqual(settings["mask_visible_dir"], str(root/"mask_vi"))
            self.assertEqual(settings["mask_infrared_dir"], str(root/"mask_ir"))
            (root/"mask_ir").rmdir()
            with self.assertRaisesRegex(ValueError, "both mask_vi and mask_ir"):
                dataset_options(options, 1, True, None)
