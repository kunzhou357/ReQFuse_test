"""Synthetic IO, supervision and editable Python entry-point integration."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from model.fusion import RobustFusionNet
from tests.test_network import tiny_config
from utils.auxiliary_loss import AuxiliaryLoss
from utils.dataset import PairedFusionDataset, ResumableBatchSampler
import train as training
import test as inference
from utils.loss import FusionLoss


ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent


def experiment_entry(directory, module, manual, internal_overrides=None, tiny=False, max_steps=None):
    """Create a test-only editable Python runner; no production CLI is needed."""
    directory = Path(directory)
    if module not in ("train", "test") or not directory.resolve().is_relative_to(TESTS.resolve()):
        raise ValueError("Experiment runners must live inside tests/.")
    directory.mkdir(parents=True, exist_ok=True)
    source = ["import sys", "sys.dont_write_bytecode = True",
              f"sys.path.insert(0, {str(ROOT)!r})", f"import {module} as entry"]
    source += [f"entry.{key} = {str(value)!r}" if isinstance(value, Path) else f"entry.{key} = {value!r}"
               for key, value in manual.items()]
    source += ["original_defaults = entry._build_settings", "def test_defaults():",
               "    settings = original_defaults()"]
    if module == "train":
        source.append(f"    settings['training'].update({internal_overrides or {}!r})")
        if tiny:
            source += ["    from tests.test_network import tiny_config", "    settings['model'] = tiny_config()"]
    else:
        source.append(f"    vars(settings).update({internal_overrides or {}!r})")
    source += ["    return settings", "entry._build_settings = test_defaults"]
    if max_steps is not None:
        source += ["original_run = entry.run", "def limited_run(settings):",
                   f"    return original_run(settings, max_steps={max_steps!r})", "entry.run = limited_run"]
    source += ["entry.main()", ""]
    script = directory/f"run_{module}.py"
    script.write_text("\n".join(source), encoding="utf-8")
    return [sys.executable, str(script)]


def create_pairs(root, count=3, height=17, width=19):
    vi,ir = root/"vi",root/"ir"
    vi.mkdir(parents=True,exist_ok=True)
    ir.mkdir(parents=True,exist_ok=True)
    yy,xx = np.meshgrid(np.arange(height),np.arange(width),indexing="ij")
    for i in range(count):
        visible = np.stack(((xx*13+i*7)%256,(yy*17+i*9)%256,((xx+yy)*8+i*11)%256),-1).astype(np.uint8)
        infrared = ((yy*9+xx*5+i*19)%256).astype(np.uint8)
        Image.fromarray(visible).save(vi/f"scene_{i:03d}.png")
        Image.fromarray(infrared).save(ir/f"scene_{i:03d}.png")
    return vi,ir


def numpy_worker_collate(items):
    # Test the worker-independent RNG without requiring the macOS sandbox's
    # blocked torch_shm_manager. Production training keeps standard collation.
    return {key:np.stack([item[key].numpy() for item in items]) if isinstance(items[0][key],torch.Tensor)
            else [item[key] for item in items] for key in items[0]}


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pipeline_",dir=TESTS)
        self.root = Path(self.temporary.name)
        self.vi,self.ir = create_pairs(self.root/"data")

    def tearDown(self):
        self.temporary.cleanup()

    def test_synthetic_degradation_is_stateless_and_preserves_intact_pixels(self):
        ds = PairedFusionDataset(self.vi,self.ir,training=True,patch_size=16,
                degradation={"pair_probabilities":[0,1,0,0],"local_probability":1,"visible_types":["noise"]})
        first,again = ds[(1,123)],ds[(1,123)]
        for key in ("visible","infrared","hq_visible","mask_visible"):
            torch.testing.assert_close(first[key],again[key],rtol=0,atol=0)
        outside = 1-first["mask_visible"]
        torch.testing.assert_close(outside*(first["visible"]-first["hq_visible"]),torch.zeros_like(first["visible"]),rtol=0,atol=0)
        torch.testing.assert_close(first["infrared"],first["hq_infrared"],rtol=0,atol=0)
        self.assertEqual(first["mask_infrared"].sum().item(),0)
        self.assertGreater((first["visible"]-first["hq_visible"]).abs().sum().item(),0)

    def test_strict_pairing_and_shape_validation(self):
        (self.ir/"scene_002.png").unlink()
        with self.assertRaisesRegex(ValueError,"Unpaired"):
            PairedFusionDataset(self.vi,self.ir)
        Image.new("L",(11,13)).save(self.ir/"scene_002.png")
        ds = PairedFusionDataset(self.vi,self.ir)
        with self.assertRaisesRegex(ValueError,"dimensions differ"):
            ds[2]

    def test_sampler_resume_and_worker_independent_augmentation(self):
        ds = PairedFusionDataset(self.vi,self.ir,training=True,patch_size=16)
        full = list(ResumableBatchSampler(len(ds),2,5,seed=19))
        resumed = list(ResumableBatchSampler(len(ds),2,5,start_batch=2,seed=19))
        self.assertEqual(full[2:],resumed)
        a = list(DataLoader(ds,batch_sampler=full,num_workers=0))
        b = list(DataLoader(ds,batch_sampler=full,num_workers=1,collate_fn=numpy_worker_collate,
                           generator=torch.Generator().manual_seed(7)))
        for first,second in zip(a,b):
            torch.testing.assert_close(first["visible"],torch.from_numpy(second["visible"]),rtol=0,atol=0)

    def test_auxiliary_and_four_fusion_losses_have_correct_gradients(self):
        ds = PairedFusionDataset(self.vi,self.ir,patch_size=16,degradation={"pair_probabilities":[0,0,0,1]})
        batch = next(iter(DataLoader(ds,batch_size=2)))
        model = RobustFusionNet(tiny_config())
        outputs = model(batch["visible"],batch["infrared"])
        auxiliary = AuxiliaryLoss()
        aux,terms = auxiliary(outputs,batch)
        fusion = FusionLoss(10,10,1,1)(outputs["fused"],batch["hq_visible"],batch["hq_infrared"])
        (aux+fusion).backward()
        self.assertEqual(set(terms),{"reconstruction","keep","demand","error"})
        self.assertTrue(torch.isfinite(aux+fusion))
        self.assertGreater(model.error_visible.head[-1].weight.grad.abs().sum().item(),0)
        self.assertGreater(model.restorer_visible.demand_head[-1].weight.grad.abs().sum().item(),0)
        target = auxiliary.error_map(outputs["restored"]["visible"].detach(),batch["hq_visible"])
        self.assertTrue(torch.all(target >= 0))

    def test_paired_mode_reads_hq_and_gray_mask_without_synthetic_changes(self):
        hqv,hqi = create_pairs(self.root/"hq")
        mask_dir = self.root/"mask"
        mask_dir.mkdir()
        for i in range(3):
            Image.new("L",(19,17),255).save(mask_dir/f"scene_{i:03d}.png")
        with Image.open(self.vi/"scene_000.png") as image:
            degraded = (np.array(image,dtype=np.float32)*.5).astype(np.uint8)
        Image.fromarray(degraded).save(self.vi/"scene_000.png")
        ds = PairedFusionDataset(self.vi,self.ir,mode="paired",hq_visible_dir=hqv,
                                hq_infrared_dir=hqi,mask_visible_dir=mask_dir)
        sample = ds[0]
        self.assertEqual(sample["mask_visible"].shape,(1,17,19))
        self.assertTrue(torch.equal(sample["mask_visible"],torch.ones(1,17,19)))
        self.assertTrue(torch.equal(sample["mask_infrared"],torch.ones(1,17,19)))
        self.assertGreater((sample["visible"]-sample["hq_visible"]).abs().sum().item(),0)
        self.assertEqual(sample["degradation_meta"],"provided")
        with self.assertRaisesRegex(ValueError,"requires HQ"):
            PairedFusionDataset(self.vi,self.ir,mode="paired")

    def test_complete_region_loss_normalizes_soft_masks_and_handles_empty_region(self):
        outputs = {name:{} for name in ("restored","demand","error","demand_logits","error_raw")}
        batch = {}
        for modality,channels in (("visible",3),("infrared",1)):
            batch[modality] = torch.full((1,channels,1,1),.25)
            batch[f"hq_{modality}"] = batch[modality].clone()
            batch[f"mask_{modality}"] = torch.full((1,1,1,1),.5)
            outputs["restored"][modality] = torch.full((1,channels,1,1),.5,requires_grad=True)
            outputs["demand"][modality] = torch.zeros(1,1,1,1)
            outputs["error"][modality] = torch.zeros(1,1,1,1)
            outputs["demand_logits"][modality] = torch.zeros(1,1,1,1)
            outputs["error_raw"][modality] = torch.zeros(1,1,1,1)
        loss = AuxiliaryLoss(reconstruction_weight=0,keep_weight=1,demand_weight=0,error_weight=0)
        total,terms = loss(outputs,batch)
        self.assertAlmostEqual(terms["keep"].item(),.25,places=7)
        total.backward()
        for modality in ("visible","infrared"):
            batch[f"mask_{modality}"].fill_(1)
        total,terms = loss(outputs,batch)
        self.assertEqual(terms["keep"].item(),0)
        self.assertTrue(torch.isfinite(total))

    def test_python_entries_resume_exactly_and_inference_evaluates(self):
        # The two image directories have unrelated, nonstandard names.
        visible, infrared = self.root/"visible_camera", self.root/"thermal_sensor"
        self.vi.rename(visible)
        self.ir.rename(infrared)
        self.vi, self.ir = visible, infrared
        manual = {"VISIBLE_DIR": self.vi, "INFRARED_DIR": self.ir, "TOTAL_STEPS": 3,
                  "BATCH_SIZE": 1, "PATCH_SIZE": 16}
        internal = {"restoration_warmup_steps": 1, "transition_steps": 1, "lr_warmup_steps": 1,
                    "accumulation_steps": 2, "workers": 0, "cpu_threads": 1, "log_every": 1,
                    "save_every": 3, "preview_every": 2, "amp": "none", "deterministic": True,
                    "device": "cpu"}
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        def command(module, values, overrides=None, max_steps=None):
            entry = experiment_entry(self.root, module, values, overrides, tiny=True, max_steps=max_steps)
            result = subprocess.run(entry, cwd=self.root, env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout+"\n"+result.stderr)
            return result.stdout
        full, paused = self.root/"full", self.root/"paused"
        command("train", dict(manual, OUTPUT_DIR=full), internal)
        command("train", dict(manual, OUTPUT_DIR=paused), internal, max_steps=1)
        command("train", dict(manual, OUTPUT_DIR=paused, RESUME_TRAINING=True), internal)
        a,b = (torch.load(p/"last.pth",map_location="cpu",weights_only=True) for p in (full,paused))
        self.assertEqual(a["step"],3)
        self.assertEqual(b["next_micro_batch"],6)
        self.assertEqual(set(b["config"]["data"]),{"train"})
        self.assertNotIn("best_validation_loss",b)
        self.assertFalse((paused/"best.pth").exists())
        self.assertFalse((paused/"validation.jsonl").exists())
        self.assertFalse((paused/"config.yaml").exists())
        self.assertTrue((paused/"run_info.json").is_file())
        self.assertTrue((paused/"log.txt").is_file())
        self.assertTrue((paused/"loss_curves.png").is_file())
        self.assertTrue((paused/"state_curves.png").is_file())
        self.assertEqual(b["format_version"], 2)
        self.assertEqual(b["implementation_version"], "restormer_v2")
        last_record = json.loads((paused/"train.jsonl").read_text().splitlines()[-1])
        self.assertIn("vi_residual_saturated_fraction", last_record)
        self.assertIn("ir_error_grad_norm", last_record)
        self.assertIn("vi_damaged_error_mae", last_record)
        self.assertEqual(last_record["diagnostic_source"], "fixed_training_preview")
        self.assertTrue((paused/"previews/step_000001.png").is_file())
        self.assertTrue((paused/"previews/step_000003.png").is_file())
        for name,tensor in a["model"].items():
            torch.testing.assert_close(tensor,b["model"][name],rtol=0,atol=0)
        output = self.root/"inference"
        test_manual = {"CHECKPOINT_PATH": paused/"last.pth", "VISIBLE_DIR": self.vi,
                       "INFRARED_DIR": self.ir, "HQ_VISIBLE_DIR": self.vi,
                       "HQ_INFRARED_DIR": self.ir, "USE_HQ_REFERENCE": True, "OUTPUT_DIR": output}
        command("test", test_manual, {"device": "cpu", "cpu_threads": 1, "limit": 2})
        summary = json.loads((output/"summary.json").read_text())
        self.assertEqual(summary["images"],2)
        self.assertIn("restored_visible_psnr",summary["metrics"])
        self.assertTrue((output/"metrics.csv").is_file())
        self.assertEqual(len(list((output/"rgb").glob("*.png"))),2)
        self.assertEqual(Image.open(output/"rgb/scene_000.png").size,(19,17))
        self.assertEqual(Image.open(output/"gray/scene_000.png").mode,"L")
        self.assertTrue((output/"comparisons/scene_000.png").is_file())
        self.assertTrue((output/"quality_visible/scene_000.png").is_file())
        self.assertTrue((output/"applied_demand_visible/scene_000.png").is_file())
        self.assertTrue((output/"error_visible/scene_000.pt").is_file())
        plain = self.root/"plain_inference"
        command("test", dict(test_manual, OUTPUT_DIR=plain, USE_HQ_REFERENCE=False),
                {"device": "cpu", "cpu_threads": 1, "limit": 1, "save_aux": False,
                 "comparisons": False, "evaluate": False, "progress": False})
        self.assertTrue((plain/"rgb/scene_000.png").is_file())
        self.assertTrue((plain/"gray/scene_000.png").is_file())
        self.assertFalse((plain/"comparisons").exists())
        self.assertFalse((plain/"metrics.csv").exists())

    def test_manual_paths_and_boolean_switches_without_validation(self):
        defaults = training._build_settings()
        self.assertEqual(set(defaults["data"]), {"train"})
        self.assertNotIn("validate_every", defaults["training"])
        self.assertTrue(defaults["model"]["use_demand"])
        self.assertTrue(defaults["model"]["use_quality"])
        with patch.multiple(training, VISIBLE_DIR=str(self.vi), INFRARED_DIR=str(self.ir),
                            TOTAL_STEPS=3, BATCH_SIZE=1, RESUME_TRAINING=True):
            settings = training._build_settings()
            self.assertEqual(settings["data"]["train"]["visible_dir"], str(self.vi))
            self.assertEqual(settings["data"]["train"]["infrared_dir"], str(self.ir))
            self.assertEqual(settings["training"]["batch_size"], 1)
            self.assertTrue(settings["experiment"]["resume_training"])
            self.assertLessEqual(settings["training"]["lr_warmup_steps"], 3)
        with patch.multiple(training, ONLINE_DEGRADATION=False,
                            HQ_VISIBLE_DIR=str(self.vi), HQ_INFRARED_DIR=str(self.ir)):
            self.assertEqual(training._build_settings()["data"]["train"]["mode"], "paired")
        with patch.object(training, "ONLINE_DEGRADATION", "paired"):
            with self.assertRaises(TypeError):
                training._build_settings()
        with patch.multiple(inference, USE_HQ_REFERENCE=False,
                            HQ_VISIBLE_DIR=str(self.vi), HQ_INFRARED_DIR=str(self.ir)):
            self.assertIsNone(inference._build_settings().hq_visible_dir)
        self.assertFalse(hasattr(training, "parse_args"))
        self.assertFalse(hasattr(inference, "parse_args"))
        settings["model"]["fusion"]["widths"][0] = 16
        self.assertEqual(training._build_settings()["model"]["fusion"]["widths"][0], 32)
