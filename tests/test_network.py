"""Meaningful model contract, gradient routing and source-gate tests."""
import unittest

import torch

from model.blocks import RestormerBlock
from model.fusion import RobustFusionNet
from model.quality import PostRestorationErrorHead, WindowChannelCrossAttention
from utils.auxiliary_loss import AuxiliaryLoss
from utils.loss import FusionLoss


def tiny_config():
    return {"restorer":{"widths":[8,16,32],"heads":[1,2,4],"encoder_blocks":[1,1,1],"decoder_blocks":[1,1],"expansion":2.0},
            "fusion":{"widths":[8,16,32,64],"heads":[1,2,4,8],"encoder_blocks":[1,1,1,1],"decoder_blocks":[1,1,1],"expansion":2.0},
            "state_hidden":4,"window_size":4}


class NetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(31)

    def test_default_parameter_count_and_real_forward_backward(self):
        model = RobustFusionNet()
        self.assertEqual(sum(p.numel() for p in model.parameters()),8725358)
        vi,ir = torch.rand(1,3,17,19),torch.rand(1,1,17,19)
        outputs = model(vi,ir)
        self.assertEqual(outputs["fused"].shape,vi.shape)
        outputs["fused"].square().mean().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_small_odd_shapes_identity_initialization_and_every_block_executed(self):
        model = RobustFusionNet(tiny_config()).eval()
        hits = {}
        handles = []
        for name,block in model.named_modules():
            if isinstance(block,RestormerBlock):
                handles.append(block.register_forward_hook(lambda m,i,o,n=name:hits.__setitem__(n,hits.get(n,0)+1)))
        for h,w in ((1,1),(3,5),(17,19)):
            vi,ir = torch.rand(2,3,h,w),torch.rand(2,1,h,w)
            with torch.no_grad():
                outputs = model(vi,ir)
            torch.testing.assert_close(outputs["restored"]["visible"],vi,rtol=0,atol=0)
            torch.testing.assert_close(outputs["restored"]["infrared"],ir,rtol=0,atol=0)
            self.assertTrue(((outputs["fused"] >= 0)&(outputs["fused"] <= 1)).all())
            for key in ("demand","demand_logits","applied_demand","error_raw","error","quality"):
                self.assertEqual(outputs[key]["visible"].shape,(2,1,h,w))
                self.assertTrue(torch.isfinite(outputs[key]["visible"]).all())
        self.assertEqual(len(hits),len(handles))
        self.assertTrue(all(n==3 for n in hits.values()))
        for handle in handles:
            handle.remove()

    def test_source_gate_zero_positions_and_zero_window(self):
        cross = WindowChannelCrossAttention(16,4,window_size=4)
        target,source = torch.randn(2,16,5,7),torch.randn(2,16,5,7)
        quality = torch.rand(2,1,5,7)
        quality[:,:,:,2:4] = 0
        message = cross(target,source,quality)
        self.assertTrue(torch.equal(message[:,:,:,2:4],torch.zeros_like(message[:,:,:,2:4])))
        zero = cross(target,source,torch.zeros_like(quality))
        self.assertTrue(torch.equal(zero,torch.zeros_like(zero)))
        self.assertGreater(cross(target,source,torch.ones_like(quality)).abs().sum().item(),0)

    def test_single_padded_window_matches_global_channel_relation(self):
        local = WindowChannelCrossAttention(8,2,window_size=8)
        global_cross = WindowChannelCrossAttention(8,2,local=False)
        global_cross.load_state_dict(local.state_dict())
        t,s,q = torch.randn(1,8,5,7),torch.randn(1,8,5,7),torch.rand(1,1,5,7)
        torch.testing.assert_close(local(t,s,q),global_cross(t,s,q),atol=2e-6,rtol=2e-5)

    def test_error_head_detaches_all_inputs(self):
        head = PostRestorationErrorHead(3,8,hidden=4)
        inputs = [torch.rand(1,c,5,7,requires_grad=True) for c in (3,3,1,8)]
        # Nonzero final weights also exercise this head's own learnable norm.
        torch.nn.init.normal_(head.head[-1].weight, std=.1)
        head(*inputs)["raw"].sum().backward()
        self.assertTrue(all(x.grad is None for x in inputs))
        self.assertGreater(head.head[-1].weight.grad.abs().sum().item(),0)
        self.assertGreater(head.feature_norm.weight.grad.abs().sum().item(),0)

    def test_fusion_gradient_reaches_restoration_but_not_error_head(self):
        model = RobustFusionNet(tiny_config())
        outputs = model(torch.rand(1,3,9,11),torch.rand(1,1,9,11))
        outputs["fused"].mean().backward()
        self.assertTrue(all(p.grad is None for p in model.error_visible.parameters()))
        self.assertTrue(all(p.grad is None for p in model.error_infrared.parameters()))
        self.assertGreater(model.restorer_visible.residual_head.weight.grad.abs().sum().item(),0)
        self.assertGreater(model.cross_i_to_v["2"].kv.weight.grad.abs().sum().item(),0)
        self.assertTrue(all(p.grad is None for p in model.restorer_visible.demand_head.parameters()))

    def test_restore_gate_curriculum_detaches_demand_and_preserves_residual_gradient(self):
        model = RobustFusionNet(tiny_config())
        restorer = model.restorer_visible
        torch.nn.init.constant_(restorer.residual_head.bias, .2)
        image = torch.rand(1,3,8,8)
        for strength, expected in ((0, 1), (.5, .75), (1, .5)):
            restorer.zero_grad(set_to_none=True)
            outputs = restorer(image, demand_strength=strength)
            torch.testing.assert_close(outputs["applied_demand"], torch.full_like(outputs["applied_demand"], expected))
            self.assertFalse(outputs["applied_demand"].requires_grad)
            outputs["restored"].sum().backward()
            self.assertGreater(restorer.residual_head.weight.grad.abs().sum().item(), 0)
            self.assertTrue(all(p.grad is None for p in restorer.demand_head.parameters()))
            self.assertLess(outputs["feature"].std(unbiased=False).item(), 1.01)
        with self.assertRaises(ValueError):
            restorer(image, demand_strength=-1)

    def test_cpu_bfloat16_autocast_is_finite(self):
        model = RobustFusionNet(tiny_config())
        vi, ir = torch.rand(1,3,9,11), torch.rand(1,1,9,11)
        batch = {"visible":vi, "infrared":ir, "hq_visible":(vi+.05).clamp(0,1),
                 "hq_infrared":(ir+.05).clamp(0,1),
                 "mask_visible":torch.ones(1,1,9,11), "mask_infrared":torch.ones(1,1,9,11)}
        with torch.autocast("cpu",dtype=torch.bfloat16):
            outputs = model(vi, ir, demand_strength=.5)
            auxiliary, _ = AuxiliaryLoss()(outputs, batch)
            loss = auxiliary+FusionLoss(10,2,1,1)(outputs["fused"], batch["hq_visible"], batch["hq_infrared"])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(model.restorer_visible.demand_head[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.error_visible.head[-1].weight.grad.abs().sum().item(), 0)

    def test_activation_checkpoint_matches_outputs_and_gradients(self):
        plain = RobustFusionNet(tiny_config())
        config = tiny_config()
        config["checkpoint_blocks"] = True
        checked = RobustFusionNet(config)
        checked.load_state_dict(plain.state_dict())
        vi,ir = torch.rand(1,3,9,11),torch.rand(1,1,9,11)
        a,b = plain(vi,ir)["fused"],checked(vi,ir)["fused"]
        torch.testing.assert_close(a,b)
        a.mean().backward()
        b.mean().backward()
        for (_,p),(_,q) in zip(plain.named_parameters(),checked.named_parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad,q.grad)

    def test_ablation_switches_and_restoration_only_contract(self):
        config = tiny_config()
        config.update(use_demand=False,use_quality=False,gate_private=False,gate_messages=False,local_exchange=False)
        model = RobustFusionNet(config)
        vi,ir = torch.rand(1,3,9,11),torch.rand(1,1,9,11)
        outputs = model(vi,ir,restoration_only=True)
        self.assertNotIn("fused",outputs)
        self.assertEqual(model(vi,ir,return_aux=False).shape,vi.shape)
        with self.assertRaises(ValueError):
            model(vi,ir,quality_strength=2)
        with self.assertRaises(ValueError):
            RobustFusionNet({"fusion":{"widths":[32,64,128,128]}})

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA unavailable")
    def test_cuda_mixed_precision_forward_backward(self):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = RobustFusionNet(tiny_config()).cuda()
        with torch.autocast("cuda",dtype=dtype):
            out = model(torch.rand(1,3,17,19,device="cuda"),torch.rand(1,1,17,19,device="cuda"))
            loss = out["fused"].float().mean()+out["error"]["visible"].mean()
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
