"""从仓库根目录运行：python tests/run_tests.py。"""

import unittest

import torch

from utils.loss import (
    ColorLoss, FusionLoss, GradientLoss, IntensityLoss, SSIMLoss,
    SobelGradient, rgb_to_ycbcr, ssim, to_luminance,
)


class FusionLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(17)

    def test_identical_inputs_have_zero_loss_including_black_and_small_images(self):
        for channels, height, width in ((1, 1, 1), (3, 2, 3), (1, 15, 17), (3, 15, 17)):
            for fill in (0.0, 0.4, 1.0):
                image = torch.full((2, channels, height, width), fill)
                total, terms = FusionLoss()(image, image, image, return_components=True)
                self.assertLess(abs(total.item()), 1e-6)
                self.assertEqual(set(terms), {"intensity", "gradient", "ssim", "color"})

    def test_intensity_matches_known_max_mean_and_single_reference(self):
        fused = torch.full((1, 1, 3, 4), 0.25)
        a, b = torch.full_like(fused, 0.3), torch.full_like(fused, 0.7)
        self.assertAlmostEqual(IntensityLoss()(fused, a, b).item(), 0.45, places=6)
        self.assertAlmostEqual(IntensityLoss("mean")(fused, a, b).item(), 0.25, places=6)
        self.assertAlmostEqual(IntensityLoss()(fused, a).item(), 0.05, places=6)

    def test_sobel_constant_and_linear_ramp(self):
        sobel = SobelGradient()
        self.assertTrue(torch.equal(sobel(torch.ones(2, 3, 5, 6)), torch.zeros(2, 3, 5, 6)))
        ramp = torch.arange(6, dtype=torch.float32).reshape(1, 1, 1, 6).expand(1, 1, 5, 6)
        expected = torch.tensor([4., 8., 8., 8., 8., 4.]).reshape(1, 1, 1, 6).expand_as(ramp)
        torch.testing.assert_close(sobel(ramp), expected)
        self.assertEqual(GradientLoss()(ramp, ramp, torch.zeros_like(ramp)).item(), 0)

    def test_ssim_constant_images_match_closed_form(self):
        a = torch.full((1, 1, 3, 4), 0.2, dtype=torch.float64)
        b = torch.full_like(a, 0.6)
        c1 = 0.01 ** 2
        expected = (2 * 0.2 * 0.6 + c1) / (0.2 ** 2 + 0.6 ** 2 + c1)
        self.assertAlmostEqual(ssim(a, b).item(), expected, places=10)
        self.assertGreater(SSIMLoss()(a, b).item(), SSIMLoss()(a, a).item())

    def test_ssim_is_symmetric_and_respects_data_range(self):
        a, b = torch.rand(2, 3, 13, 15), torch.rand(2, 3, 13, 15)
        torch.testing.assert_close(ssim(a, b), ssim(b, a))
        torch.testing.assert_close(ssim(a, b), ssim(a * 255, b * 255, data_range=255), atol=2e-6, rtol=2e-5)

    def test_color_changes_hue_but_ignores_uniform_brightness_shift(self):
        reference = torch.full((1, 3, 4, 5), 0.4, dtype=torch.float64)
        self.assertLess(ColorLoss()(reference + 0.1, reference).item(), 1e-12)
        predicted = reference.clone()
        predicted[:, 0] += 0.1
        predicted[:, 1] -= 0.299 * 0.1 / 0.587
        torch.testing.assert_close(to_luminance(predicted), to_luminance(reference))
        predicted.requires_grad_()
        loss = FusionLoss(intensity_weight=0, gradient_weight=0, ssim_weight=0)(predicted, reference, reference[:, :1])
        self.assertGreater(loss.item(), 0)
        loss.backward()
        self.assertGreater(predicted.grad.abs().sum().item(), 0)

    def test_color_conversion_known_primaries(self):
        red = torch.tensor([1., 0., 0.]).reshape(1, 3, 1, 1)
        torch.testing.assert_close(to_luminance(red), torch.tensor([[[[0.299]]]]))
        torch.testing.assert_close(rgb_to_ycbcr(red).flatten(), torch.tensor([0.299, 0.331264, 1.0]))

    def test_rgb_fusion_backward_weighted_sum_and_custom_color_reference(self):
        fused = torch.rand(2, 3, 9, 13, requires_grad=True)
        visible, infrared = torch.rand_like(fused), torch.rand(2, 1, 9, 13)
        criterion = FusionLoss(2, 3, 4, 5)
        total, terms = criterion(fused, visible, infrared, return_components=True)
        expected = 2 * terms["intensity"] + 3 * terms["gradient"] + 4 * terms["ssim"] + 5 * terms["color"]
        torch.testing.assert_close(total, expected)
        total.backward()
        self.assertTrue(torch.isfinite(fused.grad).all())
        self.assertGreater(fused.grad.abs().sum().item(), 0)
        _, custom = criterion(fused, visible, infrared, color_reference=fused.detach(), return_components=True)
        self.assertEqual(custom["color"].item(), 0)

    def test_grayscale_color_policy_and_disabled_terms(self):
        fused = torch.rand(2, 1, 4, 5, requires_grad=True)
        source = torch.rand_like(fused)
        _, terms = FusionLoss()(fused, source, source, return_components=True)
        self.assertEqual(terms["color"].item(), 0)
        with self.assertRaisesRegex(ValueError, "requires RGB"):
            FusionLoss(require_color=True)(fused, source, source)
        total, terms = FusionLoss(0, 0, 0, 0)(fused, source, source, return_components=True)
        total.backward()
        self.assertEqual(total.item(), 0)
        self.assertTrue(torch.equal(fused.grad, torch.zeros_like(fused)))
        self.assertTrue(all(term.item() == 0 for term in terms.values()))

    def test_reductions_have_consistent_maps(self):
        fused, a, b = torch.rand(2, 3, 5, 7), torch.rand(2, 3, 5, 7), torch.rand(2, 1, 5, 7)
        loss_map = FusionLoss(reduction="none")(fused, a, b)
        self.assertEqual(loss_map.shape, (2, 1, 5, 7))
        torch.testing.assert_close(loss_map.mean(), FusionLoss()(fused, a, b))
        torch.testing.assert_close(loss_map.sum(), FusionLoss(reduction="sum")(fused, a, b))

    def test_half_precision_and_cpu_autocast_have_finite_gradients(self):
        for dtype in (torch.float16, torch.bfloat16):
            fused = torch.rand(1, 3, 4, 5, dtype=dtype, requires_grad=True)
            a, b = torch.rand_like(fused), torch.rand(1, 1, 4, 5, dtype=dtype)
            loss = FusionLoss()(fused, a, b)
            self.assertEqual(loss.dtype, torch.float32)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(fused.grad).all())
        fused = torch.rand(1, 3, 4, 5, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = FusionLoss()(fused, torch.rand_like(fused), torch.rand(1, 1, 4, 5))
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(fused.grad).all())

    def test_ssim_gradient_matches_finite_differences(self):
        predicted = torch.rand(1, 1, 2, 3, dtype=torch.float64, requires_grad=True)
        reference = torch.rand_like(predicted)
        self.assertTrue(torch.autograd.gradcheck(lambda x: SSIMLoss()(x, reference), (predicted,)))

    def test_validation_rejects_wrong_layout_dtype_shape_and_configuration(self):
        image = torch.rand(1, 1, 3, 4)
        with self.assertRaises(TypeError):
            FusionLoss()(image.to(torch.uint8), image, image)
        with self.assertRaises(ValueError):
            FusionLoss()(image.squeeze(0), image, image)
        with self.assertRaises(ValueError):
            FusionLoss()(image, torch.rand(2, 1, 3, 4), image)
        with self.assertRaises(ValueError):
            FusionLoss()(torch.rand(1, 2, 3, 4), image, image)
        with self.assertRaises(ValueError):
            ColorLoss()(image, image)
        for options in ({"gradient_weight": -1}, {"ssim_weight": float("nan")},
                        {"ssim_visible_weight": 2}, {"data_range": 0},
                        {"window_size": 4}, {"intensity_mode": "unsupported"},
                        {"reduction": "unsupported"}):
            with self.assertRaises(ValueError):
                FusionLoss(**options)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_forward_and_backward(self):
        fused = torch.rand(1, 3, 8, 9, device="cuda", requires_grad=True)
        loss = FusionLoss().to("cuda")(fused, torch.rand_like(fused), torch.rand(1, 1, 8, 9, device="cuda"))
        loss.backward()
        self.assertTrue(torch.isfinite(fused.grad).all())

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_mps_module_transfer_forward_and_backward(self):
        fused = torch.rand(1, 3, 8, 9, device="mps", requires_grad=True)
        loss = FusionLoss().to("mps")(fused, torch.rand_like(fused), torch.rand(1, 1, 8, 9, device="mps"))
        loss.backward()
        self.assertTrue(torch.isfinite(fused.grad).all())


if __name__ == "__main__":
    unittest.main()
