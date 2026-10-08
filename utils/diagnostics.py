"""Read-only restoration/state diagnostics on fixed training examples."""
import torch

from .auxiliary_loss import masked_mean


@torch.no_grad()
def reference_error_maps(outputs, batch, criterion):
    return {kind: {m: criterion.error_map(images[m], batch[f"hq_{m}"])
                   for m in ("visible", "infrared")}
            for kind, images in (("input", batch), ("restored", outputs["restored"]))}


@torch.no_grad()
def collect_diagnostics(outputs, batch, criterion, error_maps=None):
    """Flat numeric records; no weight/RNG/mode changes or validation selection."""
    errors = error_maps if error_maps is not None else reference_error_maps(outputs, batch, criterion)
    record = {}
    for m, label in (("visible", "vi"), ("infrared", "ir")):
        source, reference = batch[m].float(), batch[f"hq_{m}"].float()
        restored = outputs["restored"][m].float()
        mask = batch[f"mask_{m}"].float()
        before = (source-reference).abs().mean(1, keepdim=True)
        after = (restored-reference).abs().mean(1, keepdim=True)
        change = (restored-source).abs().mean(1, keepdim=True)
        truth = errors["restored"][m]
        demand_target = (errors["input"][m]/criterion.demand_scale).clamp(0, 1)
        predicted = outputs["error"][m].float()
        metrics = {
            "input_mae": before.mean(), "restored_mae": after.mean(),
            "damaged_input_mae": masked_mean(before, mask),
            "damaged_restored_mae": masked_mean(after, mask),
            "intact_change_mae": masked_mean(change, 1-mask),
            "damaged_fraction": mask.mean(), "intact_fraction": (1-mask).mean(),
            "demand_target_mean": demand_target.mean(),
            "demand_mean": outputs["demand"][m].float().mean(),
            "demand_mae": (outputs["demand"][m].float()-demand_target).abs().mean(),
            "applied_demand_mean": outputs["applied_demand"][m].float().mean(),
            "error_target_mean": truth.mean(), "error_mean": predicted.mean(),
            "error_mae": (predicted-truth).abs().mean(),
            "damaged_error_mae": masked_mean((predicted-truth).abs(), mask),
            "damaged_zero_error_mae": masked_mean(truth, mask),
            "error_std": predicted.std(unbiased=False),
            "quality_mean": outputs["quality"][m].float().mean(),
            "head_feature_std": outputs["head_feature_std"][m],
            "residual_saturated_fraction": (outputs["residual_logits"][m].abs() > 3).float().mean(),
        }
        for kind in ("demand_logits", "error_raw", "quality"):
            values = outputs[kind][m].detach().float().flatten()
            quantiles = torch.quantile(values, values.new_tensor([0.01, 0.5, 0.99]))
            metrics.update({f"{kind}_{name}": v for name,v in zip(("p01", "p50", "p99"), quantiles)})
        # Calibrate quality against actual remaining error, not degradation labels.
        for i, (low, high) in enumerate(((0, .01), (.01, .03), (.03, .1), (.1, float("inf")))):
            selected = ((truth >= low) & (truth < high)).float()
            metrics[f"error_bin{i}_fraction"] = selected.mean()
            metrics[f"error_bin{i}_predicted"] = masked_mean(predicted, selected)
            metrics[f"error_bin{i}_quality"] = masked_mean(outputs["quality"][m].float(), selected)
        record.update({f"{label}_{name}": float(value) for name,value in metrics.items()})
    return record


@torch.no_grad()
def head_gradient_norms(model):
    """Unscaled, pre-clipping parameter gradients at the optimizer boundary."""
    record = {}
    for m, label in (("visible", "vi"), ("infrared", "ir")):
        restorer = getattr(model, f"restorer_{m}")
        modules = {"demand": restorer.demand_head, "residual": restorer.residual_head,
                   "head_norm": restorer.head_norm, "error": getattr(model, f"error_{m}")}
        for name, module in modules.items():
            gradients = [p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
            record[f"{label}_{name}_grad_norm"] = float(torch.stack(gradients).sum().sqrt()) if gradients else 0.0
    return record
