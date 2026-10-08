"""Read-only checkpoint probe: head saturation and local gradient mechanisms.

Uses two existing MSRS training pairs and the checkpoint's original synthetic
degradations. No optimizer steps or production modifications are performed.
"""
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.fusion import RobustFusionNet
from utils.auxiliary_loss import AuxiliaryLoss
from utils.dataset import image_index
from utils.degradation import LocalDegrader
from utils.image import read_image


def statistics(value):
    value = value.float()
    return {"mean": float(value.mean()), "std": float(value.std(unbiased=False)),
            "min": float(value.min()), "max": float(value.max())}


def gradient_probe(raw, target, criterion):
    variable = raw.detach().float().clone().requires_grad_(True)
    loss = criterion(variable, target.detach().float())
    gradient, = torch.autograd.grad(loss, variable)
    return {"loss": float(loss.detach()), "gradient_abs_sum": float(gradient.abs().sum()),
            "gradient_abs_max": float(gradient.abs().max())}


def main():
    torch.set_num_threads(4)
    checkpoint = ROOT / "ckpt/last.pth"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = RobustFusionNet(state["config"]["model"]).eval()
    model.load_state_dict(state["model"], strict=True)
    auxiliary = AuxiliaryLoss(**state["config"]["auxiliary_loss"])
    sources = ROOT.parent / "datasets/train"
    indices = {"visible": image_index(sources/"vis"), "infrared": image_index(sources/"ir")}
    if set(indices["visible"]) != set(indices["infrared"]):
        raise ValueError("MSRS training pairs do not match.")
    samples = [next(key for key in sorted(indices["visible"]) if Path(key).name.endswith(suffix))
               for suffix in ("D", "N")]
    rows = []
    for sample in samples:
        vi, ir = (read_image(indices[modality][sample], channels)
                  for modality, channels in (("visible", 3), ("infrared", 1)))
        if vi.shape[-2:] != ir.shape[-2:]:
            raise ValueError(f"Unregistered training pair: {sample}")
        h, w = vi.shape[-2:]
        top, left = (h-128)//2, (w-128)//2
        vi, ir = vi[:, top:top+128, left:left+128], ir[:, top:top+128, left:left+128]
        for kind in ("noise", "blur", "contrast", "exposure"):
            degrader = LocalDegrader(dict(pair_probabilities=[0, 0, 0, 1], local_probability=0.0,
                                         visible_types=[kind], infrared_types=["noise"]))
            dv, di, _, _, _ = degrader(vi, ir, torch.Generator().manual_seed(123))
            tensors = {}
            handles = []
            for modality in ("visible", "infrared"):
                restorer = getattr(model, f"restorer_{modality}")
                error = getattr(model, f"error_{modality}")
                for label, module in (("feature", restorer.refinement), ("demand_raw", restorer.demand_head),
                                      ("residual_raw", restorer.residual_head), ("error_raw", error.head)):
                    def record(module, args, output, key=(modality, label)):
                        tensors[key] = output.detach().float().clone()
                    handles.append(module.register_forward_hook(record))
            try:
                with torch.no_grad():
                    outputs = model(dv[None], di[None], restoration_only=True)
            finally:
                for handle in handles:
                    handle.remove()
            for modality, source, reference in (("visible", dv[None], vi[None]), ("infrared", di[None], ir[None])):
                raw = {label: tensors[(modality, label)] for label in ("feature", "demand_raw", "residual_raw", "error_raw")}
                demand_target = (auxiliary.error_map(source, reference)/auxiliary.demand_scale).clamp(0, 1)
                restored = outputs["restored"][modality]
                error_target = auxiliary.error_map(restored, reference)
                demand = outputs["demand"][modality].detach()
                bound = getattr(model, f"restorer_{modality}").residual_bound
                row = dict(sample_id=sample, vi_type=kind, ir_type="noise", modality=modality,
                    input_mae=float((source-reference).abs().mean()),
                    restored_mae=float((restored-reference).abs().mean()),
                    demand_mean=float(demand.mean()), demand_target_mean=float(demand_target.mean()),
                    error_mean=float(outputs["error"][modality].mean()), error_target_mean=float(error_target.mean()),
                    quality_mean=float(outputs["quality"][modality].mean()),
                    residual_tanh_saturation_fraction=float((raw["residual_raw"].abs()>3).float().mean()),
                    head_statistics={name:statistics(value) for name,value in raw.items()},
                    demand_l1_sigmoid=gradient_probe(raw["demand_raw"], demand_target,
                        lambda z,y:F.l1_loss(z.sigmoid(),y)),
                    demand_bce_logits=gradient_probe(raw["demand_raw"], demand_target,
                        F.binary_cross_entropy_with_logits),
                    error_l1_softplus=gradient_probe(raw["error_raw"], error_target,
                        lambda z,y:F.l1_loss(F.softplus(z),y)),
                    error_unconstrained_regression=gradient_probe(raw["error_raw"], error_target/0.1,
                        F.smooth_l1_loss),
                    residual_gated_gradient=gradient_probe(raw["residual_raw"], reference,
                        lambda z,y:F.l1_loss(source+demand*bound*z.tanh(),y)),
                    residual_ungated_gradient=gradient_probe(raw["residual_raw"], reference,
                        lambda z,y:F.l1_loss(source+bound*z.tanh(),y)))
                rows.append(row)
                print(json.dumps({key:row[key] for key in ("sample_id", "vi_type", "modality", "input_mae", "restored_mae",
                    "residual_tanh_saturation_fraction", "demand_mean")}), flush=True)
    output = ROOT / "tests/diagnosis_20261007/restoration_gradient_probes.json"
    output.write_text(json.dumps(dict(checkpoint=str(checkpoint), step=state["step"], training_sample_ids=samples,
        description="Detached-logit gradient mechanisms; not retrained-model performance.", records=rows), indent=2)+"\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
