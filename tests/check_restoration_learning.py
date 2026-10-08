"""E3: short learning check on training pairs, with the complete v2 restorers.

No validation split or production checkpoint is created. Edit this test-only
file to change the source directories; every generated image/log stays here.
"""
from datetime import datetime
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F
from torch.utils.data import default_collate

from model.fusion import RobustFusionNet
from utils.auxiliary_loss import AuxiliaryLoss
from utils.degradation import LocalDegrader
from utils.diagnostics import collect_diagnostics, head_gradient_norms, reference_error_maps
from utils.dataset import image_index
from utils.experiment import append_json, resolve_path, save_json, seed_everything
from utils.image import read_image, save_image
from utils.visualization import save_comparison, save_state_curves


VISIBLE_DIR = "../datasets/train/vis"
INFRARED_DIR = "../datasets/train/ir"
USE_CUDA = False
UPDATES = 250
PATCH_SIZE = 128
BATCH_SIZE = 2
SAMPLES = 16


def main():
    torch.set_num_threads(4)
    seed_everything(107, deterministic=True)
    device = torch.device("cuda" if USE_CUDA else "cpu")
    output = ROOT/"tests/artifacts"/datetime.now().strftime("restoration_v2_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    vi, ir = image_index(resolve_path(VISIBLE_DIR)), image_index(resolve_path(INFRARED_DIR))
    if set(vi) != set(ir):
        raise ValueError("E3 requires exactly matching training source IDs.")
    day = [s for s in sorted(vi) if s.endswith("D")]
    night = [s for s in sorted(vi) if s.endswith("N")]
    ids = (day[:SAMPLES//2]+night[:SAMPLES-SAMPLES//2]) if day and night else sorted(vi)[:SAMPLES]
    if len(ids) != SAMPLES:
        raise ValueError(f"E3 needs {SAMPLES} pairs; found {len(ids)} selected pairs.")
    # A fixed, known-noise engineering check. The main training distribution is
    # unchanged; this small exercise is not a general robustness benchmark.
    degrader = LocalDegrader({"pair_probabilities": [0,0,0,1], "local_probability": .5,
                              "visible_types": ["noise"], "infrared_types": ["noise"],
                              "noise_std": [.06, .10]})
    examples = []
    for i, sample in enumerate(ids):
        visible, infrared = read_image(vi[sample], 3), read_image(ir[sample], 1)
        if visible.shape[-2:] != infrared.shape[-2:]:
            raise ValueError(f"Unregistered pair: {sample}")
        h, w = visible.shape[-2:]
        ph, pw = max(0, PATCH_SIZE-h), max(0, PATCH_SIZE-w)
        visible, infrared = (F.pad(x, (0,pw,0,ph), mode="replicate") for x in (visible,infrared))
        top, left = (h+ph-PATCH_SIZE)//2, (w+pw-PATCH_SIZE)//2
        visible, infrared = (x[:,top:top+PATCH_SIZE,left:left+PATCH_SIZE] for x in (visible,infrared))
        dv, di, mv, mi, _ = degrader(visible, infrared, torch.Generator().manual_seed(601+i))
        examples.append(dict(visible=dv, infrared=di, hq_visible=visible, hq_infrared=infrared,
                             mask_visible=mv, mask_infrared=mi, sample_id=sample))
        for kind,image in (("hq_vi",visible), ("hq_ir",infrared), ("input_vi",dv), ("input_ir",di),
                           ("mask_vi",mv), ("mask_ir",mi)):
            save_image(image, output/"data"/kind/(sample+".png"))
    model = RobustFusionNet().to(device)
    criterion = AuxiliaryLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    def batch_at(indices):
        batch = default_collate([examples[int(i)] for i in indices])
        return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}

    @torch.no_grad()
    def evaluate(strength, step, save_preview=False):
        was_training = model.training
        model.eval()
        records = []
        try:
            for start in range(0, len(examples), BATCH_SIZE):
                batch = batch_at(range(start, min(start+BATCH_SIZE, len(examples))))
                result = model(batch["visible"], batch["infrared"], demand_strength=strength,
                               quality_strength=0, restoration_only=True)
                errors = reference_error_maps(result, batch, criterion)
                records.append(collect_diagnostics(result, batch, criterion, errors))
                if save_preview:
                    save_comparison(batch, result, output/"previews"/f"step_{step:04d}_gate{strength:g}_group{start//BATCH_SIZE}.png",
                                    max_samples=len(batch["sample_id"]), reference_errors=errors,
                                    title=f"E3 training samples | update {step} | applied gate strength {strength:g}")
            return {key:sum(row[key] for row in records)/len(records) for key in records[0]}
        finally:
            model.train(was_training)

    baseline = evaluate(0, 0, save_preview=True)
    generator = torch.Generator().manual_seed(912)
    order = torch.randperm(len(examples), generator=generator)
    batches_per_epoch = (len(examples)+BATCH_SIZE-1)//BATCH_SIZE
    for step in range(1, UPDATES+1):
        if (step-1) % batches_per_epoch == 0:
            order = torch.randperm(len(examples), generator=generator)
        offset = ((step-1) % batches_per_epoch)*BATCH_SIZE
        batch = batch_at(order[offset:offset+BATCH_SIZE])
        optimizer.zero_grad(set_to_none=True)
        result = model(batch["visible"], batch["infrared"], demand_strength=0,
                       quality_strength=0, restoration_only=True)
        loss, terms = criterion(result, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"E3 nonfinite loss at {step}.")
        loss.backward()
        gradients = head_gradient_norms(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == UPDATES:
            metrics = evaluate(0, step, save_preview=step == UPDATES)
            record = dict(metrics, **gradients, step=step, loss=float(loss.detach()),
                          **{f"aux_{k}":float(v.detach()) for k,v in terms.items()})
            append_json(output/"train.jsonl", record)
            print(f"E3 {step}/{UPDATES}: VI {metrics['vi_input_mae']:.5f}->{metrics['vi_restored_mae']:.5f}; "
                  f"IR {metrics['ir_input_mae']:.5f}->{metrics['ir_restored_mae']:.5f}; "
                  f"sat VI/IR={metrics['vi_residual_saturated_fraction']:.3%}/{metrics['ir_residual_saturated_fraction']:.3%}", flush=True)
    after = evaluate(0, UPDATES)
    inference = evaluate(1, UPDATES, save_preview=True)
    save_json(output/"summary.json", {"scope": "16 fixed training pairs, known noise, restoration-only engineering check",
              "updates": UPDATES, "patch_size": PATCH_SIZE, "batch_size": BATCH_SIZE,
              "parameters": sum(p.numel() for p in model.parameters()), "sample_ids": ids,
              "baseline": baseline, "warmup_gate_after": after, "inference_gate_after": inference,
              "note": "No fusion training or generic robustness claim. Constant LR for this short learning check."})
    save_state_curves(output/"train.jsonl", output/"state_curves.png")
    print(f"E3 artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
