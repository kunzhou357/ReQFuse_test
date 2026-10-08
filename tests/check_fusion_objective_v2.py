"""E2: compare four-loss recipes by fitting pixels on eight training crops.

This tests objective preferences, not a fusion network or test-set performance.
All images and numeric evidence stay under tests/artifacts.
"""
from datetime import datetime
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw
import torch

from utils.dataset import image_index
from utils.experiment import resolve_path, save_json
from utils.image import read_image, save_image
from utils.loss import FusionLoss, to_luminance
from utils.visualization import _pil


VISIBLE_DIR = "../datasets/train/vis"
INFRARED_DIR = "../datasets/train/ir"
UPDATES = 300
PATCH_SIZE = 128


def output_gradient_norms(image, vi, ir, criterion):
    image = image.detach().requires_grad_()
    _, terms = criterion(image, vi, ir, return_components=True)
    weights = dict(intensity=10, gradient=criterion.weights["gradient"], ssim=1, color=1)
    return {key:float(torch.autograd.grad(weights[key]*value, image, retain_graph=True)[0].norm())
            for key,value in terms.items()}


def component_records(image, vi, ir, criterion):
    records = []
    with torch.no_grad():
        for i in range(image.shape[0]):
            total, terms = criterion(image[i:i+1], vi[i:i+1], ir[i:i+1], return_components=True)
            records.append(dict(total=float(total), **{key:float(value) for key,value in terms.items()}))
    return records


def main():
    torch.set_num_threads(4)
    output = ROOT/"tests/artifacts"/datetime.now().strftime("objective_v2_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    vis, infrared = image_index(resolve_path(VISIBLE_DIR)), image_index(resolve_path(INFRARED_DIR))
    if set(vis) != set(infrared):
        raise ValueError("E2 requires matching training source IDs.")
    ids = [s for s in sorted(vis) if s.endswith("D")][:4]+[s for s in sorted(vis) if s.endswith("N")][:4]
    if len(ids) != 8:
        raise ValueError("E2 needs four day and four night training pairs.")
    vi, ir = [], []
    for sample in ids:
        v, i = read_image(vis[sample], 3), read_image(infrared[sample], 1)
        h, w = v.shape[-2:]
        if i.shape[-2:] != (h,w) or min(h,w) < PATCH_SIZE:
            raise ValueError(f"E2 needs registered pairs of at least {PATCH_SIZE}px: {sample}")
        top, left = (h-PATCH_SIZE)//2, (w-PATCH_SIZE)//2
        vi.append(v[:,top:top+PATCH_SIZE,left:left+PATCH_SIZE])
        ir.append(i[:,top:top+PATCH_SIZE,left:left+PATCH_SIZE])
    vi, ir = torch.stack(vi), torch.stack(ir)
    target = torch.maximum(to_luminance(vi), ir)
    initial = (vi+target-to_luminance(vi)).clamp(.0001,.9999)
    results, evidence = {}, {}
    for name, gradient_weight in (("old_10_10_1_1",10), ("v2_10_2_1_1",2)):
        criterion = FusionLoss(10,gradient_weight,1,1,require_color=True)
        # Identical bounded parameterization, initial image and optimizer.
        parameter = torch.nn.Parameter(torch.logit(initial.clone()))
        optimizer = torch.optim.Adam([parameter], lr=.04)
        # A fixed large LR oscillates around the nonsmooth intensity minimum;
        # decay identically for both recipes to inspect a more settled fit.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UPDATES, eta_min=.001)
        history = []
        for step in range(1, UPDATES+1):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(parameter.sigmoid(), vi, ir)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step == 1 or step % 100 == 0 or step == UPDATES:
                history.append(dict(step=step, loss=float(loss.detach())))
                print(f"E2 {name} {step}/{UPDATES}: loss={float(loss.detach()):.5f}", flush=True)
        result = parameter.detach().sigmoid()
        results[name] = result
        evidence[name] = dict(history=history, initial=component_records(initial,vi,ir,criterion),
                              final=component_records(result,vi,ir,criterion),
                              initial_weighted_output_gradients=output_gradient_norms(initial,vi,ir,criterion),
                              final_weighted_output_gradients=output_gradient_norms(result,vi,ir,criterion))
    canvas = Image.new("RGB", (5*166+6,8*192+6), (24,27,32))
    draw = ImageDraw.Draw(canvas)
    for row,sample in enumerate(ids):
        tiles = (("VI",vi[row]), ("IR",ir[row]), ("Initial",initial[row]),
                 ("Old weights",results["old_10_10_1_1"][row]), ("v2 weights",results["v2_10_2_1_1"][row]))
        for column,(label,tensor) in enumerate(tiles):
            x, y = 6+column*166, 6+row*192
            draw.text((x,y), f"{sample} | {label}", fill=(235,235,235))
            canvas.paste(_pil(tensor).resize((160,160), Image.Resampling.NEAREST), (x,y+25))
            save_image(tensor, output/label.replace(" ","_")/(sample+".png"))
    canvas.save(output/"comparison.png")
    save_json(output/"summary.json", dict(sample_ids=ids, crop="128 center crop", updates=UPDATES,
              optimizer="Adam", initial_lr=.04, minimum_lr=.001, scheduler="cosine",
              scope="Training-image pixel fit; no network training, no fused ground truth, no generic quality claim",
              evidence=evidence))
    print(f"E2 artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
