"""Infer and evaluate Ours. Edit the defaults below, then run: python test.py."""
import sys
from types import SimpleNamespace
import csv
from itertools import islice
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__:
    from .model.fusion import MODEL_VERSION, RobustFusionNet
    from .utils.auxiliary_loss import AuxiliaryLoss
    from .utils.dataset import PairedFusionDataset
    from .utils.diagnostics import reference_error_maps
    from .utils.experiment import (amp_settings, autocast_context, choose_device, move_batch,
                                 read_checkpoint, resolve_path, save_json, strict_load)
    from .utils.image import fusion_statistics, reconstruction_metrics, save_image
    from .utils.loss import FusionLoss
    from .utils.loss import to_luminance
    from .utils.visualization import save_comparison
else:
    from model.fusion import MODEL_VERSION, RobustFusionNet
    from utils.auxiliary_loss import AuxiliaryLoss
    from utils.dataset import PairedFusionDataset
    from utils.diagnostics import reference_error_maps
    from utils.experiment import (amp_settings, autocast_context, choose_device, move_batch,
                                 read_checkpoint, resolve_path, save_json, strict_load)
    from utils.image import fusion_statistics, reconstruction_metrics, save_image
    from utils.loss import FusionLoss
    from utils.loss import to_luminance
    from utils.visualization import save_comparison


# ===================== 只需修改这里 =====================
# 直接填写图像目录，不要求名称为 vi/ir。
VISIBLE_DIR = "data/test/vi"
INFRARED_DIR = "data/test/ir"
CHECKPOINT_PATH = "outputs/restormer_v2/last.pth"
OUTPUT_DIR = "outputs/inference_v2"

USE_HQ_REFERENCE = False                  # True：额外评估同模态恢复质量
HQ_VISIBLE_DIR = None                     # 仅 True 时填写
HQ_INFRARED_DIR = None
# =======================================================


def _build_settings():
    """Automatic runtime/display defaults; no terminal parameter setup."""
    if not isinstance(USE_HQ_REFERENCE, bool):
        raise TypeError("USE_HQ_REFERENCE must be True or False.")
    if USE_HQ_REFERENCE and (not HQ_VISIBLE_DIR or not HQ_INFRARED_DIR):
        raise ValueError("USE_HQ_REFERENCE=True requires HQ_VISIBLE_DIR and HQ_INFRARED_DIR.")
    return SimpleNamespace(
        checkpoint=CHECKPOINT_PATH, visible_dir=VISIBLE_DIR, infrared_dir=INFRARED_DIR,
        output=OUTPUT_DIR,
        hq_visible_dir=HQ_VISIBLE_DIR if USE_HQ_REFERENCE else None,
        hq_infrared_dir=HQ_INFRARED_DIR if USE_HQ_REFERENCE else None,
        device="auto", amp="none", cpu_threads=4, limit=None,
        save_aux=True, comparisons=True, evaluate=True, progress=True,
    )


@torch.no_grad()
def run(args):
    if args.cpu_threads < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("Thread count and image limit must be positive.")
    torch.set_num_threads(args.cpu_threads)
    device = choose_device(args.device)
    dtype = amp_settings(device,args.amp)
    state = read_checkpoint(args.checkpoint, expected_implementation=MODEL_VERSION)
    model = RobustFusionNet(state["config"]["model"]).to(device)
    strict_load(model,state["model"])
    model.eval()
    dataset = PairedFusionDataset(str(resolve_path(args.visible_dir)),str(resolve_path(args.infrared_dir)),mode="inference",
              hq_visible_dir=str(resolve_path(args.hq_visible_dir)) if args.hq_visible_dir else None,
              hq_infrared_dir=str(resolve_path(args.hq_infrared_dir)) if args.hq_infrared_dir else None)
    has_reference = args.hq_visible_dir is not None
    fusion_criterion = FusionLoss(**state["config"]["fusion_loss"]).to(device)
    auxiliary = AuxiliaryLoss(**state["config"]["auxiliary_loss"]).to(device)
    output = resolve_path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    rows = []
    print(f"Loaded step={state['step']} parameters={sum(p.numel() for p in model.parameters()):,} device={device}")
    total = min(len(dataset),args.limit or len(dataset))
    loader = DataLoader(dataset,batch_size=1,num_workers=0)
    progress = tqdm(islice(loader,total),total=total,desc="Fusing",unit="image",dynamic_ncols=True,disable=not args.progress)
    for batch in progress:
        batch = move_batch(batch,device)
        sample = batch["sample_id"][0]
        need_aux = args.save_aux or args.comparisons or (args.evaluate and has_reference)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()
        start = time.perf_counter()
        with autocast_context(device,dtype):
            result = model(batch["visible"],batch["infrared"],return_aux=need_aux)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter()-start
        fused = result["fused"] if need_aux else result
        save_image(fused,output/"rgb"/(sample+".png"))
        save_image(to_luminance(fused),output/"gray"/(sample+".png"))
        if args.comparisons:
            errors = reference_error_maps(result, batch, auxiliary) if has_reference else None
            save_comparison(batch,result,output/"comparisons"/(sample+".png"),max_samples=1,
                            title=f"Test | checkpoint step {state['step']}", reference_errors=errors)
        if args.save_aux:
            for kind in ("restored","demand","applied_demand","quality"):
                for modality,image in result[kind].items():
                    save_image(image,output/f"{kind}_{modality}"/(sample+".png"))
            # Store actual error values losslessly; visualization is fixed-scale.
            for modality,error in result["error"].items():
                path = output/f"error_{modality}"/(sample+".pt")
                path.parent.mkdir(parents=True,exist_ok=True)
                torch.save(error.detach().float().cpu(),path)
                save_image(error/.2,output/f"error_{modality}_visual"/(sample+".png"))
        row = {"sample_id":sample,"seconds":elapsed}
        if args.evaluate:
            row.update(fusion_statistics(fused))
            if has_reference:
                loss,terms = fusion_criterion(fused,batch["hq_visible"],batch["hq_infrared"],return_components=True)
                row["fusion_loss_hq"] = float(loss)
                row.update({f"fusion_{k}":float(v) for k,v in terms.items()})
                for modality in ("visible","infrared"):
                    restored,reference = result["restored"][modality],batch[f"hq_{modality}"]
                    row.update({f"restored_{modality}_{k}":v for k,v in reconstruction_metrics(restored,reference).items()})
                    target = auxiliary.error_map(restored,reference)
                    row[f"error_{modality}_mae"] = float((result["error"][modality]-target).abs().mean())
        rows.append(row)
        progress.set_postfix(sample=sample,seconds=f"{elapsed:.3f}")
    progress.close()
    if not rows:
        raise ValueError("No images processed.")
    summary = {k:sum(row[k] for row in rows)/len(rows) for k in rows[0] if k != "sample_id"}
    save_json(output/"summary.json",{"checkpoint":str(resolve_path(args.checkpoint)),"step":state["step"],"images":len(rows),
              "has_hq_reference":has_reference,"metrics":summary,"timing":"model wall time; first image included; excludes image IO"})
    if args.evaluate:
        with (output/"metrics.csv").open("w",newline="",encoding="utf-8") as handle:
            writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("  ".join(f"{name}={value:.4f}" for name,value in summary.items() if name != "seconds"))
    print(f"Saved {len(rows)} RGB/gray pairs to {output/'rgb'} and {output/'gray'}")


def main():
    run(_build_settings())


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("Edit the settings at the top of test.py, then run: python test.py")
    main()
