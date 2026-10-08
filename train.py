"""Train the complete Ours model: edit the small block below and run train.py.

All supplied training pairs are used. No external configuration file,
validation split or command-line experiment options are required.
"""
from copy import deepcopy
import sys
import time

import torch
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm

if __package__:
    from .model.fusion import DEFAULT_MODEL_CONFIG, MODEL_VERSION, RobustFusionNet
    from .utils.auxiliary_loss import AuxiliaryLoss
    from .utils.dataset import PairedFusionDataset, ResumableBatchSampler
    from .utils.degradation import DEFAULT_DEGRADATION
    from .utils.diagnostics import collect_diagnostics, head_gradient_norms, reference_error_maps
    from .utils.experiment import (amp_settings, append_json, atomic_checkpoint, autocast_context,
        capture_rng, choose_device, dataset_fingerprint, dataset_options, joint_strength,
        learning_rate, move_batch, read_checkpoint, resolve_path, restore_rng,
        save_json, seed_everything, strict_load, training_signature)
    from .utils.loss import FusionLoss
    from .utils.visualization import save_comparison, save_loss_curves, save_state_curves
else:
    from model.fusion import DEFAULT_MODEL_CONFIG, MODEL_VERSION, RobustFusionNet
    from utils.auxiliary_loss import AuxiliaryLoss
    from utils.dataset import PairedFusionDataset, ResumableBatchSampler
    from utils.degradation import DEFAULT_DEGRADATION
    from utils.diagnostics import collect_diagnostics, head_gradient_norms, reference_error_maps
    from utils.experiment import (amp_settings, append_json, atomic_checkpoint, autocast_context,
        capture_rng, choose_device, dataset_fingerprint, dataset_options, joint_strength,
        learning_rate, move_batch, read_checkpoint, resolve_path, restore_rng,
        save_json, seed_everything, strict_load, training_signature)
    from utils.loss import FusionLoss
    from utils.visualization import save_comparison, save_loss_curves, save_state_curves


# ===================== 只需修改这里 =====================
# 直接填写图片所在目录；目录名可以是 vi、vis、visible 等任意名称。
VISIBLE_DIR = "data/train/vi"
INFRARED_DIR = "data/train/ir"
OUTPUT_DIR = "outputs/restormer_v2"

BATCH_SIZE = 4
PATCH_SIZE = 128
TOTAL_STEPS = 30000                       # optimizer 更新次数
LEARNING_RATE = 2e-4
RESUME_TRAINING = False                   # True：从 OUTPUT_DIR/last.pth 续训

ONLINE_DEGRADATION = True                 # True：从高质量源对在线生成退化
HQ_VISIBLE_DIR = None                     # 仅 False 时填写对应的高质量参考目录
HQ_INFRARED_DIR = None
# =======================================================


def _build_settings():
    """Internal defaults for the complete method; not a user options catalog."""
    if not isinstance(ONLINE_DEGRADATION, bool) or not isinstance(RESUME_TRAINING, bool):
        raise TypeError("ONLINE_DEGRADATION and RESUME_TRAINING must be True or False.")
    if not ONLINE_DEGRADATION and (not HQ_VISIBLE_DIR or not HQ_INFRARED_DIR):
        raise ValueError("ONLINE_DEGRADATION=False requires HQ_VISIBLE_DIR and HQ_INFRARED_DIR.")
    device = choose_device()
    precision = ("bfloat16" if torch.cuda.is_bf16_supported() else "float16") if device.type == "cuda" else "none"
    return {
        "implementation": {"version": MODEL_VERSION, "demand_curriculum": "joint_strength_detached",
                           "demand_supervision": "logits_bce_region_balanced",
                           "error_supervision": "linear_scaled_smooth_l1_region_balanced"},
        "model": deepcopy(DEFAULT_MODEL_CONFIG),
        "data": {"train": {
            "mode": "synthetic" if ONLINE_DEGRADATION else "paired",
            "visible_dir": VISIBLE_DIR, "infrared_dir": INFRARED_DIR,
            "hq_visible_dir": None if ONLINE_DEGRADATION else HQ_VISIBLE_DIR,
            "hq_infrared_dir": None if ONLINE_DEGRADATION else HQ_INFRARED_DIR,
            "mask_visible_dir": None, "mask_infrared_dir": None, "patch_size": PATCH_SIZE,
        }},
        "training": {
            "seed": 100, "deterministic": False, "device": str(device), "amp": precision,
            "cpu_threads": 4, "workers": 4 if device.type == "cuda" else 0,
            "batch_size": BATCH_SIZE, "accumulation_steps": 2, "total_steps": TOTAL_STEPS,
            "restoration_warmup_steps": min(5000, TOTAL_STEPS//14),
            "transition_steps": min(5000, TOTAL_STEPS//14),
            "lr": LEARNING_RATE, "min_lr": min(1e-6, LEARNING_RATE),
            "lr_warmup_steps": min(1000, TOTAL_STEPS//70),
            "weight_decay": 1e-4, "betas": [0.9, 0.999], "grad_clip": 1.0,
            "log_every": 100, "save_every": 2000, "progress": True,
            "save_previews": True, "preview_every": 500,
            "preview_images": 2, "preview_crop_size": 128, "save_loss_plots": True,
        },
        "degradation": deepcopy(DEFAULT_DEGRADATION),
        "fusion_loss": {"intensity_weight": 10.0, "gradient_weight": 2.0,
                        "ssim_weight": 1.0, "color_weight": 1.0, "require_color": True},
        "auxiliary_loss": {"reconstruction_weight": 5.0, "keep_weight": 2.0,
                           "demand_weight": 1.0, "error_weight": 1.0, "demand_scale": 0.1,
                           "average_kernel": 5, "gradient_weight": 0.25, "sobel_divisor": 8.0,
                           "error_scale": 0.1},
        "experiment": {"output_dir": OUTPUT_DIR, "resume_training": RESUME_TRAINING},
    }


def check_training(settings):
    positive = ("batch_size", "accumulation_steps", "total_steps", "cpu_threads", "log_every",
                "save_every", "preview_every", "preview_images", "preview_crop_size")
    if any(not isinstance(settings[k], int) or settings[k] < 1 for k in positive):
        raise ValueError(f"Positive integer training options required: {positive}")
    if settings["workers"] < 0 or min(settings["restoration_warmup_steps"], settings["transition_steps"], settings["lr_warmup_steps"]) < 0:
        raise ValueError("Worker and warmup counts cannot be negative.")
    if settings["lr"] <= 0 or not 0 <= settings["min_lr"] <= settings["lr"] or settings["grad_clip"] <= 0:
        raise ValueError("Invalid learning rate or gradient clipping.")
    if settings["lr_warmup_steps"] > settings["total_steps"]:
        raise ValueError("LR warmup cannot exceed total_steps.")


@torch.no_grad()
def render_training_preview(model, batch, device, dtype, output, step, strength,
                            restoration_only, auxiliary_criterion, save_preview=True):
    rng, was_training = capture_rng(), model.training
    try:
        model.eval()
        batch = move_batch(batch, device)
        with autocast_context(device, dtype):
            outputs = model(batch["visible"], batch["infrared"], quality_strength=strength,
                            demand_strength=strength, restoration_only=restoration_only)
        errors = reference_error_maps(outputs, batch, auxiliary_criterion)
        diagnostics = collect_diagnostics(outputs, batch, auxiliary_criterion, errors)
        if save_preview:
            save_comparison(batch, outputs, output/"previews"/f"step_{step:06d}.png",
                            max_samples=batch["visible"].shape[0],
                            title=f"Train preview | step {step} | demand strength {strength:.3f}",
                            reference_errors=errors)
        return diagnostics
    finally:
        model.train(was_training)
        restore_rng(rng)


def run(config, resume=None, max_steps=None):
    settings = config["training"]
    check_training(settings)
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive.")
    target = min(settings["total_steps"], max_steps or settings["total_steps"])
    torch.set_num_threads(settings["cpu_threads"])
    seed_everything(settings["seed"], settings["deterministic"])
    device = choose_device(settings["device"])
    dtype = amp_settings(device, settings["amp"])
    dataset = PairedFusionDataset(**dataset_options(config["data"]["train"], settings["seed"], True, config["degradation"]))
    if dataset.mode == "inference":
        raise ValueError("Training requires synthetic or paired reference mode.")
    fingerprint = dataset_fingerprint(dataset)
    model = RobustFusionNet(config["model"]).to(device)
    config["model"] = model.config
    if config.get("implementation", {}).get("version") != MODEL_VERSION:
        raise ValueError("Training settings must describe the current implementation.")
    signature = training_signature(config)
    fusion_criterion = FusionLoss(**config["fusion_loss"]).to(device)
    auxiliary_criterion = AuxiliaryLoss(**config["auxiliary_loss"]).to(device)
    if auxiliary_criterion.error_scale != model.config["error_scale"]:
        raise ValueError("Model and auxiliary error regression scales must match.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"], betas=tuple(settings["betas"]))
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    completed = 0
    output = resolve_path(config["experiment"]["output_dir"])
    if resume is None and config["experiment"].get("resume_training", False):
        resume = output/"last.pth"
    if (output/"last.pth").exists() and not resume:
        raise FileExistsError(f"{output}/last.pth exists; set RESUME_TRAINING=True or select a fresh OUTPUT_DIR.")
    if resume:
        state = read_checkpoint(resume, expected_implementation=MODEL_VERSION)
        if state["training_signature"] != signature or state["dataset_fingerprint"] != fingerprint:
            raise ValueError("Resume parameters/data differ from the checkpoint. Keep the original training values and data.")
        strict_load(model, state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        completed = state["step"]
        if state["next_micro_batch"] != completed*settings["accumulation_steps"]:
            raise ValueError("Checkpoint sampler cursor is inconsistent.")
        restore_rng(state["rng"])
    if completed >= target:
        print(f"Checkpoint already reached {completed} updates; requested stop={target}.")
        return
    output.mkdir(parents=True, exist_ok=True)
    # A record of this run, never an input configuration file. Inference reads
    # the network specification from the checkpoint itself.
    save_json(output/"run_info.json", config)
    save_json(output/"data_manifest.json", {"fingerprint": fingerprint, "sample_ids": dataset.sample_ids,
              "paths": {k: {s: str(p) for s, p in v.items()} for k, v in dataset.indices.items()}})
    sampler = ResumableBatchSampler(len(dataset), settings["batch_size"], target*settings["accumulation_steps"],
                                    completed*settings["accumulation_steps"], settings["seed"])
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=settings["workers"], pin_memory=device.type == "cuda",
                        generator=torch.Generator().manual_seed(settings["seed"]+999))
    iterator = iter(loader)
    preview_batch = None
    if settings["save_previews"]:
        options = dict(config["data"]["train"])
        options["patch_size"] = min(options["patch_size"], settings["preview_crop_size"])
        preview_dataset = PairedFusionDataset(**dataset_options(options, settings["seed"]+50000, False, config["degradation"]))
        preview_batch = default_collate([preview_dataset[i] for i in range(min(len(preview_dataset), settings["preview_images"]))])
    parameters = sum(p.numel() for p in model.parameters())
    print(f"device={device} parameters={parameters:,} pairs={len(dataset)} updates={completed}->{target} amp={dtype}")
    model.train()
    start, first_step = time.perf_counter(), completed
    progress = tqdm(total=target, initial=completed, desc="Train", unit="step", dynamic_ncols=True, disable=not settings["progress"])

    def checkpoint_state():
        return {"format_version": 2, "implementation_version": MODEL_VERSION,
                "model": model.state_dict(), "config": config, "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "scheduler": {"step": completed, "total_steps": settings["total_steps"], "kind": "warmup_cosine"},
                "step": completed, "next_micro_batch": completed*settings["accumulation_steps"], "rng": capture_rng(),
                "training_signature": signature, "dataset_fingerprint": fingerprint,
                "parameters": parameters, "torch_version": str(torch.__version__)}

    try:
        while completed < target:
            optimizer.zero_grad(set_to_none=True)
            lr = learning_rate(completed, settings)
            for group in optimizer.param_groups:
                group["lr"] = lr
            strength = joint_strength(completed, settings)
            restoration_only = completed < settings["restoration_warmup_steps"]
            next_step = completed+1
            log_now = next_step == first_step+1 or next_step % settings["log_every"] == 0 or next_step == target
            preview_now = next_step % settings["preview_every"] == 0 or next_step == target
            accumulated = {}
            for _ in range(settings["accumulation_steps"]):
                batch = move_batch(next(iterator), device)
                with autocast_context(device, dtype):
                    outputs = model(batch["visible"], batch["infrared"], quality_strength=strength,
                                    demand_strength=strength, restoration_only=restoration_only)
                    auxiliary, aux_terms = auxiliary_criterion(outputs, batch)
                    if restoration_only:
                        fusion = auxiliary.new_zeros(())
                        fusion_terms = {name: fusion for name in ("intensity", "gradient", "ssim", "color")}
                    else:
                        fusion, fusion_terms = fusion_criterion(outputs["fused"], batch["hq_visible"], batch["hq_infrared"], return_components=True)
                    total = auxiliary+strength*fusion
                if not torch.isfinite(total):
                    raise FloatingPointError(f"Non-finite loss at update {completed+1}.")
                scaler.scale(total/settings["accumulation_steps"]).backward()
                values = {"loss": total.detach(), "fusion_loss": fusion.detach(), "auxiliary_loss": auxiliary.detach()}
                values.update({f"fusion_{k}": v.detach() for k, v in fusion_terms.items()})
                values.update({f"aux_{k}": v.detach() for k, v in aux_terms.items()})
                for key, value in values.items():
                    accumulated[key] = accumulated.get(key, 0.0)+float(value)/settings["accumulation_steps"]
            scaler.unscale_(optimizer)
            gradients = head_gradient_norms(model) if log_now else {}
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"], error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            completed += 1
            phase = "restoration" if restoration_only else ("transition" if strength < 1 else "joint")
            progress.update(1)
            progress.set_postfix(loss=f"{accumulated['loss']:.4f}", fusion=f"{accumulated['fusion_loss']:.4f}", phase=phase)
            diagnostics = {}
            if preview_batch is not None and (log_now or preview_now):
                diagnostics = render_training_preview(model, preview_batch, device, dtype, output, completed,
                                                       strength, restoration_only, auxiliary_criterion,
                                                       save_preview=preview_now or completed == first_step+1)
            elif log_now:
                # Programmatic test runs may disable previews. Use the last
                # training micro-batch, explicitly labeled as such in the log.
                diagnostics = collect_diagnostics(outputs, batch, auxiliary_criterion)
            if log_now:
                record = dict(accumulated, step=completed, lr=lr, quality_strength=strength,
                              demand_strength=strength, grad_norm=float(norm),
                              seconds_per_update=(time.perf_counter()-start)/(completed-first_step), phase=phase)
                record.update(gradients)
                record.update(diagnostics)
                record["diagnostic_source"] = "fixed_training_preview" if preview_batch is not None else "last_training_micro_batch"
                if device.type == "cuda":
                    record["peak_memory_mb"] = torch.cuda.max_memory_allocated(device)/1024**2
                append_json(output/"train.jsonl", record)
                line = (f"step={completed} phase={phase} loss={record['loss']:.5f} fusion={record['fusion_loss']:.5f} "
                        f"aux={record['auxiliary_loss']:.5f} lr={lr:.3g} joint_strength={strength:.3f} "
                        f"VI_MAE={record['vi_input_mae']:.4f}->{record['vi_restored_mae']:.4f} "
                        f"IR_MAE={record['ir_input_mae']:.4f}->{record['ir_restored_mae']:.4f} "
                        f"VI_demand={record['vi_demand_mean']:.3f} VI_q={record['vi_quality_mean']:.3f} "
                        f"VI_sat={record['vi_residual_saturated_fraction']:.3%}")
                tqdm.write(line)
                with (output/"log.txt").open("a", encoding="utf-8") as handle:
                    handle.write(line+"\n")
            if preview_now:
                if settings["save_loss_plots"]:
                    save_loss_curves(output/"train.jsonl", output/"loss_curves.png")
                    save_state_curves(output/"train.jsonl", output/"state_curves.png")
            if completed % settings["save_every"] == 0 or completed == target:
                state = checkpoint_state()
                atomic_checkpoint(state, output/"last.pth")
                if completed % settings["save_every"] == 0:
                    atomic_checkpoint(state, output/f"step_{completed:06d}.pth")
    finally:
        progress.close()
    print(f"Saved checkpoint: {output/'last.pth'}")


def main():
    run(_build_settings())


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("Edit the settings at the top of train.py, then run: python train.py")
    main()
