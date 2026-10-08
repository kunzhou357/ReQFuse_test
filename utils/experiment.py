"""Portable paths, deterministic training state, checkpoints and logs."""
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def dataset_options(options, seed, training, degradation):
    options = dict(options)
    for key in list(options):
        if key.endswith("_dir") and options[key] is not None:
            options[key] = str(resolve_path(options[key]))
    # The offline generator uses vi/ir with sibling mask_vi/mask_ir folders.
    # Discover those annotations without adding more daily manual settings.
    if options.get("mode") == "paired" and not options.get("mask_visible_dir") and not options.get("mask_infrared_dir"):
        masks = {"mask_visible_dir": Path(options["visible_dir"]).parent/"mask_vi",
                 "mask_infrared_dir": Path(options["infrared_dir"]).parent/"mask_ir"}
        if any(path.exists() for path in masks.values()):
            if not all(path.is_dir() for path in masks.values()):
                raise ValueError("Offline masks require both mask_vi and mask_ir directories.")
            options.update({key:str(path) for key,path in masks.items()})
    return dict(options,seed=seed,training=training,degradation=degradation)


def dataset_fingerprint(dataset):
    rows = []
    for name,index in sorted(dataset.indices.items()):
        for sample_id,path in sorted(index.items()):
            stat = path.stat()
            rows.append((name,sample_id,stat.st_size,stat.st_mtime_ns))
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def training_signature(config):
    t = config["training"]
    keys = ("seed","deterministic","batch_size","accumulation_steps","total_steps",
            "restoration_warmup_steps","transition_steps","lr","min_lr","lr_warmup_steps",
            "weight_decay","betas","grad_clip","amp")
    dataset = {k:v for k,v in config["data"]["train"].items() if not k.endswith("_dir")}
    payload = {"implementation":config["implementation"],
               "model":config["model"],"training":{k:t[k] for k in keys},"dataset":dataset,
               "degradation":config["degradation"],"fusion_loss":config["fusion_loss"],"auxiliary_loss":config["auxiliary_loss"]}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def seed_everything(seed, deterministic=False):
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic


def choose_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable.")
    if device.type not in ("cpu","cuda","mps"):
        raise ValueError("Supported devices are cpu, cuda and mps.")
    return device


def amp_settings(device, name):
    name = {"bf16":"bfloat16", "fp16":"float16"}.get(name, name)
    if name not in ("none","bfloat16","float16"):
        raise ValueError("amp must be none, bfloat16 or float16.")
    if device.type != "cuda" or name == "none":
        return None
    if name == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16; set training.amp=float16 or none.")
    return torch.bfloat16 if name == "bfloat16" else torch.float16


def autocast_context(device, dtype):
    return torch.autocast(device.type,dtype=dtype) if dtype is not None else nullcontext()


def move_batch(batch, device):
    return {k:v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}


def learning_rate(step, settings):
    total,warm = settings["total_steps"],settings["lr_warmup_steps"]
    if warm and step < warm:
        return settings["lr"]*(step+1)/warm
    progress = min(1,max(0,(step-warm)/max(1,total-warm-1)))
    return settings["min_lr"]+(settings["lr"]-settings["min_lr"])*.5*(1+math.cos(math.pi*progress))


def joint_strength(step, settings):
    warm,transition = settings["restoration_warmup_steps"],settings["transition_steps"]
    if step < warm:
        return 0.0
    return min(1.0,(step-warm+1)/max(1,transition))


def capture_rng():
    n = np.random.get_state()
    state = {"python":random.getstate(),"numpy":(n[0],n[1].tolist(),n[2],n[3],n[4]),"torch":torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())


def atomic_checkpoint(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    try:
        torch.save(state,temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_checkpoint(path, expected_implementation=None):
    state = torch.load(resolve_path(path),map_location="cpu",weights_only=True)
    if not isinstance(state,dict) or state.get("format_version") != 2 or "model" not in state or "config" not in state:
        raise ValueError("Expected an Ours v2 checkpoint. Use the backed-up v1 project for old weights; v2 requires fresh training.")
    version = state.get("implementation_version")
    if not version or state["config"].get("implementation", {}).get("version") != version:
        raise ValueError("Checkpoint implementation metadata is missing or inconsistent.")
    if expected_implementation is not None and version != expected_implementation:
        raise ValueError(f"Incompatible implementation: {version}; expected {expected_implementation}.")
    return state


def strict_load(model, state):
    # Explicitly fail on incompatible keys; never silently accept an empty load.
    model.load_state_dict(state,strict=True)


def append_json(path, record):
    with Path(path).open("a",encoding="utf-8") as handle:
        handle.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n")


def save_json(path, data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
