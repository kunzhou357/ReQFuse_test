"""Train, resume and infer on generated images; everything stays under tests/."""
from pathlib import Path
import os
import subprocess
import sys
from datetime import datetime

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_pipeline import create_pairs, experiment_entry

# Test-only settings. Run this file directly, without terminal options.
USE_CUDA = False
USE_TINY_MODEL = False
KEEP_CHECKPOINTS = False


def main():
    output = ROOT/"tests/artifacts"/datetime.now().strftime("smoke_%Y%m%d_%H%M%S_%f")
    vi, ir = create_pairs(output/"data", count=3, height=25, width=29)
    # Deliberately use folder names other than vi/ir.
    visible, infrared = vi.with_name("visible"), ir.with_name("thermal")
    vi.rename(visible)
    ir.rename(infrared)
    device, amp = ("cuda", "bfloat16") if USE_CUDA else ("cpu", "none")
    manual = {"VISIBLE_DIR": visible, "INFRARED_DIR": infrared,
              "OUTPUT_DIR": output/"checkpoints", "TOTAL_STEPS": 3,
              "PATCH_SIZE": 24, "BATCH_SIZE": 1}
    internal = {"device": device, "amp": amp, "workers": 0, "cpu_threads": 1,
                "restoration_warmup_steps": 1, "transition_steps": 1, "lr_warmup_steps": 1,
                "log_every": 1, "save_every": 3, "preview_every": 2, "deterministic": True}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    def command(module, values, overrides, max_steps=None):
        entry = experiment_entry(output, module, values, overrides, USE_TINY_MODEL, max_steps)
        subprocess.run(entry, cwd=ROOT, env=env, check=True)
    command("train", manual, internal, max_steps=1)
    command("train", dict(manual, RESUME_TRAINING=True), internal)
    command("test", {"CHECKPOINT_PATH": output/"checkpoints/last.pth",
            "VISIBLE_DIR": visible, "INFRARED_DIR": infrared, "USE_HQ_REFERENCE": True,
            "HQ_VISIBLE_DIR": visible, "HQ_INFRARED_DIR": infrared, "OUTPUT_DIR": output/"inference"},
            {"device": device, "amp": amp, "cpu_threads": 1})
    if not KEEP_CHECKPOINTS:
        for weight in (output/"checkpoints").glob("*.pth"):
            weight.unlink()
    print(f"Smoke artifacts: {output}")


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("Edit USE_CUDA/USE_TINY_MODEL above, then run: python tests/smoke.py")
    main()
