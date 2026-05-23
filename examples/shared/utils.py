from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed all relevant RNG sources for repeatable experiments.

    In RL, exact determinism can still vary by simulator/hardware, but seeding
    Python, NumPy, and Torch significantly reduces run-to-run drift.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_device() -> str:
    """Pick CUDA when available, else CPU.

    This keeps scripts portable: the same command works on laptops and GPUs.
    """

    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Create a directory tree if needed and return its path object."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def print_run_banner(title: str, env_id: str, timesteps: int, device: str) -> None:
    """Print run metadata so logs clearly capture experiment context."""

    print("=" * 72)
    print(title)
    print(f"Environment: {env_id}")
    print(f"Timesteps:   {timesteps}")
    print(f"Device:      {device}")
    print("=" * 72)


def print_next_steps(run_dir: str, env_id: str, eval_module: str, model: str = "mlp") -> None:
    """Print a cheat-sheet of what to do after training finishes."""

    print()
    print("=" * 72)
    print("Training complete! What to do next:")
    print()
    print("  1. View training curves (TensorBoard live view):")
    print(f"       tensorboard --logdir {run_dir}")
    print(f"       # then open http://localhost:6006")
    print()
    print("  2. Generate a static PNG summary:")
    print(f"       python -m examples.shared.plot {run_dir}")
    print()
    print("  3. Watch the trained policy play:")
    chk = f"{run_dir}/checkpoints/<agent_N>.pt"
    print(f"       python -m {eval_module} --checkpoint {chk} --render")
    print()
    print("  4. Record a video + export weights for standalone deployment:")
    print(f"       python -m examples.shared.export \\")
    print(f"           --checkpoint {chk} \\")
    print(f"           --env-id {env_id} --model {model} \\")
    print(f"           --record --video-dir videos/")
    print("=" * 72)
