"""train.py — PPO training with NOMAD encoder + gs_navsim simulator.

Usage
-----
    python -m examples.10_gs_navsim_nomad.train \\
        --goal path/to/goal.png \\
        --mask path/to/mask.json \\
        --timesteps 200000

Prerequisites
-------------
    1. gs_navsim server running:  cd gs_navsim/backend && node server.js
    2. gs_navsim open in browser: http://localhost:3000
    3. A PLY scene loaded in the browser.
    4. websocket-client installed:  pip install websocket-client
"""
from __future__ import annotations

import argparse
import os
import sys

# Parse CLI args first (no heavy imports yet)
parser = argparse.ArgumentParser(
    description="Example 10 — gs_navsim + NOMAD PPO training"
)
parser.add_argument("--goal",            type=str,   required=True,
                    help="Path to goal image (PNG/JPG).")
parser.add_argument("--mask",            type=str,   default=None,
                    help="Path to obstacle mask JSON (from mask_editor).")
parser.add_argument("--timesteps",       type=int,   default=200_000)
parser.add_argument("--max-steps",       type=int,   default=500,
                    help="Max steps per episode.")
parser.add_argument("--seed",            type=int,   default=42)
parser.add_argument("--checkpoint",      type=str,   default=None,
                    help="Resume from a saved checkpoint (.pt).")
parser.add_argument("--experiment-name", type=str,   default="10_gs_navsim_nomad")
parser.add_argument("--ws-url",          type=str,   default="ws://localhost:8081")
args = parser.parse_args()

import torch
from skrl.envs.wrappers.torch import wrap_env
from skrl.trainers.torch import SequentialTrainer

# Add this example dir to sys.path so sibling imports work
sys.path.insert(0, os.path.dirname(__file__))
from env   import GsNavSimEnv                               # noqa: E402
from model import NomadPolicy, NomadValue, NOMAD_WEIGHTS    # noqa: E402

from examples.shared.ppo   import build_ppo_agent           # noqa: E402
from examples.shared.utils import (                         # noqa: E402
    default_device, print_run_banner, set_seed,
)


def main() -> None:
    set_seed(args.seed)
    device = default_device()

    # ── Environment ────────────────────────────────────────────────────────────
    raw_env = GsNavSimEnv(
        goal_image_path=args.goal,
        mask_path=args.mask,
        ws_url=args.ws_url,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    env = wrap_env(raw_env)

    # ── Models ─────────────────────────────────────────────────────────────────
    common = dict(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        weights_path=NOMAD_WEIGHTS,
    )
    models = {
        "policy": NomadPolicy(**common),
        "value":  NomadValue(**common),
    }

    # ── PPO agent ──────────────────────────────────────────────────────────────
    # Single-env → no RunningStandardScaler on observations (images already normalised).
    agent = build_ppo_agent(
        env=env,
        device=device,
        models=models,
        experiment_directory="runs/10_gs_navsim_nomad",
        experiment_name=args.experiment_name,
        overrides={
            "rollouts":            256,
            "learning_rate":       1e-4,
            "entropy_loss_scale":  0.01,
            "mini_batches":        1,
            # Disable obs scaler — images are already ImageNet-normalised
            "state_preprocessor":       None,
            "state_preprocessor_kwargs": {},
        },
    )

    if args.checkpoint:
        agent.load(args.checkpoint)

    print_run_banner(
        "Example 10 — gs_navsim + NOMAD PPO",
        env_id="GsNavSimEnv",
        timesteps=args.timesteps,
        device=device,
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    trainer = SequentialTrainer(
        cfg={"timesteps": args.timesteps, "headless": True},
        env=env,
        agents=agent,
    )
    trainer.train()

    env.close()
    print("\nTraining complete.")
    print(f"Checkpoints saved to: runs/10_gs_navsim_nomad/")
    print(f"\nTo run the trained policy:")
    print(f"  python -m examples.10_gs_navsim_nomad.run \\")
    print(f"      --checkpoint runs/10_gs_navsim_nomad/checkpoints/<agent_N>.pt \\")
    print(f"      --goal {args.goal}")
    if args.mask:
        print(f"      --mask {args.mask}")


if __name__ == "__main__":
    main()
