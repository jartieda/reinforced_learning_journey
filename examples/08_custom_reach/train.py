from __future__ import annotations

# ── Step 1: Parse CLI args BEFORE starting the Isaac Sim runtime ──────────────
import argparse
import sys
import os

parser = argparse.ArgumentParser(
    description="Example 08 — Train a Franka Panda reaching policy from a custom DirectRLEnv."
)
parser.add_argument("--timesteps",       type=int,   default=500_000)
parser.add_argument("--num-envs",        type=int,   default=None,
                    help="Override number of parallel environments (default: cfg value = 4096).")
parser.add_argument("--seed",            type=int,   default=42)
parser.add_argument("--checkpoint",      type=str,   default=None,
                    help="Resume training from a saved checkpoint (.pt file).")
parser.add_argument("--experiment-name", type=str,   default="08_custom_reach")
parser.add_argument("--livestream",      type=int,   default=0, choices=[0, 1, 2],
                    help="0=off, 1=native Omniverse client, 2=WebRTC browser (port 8080).")
args = parser.parse_args()

# ── Step 2: Boot Isaac Lab's Omniverse Kit runtime ────────────────────────────
# All omni.* and isaaclab.* imports must happen AFTER AppLauncher is created.
try:
    from isaaclab.app import AppLauncher
except ImportError as exc:
    raise SystemExit(
        "\n[Example 08] Isaac Lab is not installed.\n"
        "Run inside the Isaac Sim container or a native Isaac Lab install.\n"
        "See: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/\n"
    ) from exc

simulation_app = AppLauncher(
    {"headless": True, "enable_cameras": False, "livestream": args.livestream}
).app

# ── Step 3: Safe to import isaaclab modules now ───────────────────────────────
import torch
import gymnasium as gym
from skrl.envs.wrappers.torch import wrap_env
from skrl.trainers.torch import SequentialTrainer

from examples.shared.models import MLPGaussianPolicy, MLPValue
from examples.shared.ppo import build_ppo_agent
from examples.shared.utils import default_device, print_run_banner, set_seed

# ── Step 4: Import our custom environment (also uses isaaclab.*) ──────────────
# Add this example's directory to sys.path so reach_env.py can import env_cfg.py
# using a plain "from env_cfg import ..." statement.
sys.path.insert(0, os.path.dirname(__file__))
from env_cfg import FrankaReachEnvCfg   # noqa: E402
from reach_env import FrankaReachEnv    # noqa: E402


def main() -> None:
    set_seed(args.seed)
    device = default_device()

    # ── Build environment ─────────────────────────────────────────────────────
    cfg = FrankaReachEnvCfg()

    # Allow overriding num_envs from the command line without editing the cfg.
    # With 4096 envs the training is fast but needs ~12 GB GPU RAM.
    # Try 512–1024 on a 12 GB card.
    if args.num_envs is not None:
        cfg.scene.num_envs = args.num_envs

    # FrankaReachEnv is a DirectRLEnv subclass — instantiate directly, no
    # gym.make() needed since this env is not registered in the gym registry.
    raw_env = FrankaReachEnv(cfg=cfg)
    env = wrap_env(raw_env)

    # ── Models: MLP (not CNN) because observations are proprioceptive ──────────
    # obs = 20-dim vector  →  MLP is the right architecture here.
    # The PPO agent, memory, and trainer code is IDENTICAL to earlier examples.
    # Only the model class and the environment changed — that is the lesson.
    models = {
        "policy": MLPGaussianPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        ),
        "value": MLPValue(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        ),
    }

    # ── PPO agent ─────────────────────────────────────────────────────────────
    # Larger mini_batches and more learning_epochs suit the high num_envs
    # (more data per rollout → we can afford more gradient steps).
    agent = build_ppo_agent(
        env=env,
        device=device,
        models=models,
        experiment_directory="runs/08_custom_reach",
        experiment_name=args.experiment_name,
        overrides={
            "learning_rate": 3e-4,
            "mini_batches": 8,
            "learning_epochs": 8,
            "entropy_loss_scale": 0.01,  # a little exploration helps early on
        },
    )

    if args.checkpoint:
        agent.load(args.checkpoint)

    print_run_banner(
        "Example 08 — Custom Franka Reach (DirectRLEnv)",
        env_id="FrankaReachEnv",
        timesteps=args.timesteps,
        device=device,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = SequentialTrainer(
        cfg={"timesteps": args.timesteps, "headless": True},
        env=env,
        agents=agent,
    )
    trainer.train()

    env.close()
    simulation_app.close()  # always required — shuts down the Kit runtime


if __name__ == "__main__":
    main()
