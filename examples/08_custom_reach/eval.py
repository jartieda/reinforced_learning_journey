from __future__ import annotations

# ── Step 1: CLI before AppLauncher ────────────────────────────────────────────
import argparse
import sys
import os

parser = argparse.ArgumentParser(
    description="Example 08 — Evaluate a trained Franka reaching policy."
)
parser.add_argument("--checkpoint", required=True,
                    help="Path to the saved checkpoint (.pt file).")
parser.add_argument("--episodes",   type=int, default=10,
                    help="Number of evaluation episodes to run.")
parser.add_argument("--num-envs",   type=int, default=1,
                    help="Number of parallel environments (1 for clean logging).")
parser.add_argument("--seed",       type=int, default=0)
parser.add_argument("--livestream", type=int, default=0, choices=[0, 1, 2],
                    help="0=off, 1=native Omniverse client, 2=WebRTC browser (port 8080).")
args = parser.parse_args()

# ── Step 2: Boot AppLauncher ──────────────────────────────────────────────────
try:
    from isaaclab.app import AppLauncher
except ImportError as exc:
    raise SystemExit("[Example 08 eval] Isaac Lab not found.") from exc

simulation_app = AppLauncher(
    {"headless": True, "enable_cameras": False, "livestream": args.livestream}
).app

# ── Step 3: Safe imports ──────────────────────────────────────────────────────
import torch
from skrl.envs.wrappers.torch import wrap_env

from examples.shared.models import MLPGaussianPolicy, MLPValue
from examples.shared.utils import default_device, set_seed

sys.path.insert(0, os.path.dirname(__file__))
from env_cfg import FrankaReachEnvCfg
from reach_env import FrankaReachEnv


def main() -> None:
    set_seed(args.seed)
    device = default_device()

    # ── Environment ───────────────────────────────────────────────────────────
    cfg = FrankaReachEnvCfg()
    cfg.scene.num_envs = args.num_envs

    raw_env = FrankaReachEnv(cfg=cfg)
    env = wrap_env(raw_env)

    # ── Load policy ───────────────────────────────────────────────────────────
    policy = MLPGaussianPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get("policy", ckpt.get("model", ckpt))
    policy.load_state_dict(state_dict, strict=False)
    policy.set_mode("eval")  # deterministic: use mean action, no exploration noise
    print(f"[eval] Loaded checkpoint: {args.checkpoint}")
    print(f"[eval] Running {args.episodes} episodes with {args.num_envs} env(s).")

    # ── Rollout loop ──────────────────────────────────────────────────────────
    episode_returns = []
    episode_successes = []
    obs, _ = env.reset()

    current_returns = torch.zeros(args.num_envs, device=device)
    episodes_done = 0

    while episodes_done < args.episodes:
        with torch.no_grad():
            action, _ = policy.act({"observations": obs}, role="policy")

        obs, reward, terminated, truncated, info = env.step(action)
        current_returns += reward.squeeze(-1)

        done = terminated | truncated
        for i in done.nonzero(as_tuple=False).squeeze(-1):
            episode_returns.append(current_returns[i].item())
            # Count success if at least one timestep had a success-level reward.
            # (A cleaner approach would track successes inside the env.)
            episode_successes.append(float(current_returns[i].item() > 0))
            current_returns[i] = 0.0
            episodes_done += 1
            if episodes_done >= args.episodes:
                break

    # ── Summary ───────────────────────────────────────────────────────────────
    returns = torch.tensor(episode_returns[:args.episodes])
    print("\n── Evaluation Summary ─────────────────────────────────────────")
    print(f"  Episodes:       {args.episodes}")
    print(f"  Mean return:    {returns.mean():.2f}")
    print(f"  Std  return:    {returns.std():.2f}")
    print(f"  Min / Max:      {returns.min():.2f} / {returns.max():.2f}")
    print("───────────────────────────────────────────────────────────────")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
