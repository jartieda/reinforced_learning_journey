"""run.py — Execute a trained PPO policy in gs_navsim (no training).

Usage
-----
    python -m examples.10_gs_navsim_nomad.run \\
        --checkpoint runs/10_gs_navsim_nomad/checkpoints/agent_200000.pt \\
        --goal path/to/goal.png \\
        --mask path/to/mask.json \\
        --episodes 5

The script loads the saved policy weights, runs N episodes deterministically,
and prints per-episode statistics.
"""
from __future__ import annotations

import argparse
import os
import sys

parser = argparse.ArgumentParser(
    description="Example 10 — run trained gs_navsim + NOMAD policy"
)
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to a saved agent checkpoint (.pt).")
parser.add_argument("--goal",       type=str, required=True,
                    help="Path to goal image (PNG/JPG).")
parser.add_argument("--mask",       type=str, default=None,
                    help="Path to obstacle mask JSON (from mask_editor).")
parser.add_argument("--episodes",   type=int, default=5)
parser.add_argument("--max-steps",  type=int, default=500)
parser.add_argument("--seed",       type=int, default=0)
parser.add_argument("--ws-url",     type=str, default="ws://localhost:8081")
args = parser.parse_args()

import torch
from skrl.envs.wrappers.torch import wrap_env

sys.path.insert(0, os.path.dirname(__file__))
from env   import GsNavSimEnv                               # noqa: E402
from model import NomadPolicy, NomadValue, NOMAD_WEIGHTS    # noqa: E402

from examples.shared.utils import default_device            # noqa: E402


def main() -> None:
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

    # ── Policy ─────────────────────────────────────────────────────────────────
    policy = NomadPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        weights_path=NOMAD_WEIGHTS,
    )
    policy.to(device)

    # Load saved weights — skrl saves {"policy": state_dict, "value": state_dict}
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "policy" in ckpt:
        policy.load_state_dict(ckpt["policy"])
    else:
        policy.load_state_dict(ckpt)
    policy.eval()

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Running {args.episodes} episode(s) — deterministic policy\n")

    # ── Evaluation loop ────────────────────────────────────────────────────────
    for ep in range(args.episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        steps        = 0
        success      = False
        collision    = False

        done = False
        while not done:
            # Build input dict expected by skrl models
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = policy.act({"observations": obs_tensor}, role="policy")
            action_np = action.squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action_np)
            total_reward += reward
            steps        += 1
            success    = info.get("success",   False)
            collision  = info.get("collision", False)
            done = terminated or truncated

        status = "SUCCESS" if success else ("COLLISION" if collision else "TIMEOUT")
        pixel_d = info.get("pixel_dist", float("nan"))
        print(
            f"Episode {ep+1:3d}  |  steps={steps:4d}  "
            f"reward={total_reward:+8.2f}  "
            f"pixel_dist={pixel_d:.4f}  [{status}]"
        )

    env.close()


if __name__ == "__main__":
    main()
