from __future__ import annotations

import argparse

parser = argparse.ArgumentParser(description="Record a video of a trained policy in Isaac Sim.")
parser.add_argument("--checkpoint", required=True, help="Path to agent checkpoint (.pt file)")
parser.add_argument(
    "--env-id",
    default="Isaac-Cartpole-Camera-Showcase-Box-Box-Direct-v0",
    help="Isaac Lab gym environment ID",
)
parser.add_argument("--steps", type=int, default=500, help="Number of simulation steps to record")
parser.add_argument("--output", default="/workspace/recording.mp4", help="Output video file path")
parser.add_argument("--fps", type=int, default=30, help="Output video frame rate")
parser.add_argument("--env-idx", type=int, default=0, help="Which parallel env to record (0-based)")
parser.add_argument("--livestream", type=int, default=0, help="Livestream mode (0=off)")
args = parser.parse_args()

try:
    from isaaclab.app import AppLauncher
except ImportError as exc:
    raise SystemExit("[record.py] Isaac Lab not found. Run inside the isaac-sim container.") from exc

simulation_app = AppLauncher(
    {"headless": True, "enable_cameras": True, "livestream": args.livestream}
).app

import importlib
import gymnasium as gym
import torch
import numpy as np
import isaaclab_tasks  # noqa: F401

from skrl.envs.wrappers.torch import wrap_env
from examples.shared.models import CNNGaussianPolicy, CNNValue
from examples.shared.utils import default_device

device = default_device()

# ── Build environment ────────────────────────────────────────────────────────
spec = gym.spec(args.env_id)
cfg_entry = (spec.kwargs or {}).get("env_cfg_entry_point", "")
if cfg_entry:
    module_path, class_name = cfg_entry.split(":")
    env_cfg = getattr(importlib.import_module(module_path), class_name)()
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs"):
        env_cfg.scene.num_envs = 1  # only need 1 env for recording
    env = wrap_env(gym.make(args.env_id, cfg=env_cfg))
else:
    env = wrap_env(gym.make(args.env_id))

# ── Load policy ──────────────────────────────────────────────────────────────
policy = CNNGaussianPolicy(
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)
ckpt = torch.load(args.checkpoint, map_location=device)
# skrl checkpoints store model weights under 'policy' key
state_dict = ckpt.get("policy", ckpt.get("model", ckpt))
policy.load_state_dict(state_dict, strict=False)
policy.to(device)  # ensure all params are on device
policy.eval()
print(f"[record] Loaded checkpoint: {args.checkpoint}")

# ── Roll out and collect frames ──────────────────────────────────────────────
frames = []
obs, _ = env.reset()
print(f"[record] Observation shape: {obs.shape}  dtype: {obs.dtype}")

for step in range(args.steps):
    with torch.no_grad():
        action, _ = policy.act({"states": obs}, role="policy")
    obs, reward, terminated, truncated, _ = env.step(action)

    # obs is (num_envs, H, W, C) or (num_envs, C*H*W) — extract one env frame
    frame = obs[args.env_idx].cpu().numpy()
    if frame.ndim == 1:
        # flat → try to infer HxWxC from observation space
        shape = env.observation_space.shape  # e.g. (H, W, C)
        frame = frame.reshape(shape)
    # Normalise to uint8
    if frame.dtype != np.uint8:
        lo, hi = frame.min(), frame.max()
        if hi > lo:
            frame = ((frame - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            frame = np.zeros_like(frame, dtype=np.uint8)
    frames.append(frame)

    if (step + 1) % 100 == 0:
        print(f"[record] step {step+1}/{args.steps}")

    if terminated[args.env_idx] or truncated[args.env_idx]:
        obs, _ = env.reset()

env.close()

# ── Write video ──────────────────────────────────────────────────────────────
print(f"[record] Writing {len(frames)} frames to {args.output} …")
try:
    import imageio
    imageio.mimwrite(args.output, frames, fps=args.fps, quality=8)
except ImportError:
    try:
        import cv2
        h, w = frames[0].shape[:2]
        fmt = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(args.output, fmt, args.fps, (w, h))
        for f in frames:
            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR) if f.ndim == 3 else f)
        out.release()
    except ImportError:
        # Last resort: save as individual PNGs
        import os
        png_dir = args.output.replace(".mp4", "_frames")
        os.makedirs(png_dir, exist_ok=True)
        from PIL import Image
        for i, f in enumerate(frames):
            Image.fromarray(f).save(f"{png_dir}/frame_{i:05d}.png")
        print(f"[record] No imageio/cv2. Saved frames to {png_dir}/")
        print(f"[record] Convert with: ffmpeg -r {args.fps} -i {png_dir}/frame_%05d.png {args.output}")

print(f"[record] Done: {args.output}")
simulation_app.close()
