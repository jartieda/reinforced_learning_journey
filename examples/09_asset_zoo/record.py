"""
record.py — Record any Isaac Lab environment using a dedicated high-quality camera.

Frames are captured from an omni.replicator render product that is completely
separate from the environment's policy observation cameras.  The environment
does NOT need to expose image observations.
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import torch

parser = argparse.ArgumentParser(
    description="Example 09 — Record a high-quality video of any Isaac Lab simulation."
)
parser.add_argument(
    "--env-id",
    default="Isaac-Cartpole-v0",
    help="Any Isaac Lab gym environment ID (does not need camera observations).",
)
parser.add_argument(
    "--checkpoint",
    default=None,
    help="Optional policy checkpoint (.pt). If omitted, zero actions are used.",
)
parser.add_argument("--steps", type=int, default=500, help="Number of simulation steps.")
parser.add_argument("--fps", type=int, default=30, help="Output video FPS.")
parser.add_argument("--width", type=int, default=1280, help="Recording width in pixels.")
parser.add_argument("--height", type=int, default=720, help="Recording height in pixels.")
parser.add_argument(
    "--cam-pos",
    type=float,
    nargs=3,
    default=[4.0, 4.0, 3.0],
    metavar=("X", "Y", "Z"),
    help="Recording camera world position (default: 4 4 3).",
)
parser.add_argument(
    "--cam-target",
    type=float,
    nargs=3,
    default=[0.0, 0.0, 0.5],
    metavar=("X", "Y", "Z"),
    help="Point the camera looks at (default: 0 0 0.5).",
)
parser.add_argument("--output", default="runs/09_asset_zoo/video.mp4", help="Output .mp4 path.")
parser.add_argument(
    "--livestream",
    type=int,
    default=0,
    choices=[0, 1, 2],
    help="0=off, 2=WebRTC streaming.",
)
args = parser.parse_args()

try:
    from isaaclab.app import AppLauncher
except ImportError as exc:
    raise SystemExit(
        "\n[Example 09] Isaac Lab not found. Run inside an Isaac Sim/Isaac Lab environment.\n"
    ) from exc

simulation_app = AppLauncher(
    {"headless": True, "enable_cameras": True, "livestream": args.livestream}
).app

# ── imports that require a running simulation app ─────────────────────────────
import gymnasium as gym
import isaaclab_tasks  # noqa: F401  (registers Isaac Lab envs)
import omni.replicator.core as rep
from skrl.envs.wrappers.torch import wrap_env

from examples.shared.utils import default_device


def _build_env():
    spec = gym.spec(args.env_id)
    cfg_entry = (spec.kwargs or {}).get("env_cfg_entry_point", "")
    if cfg_entry:
        module_path, class_name = cfg_entry.split(":")
        env_cfg = getattr(importlib.import_module(module_path), class_name)()
        if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs"):
            env_cfg.scene.num_envs = 1
        return wrap_env(gym.make(args.env_id, cfg=env_cfg))
    return wrap_env(gym.make(args.env_id))


def _setup_recording_camera():
    """Create a high-res omni.replicator camera independent of the env sensors."""
    print(
        f"[record] Recording camera: {args.width}x{args.height}  "
        f"pos={args.cam_pos}  target={args.cam_target}"
    )
    cam = rep.create.camera(
        position=tuple(args.cam_pos),
        look_at=tuple(args.cam_target),
        focal_length=24.0,
        clipping_range=(0.1, 1_000.0),
    )
    rp = rep.create.render_product(cam, (args.width, args.height))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach(rp)
    return annotator


def _grab_frame(annotator) -> np.ndarray | None:
    """Return an HxWx3 uint8 RGB array from the annotator, or None on failure."""
    data = annotator.get_data()
    if data is None or not hasattr(data, "shape") or data.size == 0:
        return None
    # replicator returns HxWx4 RGBA; drop alpha channel
    frame = data[..., :3]
    if frame.dtype != np.uint8:
        lo, hi = float(frame.min()), float(frame.max())
        if hi > lo:
            frame = ((frame - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            frame = np.zeros_like(frame, dtype=np.uint8)
    return frame.copy()


def main() -> None:
    device = default_device()
    env = _build_env()
    annotator = _setup_recording_camera()

    # Optional policy
    policy = None
    if args.checkpoint:
        from examples.shared.models import CNNGaussianPolicy

        policy = CNNGaussianPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        )
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt.get("policy", ckpt.get("model", ckpt))
        policy.load_state_dict(state_dict, strict=False)
        policy.to(device)
        policy.eval()
        print(f"[record] Loaded checkpoint: {args.checkpoint}")
    else:
        print("[record] No checkpoint provided — using zero actions.")

    obs, _ = env.reset()

    # Warm-up: let Isaac Sim finish loading assets before we start capturing
    print("[record] Warming up renderer (10 steps) ...")
    for _ in range(10):
        simulation_app.update()

    frames: list[np.ndarray] = []
    print(f"[record] Recording {args.steps} steps from '{args.env_id}' ...")

    for step in range(args.steps):
        if policy is not None:
            with torch.no_grad():
                action, _ = policy.act({"states": obs}, role="policy")
        else:
            action = torch.zeros(
                (obs.shape[0], env.action_space.shape[0]), device=device
            )

        obs, _, terminated, truncated, _ = env.step(action)

        # Trigger a full render pass so the annotator has fresh pixel data
        simulation_app.update()

        frame = _grab_frame(annotator)
        if frame is not None:
            frames.append(frame)

        if (step + 1) % 100 == 0:
            print(f"[record] step {step + 1}/{args.steps}  frames captured: {len(frames)}")

        if bool(terminated[0]) or bool(truncated[0]):
            obs, _ = env.reset()

    env.close()

    # ── Write video ───────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not frames:
        print("[record] ERROR: No frames captured. Check --cam-pos and --cam-target.")
        simulation_app.close()
        return

    print(f"[record] Writing {len(frames)} frames → {output_path}")
    try:
        import imageio

        imageio.mimwrite(str(output_path), frames, fps=args.fps, quality=8)
    except ImportError:
        try:
            import cv2

            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(
                str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h)
            )
            for fr in frames:
                writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            writer.release()
        except ImportError:
            png_dir = output_path.with_suffix("")
            png_dir.mkdir(parents=True, exist_ok=True)
            from PIL import Image

            for i, fr in enumerate(frames):
                Image.fromarray(fr).save(png_dir / f"frame_{i:05d}.png")
            print(f"[record] Saved PNG frames to {png_dir}")
            print(
                f"[record] Convert with: ffmpeg -r {args.fps} -i "
                f"{png_dir}/frame_%05d.png {output_path}"
            )
            simulation_app.close()
            return

    print(f"[record] Done: {output_path}")


main()
simulation_app.close()
