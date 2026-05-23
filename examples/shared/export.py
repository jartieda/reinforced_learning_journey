"""Export a trained skrl policy and run standalone inference.

This script demonstrates two things:

1. How to export the policy network weights to a plain *.pt* file that can be
   loaded and used for inference **without** skrl, trainers, or wrappers.
2. How to watch (or record video of) the exported policy acting in the
   environment using only PyTorch + Gymnasium.

Usage
-----
Export weights and watch the policy:

    python -m examples.shared.export \\
        --checkpoint runs/02_ppo_classic_control/02_ppo_classic_control/checkpoints/agent_5000.pt \\
        --env-id Pendulum-v1 \\
        --model mlp \\
        --episodes 3 \\
        --render

Export and record a video:

    python -m examples.shared.export \\
        --checkpoint runs/02_ppo_classic_control/02_ppo_classic_control/checkpoints/agent_5000.pt \\
        --env-id Pendulum-v1 \\
        --model mlp \\
        --record \\
        --video-dir videos/pendulum

For pixel-based examples (CNN policy):

    python -m examples.shared.export \\
        --checkpoint runs/05_ppo_cnn/05_ppo_cnn/checkpoints/agent_5000.pt \\
        --env-id CarRacing-v3 \\
        --model cnn \\
        --record \\
        --video-dir videos/carracing

Standalone inference (no skrl)
-------------------------------
After exporting, run the policy from plain Python:

    import torch, gymnasium as gym, numpy as np

    # Load only the network weights — no skrl needed.
    state = torch.load("exported_policy.pt", map_location="cpu")
    # Reconstruct the network with the same class used during training.
    from examples.shared.models import MLPGaussianPolicy
    import gymnasium as gym
    env = gym.make("Pendulum-v1")
    model = MLPGaussianPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device="cpu",
    )
    model.load_state_dict(state)
    model.eval()

    obs, _ = env.reset()
    for _ in range(200):
        t_obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            # act() returns a deterministic mean action when deterministic=True.
            action, _, _ = model.act({"observations": t_obs}, role="policy")
        obs, _, terminated, truncated, _ = env.step(action.numpy()[0])
        if terminated or truncated:
            break
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import gymnasium as gym
import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_model(model_type: str, env: gym.Env, device: str):
    """Instantiate the right model class for the given example type."""
    if model_type == "mlp":
        from examples.shared.models import MLPGaussianPolicy
        return MLPGaussianPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        )
    if model_type == "cnn":
        from examples.shared.models import CNNGaussianPolicy
        return CNNGaussianPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        )
    raise ValueError(f"Unknown model type '{model_type}'. Choose 'mlp' or 'cnn'.")


def _load_policy_weights(checkpoint: Path, model, device: str):
    """Extract and load only the policy network weights from a skrl checkpoint.

    skrl checkpoints are dicts that bundle agent state, preprocessors, etc.
    We extract just the 'policy' sub-dict so the model loads cleanly.
    """
    raw = torch.load(str(checkpoint), map_location=device)

    # skrl saves: {"policy": state_dict, "value": state_dict, ...}
    if isinstance(raw, dict) and "policy" in raw:
        state = raw["policy"]
    else:
        # Fallback: assume the file is already a plain state_dict.
        state = raw

    model.load_state_dict(state)
    model.eval()
    return model


def _preprocess_obs(obs: np.ndarray, model_type: str, device: str) -> torch.Tensor:
    """Convert numpy observation to the format the model expects."""
    t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    if model_type == "cnn" and t.max() > 1.0:
        t = t / 255.0
    return t


def _run_episodes(
    env: gym.Env,
    model,
    model_type: str,
    device: str,
    n_episodes: int,
    record: bool,
    video_dir: Path | None,
) -> list[float]:
    """Run *n_episodes* with the exported policy; optionally record video."""
    if record and video_dir is not None:
        from gymnasium.wrappers import RecordVideo
        video_dir.mkdir(parents=True, exist_ok=True)
        env = RecordVideo(
            env,
            video_folder=str(video_dir),
            episode_trigger=lambda _: True,   # record every episode
            name_prefix="policy",
            disable_logger=True,
        )
        print(f"Recording video to: {video_dir.resolve()}")

    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        step = 0

        while True:
            t_obs = _preprocess_obs(obs, model_type, device)
            with torch.no_grad():
                # act() with deterministic=True returns the mean action.
                action, _, _ = model.act(
                    {"observations": t_obs}, role="policy"
                )
            action_np = action.cpu().numpy()[0]
            obs, reward, terminated, truncated, _ = env.step(action_np)
            total_reward += float(reward)
            step += 1
            if terminated or truncated:
                break

        rewards.append(total_reward)
        print(f"  Episode {ep + 1}/{n_episodes}: reward = {total_reward:.1f}  ({step} steps)")

    env.close()
    return rewards


# ── Export ────────────────────────────────────────────────────────────────────

def export_weights(model, output: Path) -> None:
    """Save just the policy state_dict to *output* for framework-free loading."""
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(output))
    print(f"Policy weights exported to: {output.resolve()}")
    print(
        "\nStandalone inference (no skrl needed):\n"
        f"  model.load_state_dict(torch.load('{output}', map_location='cpu'))"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a trained policy and run standalone inference."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to the skrl checkpoint .pt file.")
    parser.add_argument("--env-id", required=True,
                        help="Gymnasium environment ID used during training.")
    parser.add_argument("--model", choices=["mlp", "cnn"], default="mlp",
                        help="Model architecture: 'mlp' (classic control) or 'cnn' (pixels).")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of evaluation episodes to run.")
    parser.add_argument("--render", action="store_true",
                        help="Open a window and watch the policy play.")
    parser.add_argument("--record", action="store_true",
                        help="Save MP4 video of the policy (requires render_mode=rgb_array).")
    parser.add_argument("--video-dir", default="videos",
                        help="Directory for recorded videos (default: videos/).")
    parser.add_argument("--output", default="exported_policy.pt",
                        help="Filename for the exported weights (default: exported_policy.pt).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        sys.exit(f"Checkpoint not found: {checkpoint}")

    # ── Build environment ────────────────────────────────────────────────────
    # For pixel envs, apply the same preprocessing wrappers used in training.
    if args.model == "cnn":
        from examples.shared.wrappers import ToGrayscale
        render_mode = "rgb_array" if args.record else ("human" if args.render else None)
        _env = gym.make(args.env_id, render_mode=render_mode)
        env = ToGrayscale(_env)
    else:
        render_mode = "rgb_array" if args.record else ("human" if args.render else None)
        env = gym.make(args.env_id, render_mode=render_mode)

    # ── Load model ───────────────────────────────────────────────────────────
    model = _build_model(args.model, env, args.device)
    _load_policy_weights(checkpoint, model, args.device)
    print(f"Loaded policy from: {checkpoint}")
    print(f"Model type: {args.model.upper()}")
    print(f"Device: {args.device}")

    # ── Export weights ───────────────────────────────────────────────────────
    export_weights(model, Path(args.output))

    # ── Run inference ────────────────────────────────────────────────────────
    print(f"\nRunning {args.episodes} episode(s) in {args.env_id}...")
    rewards = _run_episodes(
        env=env,
        model=model,
        model_type=args.model,
        device=args.device,
        n_episodes=args.episodes,
        record=args.record,
        video_dir=Path(args.video_dir) if args.record else None,
    )

    print(f"\nResults over {len(rewards)} episode(s):")
    print(f"  Mean reward: {np.mean(rewards):.1f}")
    print(f"  Std  reward: {np.std(rewards):.1f}")
    print(f"  Best reward: {max(rewards):.1f}")


if __name__ == "__main__":
    main()
