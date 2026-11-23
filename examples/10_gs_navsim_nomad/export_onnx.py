"""export_onnx.py — Export Example 10 policy to fixed-shape ONNX for TPU-MLIR.

This exporter supports both architectures:
- nomad  : frozen ViNT encoder + MLP head
- cv181x : CV18xx-friendly CNN backbone + MLP head

Output contract (fixed):
- Input  name: observations, shape [1, 15, 96, 96], float32
- Output name: actions,      shape [1, 2],          float32
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Sibling imports
sys.path.insert(0, os.path.dirname(__file__))
from env import GsNavSimEnv  # noqa: E402
from model import NOMAD_WEIGHTS, NomadPolicy  # noqa: E402
from model_cv181x import Cv181xDeterministicActor, Cv181xPolicy  # noqa: E402


class NomadDeterministicActor(nn.Module):
    """Inference/export wrapper that returns deterministic action means only."""

    def __init__(self, policy: NomadPolicy):
        super().__init__()
        self.encoder = policy.encoder
        self.head = policy.head

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dim() == 3:
            observations = observations.unsqueeze(0)
        observations = observations.float()
        obs_img = observations[:, :12]
        goal_img = observations[:, 12:]
        bsz = obs_img.shape[0]
        goal_mask = torch.zeros(bsz, dtype=torch.long, device=obs_img.device)
        embedding = self.encoder(obs_img, goal_img, input_goal_mask=goal_mask)
        return self.head(embedding)


def _load_policy_state(checkpoint_path: Path, device: str) -> dict:
    raw = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if isinstance(raw, dict) and "policy" in raw:
        return raw["policy"]
    if isinstance(raw, dict):
        return raw
    raise ValueError("Unsupported checkpoint format. Expected state_dict or dict with 'policy'.")


def _build_policy(arch: str, env: GsNavSimEnv, device: str):
    if arch == "nomad":
        return NomadPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            weights_path=NOMAD_WEIGHTS,
        )
    return Cv181xPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )


def _build_export_actor(arch: str, policy) -> nn.Module:
    if arch == "nomad":
        return NomadDeterministicActor(policy)
    return Cv181xDeterministicActor(policy)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Example 10 policy to fixed-shape ONNX for TPU-MLIR"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to skrl checkpoint (.pt)")
    parser.add_argument("--goal", type=str, required=True,
                        help="Goal image path used to build env spaces")
    parser.add_argument("--arch", type=str, default="cv181x", choices=["nomad", "cv181x"],
                        help="Architecture used in the checkpoint")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (default: 17)")
    parser.add_argument("--output", type=str, default="policy_cv181x.onnx",
                        help="Output ONNX path")
    parser.add_argument("--ws-url", type=str, default="ws://localhost:8081",
                        help="Env websocket URL (only for env construction)")
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()

    device = "cpu"
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    # Build env only to obtain exact observation/action spaces.
    env = GsNavSimEnv(
        goal_image_path=args.goal,
        ws_url=args.ws_url,
        max_steps=args.max_steps,
        seed=0,
    )

    policy = _build_policy(args.arch, env, device)
    policy.load_state_dict(_load_policy_state(checkpoint, device))
    policy.eval()

    actor = _build_export_actor(args.arch, policy).to(device)
    actor.eval()

    dummy = torch.zeros((1, 15, 96, 96), dtype=torch.float32, device=device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            actor,
            dummy,
            str(output),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["observations"],
            output_names=["actions"],
            dynamic_axes=None,
        )

    env.close()

    print(f"Exported ONNX: {output.resolve()}")
    print("Input : observations [1, 15, 96, 96] float32")
    print("Output: actions      [1, 2] float32")
    print("\nSuggested TPU-MLIR commands:")
    print("  model_transform --model_name nav_policy --model_def <onnx_path> --input_shapes [[1,15,96,96]] --mlir nav_policy.mlir")
    print("  model_deploy --mlir nav_policy.mlir --processor cv181x --quantize BF16 --model nav_policy_bf16.cvimodel")


if __name__ == "__main__":
    main()
