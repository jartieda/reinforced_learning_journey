"""model.py — skrl models using NOMAD ViNT as a frozen visual encoder.

Architecture
------------
- Observation: (15, 96, 96) float32
    * channels 0..11  → 4 × RGB frames (context_size=3 + 1)  →  obs_img (B,12,96,96)
    * channels 12..14 → goal RGB frame                         →  goal_img (B,3,96,96)
- Encoder: NoMaD_ViNT (EfficientNet-b0) → embedding (B, 256), frozen
- Policy head: MLP  256 → 256 → 2   (Gaussian)
- Value  head: MLP  256 → 256 → 1   (Deterministic)

The encoder weights are loaded from the nomad.pth checkpoint and frozen.
Only the MLP heads are trained during PPO.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from gymnasium import spaces
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

# ── NOMAD model path ────────────────────────────────────────────────────────
# Add visualnav-transformer/train to sys.path so vint_train can be imported.
_ROOT = Path(__file__).resolve().parents[3]  # /robot_nav
_TRAIN_DIR = _ROOT / "visualnav-transformer" / "train"
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

from vint_train.models.nomad.nomad_vint import NoMaD_ViNT  # noqa: E402

NOMAD_WEIGHTS = _ROOT / "visualnav-transformer" / "model_weights" / "nomad.pth"

# NOMAD config (must match nomad.yaml)
_CONTEXT_SIZE      = 3
_ENCODING_SIZE     = 256
_MHA_HEADS         = 4
_MHA_LAYERS        = 4
_MHA_FF_FACTOR     = 4


def build_nomad_encoder(weights_path: str | Path = NOMAD_WEIGHTS) -> NoMaD_ViNT:
    """Instantiate NoMaD_ViNT, load pretrained weights, freeze all params."""
    encoder = NoMaD_ViNT(
        context_size=_CONTEXT_SIZE,
        obs_encoder="efficientnet-b0",
        obs_encoding_size=_ENCODING_SIZE,
        mha_num_attention_heads=_MHA_HEADS,
        mha_num_attention_layers=_MHA_LAYERS,
        mha_ff_dim_factor=_MHA_FF_FACTOR,
    )

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    # The checkpoint may be either the full state_dict or a nested dict.
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    encoder.load_state_dict(ckpt, strict=False)

    encoder.requires_grad_(False)
    encoder.eval()
    return encoder


def _split_obs(obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split (B,15,H,W) observation into obs_img (B,12,H,W) and goal_img (B,3,H,W)."""
    return obs[:, :12], obs[:, 12:]


class NomadPolicy(GaussianMixin, Model):
    """Continuous-action policy: NOMAD encoder (frozen) + trainable MLP head."""

    def __init__(self, *args, weights_path: str | Path = NOMAD_WEIGHTS, **kwargs):
        Model.__init__(self, *args, **kwargs)
        GaussianMixin.__init__(
            self,
            clip_actions=True,
            clip_log_std=True,
            min_log_std=-20,
            max_log_std=2,
            reduction="sum",
        )

        self.encoder = build_nomad_encoder(weights_path)
        self.head = nn.Sequential(
            nn.Linear(_ENCODING_SIZE, 256),
            nn.Tanh(),
            nn.Linear(256, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs: dict, role: str):
        obs = inputs.get("observations", inputs.get("states")).float()
        # Reshape from flat (B, 15*96*96) back to (B, 15, 96, 96) if needed
        if obs.dim() == 2:
            obs = obs.view(obs.shape[0], *self.observation_space.shape)

        obs_img, goal_img = _split_obs(obs)

        # NOMAD encoder is frozen → no grad needed
        with torch.no_grad():
            B = obs_img.shape[0]
            goal_mask = torch.zeros(B, dtype=torch.long, device=obs_img.device)
            embedding = self.encoder(obs_img, goal_img, input_goal_mask=goal_mask)

        actions = self.head(embedding)
        return actions, {"log_std": self.log_std_parameter}


class NomadValue(DeterministicMixin, Model):
    """State-value estimator: NOMAD encoder (shared, frozen) + trainable MLP head."""

    def __init__(self, *args, weights_path: str | Path = NOMAD_WEIGHTS, **kwargs):
        Model.__init__(self, *args, **kwargs)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.encoder = build_nomad_encoder(weights_path)
        self.head = nn.Sequential(
            nn.Linear(_ENCODING_SIZE, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

    def compute(self, inputs: dict, role: str):
        obs = inputs.get("observations", inputs.get("states")).float()
        if obs.dim() == 2:
            obs = obs.view(obs.shape[0], *self.observation_space.shape)

        obs_img, goal_img = _split_obs(obs)

        with torch.no_grad():
            B = obs_img.shape[0]
            goal_mask = torch.zeros(B, dtype=torch.long, device=obs_img.device)
            embedding = self.encoder(obs_img, goal_img, input_goal_mask=goal_mask)

        value = self.head(embedding)
        return value, {}
