"""model_cv181x.py — skrl models with a CV181x-friendly pure CNN backbone.

Design goals
------------
- Fixed input layout: (B, 15, 96, 96)
- No GroupNorm / LayerNorm / attention blocks
- Use only mobile vision-friendly layers (Conv / DWConv / ReLU / GAP / Linear)

Observation layout is unchanged from the NOMAD model:
- channels 0..11  -> 4 x RGB context/current frames
- channels 12..14 -> goal RGB frame
"""
from __future__ import annotations

import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise 3x3 + pointwise 1x1 convolution block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=True,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(x)


class Cv181xBackbone(nn.Module):
    """Compact CNN backbone designed to compile on CV18xx TPU-MLIR paths."""

    def __init__(self, in_channels: int = 15):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.block1 = DepthwiseSeparableBlock(32, 64, stride=1)
        self.block2 = DepthwiseSeparableBlock(64, 128, stride=2)
        self.block3 = DepthwiseSeparableBlock(128, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


class Cv181xPolicy(GaussianMixin, Model):
    """PPO policy head for training (Gaussian in skrl; mean used at inference)."""

    def __init__(self, *args, **kwargs):
        Model.__init__(self, *args, **kwargs)
        GaussianMixin.__init__(
            self,
            clip_actions=True,
            clip_log_std=True,
            min_log_std=-20,
            max_log_std=2,
            reduction="sum",
        )

        self.backbone = Cv181xBackbone(in_channels=self.observation_space.shape[0])
        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.num_actions),
            nn.Tanh(),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def _prepare_obs(self, inputs: dict) -> torch.Tensor:
        obs = inputs.get("observations", inputs.get("states")).float()
        if obs.dim() == 2:
            obs = obs.view(obs.shape[0], *self.observation_space.shape)
        elif obs.dim() == 3:
            obs = obs.unsqueeze(0)
        return obs

    def forward_actions(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(obs)
        return self.head(features)

    def compute(self, inputs: dict, role: str):
        obs = self._prepare_obs(inputs)
        actions_mean = self.forward_actions(obs)
        return actions_mean, {"log_std": self.log_std_parameter}


class Cv181xValue(DeterministicMixin, Model):
    """Value model with the same CV181x-compatible CNN backbone."""

    def __init__(self, *args, **kwargs):
        Model.__init__(self, *args, **kwargs)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.backbone = Cv181xBackbone(in_channels=self.observation_space.shape[0])
        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def _prepare_obs(self, inputs: dict) -> torch.Tensor:
        obs = inputs.get("observations", inputs.get("states")).float()
        if obs.dim() == 2:
            obs = obs.view(obs.shape[0], *self.observation_space.shape)
        elif obs.dim() == 3:
            obs = obs.unsqueeze(0)
        return obs

    def compute(self, inputs: dict, role: str):
        obs = self._prepare_obs(inputs)
        features = self.backbone(obs)
        value = self.head(features)
        return value, {}


class Cv181xDeterministicActor(nn.Module):
    """Inference/export wrapper that emits only deterministic actions."""

    def __init__(self, policy: Cv181xPolicy):
        super().__init__()
        self.backbone = policy.backbone
        self.head = policy.head

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dim() == 3:
            observations = observations.unsqueeze(0)
        features = self.backbone(observations.float())
        return self.head(features)
