from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


def _obs_size(observation_space: spaces.Space) -> int:
    """Compute flattened observation size for MLP-based models."""

    if isinstance(observation_space, spaces.Box):
        return int(np.prod(observation_space.shape))
    return int(np.prod(observation_space.shape))



class MLPGaussianPolicy(GaussianMixin, Model):
    """Policy network for continuous actions with diagonal Gaussian outputs.

    The network predicts action means while a learned log-standard-deviation
    parameter controls exploration noise. PPO then optimizes this distribution.
    """

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

        in_features = _obs_size(self.observation_space)
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, self.num_actions),
        )
        # One learnable std value per action dimension.
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        # skrl 2.x uses "observations" as the primary key; fall back to
        # "states" for compatibility if needed.
        obs = inputs.get("observations", inputs.get("states"))
        obs = obs.view(obs.shape[0], -1)
        return self.net(obs), {"log_std": self.log_std_parameter}



class MLPValue(DeterministicMixin, Model):
    """State-value estimator V(s) used by PPO for advantage computation."""

    def __init__(self, *args, **kwargs):
        Model.__init__(self, *args, **kwargs)
        DeterministicMixin.__init__(self, clip_actions=False)

        in_features = _obs_size(self.observation_space)
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def compute(self, inputs, role):
        obs = inputs.get("observations", inputs.get("states"))
        obs = obs.view(obs.shape[0], -1)
        return self.net(obs), {}



class CNNEncoder(nn.Module):
    """Convolutional feature extractor for image observations.

    This is a classic Atari-style stack that progressively reduces spatial size
    while increasing channel depth. The output is a flat latent feature vector.
    """

    def __init__(self, in_channels: int, height: int, width: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Infer latent feature size dynamically so we can support any input
        # image resolution that survives the convolution stack.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, height, width)
            self.out_features = self.cnn(dummy).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn(x)


def _parse_image_shape(shape: tuple) -> tuple[int, int, int, bool]:
    """Return (in_channels, height, width, is_hwc) for an image observation shape.

    Handles both CHW (Gymnasium/CarRacing) and HWC (Isaac Lab camera) formats.
    HWC is detected when the last dim is a small typical channel count (1, 3, 4)
    and the spatial dims are larger.
    """
    if len(shape) == 3 and shape[-1] in (1, 3, 4) and shape[0] > shape[-1]:
        # HWC layout — Isaac Lab camera envs
        return shape[2], shape[0], shape[1], True
    # CHW layout — standard Gymnasium pixel envs
    return shape[0], shape[1], shape[2], False



class CNNGaussianPolicy(GaussianMixin, Model):
    """Continuous-action policy that first encodes images with a CNN."""

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

        in_channels, height, width, self._hwc = _parse_image_shape(self.observation_space.shape)
        self.encoder = CNNEncoder(in_channels=in_channels, height=height, width=width)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.out_features, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        obs = inputs.get("observations", inputs.get("states")).float()
        # skrl may flatten observations internally; reshape back to image tensor.
        if obs.dim() == 2:
            obs = obs.view(obs.shape[0], *self.observation_space.shape)
        # Isaac Lab camera envs emit HWC; convert to CHW for Conv2d.
        if self._hwc:
            obs = obs.permute(0, 3, 1, 2).contiguous()
        # Most pixel envs emit uint8 images in [0, 255]. Normalize on-the-fly
        # so optimization sees a stable input scale.
        if obs.max() > 1.0:
            obs = obs / 255.0
        features = self.encoder(obs)
        return self.head(features), {"log_std": self.log_std_parameter}



class CNNValue(DeterministicMixin, Model):
    """Image-based value function using the same encoder pattern as policy."""

    def __init__(self, *args, **kwargs):
        Model.__init__(self, *args, **kwargs)
        DeterministicMixin.__init__(self, clip_actions=False)

        in_channels, height, width, self._hwc = _parse_image_shape(self.observation_space.shape)
        self.encoder = CNNEncoder(in_channels=in_channels, height=height, width=width)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.out_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def compute(self, inputs, role):
        obs = inputs.get("observations", inputs.get("states")).float()
        if obs.dim() == 2:
            obs = obs.view(obs.shape[0], *self.observation_space.shape)
        if self._hwc:
            obs = obs.permute(0, 3, 1, 2).contiguous()
        if obs.max() > 1.0:
            obs = obs / 255.0
        features = self.encoder(obs)
        return self.head(features), {}
