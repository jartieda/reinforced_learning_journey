import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

class BasicCNN(nn.Module):
    def __init__(self, in_channels: int, out_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4096, 512), # Fixed the input size
            nn.ReLU(),
            nn.Linear(512, out_features),
        )

    def forward(self, x):
        return self.net(x)

class CNNPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum", **kwargs):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device, **kwargs)
        GaussianMixin.__init__(self, clip_actions=clip_actions, clip_log_std=clip_log_std, min_log_std=min_log_std, max_log_std=max_log_std, reduction=reduction)

        self.cnn = BasicCNN(in_channels=observation_space.shape[0], out_features=256)
        self.head = nn.Sequential(
            nn.Linear(256, self.num_actions),
            nn.Tanh() # Keep output in [-1, 1] then scaled by env
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        obs = inputs.get("observations").float() / 255.0
        if obs.dim() == 2:
            obs = obs.view(obs.shape[0], *self.observation_space.shape)
        elif obs.dim() == 3:
            obs = obs.unsqueeze(0)

        if torch.isnan(obs).any():
            print(f"[CNNPolicy] Warning: obs has NaNs!")

        features = self.cnn(obs)
        if torch.isnan(features).any():
            print(f"[CNNPolicy] Warning: features has NaNs!")

        out = self.head(features)
        if torch.isnan(out).any():
            print(f"[CNNPolicy] Warning: output has NaNs!")

        return out, {"log_std": self.log_std_parameter, "features": features}

class CNNValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False, **kwargs):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device, **kwargs)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.cnn = BasicCNN(in_channels=observation_space.shape[0], out_features=256)
        self.head = nn.Linear(256, 1)

    def compute(self, inputs, role):
        obs = inputs.get("observations").float() / 255.0
        if obs.dim() == 2:
            obs = obs.view(obs.shape[0], *self.observation_space.shape)
        elif obs.dim() == 3:
            obs = obs.unsqueeze(0)
        
        features = self.cnn(obs)
        value = self.head(features)

        if not self.training:
            value = value.detach()
        return value, {"features": features}
