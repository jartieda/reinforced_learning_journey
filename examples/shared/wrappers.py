from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ToChannelFirst(gym.ObservationWrapper):
    """Convert HWC images to CHW format expected by PyTorch CNNs."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        h, w, c = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(c, h, w),
            dtype=np.uint8,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.transpose(observation, (2, 0, 1))


class ToGrayscale(gym.ObservationWrapper):
    """Project RGB observations to a single luminance-like channel.

    This cuts input dimensionality and can speed up learning when color is not
    crucial for control.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        h, w, _ = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(1, h, w),
            dtype=np.uint8,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        # Simple channel average to keep dependencies minimal for tutorial use.
        gray = observation.mean(axis=2).astype(np.uint8)
        return np.expand_dims(gray, axis=0)


class FrameStackCHW(gym.Wrapper):
    """Stack recent CHW frames along channel axis for temporal context.

    Single images often lose motion information. Frame stacking gives the policy
    short-term dynamics (velocity/heading cues) without a recurrent model.
    """

    def __init__(self, env: gym.Env, num_frames: int):
        super().__init__(env)
        self.num_frames = num_frames
        c, h, w = env.observation_space.shape
        self.frames: deque[np.ndarray] = deque(maxlen=num_frames)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(c * num_frames, h, w),
            dtype=np.uint8,
        )

    def _stack(self) -> np.ndarray:
        return np.concatenate(list(self.frames), axis=0)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.frames.clear()
        # Seed stack with repeated initial frame so output shape is consistent
        # from timestep zero.
        for _ in range(self.num_frames):
            self.frames.append(observation)
        return self._stack(), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(observation)
        return self._stack(), reward, terminated, truncated, info


def make_carracing_pixels(frame_stack: int = 1, render_mode: str = "rgb_array") -> gym.Env:
    """Create a pixel-observation CarRacing environment used in vision lessons."""

    env = gym.make("CarRacing-v3", render_mode=render_mode)
    env = ToGrayscale(env)
    if frame_stack > 1:
        env = FrameStackCHW(env, num_frames=frame_stack)
    return env
