"""env.py — Gymnasium environment wrapping gs_navsim via WebSocket.

The robot lives inside the gs_navsim browser simulator.  This environment:
  1. Sends movement commands over WebSocket and receives rendered images back.
  2. Stacks the last `context_size+1` = 4 RGB frames as the observation,
     plus a fixed goal image → tensor shape (15, 96, 96).
  3. Uses an obstacle mask JSON (same format as mask_editor.js) to detect
     collisions without relying on the frontend; collision → reward = -5.0
     and episode termination.
  4. Computes a pixel-distance reward toward the goal image.
"""
from __future__ import annotations

import base64
import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PIL import Image
import io
import websocket  # websocket-client (synchronous)


# ── ImageNet normalisation constants ──────────────────────────────────────────
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMG_SIZE    = 96      # spatial size expected by NOMAD
CONTEXT     = 3       # number of *past* frames kept (total obs frames = CONTEXT+1)
OBS_FRAMES  = CONTEXT + 1          # 4
OBS_CHANNELS = OBS_FRAMES * 3     # 12 obs + 3 goal = 15


def _preprocess(img_array: np.ndarray) -> np.ndarray:
    """Resize to IMG_SIZExIMG_SIZE, convert to float32 CHW, ImageNet-normalise."""
    pil = Image.fromarray(img_array).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0        # HWC [0,1]
    arr = (arr - _MEAN) / _STD                              # ImageNet normalise
    return arr.transpose(2, 0, 1)                           # CHW


def _load_goal(path: str | Path) -> np.ndarray:
    """Load a goal image from disk and preprocess it → (3, H, W) float32."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    return _preprocess(arr)


# ── Obstacle mask (matches mask.js MaskManager) ───────────────────────────────

class _ObstacleMask:
    """Reads the JSON exported by mask_editor.js and answers collision queries."""

    def __init__(self, json_path: str | Path):
        with open(json_path) as f:
            data = json.load(f)
        b = data["bounds"]
        self.min_x  = float(b["minX"])
        self.min_z  = float(b["minZ"])
        self.res    = float(data["resolution"])
        self.grid_w = int(data["gridW"])
        self.grid_h = int(data["gridH"])
        arr         = np.array(data["data"], dtype=bool)
        self.grid   = arr.reshape(self.grid_h, self.grid_w)
        # Pre-compute free cells for random spawn
        rows, cols = np.where(~self.grid)
        self._free_cells = list(zip(rows.tolist(), cols.tolist()))

    def is_blocked(self, x: float, z: float) -> bool:
        col = int((x - self.min_x) / self.res)
        row = int((z - self.min_z) / self.res)
        if col < 0 or col >= self.grid_w or row < 0 or row >= self.grid_h:
            return False   # outside bounds → not blocked
        return bool(self.grid[row, col])

    def is_blocked_circle(self, x: float, z: float, radius: float) -> bool:
        """Sample a few points on the footprint circle for a quick check."""
        if self.is_blocked(x, z):
            return True
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            cx = x + radius * np.cos(angle)
            cz = z + radius * np.sin(angle)
            if self.is_blocked(cx, cz):
                return True
        return False

    def random_free_position(
        self, rng: np.random.Generator, robot_radius: float = 0.35
    ) -> tuple[float, float]:
        """Return a random (x, z_mask) whose footprint is entirely unblocked.

        z_mask is in mask coordinates (same as what the mask stores).
        The caller must negate it before sending to the Three.js sim
        because the sim uses  -z_threejs  when querying the mask.
        """
        if not self._free_cells:
            return 0.0, 0.0
        # Shuffle a copy so we try cells in random order without replacement.
        indices = rng.permutation(len(self._free_cells))
        for idx in indices:
            row, col = self._free_cells[int(idx)]
            x = self.min_x + (col + 0.5) * self.res
            z = self.min_z + (row + 0.5) * self.res
            if not self.is_blocked_circle(x, z, robot_radius):
                return float(x), float(z)
        # Fallback: return centre of first free cell even if footprint overlaps
        row, col = self._free_cells[int(indices[0])]
        return (
            float(self.min_x + (col + 0.5) * self.res),
            float(self.min_z + (row + 0.5) * self.res),
        )


# ── Environment ────────────────────────────────────────────────────────────────

class GsNavSimEnv(gym.Env):
    """Gymnasium environment backed by the gs_navsim WebSocket simulator.

    Parameters
    ----------
    goal_image_path : str | Path
        Path to the goal image loaded from disk (PNG or JPG).
    mask_path : str | Path | None
        Path to an obstacle mask JSON (exported by mask_editor.js).
        If None, collision checking is disabled.
    ws_url : str
        WebSocket server URL (default ``ws://localhost:8081``).
    max_steps : int
        Maximum steps per episode before truncation.
    success_threshold : float
        Pixel-distance below which the episode is considered a success.
    robot_radius : float
        Robot footprint radius used for collision checks (world units).
    seed : int | None
        RNG seed.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        goal_image_path: str | Path,
        mask_path:        str | Path | None = None,
        ws_url:           str               = "ws://localhost:8081",
        max_steps:        int               = 500,
        success_threshold: float            = 0.02,
        seed:             int | None        = None,
    ):
        super().__init__()

        self.goal_image_path  = Path(goal_image_path)
        self.ws_url           = ws_url
        self.max_steps        = max_steps
        self.success_threshold = success_threshold

        # Obstacle mask (optional)
        self.mask = _ObstacleMask(mask_path) if mask_path else None

        # RNG
        self.rng = np.random.default_rng(seed)

        # Gymnasium spaces
        # Observation: (15, 96, 96) float32 — 4 obs frames + 1 goal frame, all CHW
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(OBS_CHANNELS + 3, IMG_SIZE, IMG_SIZE),  # 15 channels
            dtype=np.float32,
        )
        # Action: continuous 2D waypoint [dx, dy] in robot frame
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([ 1.0,  1.0], dtype=np.float32),
        )

        # Frame buffer (most recent frame last)
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=OBS_FRAMES)

        # Goal image (loaded once)
        self._goal_chw = _load_goal(self.goal_image_path)

        # WebSocket state
        self._ws:          websocket.WebSocket | None = None
        self._last_image:  np.ndarray | None = None
        self._image_event  = threading.Event()
        self._collision_flag = False  # set by _recv_loop when sim reports collision

        # Episode counters / tracking
        self._step_count = 0
        self._prev_pixel_dist = 1.0

    # ── Connection ─────────────────────────────────────────────────────────────

    def _connect(self):
        if self._ws is not None:
            return
        self._ws = websocket.create_connection(self.ws_url, timeout=10)
        self._ws.send(json.dumps({"type": "identify", "client": "robot"}))
        # Start receiver thread
        self._receiver_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._receiver_thread.start()

    def _recv_loop(self):
        """Background thread that reads all WebSocket messages."""
        while self._ws is not None:
            try:
                raw = self._ws.recv()
                msg = json.loads(raw)
                mtype = msg.get("type")
                if mtype in ("image_data", "rendered_image"):
                    data_url = msg.get("data", "")
                    if data_url.startswith("data:image"):
                        b64 = data_url.split(",", 1)[1]
                    else:
                        b64 = data_url
                    img_bytes = base64.b64decode(b64)
                    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    self._last_image = np.asarray(pil)
                    self._image_event.set()
                elif mtype == "collision":
                    # Sim blocked the move and is reporting a collision.
                    # The image for this step is still coming (sent right after).
                    self._collision_flag = True
            except Exception:
                break

    def _wait_for_image(self, timeout: float = 5.0) -> np.ndarray:
        self._image_event.clear()
        self._image_event.wait(timeout=timeout)
        if self._last_image is None:
            # Return black frame as fallback
            return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        return self._last_image

    def _send(self, msg: dict):
        self._ws.send(json.dumps(msg))

    # ── Gymnasium API ──────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        print("resetting environment...")
        self._connect()
        self._step_count = 0

        # Choose a free spawn position.
        # random_free_position returns mask coordinates where z_mask = -z_threejs
        # (the editor stores obstacles at positive Z; the sim checks at -robot.z).
        if self.mask:
            x0, z0_mask = self.mask.random_free_position(self.rng)
            z0 = -z0_mask   # convert mask → Three.js
        else:
            x0, z0 = 0.0, 0.0
        rot0 = float(self.rng.uniform(0, 2 * np.pi))

        # Reset robot in the simulator
        self._send({
            "type":     "reset_robot",
            "x":        x0,
            "y":        0.3,
            "z":        z0,
            "rotation": rot0,
        })
        frame = self._wait_for_image()

        # Fill frame buffer with the initial frame
        self._frame_buffer.clear()
        chw = _preprocess(frame)
        for _ in range(OBS_FRAMES):
            self._frame_buffer.append(chw)

        self._prev_pixel_dist = self._pixel_dist(chw)

        obs = self._build_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        dx, dy = float(action[0]), float(action[1])

        # Convert 2D waypoint to [v, w] via PD controller
        v, w = _pd_controller(dx, dy)

        # Clear collision flag, then send the movement command.
        # If the sim detects a collision it will send a 'collision' WS message
        # BEFORE the image (socket.js uses a 100 ms delay before captureAndSendImage),
        # so _recv_loop will set _collision_flag before _image_event fires.
        self._collision_flag = False
        self._send({
            "type":    "robot_command",
            "command": "set_velocity",
            "value":   [v, w],
        })
        frame = self._wait_for_image()
        chw   = _preprocess(frame)
        self._frame_buffer.append(chw)

        # Check if sim reported a collision (movement was NOT applied)
        if self._collision_flag:
            obs = self._build_obs()
            return obs, -5.0, True, False, {"collision": True}

        # Reward
        curr_dist  = self._pixel_dist(chw)
        reward     = (self._prev_pixel_dist - curr_dist) * 100.0
        reward    -= 0.01  # time penalty
        self._prev_pixel_dist = curr_dist

        # Success
        success = curr_dist < self.success_threshold
        if success:
            reward += 10.0

        self._step_count += 1
        truncated  = self._step_count >= self.max_steps
        terminated = success

        obs = self._build_obs()
        return obs, reward, terminated, truncated, {"pixel_dist": curr_dist, "success": success}

    def close(self):
        if self._ws is not None:
            self._ws.close()
            self._ws = None

    # ── Internals ──────────────────────────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        """Stack frame buffer + goal image → (15, 96, 96) float32."""
        frames = np.concatenate(list(self._frame_buffer), axis=0)  # (12, H, W)
        return np.concatenate([frames, self._goal_chw], axis=0).astype(np.float32)

    def _pixel_dist(self, frame_chw: np.ndarray) -> float:
        """Mean squared pixel difference between current frame and goal."""
        return float(np.mean((frame_chw - self._goal_chw) ** 2))


# ── PD controller ──────────────────────────────────────────────────────────────

def _pd_controller(
    dx: float, dy: float,
    dt: float = 0.1,
    max_v: float = 0.5,
    max_w: float = 1.5,
) -> tuple[float, float]:
    """Convert a 2D waypoint [dx, dy] in robot frame to [v, w]."""
    EPS = 1e-8
    if abs(dx) < EPS:
        v = 0.0
        w = float(np.sign(dy)) * max_w
    else:
        v = dx / dt
        w = float(np.arctan2(dy, dx)) / dt
    v = float(np.clip(v, 0.0, max_v))
    w = float(np.clip(w, -max_w, max_w))
    return v, w
