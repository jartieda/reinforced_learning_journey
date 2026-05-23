# Example 07: Isaac Lab Transfer (Image Observations)

## Goal

Apply everything from Examples 05–06 (CNN + PPO on pixels) inside **Isaac Lab**,
NVIDIA's physics simulator for robot learning. The PPO agent, CNN models, and
training loop are copy-pasted unchanged. Only the environment integration differs.

---

## What Isaac Lab Is and Why It Matters

| Component | Role |
|---|---|
| **Isaac Sim** | NVIDIA's Omniverse-based robot simulator (closed-source) |
| **Isaac Lab** | Open-source RL framework built on top of Isaac Sim |
| **Omniverse Kit** | The underlying rendering/physics runtime (GPU-accelerated) |
| **AppLauncher** | Python class that boots the Kit process before any omni.* import |

Isaac Lab provides environments identical in API to Gymnasium but backed by
GPU-accelerated rigid-body physics and photorealistic rendering. This means:

- **Vectorised by default**: a single `gym.make()` call spawns many parallel robot
  instances on one GPU, making sample collection dramatically faster than Gymnasium.
- **Photo-realistic cameras**: observations can be RGB, depth, or instance-segmented
  images from any virtual camera angle.
- **Sim-to-real**: policies trained here can be transferred to physical robots
  because the physics and visual fidelity are high enough to close the gap.

---

## The One New Concept: AppLauncher Bootstrap Order

Isaac Lab wraps a closed-source Omniverse runtime. That runtime must be
**fully started before any `omni.*` or `isaaclab.*` module is imported**.
Importing those modules before the runtime exists causes a hard crash with no
useful error message.

The train script enforces the correct order explicitly:

```
1. Parse CLI args               ← fails fast on typos, before 30-sec simulator boot
        ↓
2. from isaaclab.app import AppLauncher
   simulation_app = AppLauncher({...}).app    ← boots Omniverse Kit (~20–40 s)
        ↓
3. import gymnasium              ← safe: runtime is running
   import isaaclab_tasks         ← side-effect: registers all Isaac Lab envs
        ↓
4. env = wrap_env(gym.make(...)) ← identical to every prior example
5. build_ppo_agent(...)          ← identical
6. trainer.train()               ← identical
        ↓
7. simulation_app.close()        ← graceful shutdown — always required
```

Steps 4–6 are **byte-for-byte the same** as Example 06. The only
Isaac-specific lines are 2, 3 (import side-effect), and 7.

---

## Why WSL2 Cannot Run This

Isaac Sim requires **Vulkan GPU rendering**. Vulkan needs a DRM kernel
driver (`/dev/dri/renderD128`), which the Microsoft WSL2 kernel intentionally
omits. Both the NVIDIA and Mesa Vulkan ICDs fail at ICD enumeration. There is
no workaround short of patching the WSL2 kernel.

Use one of these alternatives:

| Option | Notes |
|---|---|
| Native Ubuntu 22.04 / 24.04 | Recommended; full driver support |
| Cloud GPU via vast.ai | `bash scripts/vast_launch.sh --isaac` (see below) |

---

## Isaac Lab Installation

Isaac Lab is **not on PyPI**. Install it by cloning from GitHub into Isaac
Sim's Python environment:

```bash
git clone --depth 1 https://github.com/isaac-sim/IsaacLab.git /workspace/isaaclab_src
/isaac-sim/python.sh -m pip install -e /workspace/isaaclab_src/source/isaaclab
/isaac-sim/python.sh -m pip install "skrl>=2.0.0" "gymnasium>=1.0.0" h5py
```

`h5py` is required by the `isaaclab_tasks` extension at startup
(`recorder_manager` imports it). Without it, Isaac Sim exits silently.

The `vast_launch.sh --isaac` flag automates all of this on a fresh instance.

---

## Isaac Lab "Direct" Workflow: env_cfg_entry_point

Isaac Lab's *direct* environments store their config class in the Gymnasium
registry under the key `"env_cfg_entry_point"` rather than passing it as a
normal kwarg. Gymnasium does **not** auto-resolve this — it must be done
manually before calling `gym.make()`:

```python
import importlib
spec = gym.spec(args.env_id)
cfg_entry = (spec.kwargs or {}).get("env_cfg_entry_point", "")
if cfg_entry:
    module_path, class_name = cfg_entry.split(":")
    env_cfg = getattr(importlib.import_module(module_path), class_name)()
    # Cap num_envs to avoid OOM on camera obs (see VRAM section below)
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs"):
        env_cfg.scene.num_envs = min(env_cfg.scene.num_envs, 32)
    env = wrap_env(gym.make(args.env_id, cfg=env_cfg))
else:
    env = wrap_env(gym.make(args.env_id))
```

Without this, `gym.make()` raises:
```
TypeError: CartpoleCameraEnv.__init__() missing 1 required positional argument: 'cfg'
```

---

## Camera Observation Format: HWC vs CHW

Isaac Lab camera environments return observations in **HWC format** `(H, W, C)`,
e.g. `(100, 100, 3)`. PyTorch `Conv2d` expects **CHW format** `(C, H, W)`.

`shared/models.py` handles this transparently via `_parse_image_shape()`: if the
last dimension is a small channel count (1, 3, 4) and the spatial dims are
larger, the observation is detected as HWC and permuted to CHW inside `compute()`:

```python
if self._hwc:
    obs = obs.permute(0, 3, 1, 2).contiguous()
```

No manual shape configuration is needed when switching between Isaac Lab
and standard Gymnasium pixel environments.

---

## VRAM Budget for Camera Environments

Isaac Lab defaults to ~512 parallel environments. With camera obs `(100, 100, 3)`:

```
rollout buffer = num_envs × memory_size × obs_size × 4 bytes
512 × 1024 × 30,000 × 4 = ~59 GB  ← exceeds RTX 4090 (24 GB)
 32 × 1024 × 30,000 × 4 = ~3.8 GB ← safe
```

`train.py` caps `env_cfg.scene.num_envs` to 32 automatically. Adjust if you
have more VRAM.

---

## What Transfers Unchanged From Examples 05–06

| Component | Transferred as-is? |
|---|---|
| `CNNGaussianPolicy` / `CNNValue` | ✅ yes (HWC auto-detected) |
| `build_ppo_agent()` with overrides | ✅ yes |
| `SequentialTrainer` training loop | ✅ yes |
| `RunningStandardScaler` preprocessors | ✅ yes |
| `--checkpoint` resume logic | ✅ yes |
| `gym.make()` + `wrap_env()` | ✅ yes (with cfg resolution) |
| Environment creation (pre-AppLauncher) | ❌ must be after AppLauncher |
| `env.close()` | ✅ yes |
| `simulation_app.close()` | ❌ Isaac-specific — new requirement |

---

## Isaac Lab vs Gymnasium: Key Differences

| Aspect | Gymnasium (Ex. 01–06) | Isaac Lab (Ex. 07) |
|---|---|---|
| Physics | CPU box2d / MuJoCo | GPU rigid-body (PhysX) |
| Parallelism | 1 env per process | 100s of envs on 1 GPU |
| Startup time | < 1 s | 20–40 s (Omniverse boot) |
| Camera obs format | CHW `(C, H, W)` | HWC `(H, W, C)` |
| Camera obs speed | Optional, slow | Native, GPU-rendered |
| Requires GPU | Optional | Required (Vulkan) |
| Install size | ~200 MB | ~50 GB |
| Env config | `gym.make(id)` | `gym.make(id, cfg=env_cfg)` |

---

## PPO Overrides Used Here

```python
overrides = {
    "learning_rate": 1e-4,   # conservative; Isaac envs have complex reward landscapes
    "mini_batches": 8,        # smaller batches suit high-dim camera observations
}
```

Same values as Example 06 — intentionally. The CNN-on-pixels settings that
worked for CarRacing generalise to Isaac Lab out of the box.

---

## Requirements

- Isaac Sim 5.1.0+ (`nvcr.io/nvidia/isaac-sim:5.1.0`)
- Isaac Lab cloned from GitHub (not on PyPI — see Installation above)
- Python deps installed into Isaac Sim's Python: `skrl>=2.0.0`, `gymnasium>=1.0.0`, `h5py`
- Native Linux with NVIDIA GPU and working Vulkan (WSL2 not supported)

---

## Run

> ⚠ **WSL2 is not supported.** Isaac Sim requires Vulkan GPU rendering.

All commands must be run from the **project root** using the `-m` flag.

```bash
# Option A — vast.ai cloud GPU (easiest)
bash scripts/vast_launch.sh --gpu RTX_4090 --isaac --max-dph 1.00 \
    -- python -m examples.07_isaac_transfer.train --timesteps 200000
# Opens TensorBoard tunnel automatically: http://localhost:6006

# Option B — native Isaac Lab install
/isaac-sim/python.sh -m examples.07_isaac_transfer.train --timesteps 200000

# Resume from checkpoint
/isaac-sim/python.sh -m examples.07_isaac_transfer.train \
    --timesteps 200000 \
    --checkpoint runs/07_isaac_transfer/07_isaac_transfer/checkpoints/agent_5000.pt

# List all available Isaac Lab environments (must run inside Isaac Sim Python)
/isaac-sim/python.sh -c "
import isaaclab_tasks, gymnasium
print('\n'.join(k for k in gymnasium.envs.registry if 'Isaac' in k))
"
```

### Monitoring

`vast_launch.sh` starts TensorBoard on the remote and opens the SSH tunnel
automatically. For manual setup:

```bash
# On remote
tensorboard --logdir /workspace/reinforced/runs --host 127.0.0.1 --port 6006 &

# On local machine
ssh -p <PORT> -L 6006:127.0.0.1:6006 -N root@<HOST> &
# then open http://localhost:6006
```

---

## Key Learning Point

Good algorithm abstractions transfer across simulators with minimal changes.
The entire PPO + CNN stack from Examples 05–06 works in Isaac Lab with four
Isaac-specific additions:

1. `AppLauncher(...)` — boots Omniverse before any omni.* import
2. `import isaaclab_tasks` — registers Isaac Lab envs as a side-effect
3. Manual `env_cfg_entry_point` resolution before `gym.make()`
4. `simulation_app.close()` — graceful Kit shutdown

Everything else — CNN models, PPO agent, trainer, checkpointing — is identical.
