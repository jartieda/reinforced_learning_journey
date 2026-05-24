# Shared Modules

All examples import from this folder. Understanding these modules is
understanding 90% of the codebase.

---

## `models.py` — Neural Network Definitions

### MLP models (Examples 01–03)

```
MLPGaussianPolicy(GaussianMixin, Model)
  Input:  flat observation vector
  Output: mean action + learned log_std_parameter
  Use:    continuous-action policy (actor)

MLPValue(DeterministicMixin, Model)
  Input:  flat observation vector
  Output: scalar V(s)
  Use:    value function (critic)
```

**GaussianMixin** wraps the network output in a diagonal Gaussian N(μ, σ).
During training actions are **sampled** (stochastic exploration); during eval
the **mean** μ is used (deterministic). `log_std_parameter` is a learnable
per-action-dimension vector that controls exploration noise.

**DeterministicMixin** is used for the value network — always returns a scalar.

### CNN models (Examples 05–06)

```
CNNEncoder(nn.Module)
  Input:  (batch, C, H, W) image tensor
  Output: flat latent feature vector
  Layers: Conv2d(8×8 s4) → Conv2d(4×4 s2) → Conv2d(3×3 s1) → Flatten
  Note:   output size inferred dynamically via a dummy forward pass

CNNGaussianPolicy(GaussianMixin, Model)
  Input:  (batch, C, H, W) — reshapes from skrl's flat format internally
  Output: mean action + log_std_parameter
  Note:   normalises pixels /255 if max > 1.0

CNNValue(DeterministicMixin, Model)
  Input:  (batch, C, H, W)
  Output: scalar V(s)
```

**Why separate encoders?** Policy and value share the observation but are
optimised with different losses. Separate encoders avoid interference and keep
the code simpler than weight-sharing.

**Why `Model.__init__(self, *args, **kwargs)` instead of `super()`?** skrl's
multiple-inheritance requires that `Model.__init__` and each Mixin's `__init__`
are called explicitly in order. `super()` MRO does not guarantee this.

---

## `ppo.py` — PPO Agent Builder

`build_ppo_agent()` constructs an skrl `PPO` agent with teaching-oriented
defaults and accepts an `overrides` dict for per-example customisation.

```python
agent = build_ppo_agent(
    env=env, device=device, models=models,
    experiment_directory="runs/02",
    experiment_name="my_run",
    overrides={"learning_rate": 1e-4, "entropy_loss_scale": 0.01},
)
```

Internally this sets up:
- **`RandomMemory`** — the rollout buffer (on-policy; discarded after each update).
  Size is `rollouts × num_envs`.
- **`RunningStandardScaler`** — normalises observations and value targets online
  using a running mean/variance.
- **`experiment` sub-dict** — checkpoint and TensorBoard log location and frequency.

---

## `wrappers.py` — Image Preprocessing Chain

```
gym.make("CarRacing-v3")        → (96, 96, 3) uint8  HWC RGB
  └─ ToGrayscale               → (1, 96, 96)  uint8  CHW Gray
       └─ FrameStackCHW(n=4)   → (4, 96, 96)  uint8  CHW 4-frame stack
```

| Class | Input | Output | Purpose |
|-------|-------|--------|---------|
| `ToChannelFirst` | (H,W,C) | (C,H,W) | Reorder axes for PyTorch |
| `ToGrayscale` | (H,W,3) RGB | (1,H,W) | Reduce channels; colour rarely needed |
| `FrameStackCHW` | (C,H,W) | (C×N,H,W) | Temporal context via stacking |

`make_carracing_pixels(frame_stack=N)` wires the full chain.

**Why not `gymnasium.wrappers.GrayScaleObservation`?** It outputs `(H,W)`,
requiring a separate unsqueeze. `ToGrayscale` outputs `(1,H,W)` directly.

---

## `cli.py` — Unified Command-Line Interface

All train scripts share `build_train_parser` / `RunConfig`. All eval scripts
share `build_eval_parser`.

```
RunConfig
  env_id          str     Environment to train on
  timesteps       int     Total environment steps
  seed            int     RNG seed for reproducibility
  checkpoint      str?    Path to resume from
  experiment_name str     Tag for this run in logs/checkpoints
  headless        bool    Suppress rendering during training
```

Eval scripts additionally accept `--render`, `--record`, `--video-dir`.

---

## `utils.py` — Common Helpers

| Function | Purpose |
|----------|---------|
| `set_seed(seed)` | Seeds Python, NumPy, and PyTorch |
| `default_device()` | Returns `"cuda"` if available, else `"cpu"` |
| `ensure_dir(path)` | `mkdir -p`; returns a `Path` |
| `print_run_banner(...)` | Prints run metadata header |
| `print_next_steps(...)` | Prints post-training commands |

---

## `plot.py` — Training Curve Plotter

Reads TensorBoard event files and generates a matplotlib figure with reward,
loss, and entropy panels.

```bash
python -m examples.shared.plot runs/02_ppo_classic_control/02_ppo_classic_control
python -m examples.shared.plot runs/ --all
```

**What to look for:**
- Rising reward → learning
- High variance → increase `mini_batches` or lower `learning_rate`
- Plateau → more timesteps or tune entropy/LR
- Sudden drop → gradient explosion; lower LR or enable `grad_norm_clip`

---

## `export.py` — Policy Export and Standalone Inference

Exports trained weights to a plain `.pt` file and runs inference **without skrl**.

```bash
# MLP policy (examples 01–03)
python -m examples.shared.export \
    --checkpoint runs/02_ppo_classic_control/.../agent_5000.pt \
    --env-id Pendulum-v1 --model mlp --record --video-dir videos/

# CNN policy (examples 05–06)
python -m examples.shared.export \
    --checkpoint runs/05_ppo_cnn/.../agent_5000.pt \
    --env-id CarRacing-v3 --model cnn --record --video-dir videos/
```

Standalone inference (no skrl dependency):

```python
import torch
import gymnasium as gym
from examples.shared.models import MLPGaussianPolicy

env = gym.make("Pendulum-v1")
model = MLPGaussianPolicy(
    observation_space=env.observation_space,
    action_space=env.action_space,
    device="cpu",
)
model.load_state_dict(torch.load("exported_policy.pt", map_location="cpu"))
model.eval()

obs, _ = env.reset()
t_obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    action, _, _ = model.act({"observations": t_obs}, role="policy")
```
