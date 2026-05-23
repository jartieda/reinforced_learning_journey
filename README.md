# SKRL Learning Series

A hands-on progression for learning `skrl` with PyTorch — from the minimal
PPO loop to convolutional policies on image observations.

Read [docs/curriculum.md](docs/curriculum.md) for concept-by-concept learning goals.

---

## Table of Contents

1. [Setup](#setup)
2. [Reinforcement Learning Primer](#reinforcement-learning-primer)
3. [PPO — The Algorithm](#ppo--the-algorithm)
4. [Example Walkthrough](#example-walkthrough)
5. [Shared Modules Reference](#shared-modules-reference)
6. [Visualising Training Results](#visualising-training-results)
7. [Watching and Recording the Trained Policy](#watching-and-recording-the-trained-policy)
8. [Exporting the Policy](#exporting-the-policy)
9. [Example 07 — Isaac Lab](#example-07--isaac-lab-native-linux--cloud-gpu-only)

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **WSL2 note** — Examples 01–06 work on WSL2. Example 07 (Isaac Sim) requires
> native Ubuntu 22.04/24.04 or a cloud GPU; Isaac Sim does not support WSL2.

---

## Reinforcement Learning Primer

Before diving into code, here is the minimal theory you need.

### The RL Loop

An **agent** repeatedly interacts with an **environment**:

```
┌─────────┐  action aₜ   ┌─────────────┐
│  Agent  │ ───────────► │ Environment │
│         │ ◄─────────── │             │
└─────────┘  obs oₜ₊₁    └─────────────┘
              reward rₜ
```

At every step the agent observes the state, picks an action, and receives a
scalar reward. The goal is to learn a **policy** π(a|s) — a probability
distribution over actions given the current observation — that maximises the
sum of future discounted rewards:

$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

where **γ ∈ (0,1)** is the discount factor. A high γ (e.g. 0.99) makes the
agent plan far into the future; a low γ makes it short-sighted.

### Actor-Critic

PPO is an **actor-critic** algorithm. Two networks are trained jointly:

| Network | Output | Role |
|---------|--------|------|
| **Policy (actor)** | π(a\|s) — a probability distribution over actions | Decides what to do |
| **Value (critic)** | V(s) — a scalar estimate of future return | Evaluates how good the current state is |

The critic's estimates are used to compute **advantages** — whether a
particular action was better or worse than average — which guide the actor's
updates.

### Advantage and GAE

The **advantage** $A_t = Q(s_t, a_t) - V(s_t)$ measures how much better
action $a_t$ is compared to the average action from state $s_t$.

In practice we use **Generalised Advantage Estimation (GAE)** which computes a
weighted mixture of multi-step returns:

$$A_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l},
\quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

- **λ → 0**: relies heavily on the critic (low variance, high bias).
- **λ → 1**: relies on actual returns (low bias, high variance).
- **λ = 0.95** is a widely used middle ground.

---

## PPO — The Algorithm

PPO (Proximal Policy Optimization, Schulman et al. 2017) adds a crucial
stability constraint to vanilla policy gradient: it prevents the new policy
from deviating too far from the old policy in a single update.

### Clipped Surrogate Objective

The ratio $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$
measures how much the policy changed. PPO clips this ratio:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[
  \min\!\left( r_t(\theta)\, A_t,\;
               \text{clip}(r_t(\theta), 1{-}\epsilon, 1{+}\epsilon)\, A_t \right)
\right]$$

The clip parameter **ε = 0.2** (configured as `ratio_clip`) ensures no single
update shifts the policy too drastically, which is the key source of PPO's
stability over older policy gradient methods.

### Full Training Loop

```
For each iteration:
  1. Collect rollout_size transitions using the current policy (no gradient)
  2. Compute advantages with GAE
  3. For learning_epochs passes over the collected batch:
       a. Split into mini_batches
       b. Compute clipped policy loss, value loss, entropy bonus
       c. Clip gradient norm; apply Adam step
  4. Discard rollout (on-policy — data is used once then thrown away)
```

Key hyperparameters and their effect:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `discount_factor` (γ) | 0.99 | Planning horizon; higher = more far-sighted |
| `gae_lambda` (λ) | 0.95 | Bias-variance trade-off for advantages |
| `ratio_clip` (ε) | 0.2 | How much the policy can change per update |
| `learning_rate` | 3e-4 | Step size; too high → diverge, too low → slow |
| `learning_epochs` | 8 | Reuse of each rollout; more = efficient, can overfit |
| `mini_batches` | 4 | Stochastic gradient noise; usually 4–32 |
| `entropy_loss_scale` | 0.0 | Exploration bonus; >0 discourages premature convergence |
| `grad_norm_clip` | 0.5 | Prevents exploding gradients |

---

## Example Walkthrough

### Example 01 — Minimal PPO Loop

**Environment:** `Pendulum-v1` (3-D state, 1-D continuous action)  
**Concept:** the smallest possible end-to-end RL script

This example collapses everything — model definition, PPO config, training loop
— into a single file. The goal is to see every moving part in one place before
they are split into reusable modules in later examples.

Key things to notice:
- `wrap_env()` converts a raw Gymnasium env into skrl's interface.
- `SequentialTrainer` owns the train loop; you pass it an agent and a timestep
  budget. It handles rollout collection, PPO updates, and checkpoint saving.
- Checkpoints and TensorBoard logs appear automatically under `runs/`.

```bash
python -m examples.01_basics_loop.main --timesteps 10000
```

---

### Example 02 — Train / Eval Split

**Environment:** `Pendulum-v1`  
**Concept:** separate training from evaluation; reproducible comparison

A trained policy should be evaluated under a **fixed protocol** rather than
inspected during the noisy training process. `train.py` runs PPO with
stochastic actions (exploration); `eval.py` reloads the checkpoint and runs
the policy **deterministically** (mean action, no noise) for a set number of
episodes.

Separating the scripts also enforces a good habit: the evaluation code is
independent of the training code, so you can change your training loop without
accidentally changing how you measure performance.

```bash
python -m examples.02_ppo_classic_control.train --timesteps 20000
python -m examples.02_ppo_classic_control.eval \
    --checkpoint runs/02_ppo_classic_control/.../checkpoints/agent_5000.pt \
    --render
```

---

### Example 03 — Stability Settings

**Environment:** `CartPole-v1` (discrete) / `Pendulum-v1`  
**Concept:** tuning PPO for robust, reproducible training

Default hyperparameters work for simple tasks but often fail on harder ones.
This example shows three key stability knobs:

| Setting | Value | Why |
|---------|-------|-----|
| `learning_rate` | 1e-4 | Halved from default; prevents overshooting |
| `learning_epochs` | 10 | More passes per rollout — efficient for expensive envs |
| `entropy_loss_scale` | 0.01 | Small entropy bonus keeps the policy from collapsing to one action too early |
| `grad_norm_clip` | 0.5 | Clips the global gradient norm; essential for pixel-based training |

**Why does gradient clipping help?** Without it, a single catastrophic rollout
can produce a very large gradient that pushes the network weights far off a
good region of parameter space. Clipping the norm to 0.5 bounds how much one
bad batch can hurt.

---

### Example 04 — Pixel Observation Pipeline

**Environment:** `CarRacing-v3` (96×96 RGB images → 3 continuous actions)  
**Concept:** image preprocessing before the network sees a pixel

Raw images from Gymnasium arrive as `(H, W, C)` numpy arrays with uint8 values
in [0, 255]. Two things must happen before a neural network can train on them:

1. **Channel-first conversion (HWC → CHW):** PyTorch convolutions expect
   `(batch, channels, height, width)`. The `ToChannelFirst` wrapper handles this.

2. **Grayscale reduction:** Converting RGB to a single luminance channel
   reduces the input from 3×96×96 = 27,648 values to 1×96×96 = 9,216.
   Color is usually irrelevant for driving control, so this is nearly free
   dimensionality reduction.

3. **Normalisation (in the model):** The pixel values are divided by 255.0
   inside `compute()` so the network always sees inputs in [0,1]. This gives
   the Adam optimizer a stable, well-conditioned gradient scale.

In this example an **MLP** is used on the flattened image — intentionally
inefficient, to make the CNN improvement in Example 05 obvious.

---

### Example 05 — CNN Encoder

**Environment:** `CarRacing-v3` (grayscale 1×96×96)  
**Concept:** convolutional feature extraction from images

A fully-connected MLP on raw pixels is wasteful: it has no inductive bias for
spatial structure and must learn redundantly for every pixel position.

A **Convolutional Neural Network (CNN)** fixes this with two key properties:

- **Local connectivity:** each filter looks at a small patch of the image,
  learning a local feature detector (edge, curve, etc.).
- **Translation equivariance:** the same filter is applied everywhere, so a
  feature learned in one part of the image generalises to all positions.

The `CNNEncoder` in `shared/models.py` uses the classic Atari architecture:

```
Input  1×96×96
  Conv2d(1→32, 8×8, stride=4) → ReLU   # detects coarse structures
  Conv2d(32→64, 4×4, stride=2) → ReLU  # detects finer patterns
  Conv2d(64→64, 3×3, stride=1) → ReLU  # detects detail
  Flatten → latent vector (size inferred dynamically)
```

The latent vector feeds into a small MLP head that outputs action means.
The value network uses an identical encoder (weights not shared) to estimate V(s).

---

### Example 06 — Frame Stacking

**Environment:** `CarRacing-v3` (4 stacked grayscale frames → 4×96×96)  
**Concept:** restoring the Markov property for motion-dependent tasks

A single image observation violates the **Markov property**: you cannot tell
whether the car is accelerating or braking from one frame alone. The policy
needs velocity information to make sensible decisions.

**Frame stacking** concatenates the last *N* frames along the channel axis:

```
Frame t-3: grayscale 1×96×96 ┐
Frame t-2: grayscale 1×96×96 │  →  4×96×96 input tensor
Frame t-1: grayscale 1×96×96 │
Frame t  : grayscale 1×96×96 ┘
```

The CNN encoder now sees temporal context without needing a recurrent network.
With N=4 frames the agent can implicitly infer velocity, acceleration, and
short-term trajectory.

This example also tightens the PPO settings (lower LR, more mini-batches) to
reflect that pixel-based training requires more conservative updates.

---

### Example 07 — Isaac Lab Transfer

**Environment:** Isaac Lab camera-observation task  
**Concept:** the same PPO+CNN abstractions transfer to a physics simulator

The algorithmic code (PPO config, CNN model, training loop) is essentially
unchanged from Examples 05/06. The only Isaac-specific changes are:

1. **`AppLauncher`** — Isaac Sim is an Omniverse Kit application, not a plain
   Python process. The `AppLauncher` must boot the Kit runtime *before* any
   `omni.*` or `isaaclab.*` import. This is the first and only simulator-specific
   bootstrap step.

2. **`isaaclab_tasks` import** — registers Isaac Lab's gymnasium environments
   so that `gym.make("Isaac-Camera-Reach-v0")` works.

3. **`simulation_app.close()`** — cleanly shuts down the Kit process when
   training ends.

The key lesson: **RL algorithm abstractions are portable**. The PPO agent,
CNN models, and training loop are completely simulator-agnostic. Switching
from Gymnasium to Isaac Lab costs three lines of code.

> ⚠ Isaac Sim requires Vulkan GPU rendering — not available on WSL2.
> See the [Isaac Lab section](#example-07--isaac-lab-native-linux--cloud-gpu-only) below.

---

## Shared Modules Reference

All examples share a common set of utilities under `examples/shared/`.
Understanding these modules is understanding 90% of the codebase.

### `models.py` — Neural Network Definitions

Contains four model classes built on skrl's `Model` base:

#### MLP models (Examples 01–03)

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

**GaussianMixin** wraps the network output in a diagonal Gaussian distribution
N(μ, σ). During training, actions are **sampled** from this distribution
(stochastic exploration). During evaluation, the **mean** μ is used
(deterministic exploitation). The `log_std_parameter` is a learnable vector —
one value per action dimension — that controls the exploration noise.

**DeterministicMixin** is used for the value network: it always returns a
single deterministic scalar, not a distribution.

#### CNN models (Examples 05–06)

```
CNNEncoder(nn.Module)
  Input:  (batch, C, H, W) image tensor
  Output: flat latent feature vector
  Layers: Conv2d(8×8 s4) → Conv2d(4×4 s2) → Conv2d(3×3 s1) → Flatten
  Note:   output size is inferred dynamically via a dummy forward pass

CNNGaussianPolicy(GaussianMixin, Model)
  Input:  (batch, C, H, W) — reshapes from skrl's flat format internally
  Output: mean action + log_std_parameter
  Note:   normalises pixels /255 if max > 1.0

CNNValue(DeterministicMixin, Model)
  Input:  (batch, C, H, W)
  Output: scalar V(s)
```

**Why separate encoders?** Policy and value share the same observation but are
optimised with different loss functions. Sharing weights is possible (and used
in some implementations) but separate encoders keep the code simpler and avoid
interference between the two objectives.

**Why `Model.__init__(self, *args, **kwargs)` instead of `super()`?** skrl's
multiple-inheritance requires that `Model.__init__` and each Mixin's `__init__`
are called explicitly, in order. Python's `super()` MRO does not guarantee the
right call sequence here.

---

### `ppo.py` — PPO Agent Builder

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
- **`RandomMemory`** — the rollout buffer. On-policy PPO fills this buffer with
  fresh transitions each iteration and discards it after the update. Size is
  `rollouts × num_envs`.
- **`RunningStandardScaler`** — normalises observations and value targets
  online using a running mean/variance. This prevents the network from seeing
  wildly different input scales across training.
- **`experiment` sub-dict** — tells skrl where to write checkpoints and
  TensorBoard logs, and at what frequency.

---

### `wrappers.py` — Image Preprocessing Chain

Three wrapper classes and one convenience factory, applied in order:

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

`make_carracing_pixels(frame_stack=N)` is a convenience function that wires the
full chain.

**Why not use `gymnasium.wrappers.GrayScaleObservation`?** That wrapper outputs
`(H,W)` which then needs a separate channel-unsqueeze. The custom `ToGrayscale`
outputs `(1,H,W)` directly and keeps the chain cleaner.

---

### `cli.py` — Unified Command-Line Interface

All train scripts share the same parser (`build_train_parser`) and return type
(`RunConfig`). All eval scripts share `build_eval_parser`.

```
RunConfig
  env_id          str     Environment to train on
  timesteps       int     Total environment steps
  seed            int     RNG seed for reproducibility
  checkpoint      str?    Path to resume from
  experiment_name str     Tag for this run in TensorBoard / checkpoints
  headless        bool    Suppress rendering during training
```

**Why a shared CLI?** A learner can move from folder to folder without
relearning flags. Every example uses `--timesteps`, `--env-id`, `--seed`, and
`--checkpoint` consistently.

Eval scripts additionally accept:
- `--render` — open a live window to watch the policy play
- `--record` — capture an MP4 video via `gymnasium.wrappers.RecordVideo`
- `--video-dir` — destination for recorded videos

---

### `utils.py` — Common Helpers

| Function | Purpose |
|----------|---------|
| `set_seed(seed)` | Seeds Python, NumPy, and PyTorch for reproducibility |
| `default_device()` | Returns `"cuda"` if a GPU is available, else `"cpu"` |
| `ensure_dir(path)` | `mkdir -p` equivalent; returns a `Path` object |
| `print_run_banner(...)` | Prints a header line with run metadata |
| `print_next_steps(...)` | Prints post-training commands (plot, eval, export) |

**Why seed everything?** RL training has multiple sources of randomness: initial
network weights, environment resets, action sampling, and minibatch shuffling.
Seeding all of them makes results reproducible across machines.

---

### `plot.py` — Training Curve Plotter

Reads TensorBoard event files written by skrl and generates a multi-panel
matplotlib figure:

- **Reward panel** — episode return over timesteps (the primary learning signal)
- **Loss panel** — policy loss, value loss, entropy (shows optimisation health)

```bash
python -m examples.shared.plot runs/02_ppo_classic_control/02_ppo_classic_control
python -m examples.shared.plot runs/ --all      # plot every run at once
```

**What to look for in the reward panel:**
- Rising trend → the agent is learning
- High variance → try increasing `mini_batches` or reducing `learning_rate`
- Plateau → try more timesteps, or tune entropy/learning rate
- Sudden drop → gradient explosion; reduce `learning_rate` or enable `grad_norm_clip`

---

### `export.py` — Policy Export and Standalone Inference

Demonstrates how to use a trained policy **without skrl**:

1. Loads the checkpoint and extracts the policy state-dict.
2. Reconstructs the model class with plain PyTorch.
3. Runs inference in a loop — either rendering to screen (`--render`) or
   recording video (`--record`).
4. Saves `exported_policy.pt` — the raw weights, loadable anywhere PyTorch runs.

```bash
python -m examples.shared.export \
    --checkpoint runs/02.../checkpoints/agent_5000.pt \
    --env-id Pendulum-v1 --model mlp --render
```

This is the deployment pattern: train with skrl for its convenient logging and
memory management, then ship only the network weights.

---

## Visualising Training Results

### Live dashboard (TensorBoard)

```bash
tensorboard --logdir runs/
# open http://localhost:6006
```

Rewards, losses, and entropy are logged automatically by every training script.

### Static PNG plots

```bash
python -m examples.shared.plot runs/02_ppo_classic_control/02_ppo_classic_control
python -m examples.shared.plot runs/ --all
```

The PNG is saved as `training_curves.png` next to the event files.

---

## Watching and Recording the Trained Policy

```bash
# Watch live
python -m examples.02_ppo_classic_control.eval \
    --checkpoint runs/02_ppo_classic_control/.../checkpoints/agent_5000.pt \
    --env-id Pendulum-v1 --render

# Record MP4
python -m examples.02_ppo_classic_control.eval \
    --checkpoint runs/02_ppo_classic_control/.../checkpoints/agent_5000.pt \
    --env-id Pendulum-v1 --record --video-dir videos/pendulum
```

---

## Exporting the Policy

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

Standalone inference after export (no skrl dependency):

```python
import torch, gymnasium as gym
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
with torch.no_grad():
    action, _, _ = model.act(
        {"observations": torch.tensor(obs).float().unsqueeze(0)}, role="policy"
    )
```

---

## Post-Training Cheat Sheet

Every training script prints this when it finishes:

```
========================================================================
Training complete! What to do next:

  1. View training curves:   tensorboard --logdir runs/02_ppo_classic_control
  2. Generate PNG:           python -m examples.shared.plot runs/02_ppo_classic_control
  3. Watch policy play:      python -m examples.02_ppo_classic_control.eval \
                               --checkpoint ... --render
  4. Record video + export:  python -m examples.shared.export \
                               --checkpoint ... --record
========================================================================
```

---

## Example 07 — Isaac Lab (native Linux / cloud GPU only)

Isaac Sim requires Vulkan GPU rendering which is not available on WSL2.

**Option A — Docker on native Ubuntu:**
```bash
sudo apt-get install -y nvidia-container-toolkit && sudo systemctl restart docker
bash examples/07_isaac_transfer/docker_train.sh --timesteps 50000
```

**Option B — Cloud GPU (AWS g4dn.xlarge, GCP a2-highgpu-1g, etc.):**
```bash
bash examples/07_isaac_transfer/docker_train.sh --timesteps 50000
```

**Option C — Native Ubuntu install:**
Follow the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/),
then run:
```bash
python -m examples.07_isaac_transfer.train --timesteps 200000
```

---

## Visualising Training Results

### Live dashboard (TensorBoard)

```bash
tensorboard --logdir runs/
# open http://localhost:6006
```

Rewards, losses, and entropy are logged automatically by every training script.

### Static PNG plots

Generate a training-curve image without TensorBoard:

```bash
# Single run
python -m examples.shared.plot runs/02_ppo_classic_control/02_ppo_classic_control

# Every run at once
python -m examples.shared.plot runs/ --all
```

The PNG is saved next to the event files as `training_curves.png`.

---

## Watching and Recording the Trained Policy

### Watch live (opens a window)

```bash
python -m examples.02_ppo_classic_control.eval \
    --checkpoint runs/02_ppo_classic_control/02_ppo_classic_control/checkpoints/agent_5000.pt \
    --env-id Pendulum-v1 \
    --render
```

### Record an MP4 video

```bash
python -m examples.02_ppo_classic_control.eval \
    --checkpoint runs/02_ppo_classic_control/02_ppo_classic_control/checkpoints/agent_5000.pt \
    --env-id Pendulum-v1 \
    --record --video-dir videos/pendulum
```

The `--render` / `--record` flags work on the eval scripts for examples 02 and 03.  
For pixel examples (04–06) use the export tool below.

---

## Exporting the Policy

The `examples.shared.export` tool exports the trained network weights to a
plain `.pt` file and runs standalone inference **without skrl**.

```bash
# Classic-control (MLP policy)
python -m examples.shared.export \
    --checkpoint runs/02_ppo_classic_control/02_ppo_classic_control/checkpoints/agent_5000.pt \
    --env-id Pendulum-v1 \
    --model mlp \
    --record --video-dir videos/pendulum

# Pixel-based (CNN policy)
python -m examples.shared.export \
    --checkpoint runs/05_ppo_cnn/05_ppo_cnn/checkpoints/agent_5000.pt \
    --env-id CarRacing-v3 \
    --model cnn \
    --record --video-dir videos/carracing
```

This produces `exported_policy.pt` — a state-dict you can load with plain
PyTorch, with no dependency on skrl:

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

---

## Post-Training Cheat Sheet

Every training script prints the exact commands to run when it finishes:

```
========================================================================
Training complete! What to do next:

  1. View training curves (TensorBoard live view):
       tensorboard --logdir runs/02_ppo_classic_control
  2. Generate a static PNG summary:
       python -m examples.shared.plot runs/02_ppo_classic_control
  3. Watch the trained policy play:
       python -m examples.02_ppo_classic_control.eval --checkpoint ... --render
  4. Record a video + export weights for standalone deployment:
       python -m examples.shared.export --checkpoint ... --record
========================================================================
```

---

## Example 07 — Isaac Lab (native Linux / cloud GPU only)

Isaac Sim requires Vulkan GPU rendering which is not available on WSL2.

**Option A — Docker on native Ubuntu:**
```bash
# Install NVIDIA Container Toolkit first
sudo apt-get install -y nvidia-container-toolkit && sudo systemctl restart docker

bash examples/07_isaac_transfer/docker_train.sh --timesteps 50000
```

**Option B — Cloud GPU (AWS g4dn.xlarge, GCP a2-highgpu-1g, etc.):**
```bash
# Pull the pre-built container and run
bash examples/07_isaac_transfer/docker_train.sh --timesteps 50000
```

**Option C — Native Ubuntu install:**
Follow the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/),
then run:
```bash
python -m examples.07_isaac_transfer.train --timesteps 200000
```

---

## Cloud Training on vast.ai

[vast.ai](https://vast.ai) rents GPU instances by the hour. A full training run
(250 000 steps on an RTX 3090) costs roughly **$0.10–0.30** depending on the
GPU and spot availability.

### One-time setup

```bash
# 1. Get your API key from https://cloud.vast.ai/  (Account → API Key)
echo "VAST_AI_API_KEY=your_key_here" >> .env

# 2. Make sure your SSH public key is registered on vast.ai
#    (Account → SSH Keys → Add Key)
#    Your public key:
ssh-keygen -y -f ~/.ssh/id_rsa
```

### Launch, monitor, and destroy

```bash
# ── Launch ────────────────────────────────────────────────────────────────

# Train example 05 (CNN on CarRacing) on a 3090
bash scripts/vast_launch.sh \
    -- python -m examples.05_ppo_cnn.train --timesteps 250000 --headless

# Train example 06 (frame stacking) on a 4090 with higher budget
bash scripts/vast_launch.sh --gpu RTX_4090 --max-dph 0.80 \
    -- python -m examples.06_pixels_harder_task.train --timesteps 400000 --headless

# Train example 07 (Isaac Lab) — uses the Isaac Sim Docker image automatically
bash scripts/vast_launch.sh --gpu RTX_4090 --isaac \
    -- python -m examples.07_isaac_transfer.train --timesteps 200000

# Quick smoke-test on any cheap GPU
bash scripts/vast_launch.sh --gpu RTX_3080 --max-dph 0.25 \
    -- python -m examples.02_ppo_classic_control.train --timesteps 20000 --headless

# ── Monitor ───────────────────────────────────────────────────────────────

# Tail the live training log
bash scripts/vast_sync.sh --log

# Sync checkpoints + TensorBoard logs to local without stopping training
bash scripts/vast_sync.sh

# After syncing, inspect locally
tensorboard --logdir runs/

# ── Destroy ───────────────────────────────────────────────────────────────

# Final sync + confirmation prompt + destroy
bash scripts/vast_destroy.sh

# Destroy immediately (skip sync)
bash scripts/vast_destroy.sh --now
```

### What the scripts do

| Script | Action |
|---|---|
| `scripts/vast_launch.sh` | Finds cheapest matching offer, creates instance, uploads code, installs deps, starts training in a detached `tmux` session |
| `scripts/vast_sync.sh` | `rsync` `runs/` and `videos/` back to local machine; safe to run while training |
| `scripts/vast_destroy.sh` | Final sync, confirmation prompt, then destroy — **irreversible** |

### Choosing a GPU

| GPU | Typical $/hr | Good for |
|---|---|---|
| RTX 3080 | $0.15–0.25 | Examples 01–04 |
| RTX 3090 | $0.20–0.40 | Examples 05–06 (recommended) |
| RTX 4090 | $0.40–0.80 | Example 07, long runs |
| A100 40 GB | $1.00–2.00 | Isaac Lab at scale |

Browse live prices: `vastai search offers "gpu_name=RTX_3090"`
