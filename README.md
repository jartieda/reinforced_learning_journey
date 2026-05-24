# SKRL Learning Series

A hands-on progression for learning `skrl` with PyTorch — from the minimal
PPO loop to convolutional policies on image observations.

Read [docs/curriculum.md](docs/curriculum.md) for concept-by-concept learning goals.

---

## Table of Contents

1. [Setup](#setup)
2. [Examples](#examples)
3. [Monitoring Training](#monitoring-training)
4. [Watching and Recording](#watching-and-recording-the-trained-policy)
5. [Exporting the Policy](#exporting-the-policy)
6. [Post-Training Cheat Sheet](#post-training-cheat-sheet)
7. [Example 07 — Isaac Lab](#example-07--isaac-lab-native-linux--cloud-gpu-only)
8. [Cloud Training on vast.ai](#cloud-training-on-vastai)

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

## Examples

| # | Folder | Concept | GPU |
|---|---|---|---|
| 01 | [01_basics_loop](examples/01_basics_loop/) | Minimal PPO loop · RL theory · PPO algorithm | No |
| 02 | [02_ppo_classic_control](examples/02_ppo_classic_control/) | Train / eval split | No |
| 03 | [03_ppo_stability](examples/03_ppo_stability/) | Stability hyperparameters | No |
| 04 | [04_pixels_pipeline](examples/04_pixels_pipeline/) | Pixel preprocessing pipeline | No |
| 05 | [05_ppo_cnn](examples/05_ppo_cnn/) | CNN encoder | Recommended |
| 06 | [06_pixels_harder_task](examples/06_pixels_harder_task/) | Frame stacking for temporal context | Recommended |
| 07 | [07_isaac_transfer](examples/07_isaac_transfer/) | Isaac Lab (physics simulator) | Required |

Each folder contains a `README.md` with concept explanations, run commands, and key learning points.

- **RL theory and PPO algorithm**: [examples/01_basics_loop/README.md](examples/01_basics_loop/README.md)
- **Shared utilities** (models, PPO builder, wrappers, CLI, plots, export): [examples/shared/README.md](examples/shared/README.md)

---



## Monitoring Training

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
