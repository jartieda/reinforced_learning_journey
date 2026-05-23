# Example 06: Harder Vision Task (Frame Stacking)

## Goal

Add **temporal context** to the CNN policy by stacking the last 4 grayscale
frames along the channel axis, and tighten the PPO settings to match the
increased task difficulty. Same CNN architecture as Example 05 — only the
*input* and the *optimizer config* change.

---

## The Core Problem: Single Frames Are Not Markovian

An MDP requires the state $s_t$ to be **Markovian**: it must contain all
information needed to predict the next state and choose the optimal action.

A single pixel frame breaks this: you can see the car's position but not
its **velocity or direction of travel**.

```
Frame t:    🏎 at position x=48           ← where is the car?
Frame t+1:  🏎 at position x=52           ← moving right at speed 4
            BUT the policy only sees frame t+1: it cannot tell speed or direction.
```

A recurrent model (LSTM) would solve this with hidden state. Frame stacking
is a simpler, cheaper alternative: concatenate recent frames so the policy
can compute motion by comparing pixel positions across time.

---

## Frame Stacking: How It Works

`FrameStackCHW` (in `examples/shared/wrappers.py`) maintains a rolling
deque of the last `num_frames` observations and stacks them along the
channel axis:

```python
# wrappers.py (simplified)
self.frames = deque(maxlen=4)          # holds the last 4 frames

# On each step, push new frame and concatenate:
self.frames.append(new_obs)            # shape (1, 96, 96) each
obs = np.concatenate(list(self.frames), axis=0)   # → (4, 96, 96)
```

At episode start the deque is pre-filled with the initial frame repeated 4×,
so the output shape is always `(4, 96, 96)` from timestep zero.

### What the policy sees at time t

```
Channel 0 → frame at  t−3   (oldest)
Channel 1 → frame at  t−2
Channel 2 → frame at  t−1
Channel 3 → frame at  t     (newest)
```

The CNN can learn to compute differences between channels — essentially
finite-difference approximations to velocity — without any explicit
optical-flow computation.

---

## Layer-by-Layer Walkthrough

The only change from Example 05 is that **the first Conv2d now has 4 input
channels instead of 1**.

Input: `(4, 96, 96)` — 4 stacked grayscale frames.

$$H_\text{out} = \left\lfloor \frac{H_\text{in} - k}{s} \right\rfloor + 1$$

### Layer 1 — `Conv2d(4, 32, kernel_size=8, stride=4)` + ReLU

$$H_\text{out} = \left\lfloor \frac{96 - 8}{4} \right\rfloor + 1 = 23$$

| | Value |
|---|---|
| Output shape | `(32, 23, 23)` |
| Parameters | $4 \times 32 \times 8 \times 8 + 32 = \mathbf{8{,}224}$ |
| vs Example 05 | +6 144 params (extra 3 input channels) |

Each of the 32 filters now sees a **4-channel patch**: it compares the same
spatial region across 4 time steps in a single multiplication. The filter
can detect "pixel moved right between frames" as easily as it detects "edge
here".

### Layer 2 — `Conv2d(32, 64, kernel_size=4, stride=2)` + ReLU

| | Value |
|---|---|
| Output shape | `(64, 10, 10)` |
| Parameters | $32 \times 64 \times 4 \times 4 + 64 = \mathbf{32{,}832}$ |
| vs Example 05 | identical |

### Layer 3 — `Conv2d(64, 64, kernel_size=3, stride=1)` + ReLU

| | Value |
|---|---|
| Output shape | `(64, 8, 8)` |
| Parameters | $64 \times 64 \times 3 \times 3 + 64 = \mathbf{36{,}928}$ |
| vs Example 05 | identical |

### Flatten → 4 096 features

$$64 \times 8 \times 8 = 4{,}096 \quad\text{(same as Example 05)}$$

---

## Parameter Count Comparison

| Component | Example 05 (1 frame) | Example 06 (4 frames) |
|---|---|---|
| Conv layer 1 | 2 080 | **8 224** (+6 144) |
| Conv layers 2–3 | 69 760 | 69 760 (unchanged) |
| **Encoder total** | **71 840** | **78 000** |
| Policy head | 1 049 606 | 1 049 606 (unchanged) |
| **Total policy** | **≈ 1.12 M** | **≈ 1.13 M** |

Adding full temporal context costs only **6 144 extra parameters** — one
additional weight per (input-channel × output-channel × kernel position)
in layer 1. Everything downstream is unchanged.

---

## PPO Settings Changed From Example 05

| Parameter | Example 05 | Example 06 | Reason |
|---|---|---|---|
| `learning_rate` | `3e-4` | `1e-4` | Pixel gradients noisier with 4-ch input |
| `learning_epochs` | `8` | `10` | Extract more signal per rollout |
| `mini_batches` | `4` | `8` | Smaller batches reduce update variance |
| `entropy_loss_scale` | `0.0` | `0.01` | CarRacing tracks are more varied; exploration matters more |

The interaction between these settings follows the same logic as Example 03:
lower LR makes more epochs safe, entropy prevents premature lock-in to a
single driving style.

### Why more mini-batches?

Each rollout of 1 024 steps is split into `mini_batches` sub-batches of size:

$$\text{batch size} = \frac{1024}{8} = 128 \text{ transitions}$$

Smaller batches introduce more gradient noise, which can help escape shallow
local minima but requires a lower learning rate to remain stable. The
combination (more batches + lower LR) is standard practice for vision tasks.

---

## Data Flow Summary

```
(batch, 4, 96, 96)  ← 4 stacked grayscale frames, normalised to [0, 1]
        │
        ▼  Conv2d(4→32, 8, stride=4) + ReLU
(batch, 32, 23, 23)  ← motion + texture features
        │
        ▼  Conv2d(32→64, 4, stride=2) + ReLU
(batch, 64, 10, 10)  ← composed spatial-temporal patterns
        │
        ▼  Conv2d(64→64, 3, stride=1) + ReLU
(batch, 64, 8, 8)
        │
        ▼  Flatten
(batch, 4096)
        │
        ├──► Policy head → (batch, 3) means  +  log_std ∈ ℝ³
        └──► Value head  → (batch, 1) V(s)
```

---

## Run

All commands must be run from the **project root** using the `-m` flag.

```bash
# Train with 4-frame stacking and tighter PPO settings
python -m examples.06_pixels_harder_task.train --timesteps 400000

# Shorter run to verify the pipeline works
python -m examples.06_pixels_harder_task.train --timesteps 20000 --seed 0

# Export policy and record video
python -m examples.shared.export \
    --checkpoint runs/06_pixels_harder_task/06_pixels_harder_task/checkpoints/agent_5000.pt \
    --env-id CarRacing-v3 --model cnn \
    --record --video-dir videos/06_harder

# Plot and compare against Example 05
python -m examples.shared.plot runs/06_pixels_harder_task/06_pixels_harder_task
python -m examples.shared.plot runs/ --all
```

## Key Learning Point

Frame stacking converts a partially-observable problem (single frame) into an
approximately Markovian state by giving the policy access to recent history.
It costs only 6 144 extra parameters — one conv layer's worth of extra input
channels — yet lets the network compute velocity, acceleration, and heading
from pixel differences across time.
python -m examples.shared.plot runs/06_pixels_harder_task/06_pixels_harder_task
python -m examples.shared.plot runs/ --all
```

## Key Learning Point

Frame stacking helps when single images do not contain enough motion information for control.
