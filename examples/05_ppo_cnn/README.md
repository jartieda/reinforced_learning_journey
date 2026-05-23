# Example 05: PPO With Convolutional Encoder

## Goal

Replace the flat MLP from Example 04 with a **convolutional encoder** and
understand exactly why convolutions are the right tool for image observations.
Same environment, same PPO, same preprocessing — only the network changes.

---

## Files

- `train.py`: training script, uses `CNNGaussianPolicy` and `CNNValue` from `examples/shared/models.py`.

---

## The Problem With Example 04 (Recap)

In Example 04, a `Linear(9216 → 128)` layer was the first thing to see pixels.
That layer had **1.18 million parameters** and no spatial bias — a weight for
"edge at top-left" was completely independent of a weight for "edge at
top-right". A CNN fixes this with two ideas: **local filters** and **weight sharing**.

---

## Key Idea: What a Convolution Actually Does

A `Conv2d(in_ch, out_ch, kernel_size, stride)` layer slides a small
`kernel_size × kernel_size` window across the image and applies the
**same learned filter** at every position.

```
Input image (1, 96, 96)

┌───────────────────────────────┐
│                               │
│   ┌──────┐  8×8 filter slides │
│   │filter│──────────────────► │  → one feature map per filter
│   └──────┘  stride 4          │
│                               │
└───────────────────────────────┘
32 different filters → 32 feature maps
```

The same 64 weights that detect a horizontal edge detect it **anywhere** in the
image. This is called **translational equivariance**: the filter response moves
when the feature moves, instead of requiring separate weights for every location.

---

## Layer-by-Layer Walkthrough (`CNNEncoder`)

Input: grayscale frame `(1, 96, 96)` — 1 channel, 96×96 pixels.

$$H_\text{out} = \left\lfloor \frac{H_\text{in} - k}{s} \right\rfloor + 1$$

### Layer 1 — `Conv2d(1, 32, kernel_size=8, stride=4)` + ReLU

$$H_\text{out} = \left\lfloor \frac{96 - 8}{4} \right\rfloor + 1 = 23$$

| | Value |
|---|---|
| Output shape | `(32, 23, 23)` |
| Receptive field | 8×8 pixels |
| Parameters | $1 \times 32 \times 8 \times 8 + 32 = \mathbf{2{,}080}$ |
| Purpose | Detect low-level textures: edges, road boundaries |

32 filters each see an 8×8 patch and output a single number.
With stride 4 the filter jumps 4 pixels at a time, so the spatial grid shrinks
from 96 → 23.

### Layer 2 — `Conv2d(32, 64, kernel_size=4, stride=2)` + ReLU

$$H_\text{out} = \left\lfloor \frac{23 - 4}{2} \right\rfloor + 1 = 10$$

| | Value |
|---|---|
| Output shape | `(64, 10, 10)` |
| Receptive field | 8 + (4−1)×4 = **20×20 pixels** on the original image |
| Parameters | $32 \times 64 \times 4 \times 4 + 64 = \mathbf{32{,}832}$ |
| Purpose | Detect mid-level patterns: curves, car shape |

Each position in this layer "sees" a 20×20 region of the original image —
its **effective receptive field** grows with depth because it is built on top
of the previous layer's 8×8 filters.

### Layer 3 — `Conv2d(64, 64, kernel_size=3, stride=1)` + ReLU

$$H_\text{out} = \left\lfloor \frac{10 - 3}{1} \right\rfloor + 1 = 8$$

| | Value |
|---|---|
| Output shape | `(64, 8, 8)` |
| Receptive field | 20 + (3−1)×2×4 = **36×36 pixels** on the original image |
| Parameters | $64 \times 64 \times 3 \times 3 + 64 = \mathbf{36{,}928}$ |
| Purpose | Combine mid-level features; near-global spatial context |

### Flatten

`(64, 8, 8)` → **4 096** latent features.

$$64 \times 8 \times 8 = 4{,}096$$

---

## Policy Head and Value Head

After the encoder produces a 4 096-dim feature vector, two separate heads
perform the task-specific prediction.

### Policy head (`CNNGaussianPolicy`)

```
encoder(obs) → (batch, 4096)
    → Linear(4096 → 256) → ReLU
    → Linear(256 → 3)                ← 3 action means (steer, gas, brake)
    + log_std_parameter  ∈ ℝ³        ← learnable exploration noise
```

| Layer | Params |
|---|---|
| `Linear(4096 → 256)` | $4096 \times 256 + 256 = \mathbf{1{,}048{,}832}$ |
| `Linear(256 → 3)` | $256 \times 3 + 3 = \mathbf{771}$ |
| `log_std` | **3** |

### Value head (`CNNValue`)

Identical encoder (separate weights), then:

```
encoder(obs) → (batch, 4096)
    → Linear(4096 → 256) → ReLU
    → Linear(256 → 1)                ← scalar V(s)
```

---

## Full Parameter Count Comparison

| Component | Example 04 (MLP) | Example 05 (CNN) |
|---|---|---|
| Input stage | `Linear(9216→128)` = **1 180 928** | Encoder = **71 840** |
| Features produced | 128 | **4 096** (32× richer) |
| Head | `Linear(128→128→3)` = 16 899 | `Linear(4096→256→3)` = 1 049 606 |
| **Total policy** | **≈ 1.20 M** | **≈ 1.12 M** |

The CNN has *fewer* total parameters yet produces a **32× richer feature
vector** from the same pixels. The saving comes entirely from weight sharing:
the 71 840 encoder parameters are reused at every spatial position, while the
MLP's 1.18 M parameters each encode exactly one pixel location.

---

## Data Flow Summary

```
(batch, 1, 96, 96)  ← grayscale frame, normalised to [0, 1]
        │
        ▼  Conv2d(1→32, 8, stride=4) + ReLU
(batch, 32, 23, 23)
        │
        ▼  Conv2d(32→64, 4, stride=2) + ReLU
(batch, 64, 10, 10)
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
# Train with CNN policy on grayscale CarRacing
python -m examples.05_ppo_cnn.train --timesteps 250000

# Shorter run for architecture verification
python -m examples.05_ppo_cnn.train --timesteps 20000 --seed 0

# Export policy and record video
python -m examples.shared.export \
    --checkpoint runs/05_ppo_cnn/05_ppo_cnn/checkpoints/agent_5000.pt \
    --env-id CarRacing-v3 --model cnn \
    --record --video-dir videos/05_cnn

# Plot training curves
python -m examples.shared.plot runs/05_ppo_cnn/05_ppo_cnn
```

## Key Learning Point

A convolutional encoder replaces `Linear(9216 → 128)` (1.18 M params, no
spatial bias) with three conv layers (71 840 params total) that produce a
4 096-dim feature via weight sharing. The PPO training loop, memory, and
advantage computation are completely unchanged — only the model swaps.

You can keep the PPO training loop intact and swap only the model architecture when moving to vision tasks.
