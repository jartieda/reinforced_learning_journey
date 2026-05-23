# Example 04: Pixel Pipeline Without CNN Yet

## Goal

Feed **raw image pixels** into PPO while deliberately keeping the MLP
architecture from Examples 01-03.  
The intentional mismatch — a flat network applied to image data — teaches you
*exactly* why convolutional networks are needed (Example 05) and what the
preprocessing pipeline must do before any images reach the network.

---

## The Observation Pipeline Step-by-Step

CarRacing-v3 produces a colour image at every step. The code transforms it
through three stages before it reaches the network:

```
CarRacing-v3 environment
      │
      │  RGB frame: shape (96, 96, 3)  — height × width × channels
      │  dtype uint8, values 0-255
      ▼
 ToGrayscale wrapper                       (examples/shared/wrappers.py)
      │  average the 3 colour channels → single luminance channel
      │  transpose to channel-first:  (96, 96, 3) → (1, 96, 96)
      │  output shape: (1, 96, 96)
      ▼
 wrap_env (skrl)                           converts numpy → torch tensor
      │  normalises to float32 in [0.0, 1.0]
      │  output shape: (batch, 1, 96, 96)
      ▼
 MLPGaussianPolicy.compute()               (examples/shared/models.py)
      │  obs.view(batch, -1)  ← FLATTEN everything into one long vector
      │  output shape: (batch, 9216)          ← 1 × 96 × 96
      ▼
 MLP network
```

### Why 9216?

$$1 \times 96 \times 96 = 9{,}216 \text{ pixel values per frame}$$

Each pixel becomes an independent input feature. The MLP has no idea that
adjacent pixels are spatially related — it sees 9,216 disconnected numbers.

---

## Inside the MLP: Every Layer Explained

### Policy network (`MLPGaussianPolicy`)

CarRacing-v3 has **3 continuous actions**: [steering, gas, brake].

```
Input  →  Linear(9216 → 128)  →  Tanh  →  Linear(128 → 128)  →  Tanh  →  Linear(128 → 3)
                                                                             ↓
                                                              action means  μ ∈ ℝ³
                                                         + log_std_parameter σ ∈ ℝ³  (learnable)
```

| Layer | Input dim | Output dim | Parameters | Purpose |
|---|---|---|---|---|
| `Linear` | 9 216 | 128 | 9 216×128 + 128 = **1 180 800** | compress pixels to abstract features |
| `Tanh` | 128 | 128 | 0 | squash activations to (−1, 1) |
| `Linear` | 128 | 128 | 128×128 + 128 = **16 512** | second representation layer |
| `Tanh` | 128 | 128 | 0 | |
| `Linear` | 128 | 3 | 128×3 + 3 = **387** | output one mean per action |
| `log_std` | — | 3 | **3** | one learnable log-σ per action |
| **Total** | | | **≈ 1.2 M params** | |

The first linear layer alone has **1.18 million parameters** just to project
9,216 raw pixel values into 128 features. This is wasteful: nearby pixels are
highly correlated, so most of that capacity is redundant. A CNN would achieve
the same compression with ~10× fewer parameters by sharing weights spatially.

### Value network (`MLPValue`)

Identical structure, but the output is a single scalar $V(s)$:

```
Input  →  Linear(9216 → 128)  →  Tanh  →  Linear(128 → 128)  →  Tanh  →  Linear(128 → 1)
                                                                              ↓
                                                                          V(s) ∈ ℝ
```

Both networks share the same architecture but have **separate weights** — the
critic is not a branch of the actor; it is an independent model.

---

## What `ToGrayscale` Actually Does

```python
# wrappers.py — simplified
gray = observation.mean(axis=2).astype(np.uint8)   # (96, 96, 3) → (96, 96)
return np.expand_dims(gray, axis=0)                 # → (1, 96, 96)
```

It computes the average of the R, G, B channels per pixel. The result is a
single-channel image. This reduces input dimensionality by 3×:
$96 \times 96 \times 3 = 27{,}648$ → $96 \times 96 \times 1 = 9{,}216$.

Color is not needed for CarRacing: the car, track edges, and grass are
distinguishable by brightness alone.

---

## Why This Setup Struggles (and That Is the Point)

The MLP flattens the image before any spatial reasoning can happen:

- **No translation invariance**: a car slightly left vs. slightly right looks
  like a completely different 9,216-dimensional input vector.
- **No local feature detection**: whether the road curves left is encoded
  across hundreds of pixel positions, not a single learned filter.
- **Enormous first layer**: 1.18 M params to learn from scratch with no
  inductive bias about images.

Example 05 replaces the first `Linear(9216 → 128)` with a convolutional
encoder that shares weights across spatial positions. That one change fixes all
three problems above.

---

## Observation Shape Through the Stack — Quick Reference

| Stage | Shape | Dtype |
|---|---|---|
| CarRacing-v3 raw | `(96, 96, 3)` | uint8 |
| After `ToGrayscale` | `(1, 96, 96)` | uint8 |
| After `wrap_env` (skrl) | `(batch, 1, 96, 96)` | float32, /255 |
| After `obs.view(..., -1)` | `(batch, 9216)` | float32 |
| After `Linear(9216→128)` | `(batch, 128)` | float32 |
| After `Linear(128→3)` | `(batch, 3)` | float32 |

---

## Run

All commands must be run from the **project root** using the `-m` flag.

```bash
# Train
python -m examples.04_pixels_pipeline.train --timesteps 150000

# Shorter run for quick iteration
python -m examples.04_pixels_pipeline.train --timesteps 20000 --seed 0

# Export policy and record video of trained agent
python -m examples.shared.export \
    --checkpoint runs/04_pixels_pipeline/04_pixels_pipeline/checkpoints/agent_5000.pt \
    --env-id CarRacing-v3 --model mlp \
    --record --video-dir videos/04_pixels

# Plot training curves
python -m examples.shared.plot runs/04_pixels_pipeline/04_pixels_pipeline
```

## Key Learning Point

An MLP can technically accept image inputs after flattening, but it wastes
parameters and loses all spatial structure. Understanding *why* it fails —
no weight sharing, no translation invariance, huge first layer — is the exact
motivation for the convolutional encoder introduced in Example 05.
