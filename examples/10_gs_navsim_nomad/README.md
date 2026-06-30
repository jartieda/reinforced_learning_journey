# Example 10: Visual Navigation in Gaussian Splat Scenes — gs_navsim + NOMAD

## Goal

Train a robot to navigate toward a goal image inside a **photorealistic
Gaussian Splat scene** rendered live in a browser.

The key ideas introduced here:

- **WebSocket-coupled environment** — the Gymnasium env communicates with a
  running browser session over WebSocket instead of calling a local physics
  API.
- **Frozen pretrained encoder** — NOMAD's ViNT backbone (EfficientNet-b0) is
  loaded from a checkpoint and never updated; only small MLP heads are trained.
- **Image-goal navigation** — the agent is rewarded for making the current
  camera view look like a goal photograph; no GPS or map is used.
- **Obstacle masks** — a 2-D grid mask (drawn in the browser's mask editor)
  provides collision detection on the Python side, removing the need for
  physics collision callbacks.

---

## Architecture Overview

```
Browser (gs_navsim)                Python (this example)
─────────────────────              ──────────────────────────────────────────
Gaussian Splat renderer            GsNavSimEnv (env.py)
  ↕  WebSocket :8081                 │
robot_command ──────────────────►  step()
                                     │  action [dx, dy]
                                     │  → PD controller → [v, w]
rendered_image ◄────────────────   wait for image
collision      ◄────────────────   check collision flag
                                   │
                                   ▼
                              observation (15, 96, 96)
                              = 4 × RGB frames (CHW, ImageNet-norm)
                              + 1 × goal  frame (CHW, ImageNet-norm)
                                   │
                              NomadPolicy / NomadValue (model.py)
                              = frozen NoMaD_ViNT encoder (256-d)
                              + trainable MLP heads
                                   │
                              skrl PPO trainer (train.py)
```

---

## Files

| File | What it contains |
|---|---|
| `env.py` | `GsNavSimEnv` — Gymnasium env, WebSocket client, obstacle mask, reward |
| `model.py` | `NomadPolicy` and `NomadValue` — frozen NOMAD encoder + MLP heads |
| `train.py` | PPO training entry point |
| `run.py` | Load a checkpoint and run deterministic evaluation episodes |

---

## Prerequisites

### 1 — NOMAD weights

Download the pretrained NOMAD checkpoint and place it at:

```
robot_nav/visualnav-transformer/model_weights/nomad.pth
```

The ViNT/NOMAD weights are available from the
[visualnav-transformer releases](https://github.com/robodhruv/visualnav-transformer).

### 2 — gs_navsim running

Start the WebSocket + HTTP server:

```bash
cd gs_navsim/backend
node server.js          # WebSocket on :8081, static files on :3000
```

Open the frontend in a browser:

```
http://localhost:3000
```

Load a `.ply` scene file using the file picker in the sidebar, then make sure
the scene renders correctly before starting training.

### 3 — Python dependencies

```bash
pip install -r requirements.txt
pip install websocket-client   # WebSocket client for env.py
```

### 4 — Goal image

Capture or screenshot the target view from the browser and save it as a PNG:

```bash
# The browser's "📷 Export" button saves the current rendered frame.
# Move / rename it:
cp ~/Downloads/export.png goal.png
```

### 5 — Obstacle mask (optional but recommended)

Open the mask editor in the browser (🗺 **Mask** button), paint blocked cells
over walls and furniture, then click **💾 Guardar** to download `mask.json`.

---

## Training

```bash
python -m examples.10_gs_navsim_nomad.train \
    --goal path/to/goal.png \
  --scene-id your_scene_id \
    --timesteps 200000
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--goal` | *(required)* | Path to goal image (PNG/JPG) |
| `--scene-id` | `None` | Scene folder id from gs_navsim backend data dir (auto-load scene) |
| `--timesteps` | `200000` | Total environment steps to train |
| `--max-steps` | `500` | Max steps per episode before truncation |
| `--ws-url` | `ws://localhost:8081` | WebSocket URL of the gs_navsim server |
| `--seed` | `42` | RNG seed |
| `--checkpoint` | `None` | Resume from a saved checkpoint (`.pt`) |
| `--experiment-name` | `10_gs_navsim_nomad` | TensorBoard run label |

Checkpoints are saved to `runs/10_gs_navsim_nomad/checkpoints/`.

Monitor progress:

```bash
tensorboard --logdir runs/
# open http://localhost:6006
```

---

## Evaluation

```bash
python -m examples.10_gs_navsim_nomad.run \
    --checkpoint runs/10_gs_navsim_nomad/checkpoints/agent_200000.pt \
    --goal path/to/goal.png \
  --scene-id your_scene_id \
    --episodes 10
```

The script prints per-episode statistics:

```
Episode   1  |  steps= 147  reward=  +42.31  pixel_dist=0.0181  [SUCCESS]
Episode   2  |  steps= 500  reward=  -12.05  pixel_dist=0.0834  [TIMEOUT]
Episode   3  |  steps=  23  reward=  -25.00  pixel_dist=0.1203  [COLLISION]
```

---

## Observation Space

```
Shape: (15, 96, 96)  float32
```

| Channels | Content |
|---|---|
| 0 – 11 | 4 × RGB frames (most recent last), ImageNet-normalised, CHW format |
| 12 – 14 | Goal RGB frame, same normalisation |

The NOMAD encoder expects exactly this layout (context_size = 3 past frames + 1 current).

---

## Action Space

```
Shape: (2,)  float32  ∈ [−1, 1]
```

The policy outputs a 2-D waypoint `[dx, dy]` in the robot's local frame.
A PD controller in `env.py` converts it to linear velocity `v` and angular
velocity `w` before sending the `set_velocity` command to the simulator.

---

## Reward Function

| Event | Reward |
|---|---|
| Collision with obstacle | −5.0 (episode ends) |
| Progress toward goal | `(prev_pixel_dist − curr_pixel_dist) × 100` |
| Time penalty (every step) | −0.01 |
| Goal reached (`pixel_dist < 0.02`) | +10.0 (episode ends) |

Pixel distance is the mean squared difference between the current frame and
the goal frame in ImageNet-normalised CHW space.

---

## Model Details

Both `NomadPolicy` and `NomadValue` share the same frozen encoder:

```
NoMaD_ViNT (EfficientNet-b0 backbone)
  context_size        = 3
  obs_encoding_size   = 256
  mha_num_attention_heads  = 4
  mha_num_attention_layers = 4
  mha_ff_dim_factor        = 4
```

Only the MLP heads (256 → 256 → output) are updated during PPO.
Observations are already ImageNet-normalised, so no `RunningStandardScaler`
is used.

---

## Tips

- **Scene matters** — the pixel-distance reward works best when the goal image
  is distinctive (avoid featureless white walls as goals).
- **Mask quality** — a well-drawn obstacle mask prevents the robot from getting
  stuck in walls during exploration, which dramatically speeds up early
  training.
- **Spawn randomisation** — each episode starts at a random free cell sampled
  from the mask, giving the agent diverse starting positions.
- **Single-env training** — this example uses one environment instance because
  the browser can only render one scene at a time. Expect slower wall-clock
  throughput than vectorised setups.
- **WebSocket latency** — each step waits for a rendered image; keep the
  browser tab in the foreground and hardware-accelerated to minimise latency.

---

## Full Remote on vast.ai (backend + browser + training all remote)

This mode runs everything in the same vast.ai instance:

- gs_navsim backend (`node server.js`)
- browser renderer (`chromium` under `Xvfb`)
- PPO training (`python -m examples.10_gs_navsim_nomad.train`)

### 1 — Prepare remote scene data layout

Inside the remote machine/container, place scenes under one data directory:

```
/workspace/data/<scene_id>/3dgs_uncompressed.ply
/workspace/data/<scene_id>/occupancy.json
/workspace/data/<scene_id>/occupancy.png
```

`gs_navsim/backend/server.js` now reads this from `GS_DATA_DIR`.

### 2 — Start gs_navsim backend remotely

```bash
cd /workspace/robot_nav/gs_navsim/backend
npm install
GS_DATA_DIR=/workspace/data node server.js
```

### 3 — Start browser renderer remotely (no local browser needed)

```bash
apt-get update && apt-get install -y xvfb chromium
Xvfb :99 -screen 0 1280x720x24 &
export DISPLAY=:99
chromium \
  --no-sandbox \
  --disable-dev-shm-usage \
  --ignore-gpu-blocklist \
  --enable-webgl \
  --use-gl=angle \
  --app=http://127.0.0.1:3000 &
```

Keep this browser process alive during training.

### 4 — Run training remotely

```bash
cd /workspace/robot_nav/reinforced
export PYTHONPATH=/workspace/robot_nav/reinforced
python -m examples.10_gs_navsim_nomad.train \
  --goal /workspace/robot_nav/reinforced/examples/10_gs_navsim_nomad/goal.png \
  --scene-id your_scene_id \
  --ws-url ws://127.0.0.1:8081 \
  --timesteps 200000
```

### 5 — Run evaluation remotely

```bash
python -m examples.10_gs_navsim_nomad.run \
  --checkpoint runs/10_gs_navsim_nomad/checkpoints/agent_200000.pt \
  --goal /workspace/robot_nav/reinforced/examples/10_gs_navsim_nomad/goal.png \
  --scene-id your_scene_id \
  --ws-url ws://127.0.0.1:8081 \
  --episodes 10
```

### Notes for vast.ai

- If you start from `reinforced/scripts/vast_launch.sh`, the script syncs only
  the `reinforced/` directory. For full-remote Example 10 you must also upload
  `gs_navsim/` and `visualnav-transformer/model_weights/nomad.pth`.
- TensorBoard can still be tunneled with SSH (`-L 6006:127.0.0.1:6006`).
