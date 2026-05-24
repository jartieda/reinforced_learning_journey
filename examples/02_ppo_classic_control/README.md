# Example 02: PPO Train/Eval Split

## Goal

Use a cleaner project-style workflow with separate scripts for training and evaluation.

## Files

- `train.py`: trains PPO and writes checkpoints/logs.
- `eval.py`: loads a checkpoint and runs evaluation episodes.

## How It Works

- Training script builds agent and runs `trainer.train()`.
- Evaluation script recreates architecture, loads checkpoint, and runs `trainer.eval()`.

### Why separate train and eval scripts?

A trained policy should be evaluated under a **fixed protocol** rather than
inspected during the noisy training process. `train.py` runs PPO with
stochastic actions (exploration); `eval.py` reloads the checkpoint and runs
the policy **deterministically** (mean action, no noise) for a set number of
episodes.

Separating the scripts enforces a good habit: the evaluation code is
independent of the training code, so you can change your training loop without
accidentally changing how you measure performance.

## Run

All commands must be run from the **project root** using the `-m` flag.

```bash
# Train (writes checkpoints + TensorBoard logs to runs/)
python -m examples.02_ppo_classic_control.train --timesteps 75000

# Train with a different environment and seed
python -m examples.02_ppo_classic_control.train --env-id MountainCarContinuous-v0 --timesteps 100000 --seed 1

# Evaluate a checkpoint — watch it play in a window
python -m examples.02_ppo_classic_control.eval \
    --checkpoint runs/02_ppo_classic_control/02_ppo_classic_control/checkpoints/agent_5000.pt \
    --render

# Evaluate and record an MP4 video
python -m examples.02_ppo_classic_control.eval \
    --checkpoint runs/02_ppo_classic_control/02_ppo_classic_control/checkpoints/agent_5000.pt \
    --record --video-dir videos/pendulum

# Plot training curves (after training)
python -m examples.shared.plot runs/02_ppo_classic_control/02_ppo_classic_control
```

## Key Learning Point

Separate train/eval entrypoints make experiments repeatable and easier to debug.
