# Example 01: Minimal SKRL PPO Loop

## Goal

Learn the smallest complete training pipeline in SKRL.

## What This Example Contains

- A single script: `main.py`
- Continuous-control environment (`Pendulum-v1`)
- MLP policy + MLP value model
- PPO agent and sequential trainer

## How It Works

1. Build and wrap the Gymnasium environment.
2. Create policy/value models.
3. Build PPO agent (with memory and default config).
4. Train with `SequentialTrainer`.

## Run

All commands must be run from the **project root** using the `-m` flag so Python
can resolve the `examples.*` package imports.

```bash
# Minimal run
python -m examples.01_basics_loop.main

# Specify timesteps
python -m examples.01_basics_loop.main --timesteps 30000

# Different environment, fixed seed, no window
python -m examples.01_basics_loop.main --env-id MountainCarContinuous-v0 --timesteps 50000 --seed 0 --headless

# Resume from a saved checkpoint
python -m examples.01_basics_loop.main --timesteps 50000 --checkpoint runs/01_basics_loop/01_basics_loop/checkpoints/agent_5000.pt
```

## Key Learning Point

The core SKRL lifecycle is:

`env -> models -> memory/config -> agent -> trainer.train()`
