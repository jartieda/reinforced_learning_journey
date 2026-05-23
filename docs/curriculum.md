# Curriculum Guide

## Stage 1: SKRL Core Objects

- Example 01 introduces the minimal loop: environment wrapper, models, memory, PPO agent, trainer.
- Goal: understand object responsibilities and the train call path.

## Stage 2: Reusable Workflow

- Example 02 separates train and eval scripts.
- Goal: use reproducible experiment scripts and checkpoint loading.

## Stage 3: Stability Engineering

- Example 03 tunes PPO defaults for smoother optimization.
- Goal: learn practical knobs (`learning_rate`, entropy, gradient clipping, epochs).

## Stage 4: Pixel Observation Pipeline

- Example 04 changes only the observation pipeline (grayscale image stream) while keeping PPO familiar.
- Goal: see how input modality changes the setup and sample complexity.

## Stage 5: Convolutional Policies

- Example 05 introduces a CNN policy/value encoder for image observations.
- Goal: understand tensor shapes and image feature extraction before policy/value heads.

## Stage 6: Harder Vision Training

- Example 06 adds frame stacking and stronger training settings.
- Goal: handle partial observability and longer-horizon credit assignment from pixels.

## Stage 7: Isaac Transfer

- Example 07 keeps PPO + CNN design, swaps environment stack to Isaac Lab.
- Goal: recognize what generalizes across simulators and what is simulator-specific.

## Suggested Study Pattern

1. Run each example with a tiny `--timesteps` value first.
2. Read that folder's README before scaling timesteps.
3. Compare model classes and PPO config diffs between consecutive folders.
4. Keep notes on observation shapes at every stage.
