# Example 01: Minimal SKRL PPO Loop

## Goal

Learn the smallest complete training pipeline in SKRL.  
This is also the place where the core RL and PPO theory is documented.

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

The core SKRL lifecycle is: `env → models → memory/config → agent → trainer.train()`

---

## Reinforcement Learning Primer

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
distribution over actions — that maximises the sum of future discounted rewards:

$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

**γ ∈ (0,1)** is the discount factor. High γ (e.g. 0.99) = plans far ahead; low γ = short-sighted.

### Actor-Critic

PPO is an **actor-critic** algorithm. Two networks are trained jointly:

| Network | Output | Role |
|---------|--------|------|
| **Policy (actor)** | π(a\|s) — distribution over actions | Decides what to do |
| **Value (critic)** | V(s) — scalar estimate of future return | Evaluates how good the state is |

### Advantage and GAE

The **advantage** $A_t = Q(s_t, a_t) - V(s_t)$ measures how much better an
action is compared to the average from that state.

**Generalised Advantage Estimation (GAE)** computes a weighted mixture of
multi-step returns:

$$A_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l},
\quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

- **λ → 0**: relies on the critic (low variance, high bias)
- **λ → 1**: relies on actual returns (low bias, high variance)
- **λ = 0.95** is a widely used middle ground

---

## PPO — The Algorithm

PPO (Proximal Policy Optimization, Schulman et al. 2017) prevents the policy
from changing too drastically in a single update.

### Clipped Surrogate Objective

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[
  \min\!\left( r_t(\theta)\, A_t,\;
               \text{clip}(r_t(\theta), 1{-}\epsilon, 1{+}\epsilon)\, A_t \right)
\right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$.
The clip parameter **ε = 0.2** (`ratio_clip`) bounds how much one update can shift the policy.

### Training Loop

```
For each iteration:
  1. Collect rollout_size transitions with the current policy (no gradient)
  2. Compute advantages with GAE
  3. For learning_epochs passes over the collected data:
       a. Split into mini_batches
       b. Compute clipped policy loss, value loss, entropy bonus
       c. Clip gradient norm; apply Adam step
  4. Discard rollout (on-policy — data used once then thrown away)
```

### Key Hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `discount_factor` (γ) | 0.99 | Planning horizon |
| `gae_lambda` (λ) | 0.95 | Bias-variance trade-off for advantages |
| `ratio_clip` (ε) | 0.2 | How much the policy can change per update |
| `learning_rate` | 3e-4 | Step size |
| `learning_epochs` | 8 | Reuse of each rollout |
| `mini_batches` | 4 | Stochastic gradient noise |
| `entropy_loss_scale` | 0.0 | Exploration bonus |
| `grad_norm_clip` | 0.5 | Prevents exploding gradients |

---

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
