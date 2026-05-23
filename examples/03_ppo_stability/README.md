# Example 03: PPO Stability Settings

## Goal

Understand **why** certain PPO hyperparameters exist, and how changing just four
of them can be the difference between reliable convergence and training collapse.
The model architecture and environment are identical to Example 02. Only the
optimizer-level config changes.

---

## Background: What Can Go Wrong in PPO

PPO is far more stable than earlier policy-gradient methods, but it can still
fail in characteristic ways. Recognising the failure mode tells you which knob
to turn.

| Symptom | Root cause |
|---|---|
| Reward explodes then crashes | Policy update too large (high LR or no gradient clip) |
| Policy never improves past chance | Too little exploration (entropy collapses to zero) |
| Very slow improvement | Too few update epochs — data under-used |
| Value loss diverges | Critic update too aggressive; target scale not normalised |

---

## The Four Settings Changed Here

### 1 · Learning Rate: `3e-4` → `1e-4`

The Adam optimiser takes gradient steps of effective size ≈ `learning_rate`.
Each step moves the policy a small distance in parameter space.

**Problem with a large LR:** PPO clips the probability ratio
$r_t(\theta) = \pi_\theta(a|s) / \pi_{\theta_\text{old}}(a|s)$ to the interval
$[1-\varepsilon,\ 1+\varepsilon]$, but a large gradient step can push parameters
so far that the *next* rollout — collected under the new policy — is essentially
off-distribution. The update was valid at the clip boundary; the parameter
movement was not.

**Effect of lowering it:** Each epoch moves the policy a smaller distance, so the
data collected in the previous rollout stays approximately on-policy for more
epochs. This is the same reasoning behind the clipping: both the clip and the LR
are guards against the policy drifting too far from the behaviour policy.

---

### 2 · Learning Epochs: `8` → `10`

#### What is a rollout?

A **rollout** is a block of experience collected by running the current policy
in the environment for a fixed number of steps before doing any learning.

```
rollout (1024 steps)
│
├── step 1:  observe s₁ → policy outputs a₁ → env returns r₁, s₂
├── step 2:  observe s₂ → policy outputs a₂ → env returns r₂, s₃
├── ...
└── step 1024: observe s₁₀₂₄ → ...
                          ↓
              compute advantages (GAE)
              run learning_epochs gradient updates
              discard data, collect next rollout
```

The key point: **all 1024 transitions are thrown away after the update**.
PPO is an *on-policy* algorithm — it can only learn from data collected by the
policy that exists right now, because the clipping math assumes the behaviour
policy and the learning policy are close. Stale data would violate that
assumption.

This is why PPO is less sample-efficient than off-policy methods (like SAC or
DDPG, which replay old transitions): every gradient step has to be paid for with
fresh environment interaction.

After every rollout of `rollouts` steps, PPO runs `learning_epochs` full passes
over the collected data, each time sampling `mini_batches` random sub-batches.

**Trade-off:** More epochs extract more gradient signal from each rollout
(higher sample efficiency), but the data becomes increasingly off-policy as the
policy changes. PPO's clip is what makes multiple epochs safe — it prevents the
ratio from moving too far in a single epoch. Together with the lower LR, 10
epochs become sustainable: each individual step is smaller, so even more steps
stay within the safe region.

**Rule of thumb:** Increase `learning_epochs` together with decreasing
`learning_rate`. Raising epochs without lowering LR tends to worsen instability.

---

### 3 · Entropy Regularisation: `0.0` → `0.01`

The full PPO objective is:

$$L(\theta) = L^{\text{CLIP}}(\theta) - c_v \cdot L^{\text{VF}}(\theta) + c_e \cdot \mathcal{H}[\pi_\theta]$$

where $\mathcal{H}[\pi_\theta] = -\mathbb{E}[\log \pi_\theta(a|s)]$ is the
**policy entropy** — a measure of how spread-out (exploratory) the action
distribution is.

A Gaussian policy's entropy is:

$$\mathcal{H} = \frac{1}{2}\ln(2\pi e\,\sigma^2)$$

When $\sigma \to 0$, entropy $\to -\infty$ and the policy becomes deterministic.
Without regularisation, gradient descent happily collapses $\sigma$ once it finds
any locally profitable deterministic action — even if a wider search would find
better ones.

Adding `entropy_loss_scale = 0.01` adds a small positive reward for keeping
$\sigma$ large, preventing premature determinism. The coefficient $c_e = 0.01$
is intentionally small so it does not swamp the task reward signal.

---

### 4 · Gradient Norm Clipping: already `0.5` (made explicit)

Before each Adam step the code calls:

```
torch.nn.utils.clip_grad_norm_(parameters, grad_norm_clip)
```

This rescales the gradient vector so its $\ell_2$ norm never exceeds `0.5`.
Without clipping, a single bad mini-batch (e.g., a rare large advantage) can
produce a gradient whose magnitude dwarfs normal steps, blowing the policy
out of a learned region — the "exploding gradient" problem. Clipping to 0.5 is
conservative and well-established across actor-critic methods.

---

## Interaction Between the Four Settings

The settings are not independent:

```
Lower LR → smaller steps → can safely do more epochs
More epochs → more gradient updates per rollout → entropy decays faster → need entropy bonus
Entropy bonus → policy stays exploratory → better coverage → more informative rollouts
Grad clip → prevents any single outlier update from negating the above
```

Tuning one without considering the others is why naive hyperparameter search
often fails. This example changes all four together as a known-good stable
configuration.

---

## Comparison With Example 02 Defaults

| Parameter | Example 02 | Example 03 | Effect of change |
|---|---|---|---|
| `learning_rate` | `3e-4` | `1e-4` | Smaller steps, more stable updates |
| `learning_epochs` | `8` | `10` | More passes, better sample efficiency |
| `entropy_loss_scale` | `0.0` | `0.01` | Prevents premature policy collapse |
| `grad_norm_clip` | `0.5` | `0.5` | Unchanged — already conservative |

---

## Run

All commands must be run from the **project root** using the `-m` flag.

```bash
# Train with stability overrides
python -m examples.03_ppo_stability.train --timesteps 100000

# Shorter run for quick iteration
python -m examples.03_ppo_stability.train --timesteps 20000 --seed 42

# Evaluate the result
python -m examples.02_ppo_classic_control.eval \
    --checkpoint runs/03_ppo_stability/03_ppo_stability/checkpoints/agent_5000.pt \
    --render

# Plot training curves
python -m examples.shared.plot runs/03_ppo_stability/03_ppo_stability
```

## Key Learning Point

PPO hyperparameters interact: lower the learning rate when you raise the number
of epochs; add entropy regularisation when the policy collapses early; always
clip gradients. Changing one without understanding the others is the most common
source of unexplained PPO failures.
