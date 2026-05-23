from __future__ import annotations

from skrl.agents.torch.ppo import PPO
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler


def build_ppo_agent(
    env,
    device: str,
    models: dict,
    experiment_directory: str,
    experiment_name: str,
    overrides: dict | None = None,
):
    """Build a PPO agent with teaching-oriented defaults.

    The values here are not globally optimal; they are intentionally moderate so
    learners can see stable behavior across multiple tasks and then tune from a
    known baseline.
    """

    # PPO stores rollout transitions before each update phase. In on-policy RL,
    # memory is consumed quickly and replaced by fresh trajectories.
    memory = RandomMemory(memory_size=1024, num_envs=env.num_envs, device=device)

    # Define PPO config using only fields valid in skrl 2.x PPO_CFG.
    # Key renames from skrl 1.x: "lambda" -> "gae_lambda".
    # Removed keys that no longer exist: "clip_predicted_values".
    cfg = {
        # Number of environment steps collected before each optimization phase.
        "rollouts": 1024,
        # How many passes over each rollout batch during PPO updates.
        "learning_epochs": 8,
        # Split rollout into mini-batches for stochastic gradient steps.
        "mini_batches": 4,
        # Discount factor for future rewards (gamma).
        "discount_factor": 0.99,
        # GAE smoothing parameter lambda; balances bias vs variance in advantages.
        "gae_lambda": 0.95,
        # Adam learning rate; conservative default suitable for most tasks.
        "learning_rate": 3e-4,
        # Clip gradient norm to prevent destructive optimizer steps.
        "grad_norm_clip": 0.5,
        # PPO clipping epsilon: limits how far the new policy can diverge.
        "ratio_clip": 0.2,
        # Value function clipping to stabilize critic updates.
        "value_clip": 0.2,
        # Entropy regularization; set > 0 to encourage exploration.
        "entropy_loss_scale": 0.0,
        # Weight of value loss relative to policy loss.
        "value_loss_scale": 2.5,
        # Early stop threshold on KL divergence (0 = disabled).
        "kl_threshold": 0.0,
        # Running standardization of observations improves optimization stability.
        "state_preprocessor": RunningStandardScaler,
        "state_preprocessor_kwargs": {"size": env.observation_space, "device": device},
        # Running standardization of value targets reduces scale sensitivity.
        "value_preprocessor": RunningStandardScaler,
        "value_preprocessor_kwargs": {"size": 1, "device": device},
        "experiment": {
            # How often (in timesteps) to write tensorboard logs.
            "write_interval": 250,
            # How often (in timesteps) to save a checkpoint.
            "checkpoint_interval": 1000,
            "directory": experiment_directory,
            "experiment_name": experiment_name,
        },
    }
    if overrides:
        cfg.update(overrides)

    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    return agent
