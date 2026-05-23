from __future__ import annotations

import gymnasium as gym

from skrl.envs.wrappers.torch import wrap_env
from skrl.trainers.torch import SequentialTrainer

from examples.shared.cli import build_train_parser, parse_run_config
from examples.shared.models import MLPGaussianPolicy, MLPValue
from examples.shared.ppo import build_ppo_agent
from examples.shared.utils import default_device, print_next_steps, print_run_banner, set_seed


def main() -> None:
    # Example 03 keeps architecture/task fixed and changes only optimizer-style
    # settings. This isolates hyperparameter effects from model/env changes.
    parser = build_train_parser(
        default_env_id="Pendulum-v1",
        default_timesteps=100_000,
        default_experiment="03_ppo_stability",
    )
    args = parse_run_config(parser)

    set_seed(args.seed)
    device = default_device()

    render_mode = "human" if not args.headless else "rgb_array"
    env = wrap_env(gym.make(args.env_id, render_mode=render_mode))
    models = {
        "policy": MLPGaussianPolicy(observation_space=env.observation_space, action_space=env.action_space, device=device),
        "value": MLPValue(observation_space=env.observation_space, action_space=env.action_space, device=device),
    }

    # Stability-focused defaults: lower learning rate, more epochs, and entropy bonus.
    overrides = {
        # Smaller steps reduce destructive policy updates.
        "learning_rate": 1e-4,
        # More epochs extract more signal from each rollout batch.
        "learning_epochs": 10,
        # Entropy keeps policy from collapsing too early.
        "entropy_loss_scale": 0.01,
        # Clip large gradients to avoid optimizer explosions.
        "grad_norm_clip": 0.5,
    }

    agent = build_ppo_agent(
        env=env,
        device=device,
        models=models,
        experiment_directory="runs/03_ppo_stability",
        experiment_name=args.experiment_name,
        overrides=overrides,
    )

    if args.checkpoint:
        agent.load(args.checkpoint)

    print_run_banner("Example 03 - PPO stability settings", args.env_id, args.timesteps, device)

    trainer = SequentialTrainer(cfg={"timesteps": args.timesteps, "headless": args.headless}, env=env, agents=agent)
    trainer.train()

    env.close()
    print_next_steps(
        run_dir="runs/03_ppo_stability",
        env_id=args.env_id,
        eval_module="examples.02_ppo_classic_control.eval",
    )


if __name__ == "__main__":
    main()
