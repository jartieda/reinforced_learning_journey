from __future__ import annotations

import gymnasium as gym

from skrl.envs.wrappers.torch import wrap_env
from skrl.trainers.torch import SequentialTrainer

from examples.shared.cli import build_train_parser, parse_run_config
from examples.shared.models import MLPGaussianPolicy, MLPValue
from examples.shared.ppo import build_ppo_agent
from examples.shared.utils import default_device, print_next_steps, print_run_banner, set_seed


def main() -> None:
    # Example 02 introduces a production-like split: training and evaluation are
    # separate commands so experiments are easier to reproduce and compare.
    parser = build_train_parser(
        default_env_id="Pendulum-v1",
        default_timesteps=75_000,
        default_experiment="02_ppo_classic_control",
    )
    args = parse_run_config(parser)

    set_seed(args.seed)
    device = default_device()

    # Same task as Example 01, but with cleaner experiment organization.
    render_mode = "human" if not args.headless else "rgb_array"
    env = wrap_env(gym.make(args.env_id, render_mode=render_mode))
    models = {
        "policy": MLPGaussianPolicy(observation_space=env.observation_space, action_space=env.action_space, device=device),
        "value": MLPValue(observation_space=env.observation_space, action_space=env.action_space, device=device),
    }

    agent = build_ppo_agent(
        env=env,
        device=device,
        models=models,
        experiment_directory="runs/02_ppo_classic_control",
        experiment_name=args.experiment_name,
    )

    if args.checkpoint:
        # Resume mode: useful for interrupted jobs or longer schedules.
        agent.load(args.checkpoint)

    print_run_banner("Example 02 - PPO training with separated scripts", args.env_id, args.timesteps, device)

    trainer = SequentialTrainer(cfg={"timesteps": args.timesteps, "headless": args.headless}, env=env, agents=agent)
    trainer.train()

    env.close()
    print_next_steps(
        run_dir="runs/02_ppo_classic_control",
        env_id=args.env_id,
        eval_module="examples.02_ppo_classic_control.eval",
    )


if __name__ == "__main__":
    main()
