from __future__ import annotations

import gymnasium as gym

from skrl.envs.wrappers.torch import wrap_env
from skrl.trainers.torch import SequentialTrainer

from examples.shared.cli import build_train_parser, parse_run_config
from examples.shared.models import MLPGaussianPolicy, MLPValue
from examples.shared.ppo import build_ppo_agent
from examples.shared.utils import default_device, print_next_steps, print_run_banner, set_seed


def main() -> None:
    # This first lesson intentionally keeps everything in one file so you can
    # see the entire RL control flow end-to-end.
    parser = build_train_parser(
        default_env_id="Pendulum-v1",
        default_timesteps=30_000,
        default_experiment="01_basics_loop",
    )
    args = parse_run_config(parser)

    # Reproducibility: same seed gives comparable trajectories and updates.
    set_seed(args.seed)
    device = default_device()

    # wrap_env adapts Gymnasium API details to SKRL's torch-friendly interface.
    # render_mode must be set at make() time in Gymnasium 1.x; "rgb_array" avoids
    # warnings during training without opening a display window.
    render_mode = "human" if not args.headless else "rgb_array"
    env = wrap_env(gym.make(args.env_id, render_mode=render_mode))

    # PPO in actor-critic mode needs:
    # - policy: pi(a|s), a stochastic action distribution
    # - value:  V(s), a baseline for low-variance advantage estimates
    models = {
        "policy": MLPGaussianPolicy(observation_space=env.observation_space, action_space=env.action_space, device=device),
        "value": MLPValue(observation_space=env.observation_space, action_space=env.action_space, device=device),
    }
    agent = build_ppo_agent(
        env=env,
        device=device,
        models=models,
        experiment_directory="runs/01_basics_loop",
        experiment_name=args.experiment_name,
    )

    if args.checkpoint:
        # Optional warm start to continue training from a previous run.
        agent.load(args.checkpoint)

    print_run_banner("Example 01 - Minimal SKRL PPO loop", args.env_id, args.timesteps, device)

    # SequentialTrainer drives the interaction/update loop:
    # collect rollout -> optimize PPO objectives -> repeat until timesteps.
    trainer = SequentialTrainer(cfg={"timesteps": args.timesteps, "headless": args.headless}, env=env, agents=agent)
    trainer.train()

    env.close()
    print_next_steps(
        run_dir="runs/01_basics_loop",
        env_id=args.env_id,
        eval_module="examples.02_ppo_classic_control.eval",
    )


if __name__ == "__main__":
    main()
