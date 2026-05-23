from __future__ import annotations

from skrl.envs.wrappers.torch import wrap_env
from skrl.trainers.torch import SequentialTrainer

from examples.shared.cli import build_train_parser, parse_run_config
from examples.shared.models import MLPGaussianPolicy, MLPValue
from examples.shared.ppo import build_ppo_agent
from examples.shared.utils import default_device, print_next_steps, print_run_banner, set_seed
from examples.shared.wrappers import make_carracing_pixels


def main() -> None:
    # This lesson changes only the observation modality (pixels) while keeping
    # PPO mechanics familiar from vector-state examples.
    parser = build_train_parser(
        default_env_id="CarRacing-v3",
        default_timesteps=150_000,
        default_experiment="04_pixels_pipeline",
    )
    args = parse_run_config(parser)

    set_seed(args.seed)
    device = default_device()

    # Start with grayscale pixels and no frame stack to isolate preprocessing
    # concepts before introducing temporal context.
    render_mode = "human" if not args.headless else "rgb_array"
    env = wrap_env(make_carracing_pixels(frame_stack=1, render_mode=render_mode))

    models = {
        "policy": MLPGaussianPolicy(observation_space=env.observation_space, action_space=env.action_space, device=device),
        "value": MLPValue(observation_space=env.observation_space, action_space=env.action_space, device=device),
    }
    agent = build_ppo_agent(
        env=env,
        device=device,
        models=models,
        experiment_directory="runs/04_pixels_pipeline",
        experiment_name=args.experiment_name,
        # Pixel tasks are often noisier and higher-dimensional than low-dim
        # control; a lower LR is a safer baseline.
        overrides={"learning_rate": 1e-4},
    )

    if args.checkpoint:
        agent.load(args.checkpoint)

    print_run_banner("Example 04 - Pixel pipeline with PPO", args.env_id, args.timesteps, device)

    trainer = SequentialTrainer(cfg={"timesteps": args.timesteps, "headless": args.headless}, env=env, agents=agent)
    trainer.train()

    env.close()
    print_next_steps(
        run_dir="runs/04_pixels_pipeline",
        env_id=args.env_id,
        eval_module="examples.shared.export",
        model="mlp",
    )


if __name__ == "__main__":
    main()
