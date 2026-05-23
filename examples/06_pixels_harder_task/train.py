from __future__ import annotations

from skrl.envs.wrappers.torch import wrap_env
from skrl.trainers.torch import SequentialTrainer

from examples.shared.cli import build_train_parser, parse_run_config
from examples.shared.models import CNNGaussianPolicy, CNNValue
from examples.shared.ppo import build_ppo_agent
from examples.shared.utils import default_device, print_next_steps, print_run_banner, set_seed
from examples.shared.wrappers import make_carracing_pixels


def main() -> None:
    # Example 06 scales difficulty by adding temporal context and stronger
    # optimization settings, approximating a more realistic vision workflow.
    parser = build_train_parser(
        default_env_id="CarRacing-v3",
        default_timesteps=400_000,
        default_experiment="06_pixels_harder_task",
    )
    args = parse_run_config(parser)

    set_seed(args.seed)
    device = default_device()

    # Frame stacking supplies short motion history, helping policy infer latent
    # velocity/heading from pixels alone.
    render_mode = "human" if not args.headless else "rgb_array"
    env = wrap_env(make_carracing_pixels(frame_stack=4, render_mode=render_mode))

    models = {
        "policy": CNNGaussianPolicy(observation_space=env.observation_space, action_space=env.action_space, device=device),
        "value": CNNValue(observation_space=env.observation_space, action_space=env.action_space, device=device),
    }

    overrides = {
        # Conservative optimizer step size for high-variance pixel gradients.
        "learning_rate": 1e-4,
        # Extra optimization passes improve sample usage.
        "learning_epochs": 10,
        # More mini-batches reduce update variance.
        "mini_batches": 8,
        # Encourage broader exploration in harder tracks.
        "entropy_loss_scale": 0.01,
    }
    agent = build_ppo_agent(
        env=env,
        device=device,
        models=models,
        experiment_directory="runs/06_pixels_harder_task",
        experiment_name=args.experiment_name,
        overrides=overrides,
    )

    if args.checkpoint:
        agent.load(args.checkpoint)

    print_run_banner("Example 06 - Harder vision task with frame stacking", args.env_id, args.timesteps, device)

    trainer = SequentialTrainer(cfg={"timesteps": args.timesteps, "headless": args.headless}, env=env, agents=agent)
    trainer.train()

    env.close()
    print_next_steps(
        run_dir="runs/06_pixels_harder_task",
        env_id=args.env_id,
        eval_module="examples.shared.export",
        model="cnn",
    )


if __name__ == "__main__":
    main()
