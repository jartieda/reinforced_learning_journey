from __future__ import annotations

from skrl.envs.wrappers.torch import wrap_env
from skrl.trainers.torch import SequentialTrainer

from examples.shared.cli import build_train_parser, parse_run_config
from examples.shared.ppo import build_ppo_agent
from examples.shared.utils import default_device, print_next_steps, print_run_banner, set_seed
from examples.shared.wrappers import make_carracing_pixels
# model.py re-exports these from examples.shared.models.
# Importing directly because the folder name (05_...) cannot be used as a
# Python package path (starts with a digit).
from examples.shared.models import CNNGaussianPolicy as PolicyCNN
from examples.shared.models import CNNValue as ValueCNN


def main() -> None:
    # Example 05 is the architecture shift: MLP heads are replaced with CNN
    # encoders so policy/value operate directly on image features.
    parser = build_train_parser(
        default_env_id="CarRacing-v3",
        default_timesteps=250_000,
        default_experiment="05_ppo_cnn",
    )
    args = parse_run_config(parser)

    set_seed(args.seed)
    device = default_device()

    render_mode = "human" if not args.headless else "rgb_array"
    env = wrap_env(make_carracing_pixels(frame_stack=1, render_mode=render_mode))

    # Same actor-critic split as before, but both networks now begin with
    # convolutional feature extraction.
    models = {
        "policy": PolicyCNN(observation_space=env.observation_space, action_space=env.action_space, device=device),
        "value": ValueCNN(observation_space=env.observation_space, action_space=env.action_space, device=device),
    }
    agent = build_ppo_agent(
        env=env,
        device=device,
        models=models,
        experiment_directory="runs/05_ppo_cnn",
        experiment_name=args.experiment_name,
        # Slight entropy bonus helps maintain exploration in vision tasks.
        overrides={"learning_rate": 3e-4, "entropy_loss_scale": 0.005},
    )

    if args.checkpoint:
        agent.load(args.checkpoint)

    print_run_banner("Example 05 - PPO with CNN encoder", args.env_id, args.timesteps, device)

    trainer = SequentialTrainer(cfg={"timesteps": args.timesteps, "headless": args.headless}, env=env, agents=agent)
    trainer.train()

    env.close()
    print_next_steps(
        run_dir="runs/05_ppo_cnn",
        env_id=args.env_id,
        eval_module="examples.shared.export",
        model="cnn",
    )


if __name__ == "__main__":
    main()
