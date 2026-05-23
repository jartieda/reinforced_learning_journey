from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from skrl.envs.wrappers.torch import wrap_env
from skrl.trainers.torch import SequentialTrainer

from examples.shared.cli import build_eval_parser
from examples.shared.models import MLPGaussianPolicy, MLPValue
from examples.shared.ppo import build_ppo_agent
from examples.shared.utils import default_device, set_seed


def main() -> None:
    # Evaluation mirrors the training architecture exactly. If model classes or
    # spaces differ from training, checkpoint loading/evaluation will be invalid.
    parser = build_eval_parser(default_env_id="Pendulum-v1")
    args = parser.parse_args()

    set_seed(args.seed)
    device = default_device()

    # Choose render mode based on flags:
    #   --render  → opens a live window so you can watch the agent.
    #   --record  → captures rgb_array frames for video (no window needed).
    if args.render:
        render_mode = "human"
    elif args.record:
        render_mode = "rgb_array"
    else:
        render_mode = None

    raw_env = gym.make(args.env_id, render_mode=render_mode)

    if args.record:
        from gymnasium.wrappers import RecordVideo
        video_dir = Path(args.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        raw_env = RecordVideo(
            raw_env,
            video_folder=str(video_dir),
            episode_trigger=lambda _: True,
            name_prefix="eval",
            disable_logger=True,
        )
        print(f"Recording video to: {video_dir.resolve()}")

    env = wrap_env(raw_env)
    models = {
        "policy": MLPGaussianPolicy(observation_space=env.observation_space, action_space=env.action_space, device=device),
        "value": MLPValue(observation_space=env.observation_space, action_space=env.action_space, device=device),
    }

    agent = build_ppo_agent(
        env=env,
        device=device,
        models=models,
        experiment_directory="runs/02_ppo_classic_control",
        experiment_name="eval",
    )
    agent.load(args.checkpoint)

    # Convert "episodes" into a rough step budget for the evaluator.
    timesteps = max(1_000, args.episodes * 500)
    trainer = SequentialTrainer(cfg={"timesteps": timesteps, "headless": not args.render}, env=env, agents=agent)
    trainer.eval()

    env.close()


if __name__ == "__main__":
    main()
