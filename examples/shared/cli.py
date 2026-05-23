from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class RunConfig:
    """Common runtime knobs shared by all train scripts.

    Keeping these values in a typed dataclass does two things for learners:
    1) makes the training entrypoint explicit and easy to inspect, and
    2) separates algorithm logic from command-line plumbing.
    """

    env_id: str
    timesteps: int
    seed: int
    checkpoint: str | None
    experiment_name: str
    headless: bool
    livestream: int  # 0=off, 1=native client, 2=WebRTC browser


def build_train_parser(default_env_id: str, default_timesteps: int, default_experiment: str) -> argparse.ArgumentParser:
    """Create a uniform training parser used by every lesson.

    A stable CLI surface means learners can move from one folder to the next
    without relearning commands, and only focus on the new RL concept.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default=default_env_id)
    parser.add_argument("--timesteps", type=int, default=default_timesteps)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default=default_experiment)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--livestream",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Isaac Sim streaming: 0=off, 1=native client, 2=WebRTC browser (port 8080)",
    )
    return parser


def parse_run_config(parser: argparse.ArgumentParser) -> RunConfig:
    """Parse CLI args into a typed object used by training scripts."""

    args = parser.parse_args()
    return RunConfig(
        env_id=args.env_id,
        timesteps=args.timesteps,
        seed=args.seed,
        checkpoint=args.checkpoint,
        experiment_name=args.experiment_name,
        headless=args.headless,
        livestream=args.livestream,
    )


def build_eval_parser(default_env_id: str) -> argparse.ArgumentParser:
    """Create a parser for deterministic checkpoint evaluation.

    Evaluation is intentionally isolated from training to reinforce a key RL
    practice: compare policies under a fixed protocol, not during exploration.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default=default_env_id)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment in a window (requires a display).",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record MP4 video of the evaluation episodes.",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="videos",
        help="Directory for recorded videos (default: videos/).",
    )
    return parser
