from __future__ import annotations

# ── Step 1: Parse our CLI args BEFORE starting the Isaac Sim runtime ─────────
# The AppLauncher below will launch the Omniverse Kit process, which is heavy.
# We parse first so a typo in args fails fast rather than after a 30-second
# simulator boot.
from examples.shared.cli import build_train_parser, parse_run_config

parser = build_train_parser(
    default_env_id="Isaac-Cartpole-Camera-Showcase-Box-Box-Direct-v0",
    default_timesteps=200_000,
    default_experiment="07_isaac_transfer",
)
args = parse_run_config(parser)

# ── Step 2: Boot Isaac Lab's Omniverse Kit runtime ────────────────────────────
# Isaac Lab wraps the closed-source Omniverse Kit (Isaac Sim).  The AppLauncher
# must be created before ANY omni.* or isaaclab.* import; those modules hook
# into the running Kit process.  Importing them before the launcher starts
# causes hard crashes.
#
# In Docker (docker_train.sh) the container already has Isaac Lab installed.
# Outside Docker, this raises a clear message pointing to the Docker workflow.
try:
    from isaaclab.app import AppLauncher
except ImportError as exc:
    raise SystemExit(
        "\n[Example 07] Isaac Lab is not installed.\n"
        "\nIsaac Sim requires native Ubuntu 22.04/24.04 or a cloud GPU — "
        "WSL2 is NOT supported (missing Vulkan/DRM passthrough).\n"
        "\nOptions:\n"
        "  A) Native Linux:  install Isaac Lab and run directly\n"
        "       https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/\n"
        "  B) Docker on native Linux:  bash examples/07_isaac_transfer/docker_train.sh\n"
        "  C) Cloud GPU (AWS g4dn, GCP a2, Azure NC):  pull the container there\n"
    ) from exc

# enable_cameras=True is required for any image-observation environment.
# livestream: 0=off, 1=native Omniverse client, 2=WebRTC browser on port 8080.
# With livestream>0, headless can stay True (the stream IS the viewport).
simulation_app = AppLauncher(
    {"headless": True, "enable_cameras": True, "livestream": args.livestream}
).app

# ── Step 3: All Isaac / omni imports are safe from here onward ───────────────
import gymnasium as gym
import isaaclab_tasks  # noqa: F401  – side-effect: registers Isaac Lab envs

from skrl.envs.wrappers.torch import wrap_env
from skrl.trainers.torch import SequentialTrainer

from examples.shared.models import CNNGaussianPolicy, CNNValue
from examples.shared.ppo import build_ppo_agent
from examples.shared.utils import default_device, print_run_banner, set_seed


def main() -> None:
    # ── Capstone lesson: this setup is intentionally identical to Ex. 05/06 ──
    # The PPO agent, CNN models, and training loop transfer unchanged.
    # Only three things are Isaac-specific:
    #   1. The AppLauncher bootstrap above.
    #   2. The `isaaclab_tasks` import that registers the environment.
    #   3. The `simulation_app.close()` shutdown call at the end.
    set_seed(args.seed)
    device = default_device()

    # Replace the env-id with any Isaac Lab camera-observation task registered
    # in your container.  Browse available envs with:
    #   python -c "import isaaclab_tasks, gymnasium; print(list(gymnasium.envs.registry))"
    # Default: Isaac-Cartpole-Camera-Showcase-Box-Box-Direct-v0
    #   (continuous box obs from a camera + continuous box actions — matches CNN policy)
    #
    # Isaac Lab "direct" workflow envs store their config class in the gym registry
    # under the key "env_cfg_entry_point".  gymnasium does NOT auto-resolve this;
    # we must import the class and pass it as cfg=<instance> to gym.make().
    import importlib

    spec = gym.spec(args.env_id)
    cfg_entry = (spec.kwargs or {}).get("env_cfg_entry_point", "")
    if cfg_entry:
        module_path, class_name = cfg_entry.split(":")
        env_cfg = getattr(importlib.import_module(module_path), class_name)()
        # Camera obs are (100, 100, 3) = 30 k floats/env/step.  With the default
        # num_envs ~512 the rollout buffer alone needs ~59 GB — far beyond a 24 GB
        # RTX 4090.  Cap to 32 envs (≈ 3.8 GB for the 1024-step buffer).
        if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs"):
            env_cfg.scene.num_envs = min(env_cfg.scene.num_envs, 32)
        env = wrap_env(gym.make(args.env_id, cfg=env_cfg))
    else:
        env = wrap_env(gym.make(args.env_id))

    models = {
        "policy": CNNGaussianPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        ),
        "value": CNNValue(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        ),
    }
    agent = build_ppo_agent(
        env=env,
        device=device,
        models=models,
        experiment_directory="runs/07_isaac_transfer",
        experiment_name=args.experiment_name,
        # Conservative LR and larger mini-batches suit simulator complexity.
        overrides={"learning_rate": 1e-4, "mini_batches": 8},
    )

    if args.checkpoint:
        agent.load(args.checkpoint)

    print_run_banner(
        "Example 07 - Isaac transfer with image observations",
        args.env_id,
        args.timesteps,
        device,
    )

    trainer = SequentialTrainer(
        cfg={"timesteps": args.timesteps, "headless": True},
        env=env,
        agents=agent,
    )
    trainer.train()

    env.close()
    simulation_app.close()  # graceful Kit runtime shutdown — always required


if __name__ == "__main__":
    main()
