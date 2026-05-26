"""
Example 09 — Asset Zoo: browse.py
==================================
Spawns any robot from the isaaclab_assets catalogue into a bare scene,
runs 200 physics steps, then prints joint names, limits, and DOF count.

No RL, no reward.  Pure asset inspection.

Usage
-----
  python examples/09_asset_zoo/browse.py --list
  python examples/09_asset_zoo/browse.py --robot franka
  python examples/09_asset_zoo/browse.py --robot anymal_c
  python examples/09_asset_zoo/browse.py --robot ur10
  python examples/09_asset_zoo/browse.py --robot h1
  python examples/09_asset_zoo/browse.py --robot h1_minimal
  python examples/09_asset_zoo/browse.py --robot g1
  python examples/09_asset_zoo/browse.py --robot g1_minimal
  python examples/09_asset_zoo/browse.py --robot humanoid
  python examples/09_asset_zoo/browse.py --robot go2
  python examples/09_asset_zoo/browse.py --robot ur5
"""
from __future__ import annotations

# ── Step 1: CLI args (before Isaac Sim boots) ─────────────────────────────────
import argparse
import sys

# Map of short name → (import_module, config_symbol, description)
ROBOT_REGISTRY: dict[str, tuple[str, str, str]] = {
    # ── Manipulation arms ─────────────────────────────────────────────────────
    "franka":               ("isaaclab_assets.robots.franka",            "FRANKA_PANDA_CFG",          "Franka Panda (7-DOF arm + 2 fingers)"),
    "franka_high_pd":       ("isaaclab_assets.robots.franka",            "FRANKA_PANDA_HIGH_PD_CFG",  "Franka Panda with stiff PD gains"),
    "ur3":                  ("isaaclab_assets.robots.universal_robots",  "UR3_CFG",                   "Universal Robots UR3 (6-DOF, compact)"),
    "ur5":                  ("isaaclab_assets.robots.universal_robots",  "UR5_CFG",                   "Universal Robots UR5 (6-DOF, mid-reach)"),
    "ur10":                 ("isaaclab_assets.robots.universal_robots",  "UR10_CFG",                  "Universal Robots UR10 (6-DOF, long-reach)"),
    "ur10e":                ("isaaclab_assets.robots.universal_robots",  "UR10e_CFG",                 "Universal Robots UR10e (e-Series)"),
    "kinova_6dof":          ("isaaclab_assets.robots.kinova",            "KINOVA_JACO2_N6S300_CFG",   "Kinova Jaco 2 — 6-DOF"),
    "kinova_7dof":          ("isaaclab_assets.robots.kinova",            "KINOVA_JACO2_N7S300_CFG",   "Kinova Jaco 2 — 7-DOF"),
    # ── Legged quadrupeds ─────────────────────────────────────────────────────
    "anymal_b":             ("isaaclab_assets.robots.anymal",            "ANYMAL_B_CFG",              "ANYmal B quadruped (12-DOF)"),
    "anymal_c":             ("isaaclab_assets.robots.anymal",            "ANYMAL_C_CFG",              "ANYmal C quadruped (12-DOF)"),
    "anymal_d":             ("isaaclab_assets.robots.anymal",            "ANYMAL_D_CFG",              "ANYmal D quadruped (12-DOF)"),
    "a1":                   ("isaaclab_assets.robots.unitree",           "UNITREE_A1_CFG",             "Unitree A1 (12-DOF)"),
    "go1":                  ("isaaclab_assets.robots.unitree",           "UNITREE_GO1_CFG",            "Unitree Go1 (12-DOF)"),
    "go2":                  ("isaaclab_assets.robots.unitree",           "UNITREE_GO2_CFG",            "Unitree Go2 (12-DOF)"),
    # ── Legged humanoids ──────────────────────────────────────────────────────
    "h1":                   ("isaaclab_assets.robots.unitree",           "H1_CFG",                    "Unitree H1 humanoid — full (19-DOF)"),
    "h1_minimal":           ("isaaclab_assets.robots.unitree",           "H1_MINIMAL_CFG",             "Unitree H1 humanoid — legs only (10-DOF)"),
    "g1":                   ("isaaclab_assets.robots.unitree",           "G1_CFG",                    "Unitree G1 humanoid — full (37-DOF)"),
    "g1_minimal":           ("isaaclab_assets.robots.unitree",           "G1_MINIMAL_CFG",             "Unitree G1 humanoid — minimal (21-DOF)"),
    # ── Classic RL benchmarks ─────────────────────────────────────────────────
    "humanoid":             ("isaaclab_assets.robots.humanoid",          "HUMANOID_CFG",              "OpenAI Humanoid — MuJoCo classic (17-DOF)"),
    "cartpole":             ("isaaclab_assets.robots.cartpole",          "CARTPOLE_CFG",              "Cartpole — cart-on-rail balance (1-DOF)"),
}


def list_robots() -> None:
    """Print all registered robots and exit."""
    print("\nAvailable robots (pass to --robot):\n")
    col_w = max(len(k) for k in ROBOT_REGISTRY) + 2
    for key, (_, _, desc) in sorted(ROBOT_REGISTRY.items()):
        print(f"  {key:<{col_w}} {desc}")
    print()


parser = argparse.ArgumentParser(
    description="Example 09 — spawn a robot and print its joint catalogue."
)
parser.add_argument(
    "--robot", type=str, default="franka",
    help="Short robot name. Use --list to see all options.",
)
parser.add_argument(
    "--list", action="store_true",
    help="Print all available robots and exit (no simulation).",
)
parser.add_argument(
    "--num-steps", type=int, default=200,
    help="Number of physics steps to simulate (default: 200).",
)
parser.add_argument(
    "--livestream", type=int, default=0, choices=[0, 1, 2],
    help="0=off, 1=native Omniverse client, 2=WebRTC (port 8080).",
)
args = parser.parse_args()

if args.list:
    list_robots()
    sys.exit(0)

if args.robot not in ROBOT_REGISTRY:
    print(f"\n[browse.py] Unknown robot '{args.robot}'. Run with --list to see options.\n")
    sys.exit(1)

# ── Step 2: Boot the Omniverse runtime ───────────────────────────────────────
try:
    from isaaclab.app import AppLauncher
except ImportError as exc:
    raise SystemExit(
        "\n[Example 09] Isaac Lab is not installed.\n"
        "Run inside the Isaac Sim container or a native Isaac Lab install.\n"
        "See: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/\n"
    ) from exc

simulation_app = AppLauncher(
    {"headless": True, "enable_cameras": False, "livestream": args.livestream}
).app

# ── Step 3: Import isaaclab modules (runtime is now running) ─────────────────
import importlib

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass


# ── Step 4: Dynamically load the chosen robot config ─────────────────────────
mod_path, cfg_name, description = ROBOT_REGISTRY[args.robot]

try:
    module = importlib.import_module(mod_path)
    robot_cfg_template: ArticulationCfg = getattr(module, cfg_name)
except (ImportError, AttributeError) as exc:
    simulation_app.close()
    raise SystemExit(
        f"\n[browse.py] Could not load '{cfg_name}' from '{mod_path}'.\n"
        f"Error: {exc}\n"
        "This robot may require a newer version of isaaclab_assets.\n"
    ) from exc

print(f"\n{'='*60}")
print(f"  Robot  : {description}")
print(f"  Config : {mod_path}.{cfg_name}")
print(f"{'='*60}\n")


# ── Step 5: Build a minimal scene config with the chosen robot ────────────────
@configclass
class AssetZooSceneCfg(InteractiveSceneCfg):
    """Single-robot scene for asset inspection."""

    robot: ArticulationCfg = robot_cfg_template.replace(
        prim_path="/World/Robot",
    )


# ── Step 6: Simulation loop ───────────────────────────────────────────────────
def main() -> None:
    # Create simulation context (single environment — no cloning needed here)
    sim_cfg = sim_utils.SimulationCfg(dt=1 / 120, render_interval=4)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.5, 2.5, 2.0], target=[0.0, 0.0, 0.5])

    # Ground plane
    sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())

    # Light
    sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)).func(
        "/World/DomeLight", sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9))
    )

    # Scene (single env)
    scene_cfg = AssetZooSceneCfg(num_envs=1, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)

    # Reset before first step
    sim.reset()
    scene.reset()

    robot: Articulation = scene["robot"]

    # ── Print joint catalogue ─────────────────────────────────────────────────
    joint_names  = robot.data.joint_names
    pos_limits   = robot.data.soft_joint_pos_limits[0]   # shape (num_joints, 2)
    default_pos  = robot.data.default_joint_pos[0]        # shape (num_joints,)

    print(f"Joint catalogue ({len(joint_names)} joints total):\n")
    header = f"  {'#':<5} {'Joint name':<40} {'Min (rad)':>10} {'Max (rad)':>10} {'Default':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, name in enumerate(joint_names):
        lo  = pos_limits[i, 0].item()
        hi  = pos_limits[i, 1].item()
        dflt = default_pos[i].item()
        print(f"  {i:<5} {name:<40} {lo:>10.3f} {hi:>10.3f} {dflt:>10.3f}")

    print(f"\nControlled DOFs : {robot.num_joints}")
    print(f"Physics steps   : {args.num_steps}\n")

    # ── Simulate ──────────────────────────────────────────────────────────────
    print("Running physics simulation …", flush=True)
    for step in range(args.num_steps):
        # Zero-torque / hold default — robot just sits in its spawn pose
        zero_action = torch.zeros(1, robot.num_joints, device=sim.device)
        robot.set_joint_position_target(zero_action)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_cfg.dt)

        if (step + 1) % 50 == 0:
            print(f"  step {step + 1:>4}/{args.num_steps}")

    print("\nDone. Closing simulation.\n")


main()
simulation_app.close()
