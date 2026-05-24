from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import SPHERE_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG


# ── Scene configuration ───────────────────────────────────────────────────────
# InteractiveSceneCfg is a dataclass whose fields describe every asset in the
# scene.  The {ENV_REGEX_NS} token is replaced by IsaacLab at clone time so
# each parallel environment gets its own prim path subtree.
@configclass
class FrankaReachSceneCfg(InteractiveSceneCfg):
    """Scene containing one Franka Panda arm per parallel environment.

    Ground plane and dome light are spawned programmatically in _setup_scene
    (the direct-workflow pattern).  Only physics assets live here so the scene
    cloner can replicate them correctly.
    """

    # Franka Panda arm — FRANKA_PANDA_CFG ships with isaaclab_assets and
    # already defines joint limits, actuator models, and a USD stage path.
    # .replace() returns a copy with only the listed fields overridden.
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            # Safe home configuration: arm above the table, fingers open.
            # Joints are named panda_joint1…7 (arm) + panda_finger_joint1/2.
            joint_pos={
                "panda_joint1":  0.0,
                "panda_joint2": -0.569,
                "panda_joint3":  0.0,
                "panda_joint4": -2.810,
                "panda_joint5":  0.0,
                "panda_joint6":  3.037,
                "panda_joint7":  0.741,
                "panda_finger_joint.*": 0.04,  # regex matches both finger joints
            },
        ),
    )


# ── Task configuration ────────────────────────────────────────────────────────
@configclass
class FrankaReachEnvCfg(DirectRLEnvCfg):
    """Full task configuration for the Franka end-effector reaching exercise.

    Changing any of the fields below (e.g. num_envs, episode_length_s,
    goal ranges, reward scales) is the main way to experiment with the task
    without touching the environment implementation code.
    """

    # ── Simulation ────────────────────────────────────────────────────────────
    # dt=1/120 s with decimation=4 → 30 Hz policy control frequency.
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=4)
    decimation: int = 4

    # Episode time limit.  With 30 Hz control that is 150 steps per episode.
    episode_length_s: float = 5.0

    # ── Scene ─────────────────────────────────────────────────────────────────
    # Reduce num_envs when running locally on a smaller GPU (e.g. 512 or 1024).
    scene: FrankaReachSceneCfg = FrankaReachSceneCfg(num_envs=4096, env_spacing=2.5)

    # ── Observation / action spaces ───────────────────────────────────────────
    # Observation: 7 arm joint pos + 7 arm joint vel + 3 EE pos + 3 goal pos
    #            = 20-dimensional vector (all proprioceptive, no camera).
    num_observations: int = 20

    # Action: one normalised joint-position delta per arm DOF (fingers fixed).
    num_actions: int = 7

    # Scale factor: policy output is in [-1, 1]; this converts to radians.
    # Smaller → smoother but slower movements.
    action_scale: float = 0.5

    # ── Goal visualisation ────────────────────────────────────────────────────
    # A green sphere is rendered at the goal position each reset.
    # SPHERE_MARKER_CFG is a built-in; .replace() customises radius and colour.
    goal_marker: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/GoalMarker",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.05,  # 5 cm — matches the success_threshold below
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0)  # green
                ),
            ),
        },
    )

    # ── Reward shaping ────────────────────────────────────────────────────────
    # Dense distance penalty: each step the agent loses |dist| * scale.
    dist_reward_scale: float = -1.0

    # One-off success bonus when EE is within success_threshold of the goal.
    success_reward: float = 10.0
    success_threshold: float = 0.05  # metres

    # ── Goal sampling workspace (metres, relative to robot base) ─────────────
    # These ranges define where goals can appear.  They should stay inside
    # the Franka's ~0.85 m reach radius and above table height.
    goal_x_range: tuple = (0.20, 0.55)   # forward
    goal_y_range: tuple = (-0.35, 0.35)  # lateral
    goal_z_range: tuple = (0.25, 0.65)   # vertical
