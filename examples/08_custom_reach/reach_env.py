from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import sample_uniform

from env_cfg import FrankaReachEnvCfg  # local import — works after sys.path setup in train.py


class FrankaReachEnv(DirectRLEnv):
    """Franka Panda end-effector reaching task — custom DirectRLEnv.

    This class is the heart of Example 08.  Unlike the previous examples that
    called gym.make() on a pre-built Isaac Lab environment, here *we* define:
      • The scene  — which assets to spawn and how (see _setup_scene).
      • The actions — normalised joint-position deltas (_pre_physics_step).
      • The observations — proprioceptive joint + EE + goal vector (_get_observations).
      • The reward — dense distance penalty + success bonus (_get_rewards).
      • The resets — perturbed home pose + randomised goal (_reset_idx).

    The PPO agent and skrl trainer are unchanged from earlier examples.
    """

    cfg: FrankaReachEnvCfg

    # ── Initialisation ─────────────────────────────────────────────────────────
    def __init__(self, cfg: FrankaReachEnvCfg, render_mode: str | None = None, **kwargs):
        # super().__init__ boots the scene (calls _setup_scene), registers
        # gymnasium spaces from cfg.num_observations / num_actions, and starts
        # the physics simulation loop.
        super().__init__(cfg, render_mode, **kwargs)

        # ── Resolve arm joint indices ──────────────────────────────────────
        # find_joints returns (list_of_ids, list_of_names).  We only control
        # the 7 arm joints; the finger joints stay at the init_state value.
        arm_names = [f"panda_joint{i}" for i in range(1, 8)]
        self._arm_joint_ids, _ = self.robot.find_joints(arm_names)

        # ── Resolve end-effector body index ───────────────────────────────
        # "panda_hand" is the rigid body at the wrist — a reliable EE proxy.
        ee_body_ids, _ = self.robot.find_bodies("panda_hand")
        self._ee_body_id: int = ee_body_ids[0]

        # ── Joint position limits (used for action clamping) ───────────────
        # soft_joint_pos_limits: (num_envs, total_joints, 2) — lo in [:,:,0]
        limits = self.robot.data.soft_joint_pos_limits[:, self._arm_joint_ids, :]
        # Limits are the same across envs so we take env 0.
        self._joint_lower = limits[0, :, 0]  # (7,)
        self._joint_upper = limits[0, :, 1]  # (7,)

        # ── Default arm pose (offset for delta actions) ────────────────────
        self._default_arm_pos = self.robot.data.default_joint_pos[:, self._arm_joint_ids].clone()

        # ── Goal positions (world frame, one per env) ──────────────────────
        self._goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # ── Goal visualisation marker ──────────────────────────────────────
        self._goal_marker = VisualizationMarkers(self.cfg.goal_marker)

    # ── Scene construction ─────────────────────────────────────────────────────
    def _setup_scene(self) -> None:
        """Spawn all physics assets and prepare the environment grid.

        Called once by super().__init__ before any physics step.

        Key rule: everything added to self.scene.articulations / rigid_objects
        is cloned across all parallel environments by clone_environments().
        Assets spawned at absolute paths (ground, lights) are NOT cloned.
        """
        # Instantiate the Franka articulation from the config.
        # Articulation reads the USD stage path, joint limits, and actuator
        # models from cfg and registers itself with Isaac Sim's physics engine.
        self.robot = Articulation(self.cfg.scene.robot)
        self.scene.articulations["robot"] = self.robot

        # Ground plane — a single shared asset at an absolute path, so it is
        # NOT cloned per env.  Using the .func() callable pattern that
        # IsaacLab direct-workflow examples follow.
        sim_utils.GroundPlaneCfg().func(
            "/World/defaultGroundPlane", sim_utils.GroundPlaneCfg()
        )

        # Dome light — illuminate the whole scene uniformly.
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/DomeLight", light_cfg)

        # Replicate the source environment (env_0) into num_envs copies laid
        # out on a grid with env_spacing separation.
        self.scene.clone_environments(copy_from_source=False)

        # Prevent physics collisions between objects in different env slots.
        self.scene.filter_collisions(global_prim_paths=[])

    # ── Action interface ───────────────────────────────────────────────────────
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Convert normalised policy actions → absolute joint position targets.

        Called once per policy step, before the decimation loop of physics steps.

        actions: (num_envs, 7) in [-1, 1] — the raw policy output.
        Strategy: interpret as deltas on top of the home pose, then clamp to
        joint limits so the arm never hits a hardware stop.
        """
        self._actions = actions.clone().clamp(-1.0, 1.0)
        targets = self._default_arm_pos + self.cfg.action_scale * self._actions
        targets = targets.clamp(self._joint_lower, self._joint_upper)
        # Write targets to the actuator model; the sim drives joints each step.
        self.robot.set_joint_position_target(targets, joint_ids=self._arm_joint_ids)

    def _apply_action(self) -> None:
        """Flush the buffered joint targets to the simulation.

        Called every physics sub-step inside the decimation loop so the
        actuator model is kept in sync with the sim state.
        """
        self.robot.write_data_to_sim()

    # ── Observation ────────────────────────────────────────────────────────────
    def _get_observations(self) -> dict:
        """Build the 20-dim proprioceptive observation vector.

        Components (all in local frame relative to the robot base):
          [0:7]   arm joint positions (radians)
          [7:14]  arm joint velocities (rad/s)
          [14:17] end-effector position (metres, local)
          [17:20] goal position (metres, local)

        Using local frame (subtracting root_pos_w) makes the policy invariant
        to the env-grid offset: env_42 sitting at x=105 m learns the same as
        env_0 at x=0.
        """
        joint_pos  = self.robot.data.joint_pos[:, self._arm_joint_ids]   # (N, 7)
        joint_vel  = self.robot.data.joint_vel[:, self._arm_joint_ids]   # (N, 7)
        ee_pos_w   = self.robot.data.body_pos_w[:, self._ee_body_id, :]  # (N, 3)
        base_pos_w = self.robot.data.root_pos_w                           # (N, 3)

        ee_pos_local   = ee_pos_w         - base_pos_w  # (N, 3)
        goal_pos_local = self._goal_pos_w - base_pos_w  # (N, 3)

        obs = torch.cat([joint_pos, joint_vel, ee_pos_local, goal_pos_local], dim=-1)
        return {"policy": obs}

    # ── Reward ─────────────────────────────────────────────────────────────────
    def _get_rewards(self) -> torch.Tensor:
        """Dense distance penalty + discrete success bonus.

        reward = dist_reward_scale * ‖EE − goal‖  +  success_reward * (dist < threshold)

        The distance term gives a smooth gradient so the policy always knows
        which direction to move.  The success bonus accelerates convergence
        once the arm is close to the goal.
        """
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_id, :]  # (N, 3)
        dist = torch.linalg.norm(ee_pos_w - self._goal_pos_w, dim=-1)  # (N,)

        success = (dist < self.cfg.success_threshold).float()
        reward  = self.cfg.dist_reward_scale * dist + self.cfg.success_reward * success
        return reward

    # ── Episode termination ────────────────────────────────────────────────────
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Episodes end only on timeout — no early termination on success.

        Keeping the episode running after success lets the agent collect
        additional positive rewards for *holding* the goal, which is a more
        physically meaningful behaviour.
        """
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated  = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    # ── Reset ──────────────────────────────────────────────────────────────────
    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset selected environments: perturb robot pose + randomise goal.

        Called on the very first step (all envs) and whenever an episode ends.
        super()._reset_idx resets episode_length_buf for the affected envs.
        """
        super()._reset_idx(env_ids)
        n = len(env_ids)

        # ── Robot: home pose + small uniform noise ─────────────────────────
        # Starting from a perturbed pose rather than exactly the same config
        # each episode helps the policy generalise across initial conditions.
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        noise = (torch.rand(n, len(self._arm_joint_ids), device=self.device) - 0.5) * 0.1
        joint_pos[:, self._arm_joint_ids] += noise
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # ── Goal: random position inside the reachable workspace ───────────
        # sample_uniform draws from [lo, hi] uniformly along each axis.
        base_pos_w = self.robot.data.root_pos_w[env_ids]  # (n, 3)
        lo = torch.tensor(
            [self.cfg.goal_x_range[0], self.cfg.goal_y_range[0], self.cfg.goal_z_range[0]],
            device=self.device,
        )
        hi = torch.tensor(
            [self.cfg.goal_x_range[1], self.cfg.goal_y_range[1], self.cfg.goal_z_range[1]],
            device=self.device,
        )
        goal_offset = sample_uniform(lo, hi, (n, 3), self.device)
        self._goal_pos_w[env_ids] = base_pos_w + goal_offset

        # Update all marker positions with a single USD write.
        # translations expects (num_markers, 3) — one marker per env.
        self._goal_marker.visualize(translations=self._goal_pos_w)
