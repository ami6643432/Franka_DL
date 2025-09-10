# Copyright (c) 2022-2025, The Isaac Lab Project
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.envs import DirectMARLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.spawners.from_files import spawn_from_usd
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply_inverse,
    quat_inv,
    subtract_frame_transforms,
    sample_uniform,
)

from .franka_dl_marl_env_cfg import FrankaDlMarlEnvCfg


class FrankaDlMarlEnv(DirectMARLEnv):
    """Dual-arm Franka collaborative assembly with 4 agents (motion+stiffness per robot)."""

    cfg: FrankaDlMarlEnvCfg

    # -----------------------------
    # Standard DirectMARLEnv hooks
    # -----------------------------
    def __init__(self, cfg: FrankaDlMarlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # OSC controllers
        self._setup_osc_controllers()

        # Cache indices and precompute constants
        self._cache_robot_indices()
        self._prepare_kp_bounds()

        # Visualization
        if self.cfg.enable_visualization_markers:
            self._setup_visualization()

        # Buffers / state
        self._success_hold_buf = torch.zeros(self.scene.num_envs, device=self.device)
        self._last_actions: dict[str, torch.Tensor] = {}

        # Joint centers (nullspace targets) – only arm joints for OSC
        self._joint_centers_1 = self.robot.data.default_joint_pos[:, self.arm_joint_ids_1].clone()
        self._joint_centers_2 = self.robot2.data.default_joint_pos[:, self.arm_joint_ids_2].clone()
        
        # Store initial end-effector poses for fixed action testing
        self._initial_ee1_pos = None
        self._initial_ee1_quat = None
        self._initial_ee2_pos = None
        self._initial_ee2_quat = None

    def _setup_visualization(self):
        """Set up visualization markers for debugging."""
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.visualization_markers = VisualizationMarkers(marker_cfg)

    def _setup_scene(self):
        """Spawn two Frankas, ground, rod object, and goal fixture."""
        # Clone and replicate environments FIRST
        self.scene.clone_environments(copy_from_source=False)
        
        # Create robots
        self.robot = Articulation(self.cfg.robot)
        self.robot2 = Articulation(self.cfg.robot2)

        # Ground
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Table
        self.table = RigidObject(self.cfg.table)

        # Terrain
        self.terrain = TerrainImporter(self.cfg.terrain)

        # Rod object (dynamic)
        self.rod = RigidObject(self.cfg.rod)

        # Goal fixture (kinematic)
        self.goal_fixture = RigidObject(self.cfg.goal_fixture)
        
        # Register assets to scene collections
        self.scene.articulations["robot"] = self.robot
        self.scene.articulations["robot2"] = self.robot2
        self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_objects["rod"] = self.rod
        self.scene.rigid_objects["goal_fixture"] = self.goal_fixture

        # Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2200.0, color=(0.85, 0.85, 0.85))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        """Combine per-robot actions (motion + stiffness) -> 19D OSC commands."""
        self._last_actions = actions  # for rewards
        
        # TESTING: Use fixed dummy actions instead of agent actions
        # Comment out these lines when not testing
        fixed_actions = True  # Set to False to use original actions
        if fixed_actions:
            # Capture initial poses only once (first step after reset)
            if self._initial_ee1_pos is None:
                self._initial_ee1_pos = self.robot.data.body_pos_w[:, self.ee_frame_idx_1].clone()
                self._initial_ee1_quat = self.robot.data.body_quat_w[:, self.ee_frame_idx_1].clone()
                self._initial_ee2_pos = self.robot2.data.body_pos_w[:, self.ee_frame_idx_2].clone()
                self._initial_ee2_quat = self.robot2.data.body_quat_w[:, self.ee_frame_idx_2].clone()
                print("Captured initial poses for fixed action testing")
            
            # Robot 1 motion: hold initial pose with zero forces/torques
            a1_m = torch.cat([
                self._initial_ee1_pos,   # Initial position XYZ (fixed)
                self._initial_ee1_quat,  # Initial orientation (quat) (fixed)
                torch.zeros(self.scene.num_envs, 6, device=self.device)  # Zero forces and torques
            ], dim=-1)
            
            # Robot 1 stiffness: maximum values for pure position holding
            a1_s = torch.tensor([
                # Maximum stiffness for XYZ and RPY axes (pure position hold mode)
                1.0, 1.0, 1.0,    1.0, 1.0, 1.0
            ], device=self.device).unsqueeze(0).expand(self.scene.num_envs, -1)
            
            # Robot 2 motion: hold initial pose with zero forces/torques
            a2_m = torch.cat([
                self._initial_ee2_pos,   # Initial position XYZ (fixed)
                self._initial_ee2_quat,  # Initial orientation (quat) (fixed)
                torch.zeros(self.scene.num_envs, 6, device=self.device)  # Zero forces and torques
            ], dim=-1)
            
            # Robot 2 stiffness: maximum values for pure position holding
            a2_s = torch.tensor([
                # Maximum stiffness for XYZ and RPY axes (pure position hold mode)
                1.0, 1.0, 1.0,    1.0, 1.0, 1.0
            ], device=self.device).unsqueeze(0).expand(self.scene.num_envs, -1)
            
            # Print debug info every 60 steps (approximately once per second)
            if hasattr(self, 'step_counter'):
                self.step_counter += 1
            else:
                self.step_counter = 0
                
            if self.step_counter % 60 == 0:
                print(f"Step {self.step_counter}: Using fixed test actions")
                ee_pose1 = self.robot.data.body_pos_w[:, self.ee_frame_idx_1]
                print(f"  Robot 1 EE pos: {ee_pose1[0].cpu().numpy()}")
                ee_pose2 = self.robot2.data.body_pos_w[:, self.ee_frame_idx_2]
                print(f"  Robot 2 EE pos: {ee_pose2[0].cpu().numpy()}")
                rod_pos = self.rod.data.root_pos_w[0]
                print(f"  Rod position: {rod_pos.cpu().numpy()}")
        else:
            # Original code: use actions from agents
            a1_m = actions["robot_motion"]        # (N, 13): [pose7 | wrench6]
            a1_s = actions["robot_stiffness"]     # (N, 6):  [Kp6]
            a2_m = actions["robot2_motion"]
            a2_s = actions["robot2_stiffness"]
        
        cmd1 = self._build_osc_command(a1_m, a1_s)  # (N, 19)
        self._set_osc_command(self.robot, self.osc_1, self.ee_frame_idx_1, cmd1)

        # Robot 2
        cmd2 = self._build_osc_command(a2_m, a2_s)
        self._set_osc_command(self.robot2, self.osc_2, self.ee_frame_idx_2, cmd2)
    
    def _apply_action(self) -> None:
        """Compute torques via OSC and apply to both robots."""
        self._apply_osc_control(
            self.robot, self.osc_1, self.ee_frame_idx_1, self.arm_joint_ids_1, self._joint_centers_1
        )
        self._apply_osc_control(
            self.robot2, self.osc_2, self.ee_frame_idx_2, self.arm_joint_ids_2, self._joint_centers_2
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Same global observation to all agents."""
        obs_global = self._build_observation()
        return {
            "robot_motion": obs_global,
            "robot_stiffness": obs_global,
            "robot2_motion": obs_global,
            "robot2_stiffness": obs_global,
        }

    def _get_states(self) -> torch.Tensor:
        """Global state for centralized critic (MAPPO). 
        Concatenate all agent observations for centralized value function."""
        obs_global = self._build_observation()  # (N, 68)
        # For MAPPO, we concatenate observations from all agents
        # Since all agents see the same global state, we can just repeat it
        # State shape: (N, 4 * 68) = (N, 272)
        state = obs_global.repeat(1, 4)  # Repeat for 4 agents
        return state

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        """Group rewards per robot with role-specific terms added."""
        rew1 = self._compute_robot_rewards(robot_id=1)
        rew2 = self._compute_robot_rewards(robot_id=2)

        # Split/assign per agent
        return {
            "robot_motion": rew1["base"] + rew1["motion_terms"] + rew1["conflict_terms"],
            "robot_stiffness": rew1["base"] + rew1["stiffness_terms"],
            "robot2_motion": rew2["base"] + rew2["motion_terms"] + rew2["conflict_terms"],
            "robot2_stiffness": rew2["base"] + rew2["stiffness_terms"],
        }

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Success/timeout/failure conditions per agent."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Success: object near goal for hold time
        obj_pos = self.rod.data.root_pos_w
        obj_quat = self.rod.data.root_quat_w
        goal_pos = self._goal_pos.expand_as(obj_pos)
        goal_quat = self._goal_quat.expand_as(obj_quat)

        pos_ok = torch.norm(obj_pos - goal_pos, dim=-1) < self.cfg.success_position_threshold
        # quat alignment proxy: 1 - |dot(q,g)| ~ angle error; keep simple dot>cos(th)
        quat_dot = torch.sum(obj_quat * goal_quat, dim=-1).abs().clamp(max=1.0)
        ori_ok = quat_dot > torch.cos(torch.tensor(self.cfg.success_orientation_threshold, device=self.device))

        stable = self._object_stable_mask()
        success_mask = pos_ok & ori_ok & stable

        # hold timer
        dt = self.step_dt * self.cfg.decimation
        self._success_hold_buf = torch.where(success_mask, self._success_hold_buf + dt, torch.zeros_like(self._success_hold_buf))
        success = self._success_hold_buf >= self.cfg.success_hold_time_s

        # Failure: drop or safety
        dropped = obj_pos[:, 2] < self.cfg.object_drop_height
        unsafe = self._safety_limit_exceeded()

        terminated = {
            "robot_motion": success | dropped | unsafe,
            "robot_stiffness": success | dropped | unsafe,
            "robot2_motion": success | dropped | unsafe,
            "robot2_stiffness": success | dropped | unsafe,
        }
        timeouts = {
            "robot_motion": time_out,
            "robot_stiffness": time_out,
            "robot2_motion": time_out,
            "robot2_stiffness": time_out,
        }
        return terminated, timeouts

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Reset robots to defaults (per env origin)
        for robot in (self.robot, self.robot2):
            joint_pos = robot.data.default_joint_pos[env_ids]
            joint_vel = robot.data.default_joint_vel[env_ids]
            robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

            default_root_state = robot.data.default_root_state[env_ids]
            default_root_state[:, :3] += self.scene.env_origins[env_ids]
            robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset object: place on table between robots with random XY & yaw
        h = self.cfg.rod.tabletop_height + 0.02  # Table surface height + radius of rod
        pos_xy = sample_uniform(
            lower=torch.tensor([0.50, -0.15], device=self.device),  # Above table area
            upper=torch.tensor([1.00, 0.15], device=self.device),
            size=(len(env_ids), 2),
            device=self.device
        )
        yaw = sample_uniform(
            lower=torch.tensor([-0.25], device=self.device),
            upper=torch.tensor([0.25], device=self.device),
            size=(len(env_ids), 1),
            device=self.device
        )
        # Start with horizontal orientation from config, then add random yaw
        # Base horizontal orientation: (0.7071068, 0.0, 0.7071068, 0.0)
        qw_base = 0.7071068
        qx_base = 0.0
        qy_base = 0.7071068
        qz_base = 0.0
        
        # Random yaw rotation around Z-axis
        half = 0.5 * yaw
        qz_yaw = torch.sin(half)
        qw_yaw = torch.cos(half)
        qx_yaw = torch.zeros_like(half)
        qy_yaw = torch.zeros_like(half)
        
        # Combine base horizontal + yaw: q_total = q_yaw * q_base
        obj_pos = torch.zeros((len(env_ids), 3), device=self.device)
        obj_pos[:, 0:2] = pos_xy
        obj_pos[:, 2] = h  # Rod center at table surface + radius
        obj_quat = torch.zeros((len(env_ids), 4), device=self.device)
        # Quaternion multiplication: q_yaw * q_base
        obj_quat[:, 0] = (qw_yaw * qw_base - qx_yaw * qx_base - qy_yaw * qy_base - qz_yaw * qz_base).squeeze(-1)  # w
        obj_quat[:, 1] = (qw_yaw * qx_base + qx_yaw * qw_base + qy_yaw * qz_base - qz_yaw * qy_base).squeeze(-1)  # x
        obj_quat[:, 2] = (qw_yaw * qy_base - qx_yaw * qz_base + qy_yaw * qw_base + qz_yaw * qx_base).squeeze(-1)  # y
        obj_quat[:, 3] = (qw_yaw * qz_base + qx_yaw * qy_base - qy_yaw * qx_base + qz_yaw * qw_base).squeeze(-1)  # z
        # write to sim (offset by env origins)
        obj_pos += self.scene.env_origins[env_ids]
        self.rod.write_root_pose_to_sim(
            torch.cat([obj_pos, obj_quat], dim=-1), env_ids
        )
        self.rod.write_root_velocity_to_sim(
            torch.zeros((len(env_ids), 6), device=self.device), env_ids
        )

        # Goal fixture pose (fixed): position in each environment
        goal_p, goal_q = self.cfg.goal_fixture.pose_w
        goal_pos = torch.tensor(goal_p, device=self.device).unsqueeze(0).expand(len(env_ids), -1)
        goal_quat = torch.tensor(goal_q, device=self.device).unsqueeze(0).expand(len(env_ids), -1)
        
        # Offset by environment origins
        goal_pos += self.scene.env_origins[env_ids]
        self.goal_fixture.write_root_pose_to_sim(
            torch.cat([goal_pos, goal_quat], dim=-1), env_ids
        )
        self.goal_fixture.write_root_velocity_to_sim(
            torch.zeros((len(env_ids), 6), device=self.device), env_ids
        )

        # Cache goal tensors for fast checks (global frame)
        self._goal_pos = torch.tensor(goal_p, device=self.device).unsqueeze(0)
        self._goal_quat = torch.tensor(goal_q, device=self.device).unsqueeze(0)

        # Reset success timer
        self._success_hold_buf[env_ids] = 0.0

        # Reset OSC internals
        self.osc_1.reset()
        self.osc_2.reset()
        
        # Reset initial pose storage for fixed action testing
        self._initial_ee1_pos = None
        self._initial_ee1_quat = None
        self._initial_ee2_pos = None
        self._initial_ee2_quat = None

    # -----------------------------
    # Internals
    # -----------------------------
    def _setup_osc_controllers(self):
        osc_cfg = OperationalSpaceControllerCfg(
            target_types=["pose_abs", "wrench_abs"],
            impedance_mode="variable_kp",          # adds +6 Kp to command
            motion_damping_ratio_task=2.0,         # Overdamped for pure position hold
            gravity_compensation=True,
            nullspace_control="position",
        )
        self.osc_1 = OperationalSpaceController(osc_cfg, num_envs=self.scene.num_envs, device=self.device)
        self.osc_2 = OperationalSpaceController(osc_cfg, num_envs=self.scene.num_envs, device=self.device)

    def _cache_robot_indices(self):
        # Frames / joints
        ee_name = "panda_hand"
        self.ee_frame_idx_1 = self.robot.find_bodies(ee_name)[0][0]
        self.ee_frame_idx_2 = self.robot2.find_bodies(ee_name)[0][0]
        # 7 DoF arm joints
        self.arm_joint_ids_1 = self.robot.find_joints(["panda_joint.*"])[0]
        self.arm_joint_ids_2 = self.robot2.find_joints(["panda_joint.*"])[0]

        # Cache for object/goal pose tensors
        gp, gq = self.cfg.goal_fixture.pose_w
        self._goal_pos = torch.tensor(gp, device=self.device).unsqueeze(0)
        self._goal_quat = torch.tensor(gq, device=self.device).unsqueeze(0)

    def _prepare_kp_bounds(self):
        self._kp_min = torch.tensor(self.cfg.kp_min_task, device=self.device).unsqueeze(0)  # (1,6)
        self._kp_max = torch.tensor(self.cfg.kp_max_task, device=self.device).unsqueeze(0)  # (1,6)

    def _build_osc_command(self, action_motion: torch.Tensor, action_stiff: torch.Tensor) -> torch.Tensor:
        # action_motion: [pos(3), quat(wxyz)(4), force(3), torque(3)]
        pose = action_motion[:, :7]
        # normalize quaternion (w,x,y,z)
        pose[:, 3:7] = torch.nn.functional.normalize(pose[:, 3:7], dim=-1)

        wrench = action_motion[:, 7:13]
        # clamp wrench by cfg limits
        force = torch.clamp(wrench[:, 0:3], min=-self.cfg.max_force, max=self.cfg.max_force)
        torque = torch.clamp(wrench[:, 3:6], min=-self.cfg.max_torque, max=self.cfg.max_torque)
        wrench = torch.cat([force, torque], dim=-1)

        # stiffness map from [-1,1] to [kp_min, kp_max]
        kp_raw = action_stiff[:, :6]
        kp = 0.5 * (kp_raw + 1.0) * (self._kp_max - self._kp_min) + self._kp_min

        return torch.cat([pose, wrench, kp], dim=-1)  # (N, 19)

    def _set_osc_command(self, robot: Articulation, osc: OperationalSpaceController, ee_frame_idx: int, command: torch.Tensor):
        ee_pose_b, _ = self._ee_state(robot, ee_frame_idx)
        osc.set_command(command=command, current_ee_pose_b=ee_pose_b, current_task_frame_pose_b=ee_pose_b)

    def _apply_osc_control(self, robot, osc, ee_frame_idx, arm_joint_ids, joint_centers):
        jacobian_b, mass_matrix, gravity = self._get_robot_dynamics(robot, ee_frame_idx, arm_joint_ids)
        ee_pose_b, ee_vel_b = self._ee_state(robot, ee_frame_idx)
        # FIXED: Pass full 6D wrench instead of only 3D force
        ee_force_b = self._get_ee_force(robot, ee_frame_idx)  # (N,6) – pass full wrench

        q = robot.data.joint_pos[:, arm_joint_ids]
        qd = robot.data.joint_vel[:, arm_joint_ids]

        tau = osc.compute(
            jacobian_b=jacobian_b,
            current_ee_pose_b=ee_pose_b,
            current_ee_vel_b=ee_vel_b,
            current_ee_force_b=ee_force_b,
            mass_matrix=mass_matrix,
            gravity=gravity,
            current_joint_pos=q,
            current_joint_vel=qd,
            nullspace_joint_pos_target=joint_centers,
        )
        robot.set_joint_effort_target(tau, joint_ids=arm_joint_ids)

    def _get_robot_dynamics(self, robot, ee_frame_idx, arm_joint_ids):
        ee_jacobi_idx = ee_frame_idx - 1  # per-physx indexing quirk
        jacobian_w = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]
        mass_matrix = robot.root_physx_view.get_generalized_mass_matrices()[:, arm_joint_ids, :][:, :, arm_joint_ids]
        gravity = robot.root_physx_view.get_gravity_compensation_forces()[:, arm_joint_ids]

        # world->body frame (root)
        Rwb = matrix_from_quat(quat_inv(robot.data.root_quat_w))
        jacobian_b = jacobian_w.clone()
        jacobian_b[:, :3, :] = torch.bmm(Rwb, jacobian_b[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(Rwb, jacobian_b[:, 3:, :])
        return jacobian_b, mass_matrix, gravity

    def _ee_state(self, robot, ee_frame_idx):
        root_pos_w = robot.data.root_pos_w
        root_quat_w = robot.data.root_quat_w
        ee_pos_w = robot.data.body_pos_w[:, ee_frame_idx]
        ee_quat_w = robot.data.body_quat_w[:, ee_frame_idx]

        ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

        ee_vel_w = robot.data.body_vel_w[:, ee_frame_idx, :]
        root_vel_w = robot.data.root_vel_w
        rel_vel_w = ee_vel_w - root_vel_w
        ee_lin_vel_b = quat_apply_inverse(root_quat_w, rel_vel_w[:, 0:3])
        ee_ang_vel_b = quat_apply_inverse(root_quat_w, rel_vel_w[:, 3:6])
        ee_vel_b = torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)
        return ee_pose_b, ee_vel_b

    def _get_ee_force(self, robot, ee_frame_idx):
        # Placeholder: use contact sensors if available; else zeros
        return torch.zeros(self.scene.num_envs, 6, device=self.device)

    # -----------------------------
    # Observations / Rewards / Dones helpers
    # -----------------------------
    def _build_observation(self) -> torch.Tensor:
        # Joint states
        q1 = self.robot.data.joint_pos[:, self.arm_joint_ids_1]
        qd1 = self.robot.data.joint_vel[:, self.arm_joint_ids_1]
        q2 = self.robot2.data.joint_pos[:, self.arm_joint_ids_2]
        qd2 = self.robot2.data.joint_vel[:, self.arm_joint_ids_2]

        # EE poses (world)
        ee1_pose_w = torch.cat(
            [self.robot.data.body_pos_w[:, self.ee_frame_idx_1], self.robot.data.body_quat_w[:, self.ee_frame_idx_1]],
            dim=-1,
        )
        ee2_pose_w = torch.cat(
            [self.robot2.data.body_pos_w[:, self.ee_frame_idx_2], self.robot2.data.body_quat_w[:, self.ee_frame_idx_2]],
            dim=-1,
        )

        # Object & goal poses
        obj_pose = torch.cat([self.rod.data.root_pos_w, self.rod.data.root_quat_w], dim=-1)
        goal_pose = self._goal_pose_batch()

        # Wrenches (measured; placeholder zeros)
        w1 = self._get_ee_force(self.robot, self.ee_frame_idx_1)
        w2 = self._get_ee_force(self.robot2, self.ee_frame_idx_2)

        obs_parts = [
            q1, qd1, q2, qd2, ee1_pose_w, ee2_pose_w, obj_pose, goal_pose, w1, w2
        ]
        obs = torch.cat([p.view(self.scene.num_envs, -1) for p in obs_parts], dim=-1)

        # Pad/truncate to cfg._OBS_DIM if needed
        target_dim = self.cfg._OBS_DIM
        if obs.shape[-1] < target_dim:
            pad = torch.zeros(self.scene.num_envs, target_dim - obs.shape[-1], device=self.device)
            obs = torch.cat([obs, pad], dim=-1)
        elif obs.shape[-1] > target_dim:
            obs = obs[:, :target_dim]
        return obs

    def _goal_pose_batch(self) -> torch.Tensor:
        return torch.cat([self._goal_pos.expand(self.scene.num_envs, -1),
                          self._goal_quat.expand(self.scene.num_envs, -1)], dim=-1)

    def _compute_robot_rewards(self, robot_id: int) -> dict[str, torch.Tensor]:
        # Select per-robot handles
        if robot_id == 1:
            ee_pose = torch.cat(
                [self.robot.data.body_pos_w[:, self.ee_frame_idx_1],
                 self.robot.data.body_quat_w[:, self.ee_frame_idx_1]], dim=-1)
            motion_key, stiff_key = "robot_motion", "robot_stiffness"
        else:
            ee_pose = torch.cat(
                [self.robot2.data.body_pos_w[:, self.ee_frame_idx_2],
                 self.robot2.data.body_quat_w[:, self.ee_frame_idx_2]], dim=-1)
            motion_key, stiff_key = "robot2_motion", "robot2_stiffness"

        # Distances: EE to object grasp (use object COM as proxy) and object to goal
        ee_pos = ee_pose[:, :3]
        obj_pos = self.rod.data.root_pos_w
        goal_pos = self._goal_pos.expand_as(obj_pos)

        d_grasp = torch.norm(ee_pos - obj_pos, dim=-1)
        d_goal = torch.norm(obj_pos - goal_pos, dim=-1)
        distance_reward = -self.cfg.distance_reward_scale * (d_grasp + d_goal)

        # Orientation (simple quat alignment object->goal)
        obj_quat = self.rod.data.root_quat_w
        goal_quat = self._goal_quat.expand_as(obj_quat)
        ori_dot = torch.sum(obj_quat * goal_quat, dim=-1).abs().clamp(max=1.0)
        orientation_reward = self.cfg.orientation_reward_scale * ori_dot

        # Coordination (keep object level / synchronized)
        # Use small penalty for deviation of object height from goal height
        coord_err = torch.abs(obj_pos[:, 2] - goal_pos[:, 2])
        coordination_reward = -self.cfg.coordination_reward_scale * coord_err

        base = distance_reward + orientation_reward + coordination_reward

        # Action penalties
        a_m = self._last_actions.get(motion_key, torch.zeros(self.scene.num_envs, 13, device=self.device))
        a_s = self._last_actions.get(stiff_key, torch.zeros(self.scene.num_envs, 6, device=self.device))
        motion_pen = -self.cfg.action_penalty_scale_motion * torch.sum(a_m * a_m, dim=-1)
        stiff_pen = -self.cfg.action_penalty_scale_stiffness * torch.sum(a_s * a_s, dim=-1)

        # Compliance term: bonus for low Kp under (proxy) contact: when EE near object
        near_contact = (d_grasp < 0.10).float().unsqueeze(-1)  # (N,1)
        # map stiffness to [kp_min, kp_max] to compute effective magnitude
        kp = 0.5 * (a_s + 1.0) * (self._kp_max - self._kp_min) + self._kp_min  # (N,6)
        # lower kp -> higher bonus when near contact
        stiffness_terms = stiff_pen + self.cfg.compliance_contact_bonus * near_contact.squeeze(-1) * (
            1.0 - torch.tanh(kp.mean(dim=-1) / self._kp_max.mean())
        )

        # Conflict term (only for motion agents): penalize opposing forces
        f1 = self._extract_force(self._last_actions.get("robot_motion"))
        f2 = self._extract_force(self._last_actions.get("robot2_motion"))
        # conflict = -β * max(0, -F1·F2)
        dot = torch.sum(f1 * f2, dim=-1)
        conflict_pen = -self.cfg.conflict_force_penalty_scale * torch.clamp(-dot, min=0.0)
        motion_terms = motion_pen
        conflict_terms = conflict_pen

        # Task completion sparse bonus distributed to both robots
        success_bonus = self._success_bonus()
        base = base + success_bonus

        return {"base": base, "motion_terms": motion_terms, "stiffness_terms": stiffness_terms, "conflict_terms": conflict_terms}

    def _extract_force(self, a_m: torch.Tensor | None) -> torch.Tensor:
        if a_m is None:
            return torch.zeros(self.scene.num_envs, 3, device=self.device)
        return a_m[:, 7:10].clamp(min=-self.cfg.max_force, max=self.cfg.max_force)

    def _success_bonus(self) -> torch.Tensor:
        # Provide bonus when success condition is already met this step
        obj_pos = self.rod.data.root_pos_w
        goal_pos = self._goal_pos.expand_as(obj_pos)
        pos_ok = torch.norm(obj_pos - goal_pos, dim=-1) < self.cfg.success_position_threshold
        stable = self._object_stable_mask()
        hit = pos_ok & stable
        return torch.where(hit, torch.full_like(pos_ok, self.cfg.task_completion_reward, dtype=torch.float32), 0.0)

    def _object_stable_mask(self) -> torch.Tensor:
        vel = self.rod.data.root_vel_w
        lin = torch.norm(vel[:, :3], dim=-1)
        ang = torch.norm(vel[:, 3:], dim=-1)
        return (lin < 0.05) & (ang < 0.2)

    def _safety_limit_exceeded(self) -> torch.Tensor:
        # Use commanded forces as proxy if no sensor (conservative)
        f1 = self._extract_force(self._last_actions.get("robot_motion"))
        f2 = self._extract_force(self._last_actions.get("robot2_motion"))
        fmag = torch.max(torch.norm(f1, dim=-1), torch.norm(f2, dim=-1))
        return (fmag > self.cfg.safety_force_limit)
