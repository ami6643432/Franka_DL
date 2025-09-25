# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

import gymnasium as gym
import numpy as np
import torch

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform

from .controllers.custom_operational_space_cfg import CustomOperationalSpaceControllerCfg


@configclass
class FrankaCabinetEnvCfg(DirectMARLEnvCfg):
    """Direct MARL configuration for the Franka cabinet task.

    The default ``cabinet_operator`` agent mirrors the original direct-workflow policy by
    controlling all 7 arm joints plus the two finger joints. Additional agents can be
    registered by extending :attr:`agent_dof_groups`, :attr:`action_spaces`, and
    :attr:`observation_spaces`.
    """

    # env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2

    # agent metadata -------------------------------------------------------
    agent_dof_groups: dict[str, tuple[str, ...]] = {
        "cabinet_operator": ("panda_joint[1-7]", "panda_finger_joint.*"),
    }
    possible_agents: tuple[str, ...] = tuple(agent_dof_groups.keys())

    _OBS_DIM = 23
    observation_spaces = {}
    action_spaces = {
        "cabinet_operator": gym.spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32),
    }
    state_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(_OBS_DIM,), dtype=np.float32)

    def __post_init__(self):
        # Ensure the multi-agent naming stays consistent if the config is extended.
        self.possible_agents = tuple(self.agent_dof_groups.keys())

        for agent in self.possible_agents:
            if agent not in self.observation_spaces:
                self.observation_spaces[agent] = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self._OBS_DIM,), dtype=np.float32
                )

        missing_action_agents = [agent for agent in self.possible_agents if agent not in self.action_spaces]
        if missing_action_agents:
            raise ValueError(
                "Action space definitions missing for agents: " + ", ".join(missing_action_agents)
            )

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=3.0, replicate_physics=True)

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                "panda_joint4": -2.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.469,
                "panda_finger_joint.*": 0.035,
            },
            pos=(1.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit_sim=87.0,
                velocity_limit_sim=2.175,
                stiffness=0.0,
                damping=5.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=12.0,
                velocity_limit_sim=2.61,
                stiffness=0.0,
                damping=5.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit_sim=200.0,
                velocity_limit_sim=0.2,
                stiffness=0,
                damping=1e2,
            ),
        },
    )

    # cabinet
    cabinet = ArticulationCfg(
        prim_path="/World/envs/env_.*/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0, 0.4),
            rot=(0.1, 0.0, 0.0, 0.0),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit_sim=87.0,
                velocity_limit_sim=100.0,
                stiffness=10.0,
                damping=1.0,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit_sim=87.0,
                velocity_limit_sim=100.0,
                stiffness=10.0,
                damping=2.5,
            ),
        },
    )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    action_scale = 7.5
    dof_velocity_scale = 0.1

    # reward scales
    dist_reward_scale = 1.5
    rot_reward_scale = 1.5
    open_reward_scale = 10.0
    action_penalty_scale = 0.05
    finger_reward_scale = 2.0

    # reward gating and alignment thresholds
    gating_distance_threshold: float = 0.06  # meters (softer gating to learn closure earlier)
    gating_alignment_threshold: float = 0.6  # cosine threshold (softer early signal)
    closure_reward_scale: float = 0.5        # scale for finger gap closure when near handle
    contact_reward_scale: float = 1.0        # bonus when fingers are close & aligned
    contact_gap_threshold: float = 0.01      # desired finger gap in meters
    contact_vertical_weight: float = 10.0    # exponential falloff for vertical alignment

    # fine control knobs for gripper behavior
    finger_speed_scale: float = 1.0          # per-step scale on finger DOFs targets (was 0.1)
    finger_action_scale: float = 1.0         # extra scaling applied to finger action components

    # proximity-only gap reward to provide signal before perfect alignment
    near_handle_distance: float = 0.08       # meters
    gap_reward_scale: float = 0.5            # weight for proximity-only gap shaping
    # straddling parameters
    straddle_threshold: float = 0.01         # meters; fingers must be this far on opposite sides
    straddle_reward_scale: float = 0.5       # reward weight for straddling bonus

    # torque controller parameters
    controller_torque_limit: float = 50.0    # absolute torque clamp for arm joints
    use_torque_controller: bool = True       # toggle torque-based control for arm joints

    # controller configuration (default joint-space PD)
    controller_cfg: CustomOperationalSpaceControllerCfg = CustomOperationalSpaceControllerCfg(
        control_mode="joint_pd",
        pd_kp= [200.0, 200.0, 200.0, 200.0, 100.0, 100.0, 100.0],
        pd_kd= [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        joint_inertial_compensation=False,
        joint_gravity_compensation=False,
        target_types=("pose_abs",),
        enable_debug_logging=False,
        debug_log_interval=240,
    )


class FrankaCabinetEnv(DirectMARLEnv):
    """Franka cabinet manipulation task with configurable multi-agent control."""

    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: FrankaCabinetEnvCfg

    def __init__(self, cfg: FrankaCabinetEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        # configure finger speed scales and cache finger joint indices
        finger_idx1 = self._robot.find_joints("panda_finger_joint1")[0]
        finger_idx2 = self._robot.find_joints("panda_finger_joint2")[0]
        self.finger_joint_ids = torch.tensor([finger_idx1, finger_idx2], device=self.device, dtype=torch.long)
        arm_indices = self._robot.find_joints("panda_joint[1-7]")[0]
        self.arm_joint_ids = torch.tensor(arm_indices, device=self.device, dtype=torch.long)
        self.robot_dof_speed_scales[self.finger_joint_ids[0]] = self.cfg.finger_speed_scale
        self.robot_dof_speed_scales[self.finger_joint_ids[1]] = self.cfg.finger_speed_scale

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.robot_dof_targets[:] = self._robot.data.default_joint_pos.clone().to(device=self.device)
        self._arm_pos_target = self.robot_dof_targets[:, self.arm_joint_ids].clone()

        # Agent metadata for multi-agent control mappings.
        self._agent_names = tuple(self.cfg.possible_agents)
        self._agent_action_indices: dict[str, torch.Tensor] = {}
        self._agent_last_actions: dict[str, torch.Tensor] = {}
        assigned_mask = torch.zeros(self._robot.num_joints, dtype=torch.bool, device=self.device)

        # Shared state buffers for centralized critics
        self._shared_state_curr: torch.Tensor | None = None
        self._shared_state_prev: torch.Tensor | None = None

        for agent in self._agent_names:
            if agent not in self.cfg.agent_dof_groups:
                raise KeyError(f"Agent '{agent}' is missing an entry in agent_dof_groups.")

            joint_indices: list[int] = []
            for expr in self.cfg.agent_dof_groups[agent]:
                matches = self._robot.find_joints(expr)[0]
                joint_indices.extend(matches)

            unique_indices = sorted(set(joint_indices))
            if not unique_indices:
                raise ValueError(f"Agent '{agent}' resolved to no joints with expressions {self.cfg.agent_dof_groups[agent]}.")

            index_tensor = torch.tensor(unique_indices, device=self.device, dtype=torch.long)
            self._agent_action_indices[agent] = index_tensor
            assigned_mask[index_tensor] = True

            if agent not in self.cfg.action_spaces:
                raise KeyError(f"Action space for agent '{agent}' is not defined in the config.")

            expected_dim = int(self.cfg.action_spaces[agent].shape[0])
            if expected_dim != index_tensor.numel():
                raise ValueError(
                    f"Action space for agent '{agent}' expects {expected_dim} dimensions but maps to "
                    f"{index_tensor.numel()} joints. Update the config to keep them in sync."
                )

            self._agent_last_actions[agent] = torch.zeros((self.num_envs, index_tensor.numel()), device=self.device)

        unassigned = torch.nonzero(~assigned_mask, as_tuple=False).squeeze(-1)
        if unassigned.numel() > 0:
            warnings.warn(
                "Some robot joints are not assigned to any agent: "
                + ", ".join(self._robot.joint_names[i] for i in unassigned.tolist()),
                stacklevel=2,
            )

        # Buffer used to store the aggregated per-joint action commands.
        self.actions = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        stage = get_current_stage()
        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_link7")),
            self.device,
        )
        lfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_leftfinger")),
            self.device,
        )
        rfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_rightfinger")),
            self.device,
        )

        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        robot_local_pose_pos += torch.tensor([0, 0.04, 0], device=self.device)
        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

        drawer_local_grasp_pose = torch.tensor([0.3, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0], device=self.device)
        self.drawer_local_grasp_pos = drawer_local_grasp_pose[0:3].repeat((self.num_envs, 1))
        self.drawer_local_grasp_rot = drawer_local_grasp_pose[3:7].repeat((self.num_envs, 1))

        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.drawer_inward_axis = torch.tensor([-1, 0, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.drawer_up_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        # lateral axis of drawer handle frame (perpendicular to inward and up)
        self.drawer_lateral_axis = torch.nn.functional.normalize(
            torch.cross(self.drawer_up_axis, self.drawer_inward_axis, dim=-1), dim=-1
        )

        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]
        self.drawer_link_idx = self._cabinet.find_bodies("drawer_top")[0][0]

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.drawer_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.drawer_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        # debug step counter
        self._debug_step = 0

        # initialize operational-space controller (used later for torque tracking)
        self.use_torque_controller = getattr(self.cfg, "use_torque_controller", True)
        if self.use_torque_controller:
            self.controller_cfg = self.cfg.controller_cfg
            self.controller = self.controller_cfg.class_type(self.controller_cfg, self.num_envs, self.device)
        else:
            self.controller_cfg = None
            self.controller = None

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._cabinet = Articulation(self.cfg.cabinet)
        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["cabinet"] = self._cabinet

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]):
        # Aggregate per-agent actions into a single joint command tensor.
        self.actions.zero_()
        self._debug_step += 1

        for agent in self._agent_names:
            agent_action = actions[agent]
            if not isinstance(agent_action, torch.Tensor):
                agent_action = torch.as_tensor(agent_action, device=self.device)
            else:
                agent_action = agent_action.to(self.device)

            agent_action = agent_action.view(self.num_envs, -1)
            agent_action = torch.clamp(agent_action, -1.0, 1.0)

            indices = self._agent_action_indices[agent]
            self.actions[:, indices] = agent_action
            # optionally amplify finger action components for better closure exploration
            self.actions[:, self.finger_joint_ids] *= self.cfg.finger_action_scale
            self._agent_last_actions[agent].copy_(agent_action)

        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        if self.use_torque_controller:
            # cache desired arm joint positions for controller
            self._arm_pos_target = self.robot_dof_targets[:, self.arm_joint_ids].clone()

    def _apply_action(self):
        if self.use_torque_controller and self.controller is not None:
            if not hasattr(self, "_arm_pos_target"):
                self._arm_pos_target = self.robot_dof_targets[:, self.arm_joint_ids].clone()

            current_arm_pos = self._robot.data.joint_pos[:, self.arm_joint_ids]
            current_arm_vel = self._robot.data.joint_vel[:, self.arm_joint_ids]
            self.controller.set_joint_command(joint_positions=self._arm_pos_target, joint_velocities=None)
            arm_torques = self.controller.compute_joint_pd(
                current_joint_pos=current_arm_pos,
                current_joint_vel=current_arm_vel,
                mass_matrix=None,
                gravity=None,
            )
            torque_limit = self.cfg.controller_torque_limit
            arm_torques = torch.clamp(arm_torques, -torque_limit, torque_limit)
            self._robot.set_joint_effort_target(arm_torques, joint_ids=self.arm_joint_ids)
        else:
            arm_targets = self.robot_dof_targets[:, self.arm_joint_ids]
            self._robot.set_joint_position_target(arm_targets, joint_ids=self.arm_joint_ids)

        # fingers remain position controlled
        finger_targets = self.robot_dof_targets[:, self.finger_joint_ids]
        self._robot.set_joint_position_target(finger_targets, joint_ids=self.finger_joint_ids)
        if self.use_torque_controller and self.controller is not None:
            zero_finger_effort = torch.zeros_like(finger_targets)
            self._robot.set_joint_effort_target(zero_finger_effort, joint_ids=self.finger_joint_ids)

    # post-physics step calls

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        terminated_mask = self._cabinet.data.joint_pos[:, 3] > 0.39
        truncated_mask = self.episode_length_buf >= self.max_episode_length - 1

        terminated = {agent: terminated_mask for agent in self._agent_names}
        truncated = {agent: truncated_mask for agent in self._agent_names}
        return terminated, truncated

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()
        robot_left_finger_pos = self._robot.data.body_pos_w[:, self.left_finger_link_idx]
        robot_right_finger_pos = self._robot.data.body_pos_w[:, self.right_finger_link_idx]

        reward = self._compute_rewards(
            self.actions,
            self._cabinet.data.joint_pos,
            self.robot_grasp_pos,
            self.drawer_grasp_pos,
            self.robot_grasp_rot,
            self.drawer_grasp_rot,
            robot_left_finger_pos,
            robot_right_finger_pos,
            self.gripper_forward_axis,
            self.drawer_inward_axis,
            self.gripper_up_axis,
            self.drawer_up_axis,
            self.drawer_lateral_axis,
            self.num_envs,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.open_reward_scale,
            self.cfg.action_penalty_scale,
            self.cfg.finger_reward_scale,
            self._robot.data.joint_pos,
        )
        return {agent: reward.clone() for agent in self._agent_names}

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot_dof_targets[env_ids] = joint_pos

        # cabinet state
        zeros = torch.zeros((len(env_ids), self._cabinet.num_joints), device=self.device)
        self._cabinet.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids)

        # reset controller internal state
        if self.use_torque_controller and self.controller is not None:
            self.controller.reset()

    def _get_observations(self) -> dict[str, torch.Tensor]:
        prev_state = self._shared_state_curr.clone() if self._shared_state_curr is not None else None

        obs = self._build_observation()
        obs = torch.clamp(obs, -5.0, 5.0)

        self._shared_state_curr = obs.clone()
        if prev_state is None:
            prev_state = self._shared_state_curr.clone()
        self._shared_state_prev = prev_state

        # provide shared state buffers for centralized critics (e.g., MAPPO) via extras
        # the skrl Isaac Lab wrapper expects infos["shared_states"] and infos["shared_next_states"]
        self.extras["shared_states"] = self._shared_state_prev.clone()
        self.extras["shared_next_states"] = self._shared_state_curr.clone()

        return {agent: obs for agent in self._agent_names}

    def _get_states(self) -> torch.Tensor:
        if self._shared_state_curr is None:
            # fallback during initialization
            self._shared_state_curr = self._build_observation().clone()
        return self._shared_state_curr

    def _build_observation(self) -> torch.Tensor:
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        to_target = self.drawer_grasp_pos - self.robot_grasp_pos

        return torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
                to_target,
                self._cabinet.data.joint_pos[:, 3].unsqueeze(-1),
                self._cabinet.data.joint_vel[:, 3].unsqueeze(-1),
            ),
            dim=-1,
        )

    # auxiliary methods

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        hand_pos = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]
        drawer_pos = self._cabinet.data.body_pos_w[env_ids, self.drawer_link_idx]
        drawer_rot = self._cabinet.data.body_quat_w[env_ids, self.drawer_link_idx]
        (
            self.robot_grasp_rot[env_ids],
            self.robot_grasp_pos[env_ids],
            self.drawer_grasp_rot[env_ids],
            self.drawer_grasp_pos[env_ids],
        ) = self._compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.robot_local_grasp_rot[env_ids],
            self.robot_local_grasp_pos[env_ids],
            drawer_rot,
            drawer_pos,
            self.drawer_local_grasp_rot[env_ids],
            self.drawer_local_grasp_pos[env_ids],
        )

    def _compute_rewards(
        self,
        actions,
        cabinet_dof_pos,
        franka_grasp_pos,
        drawer_grasp_pos,
        franka_grasp_rot,
        drawer_grasp_rot,
        franka_lfinger_pos,
        franka_rfinger_pos,
        gripper_forward_axis,
        drawer_inward_axis,
        gripper_up_axis,
        drawer_up_axis,
        drawer_lateral_axis,
        num_envs,
        dist_reward_scale,
        rot_reward_scale,
        open_reward_scale,
        action_penalty_scale,
        finger_reward_scale,
        joint_positions,
    ):
        # distance from hand to the drawer
        d = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + d**2)
        dist_reward *= dist_reward
        dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

        axis1 = tf_vector(franka_grasp_rot, gripper_forward_axis)
        axis2 = tf_vector(drawer_grasp_rot, drawer_inward_axis)
        axis3 = tf_vector(franka_grasp_rot, gripper_up_axis)
        axis4 = tf_vector(drawer_grasp_rot, drawer_up_axis)

        dot1 = (
            torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of forward axis for gripper
        dot2 = (
            torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of up axis for gripper
        # reward for matching the orientation of the hand to the drawer (fingers wrapped)
        rot_reward = 0.5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2)

        # regularization on the actions (summed for each environment)
        action_penalty = torch.sum(actions**2, dim=-1)

        # how far the cabinet has been opened out
        open_reward_raw = cabinet_dof_pos[:, 3]  # drawer_top_joint
        # gate opening reward by proximity & alignment to encourage sequence: approach->align->open
        align_ok = (dot1.abs() > self.cfg.gating_alignment_threshold) & (
            dot2.abs() > self.cfg.gating_alignment_threshold
        )
        prox_ok = d < self.cfg.gating_distance_threshold
        gate = (align_ok & prox_ok).float()
        delta_left = franka_lfinger_pos - drawer_grasp_pos
        delta_right = franka_rfinger_pos - drawer_grasp_pos
        left_proj = torch.sum(delta_left * drawer_lateral_axis, dim=-1)
        right_proj = torch.sum(delta_right * drawer_lateral_axis, dim=-1)
        straddle_ok = (left_proj > self.cfg.straddle_threshold) & (right_proj < -self.cfg.straddle_threshold)
        gate = gate * straddle_ok.float()
        open_reward = open_reward_raw * gate

        # penalty for distance of each finger from the drawer handle
        lfinger_dist = franka_lfinger_pos[:, 2] - drawer_grasp_pos[:, 2]
        rfinger_dist = drawer_grasp_pos[:, 2] - franka_rfinger_pos[:, 2]
        finger_dist_penalty = torch.zeros_like(lfinger_dist)
        finger_dist_penalty += torch.where(lfinger_dist < 0, lfinger_dist, torch.zeros_like(lfinger_dist))
        finger_dist_penalty += torch.where(rfinger_dist < 0, rfinger_dist, torch.zeros_like(rfinger_dist))

        # encourage smaller finger gap when near handle & aligned (use Y-axis separation)
        finger_gap = (franka_lfinger_pos[:, 1] - franka_rfinger_pos[:, 1]).abs()
        closure_bonus = -self.cfg.closure_reward_scale * finger_gap * gate

        # proximity-only gap reward (not fully gated by alignment) to kick-start grasp learning
        near_handle = (d < self.cfg.near_handle_distance).float()
        gap_reward = self.cfg.gap_reward_scale * near_handle * torch.clamp(0.03 - finger_gap, min=0.0)

        # contact-style bonus when fingers are appropriately aligned
        vertical_align_left = torch.exp(-self.cfg.contact_vertical_weight * lfinger_dist.abs())
        vertical_align_right = torch.exp(-self.cfg.contact_vertical_weight * rfinger_dist.abs())
        vertical_alignment = 0.5 * (vertical_align_left + vertical_align_right)
        gap_bonus = torch.clamp(self.cfg.contact_gap_threshold - finger_gap, min=0.0)
        contact_reward = self.cfg.contact_reward_scale * gate * vertical_alignment * gap_bonus
        left_score = torch.clamp(left_proj - self.cfg.straddle_threshold, min=0.0)
        right_score = torch.clamp(-right_proj - self.cfg.straddle_threshold, min=0.0)
        straddle_reward = self.cfg.straddle_reward_scale * near_handle * (left_score + right_score)

        # position error diagnostics (drawer frame components)
        pos_err = drawer_grasp_pos - franka_grasp_pos
        pos_err_norm = torch.norm(pos_err, dim=-1)
        err_inward = torch.sum(pos_err * drawer_inward_axis, dim=-1)
        err_up = torch.sum(pos_err * drawer_up_axis, dim=-1)
        err_lateral = torch.sum(pos_err * drawer_lateral_axis, dim=-1)

        rewards = (
            dist_reward_scale * dist_reward
            + rot_reward_scale * rot_reward
            + open_reward_scale * open_reward
            + closure_bonus
            + contact_reward
            + gap_reward
            + straddle_reward
            + finger_reward_scale * finger_dist_penalty
            - action_penalty_scale * action_penalty
        )

        self.extras["log"] = {
            "dist_reward": float((dist_reward_scale * dist_reward).mean().item()),
            "rot_reward": float((rot_reward_scale * rot_reward).mean().item()),
            "open_reward": float((open_reward_scale * open_reward).mean().item()),
            "contact_reward": float(contact_reward.mean().item()),
            "rot_dot1_mean_abs": float(dot1.abs().mean().item()),
            "rot_dot2_mean_abs": float(dot2.abs().mean().item()),
            "gate_mean": float(gate.mean().item()),
            "closure_bonus": float(closure_bonus.mean().item()),
            "finger_gap": float(finger_gap.mean().item()),
            "vertical_alignment": float(vertical_alignment.mean().item()),
            "action_penalty": float((-action_penalty_scale * action_penalty).mean().item()),
            "left_finger_distance": float(lfinger_dist.mean().item()),
            "right_finger_distance": float(rfinger_dist.mean().item()),
            "finger_dist_penalty": float((finger_reward_scale * finger_dist_penalty).mean().item()),
            "gap_reward": float(gap_reward.mean().item()),
            "straddle_reward": float(straddle_reward.mean().item()),
            # alignment/position diagnostics
            "pos_err_norm": float(pos_err_norm.mean().item()),
            "err_inward": float(err_inward.mean().item()),
            "err_up": float(err_up.mean().item()),
            "err_lateral": float(err_lateral.mean().item()),
            "left_proj": float(left_proj.mean().item()),
            "right_proj": float(right_proj.mean().item()),
            "straddle_rate": float(straddle_ok.float().mean().item()),
        }

        # bonus for opening drawer properly
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.01, rewards + 0.25, rewards)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.2, rewards + 0.25, rewards)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.35, rewards + 0.25, rewards)

        # occasional console debug for env 0
        if self._debug_step % 240 == 0 and num_envs > 0:
            try:
                i = 0
                print(
                    f"[Debug] step={self._debug_step} d={float(d[i]):.3f} pos_err|norm={float(pos_err_norm[i]):.3f} "
                    f"inward={float(err_inward[i]):.3f} up={float(err_up[i]):.3f} lat={float(err_lateral[i]):.3f} "
                    f"dot1={float(dot1[i]):.3f} dot2={float(dot2[i]):.3f} gap={float(finger_gap[i]):.3f} "
                    f"left_proj={float(left_proj[i]):.3f} right_proj={float(right_proj[i]):.3f} gate={float(gate[i]):.2f}"
                )
            except Exception:
                pass

        return rewards

    def _compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
        drawer_rot,
        drawer_pos,
        drawer_local_grasp_rot,
        drawer_local_grasp_pos,
    ):
        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )
        global_drawer_rot, global_drawer_pos = tf_combine(
            drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
        )

        return global_franka_rot, global_franka_pos, global_drawer_rot, global_drawer_pos
