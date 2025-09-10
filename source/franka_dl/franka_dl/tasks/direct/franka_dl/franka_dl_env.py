# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from .franka_dl_env_cfg import FrankaDlEnvCfg


class FrankaDlEnv(DirectRLEnv):
    """
    A dual-arm Franka environment that uses differential IK 
    to grab a HORIZONTAL rod using a state machine.
    """
    cfg: FrankaDlEnvCfg

    def __init__(self, cfg: FrankaDlEnvCfg, render_mode: str | None = None, **kwargs):
        # <<<<<<<<<<<<<<<<<<<< MODIFICATION: DEBUGGING PRINT STATEMENT >>>>>>>>>>>>>>>>>>>>
        # This will confirm that the script is being executed.
        print("--- Initializing FrankaDlEnv ---")
        # Call the parent class constructor to build the scene
        super().__init__(cfg, render_mode, **kwargs)

        # Store useful buffers after the scene is created
        self.robot_dof_targets = self._robot.data.default_joint_pos.clone()
        self.robot2_dof_targets = self._robot2.data.default_joint_pos.clone()

        # Grasping state machine for horizontal pickup
        self.grasp_phase = "approach"  # "approach", "descend", "grasp", "lift", "hold"
        self.phase_timer = 0
        self.rod_grasped = False
        
        # Target poses for end-effectors will be calculated dynamically based on rod position
        self.target_poses_calculated = False

        # <<<<<<<<<<<<<<<<<<<< MODIFICATION: ADDING A VELOCITY FILTER >>>>>>>>>>>>>>>>>>>>
        # This will limit how much the joint targets can change per step to prevent shaking.
        # We are using a much smaller value now to aggressively clamp the trajectory.
        self.max_joint_speed = 0.05  # Lowered from 0.5 to a much smaller value.

        # --- Differential IK Controllers ---
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls"
        )
        self.diff_ik_controller_1 = DifferentialIKController(diff_ik_cfg, num_envs=self.num_envs, device=self.device)
        self.diff_ik_controller_2 = DifferentialIKController(diff_ik_cfg, num_envs=self.num_envs, device=self.device)

        # --- Visualization Markers (optional but recommended) ---
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.ee1_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/robot1_ee_goal"))
        self.ee2_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/robot2_ee_goal"))
        
        # --- Robot Scene Entities (needed for Jacobian) ---
        self.robot1_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        self.robot2_entity_cfg = SceneEntityCfg("robot2", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        self.robot1_entity_cfg.resolve(self.scene)
        self.robot2_entity_cfg.resolve(self.scene)

        # Calculate correct Jacobian indices
        if self._robot.is_fixed_base:
            self.ee1_jacobi_idx = self.robot1_entity_cfg.body_ids[0] - 1
        else:
            self.ee1_jacobi_idx = self.robot1_entity_cfg.body_ids[0]
            
        if self._robot2.is_fixed_base:
            self.ee2_jacobi_idx = self.robot2_entity_cfg.body_ids[0] - 1
        else:
            self.ee2_jacobi_idx = self.robot2_entity_cfg.body_ids[0]

    def _setup_scene(self):
        """Spawns and initializes assets in the scene."""
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._robot2 = Articulation(self.cfg.robot2)
        self.scene.articulations["robot2"] = self._robot2
        self._rod = RigidObject(self.cfg.rod)
        self.scene.rigid_objects["rod"] = self._rod

        # Set up terrain and clone environments
        self.cfg.terrain.num_envs = self.cfg.scene.num_envs
        self.cfg.terrain.env_spacing = self.cfg.scene.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        
        # Add a light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def calculate_target_poses_from_rod(self):
        """
        Dynamically calculate target Cartesian poses based on the actual rod position.
        This ensures the arms approach the rod at its correct location.
        """
        rod_pos = self._rod.data.root_pos_w[0]
        rod_quat = self._rod.data.root_quat_w[0]
        
        # <<<<<<<<<<<<<<<<<<<< MODIFICATION: NEW OFFSETS FOR THE PROVIDED CFG >>>>>>>>>>>>>>>>>>>>
        # The offsets have been adjusted to account for the rod's position at (1.5, 0.0, 2.0).
        # We need to ensure the arms move to the rod's x-coordinate (1.5) and approach it from the side.
        
        # The rotation for a horizontal grasp with grippers facing down.
        grasp_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device)
        
        # The right arm (Robot) needs to move towards the rod from its starting position at (1.0, 0.0, 0.0)
        # We want it to be to the right of the rod's center, approaching from the negative y-axis.
        right_arm_x = rod_pos[0] + 0.1 # Right side of the rod
        approach_y_offset = -0.25 # To approach from the front
        approach_height = rod_pos[2] + 0.15 # 15 cm above the rod's height
        
        # --- Robot 1 (Right Arm) Target Poses ---
        self.robot1_approach_pose = torch.zeros(1, 7, device=self.device)
        self.robot1_approach_pose[0, 0:3] = torch.tensor([right_arm_x, approach_y_offset, approach_height], device=self.device)
        self.robot1_approach_pose[0, 3:7] = grasp_quat

        self.robot1_descend_pose = self.robot1_approach_pose.clone()
        self.robot1_descend_pose[0, 2] = rod_pos[2] + 0.05 # 5 cm above the rod for descent

        self.robot1_grasp_pose = self.robot1_descend_pose.clone()
        self.robot1_grasp_pose[0, 1] = rod_pos[1] + 0.05 # Move into the rod to grasp

        self.robot1_lift_pose = self.robot1_grasp_pose.clone()
        self.robot1_lift_pose[0, 2] += 0.3

        # The left arm (Robot2) needs to move from its starting position at (1.0, 1.0, 0.0)
        # We want it to be to the left of the rod's center.
        left_arm_x = rod_pos[0] - 0.1
        
        # --- Robot 2 (Left Arm) Target Poses ---
        self.robot2_approach_pose = torch.zeros(1, 7, device=self.device)
        self.robot2_approach_pose[0, 0:3] = torch.tensor([left_arm_x, approach_y_offset, approach_height], device=self.device)
        self.robot2_approach_pose[0, 3:7] = grasp_quat

        self.robot2_descend_pose = self.robot2_approach_pose.clone()
        self.robot2_descend_pose[0, 2] = rod_pos[2] + 0.05

        self.robot2_grasp_pose = self.robot2_descend_pose.clone()
        self.robot2_grasp_pose[0, 1] = rod_pos[1] + 0.05

        self.robot2_lift_pose = self.robot2_grasp_pose.clone()
        self.robot2_lift_pose[0, 2] += 0.3
        
        self.target_poses_calculated = True
        print("Calculated target poses based on rod position.")
        print(f"Robot 1 Approach Pose: {self.robot1_approach_pose}")
        print(f"Robot 2 Approach Pose: {self.robot2_approach_pose}")


    def _pre_physics_step(self, actions: torch.Tensor):
        """Handle the physics step logic."""
        try:
            if not self.target_poses_calculated:
                self.calculate_target_poses_from_rod()
            
            # Use the state machine to control the grasping process
            self._execute_grasp_sequence()

            # <<<<<<<<<<<<<<<<<<<< NEW CODE: Placeholder for direct commands >>>>>>>>>>>>>>>>>>>>
            # This is where you can call the new function to give direct commands to the arms.
            # You can replace the "None" values with the joint targets you want to command.
            # joint_targets_robot1 = torch.tensor([[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]], device=self.device)
            # joint_targets_robot2 = torch.tensor([[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]], device=self.device)
            # self.set_arm_joint_targets(1, joint_targets_robot1)
            # self.set_arm_joint_targets(2, joint_targets_robot2)

            self._apply_action()
            self.phase_timer += 1
            
            # Print at every step for immediate feedback
            print("--- Current Trajectory Step ---")
            print(f"Step: {self.phase_timer}, Phase: {self.grasp_phase}")
            print(f"Left Arm Targets: {self.robot_dof_targets.tolist()}")
            # print(f"end effector: {self.robot()}")
            print(f"Right Arm Targets: {self.robot2_dof_targets.tolist()}")
            print("------------------------------")

            # Visualization
            self._visualize_goals()
        except Exception as e:
            print(f"An error occurred in _pre_physics_step: {e}")
            raise

    def _execute_grasp_sequence(self):
        """Executes the refined state machine for horizontal pickup."""
        if self.grasp_phase == "approach":
            self._move_to_target_pose(1, self.robot1_approach_pose)
            self._move_to_target_pose(2, self.robot2_approach_pose)
            
            # Gripper control (keep open)
            self.robot_dof_targets[:, -2:] = 0.04
            self.robot2_dof_targets[:, -2:] = 0.04
            
            if self._check_pose_reached(1, self.robot1_approach_pose, 0.15) and \
               self._check_pose_reached(2, self.robot2_approach_pose, 0.15):
                self.grasp_phase = "descend"
                self.phase_timer = 0
                print("Phase: APPROACH -> DESCEND")
        
        elif self.grasp_phase == "descend":
            self._move_to_target_pose(1, self.robot1_descend_pose)
            self._move_to_target_pose(2, self.robot2_descend_pose)
            
            if self._check_pose_reached(1, self.robot1_descend_pose, 0.1) and \
               self._check_pose_reached(2, self.robot2_descend_pose, 0.1) and self.phase_timer > 30: 
                self.grasp_phase = "grasp"
                self.phase_timer = 0
                print("Phase: DESCEND -> GRASP")

        elif self.grasp_phase == "grasp":
            self._move_to_target_pose(1, self.robot1_grasp_pose)
            self._move_to_target_pose(2, self.robot2_grasp_pose)
            
            # Gradually close the grippers
            grip_closure = min(1.0, self.phase_timer / 150.0)
            gripper_pos = 0.04 * (1.0 - grip_closure)
            self.robot_dof_targets[:, -2:] = gripper_pos
            self.robot2_dof_targets[:, -2:] = gripper_pos
            
            if self.phase_timer > 200:
                self.grasp_phase = "lift"
                self.phase_timer = 0
                self.rod_grasped = True
                print("Phase: GRASP -> LIFT")
        
        elif self.grasp_phase == "lift":
            self._move_to_target_pose(1, self.robot1_lift_pose)
            self._move_to_target_pose(2, self.robot2_lift_pose)
            # Grippers should remain closed here
            self.robot_dof_targets[:, -2:] = 0.0
            self.robot2_dof_targets[:, -2:] = 0.0
            
            if self._check_pose_reached(1, self.robot1_lift_pose, 0.25) and \
               self._check_pose_reached(2, self.robot2_lift_pose, 0.25) and self.phase_timer > 50:
                self.grasp_phase = "hold"
                self.phase_timer = 0
                print("Phase: LIFT -> HOLD")
        
        elif self.grasp_phase == "hold":
            # Just maintain the lift pose
            self._move_to_target_pose(1, self.robot1_lift_pose)
            self._move_to_target_pose(2, self.robot2_lift_pose)
            self.robot_dof_targets[:, -2:] = 0.0
            self.robot2_dof_targets[:, -2:] = 0.0

    def _move_to_target_pose(self, robot_num: int, target_pose_w: torch.Tensor):
        """Move robot end-effector towards a target Cartesian pose using Differential IK."""
        if robot_num not in [1, 2]:
            return
            
        if robot_num == 1:
            robot = self._robot
            controller = self.diff_ik_controller_1
            robot_entity = self.robot1_entity_cfg
            ee_jacobi_idx = self.ee1_jacobi_idx
        else:
            robot = self._robot2
            controller = self.diff_ik_controller_2
            robot_entity = self.robot2_entity_cfg
            ee_jacobi_idx = self.ee2_jacobi_idx

        # Get current poses and jacobian
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity.joint_ids]
        ee_pose_w = robot.data.body_pose_w[:, robot_entity.body_ids[0]]
        root_pose_w = robot.data.root_pose_w
        joint_pos = robot.data.joint_pos[:, robot_entity.joint_ids]
        
        # Transform current end-effector pose to base frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], 
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        
        # Transform target pose to base frame  
        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            target_pose_w[:, 0:3], target_pose_w[:, 3:7]
        )
        
        # Create target command in base frame
        target_command_b = torch.cat([target_pos_b, target_quat_b], dim=-1)
        
        # Set the command
        controller.set_command(target_command_b)
        
        # Compute desired joint positions
        joint_pos_des = controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        
        # <<<<<<<<<<<<<<<<<<<< MODIFICATION: ADDING THE VELOCITY FILTER >>>>>>>>>>>>>>>>>>>>
        # This will smooth out the trajectory and prevent shaking.
        current_joint_pos = joint_pos.clone()
        delta_q = joint_pos_des - current_joint_pos
        
        # Clamp the change in joint positions to prevent large, sudden movements
        max_delta_q = self.max_joint_speed * self.cfg.sim.dt
        delta_q = torch.clamp(delta_q, -max_delta_q, max_delta_q)
        
        # The new desired position is the current position plus the filtered change
        joint_pos_des = current_joint_pos + delta_q
        
        # Update the full joint target buffer
        if robot_num == 1:
            self.robot_dof_targets[:, robot_entity.joint_ids] = joint_pos_des
        else:
            self.robot2_dof_targets[:, robot_entity.joint_ids] = joint_pos_des

    def _check_pose_reached(self, robot_num: int, target_pose: torch.Tensor, threshold: float = 0.05) -> bool:
        """Check if the robot's end-effector has reached the target Cartesian pose within a tolerance."""
        if robot_num == 1:
            robot = self._robot
            robot_entity = self.robot1_entity_cfg
        elif robot_num == 2:
            robot = self._robot2
            robot_entity = self.robot2_entity_cfg
        else:
            return False

        # Get current end-effector pose in world frame
        ee_pose_w = robot.data.body_pose_w[0, robot_entity.body_ids[0]]
        
        # Calculate position error
        pos_error = torch.norm(target_pose[0, 0:3] - ee_pose_w[0:3])
        
        return pos_error < threshold
        
    def _visualize_goals(self):
        """Visualize the goal poses of the end-effectors."""
        if self.grasp_phase == "approach":
            self.ee1_marker.visualize(self.robot1_approach_pose[:, 0:3], self.robot1_approach_pose[:, 3:7])
            self.ee2_marker.visualize(self.robot2_approach_pose[:, 0:3], self.robot2_approach_pose[:, 3:7])
        elif self.grasp_phase == "descend":
            self.ee1_marker.visualize(self.robot1_descend_pose[:, 0:3], self.robot1_descend_pose[:, 3:7])
            self.ee2_marker.visualize(self.robot2_descend_pose[:, 0:3], self.robot2_descend_pose[:, 3:7])
        elif self.grasp_phase == "grasp":
            self.ee1_marker.visualize(self.robot1_grasp_pose[:, 0:3], self.robot1_grasp_pose[:, 3:7])
            self.ee2_marker.visualize(self.robot2_grasp_pose[:, 0:3], self.robot2_grasp_pose[:, 3:7])
        elif self.grasp_phase == "lift":
            self.ee1_marker.visualize(self.robot1_lift_pose[:, 0:3], self.robot1_lift_pose[:, 3:7])
            self.ee2_marker.visualize(self.robot2_lift_pose[:, 0:3], self.robot2_lift_pose[:, 3:7])
        else: # hold phase
            self.ee1_marker.visualize(self.robot1_lift_pose[:, 0:3], self.robot1_lift_pose[:, 3:7])
            self.ee2_marker.visualize(self.robot2_lift_pose[:, 0:3], self.robot2_lift_pose[:, 3:7])
            
    def _apply_action(self):
        """Applies the computed joint targets to the robots."""
        self._robot.set_joint_position_target(self.robot_dof_targets)
        self._robot2.set_joint_position_target(self.robot2_dof_targets)

    # <<<<<<<<<<<<<<<<<<<< NEW CODE: Function to give direct commands >>>>>>>>>>>>>>>>>>>>
    def set_arm_joint_targets(self, robot_num: int, joint_targets: torch.Tensor):
        """
        Set the joint position targets for a specified robot.
        This function allows direct control and overrides the IK logic for the current step.

        Args:
            robot_num (int): The robot to control (1 for the left arm, 2 for the right arm).
            joint_targets (torch.Tensor): A tensor of shape (num_envs, num_dofs) with the
                                          desired joint position targets.
        """
        if robot_num == 1:
            # Overwrite the joint targets for the left arm
            self.robot_dof_targets[:] = joint_targets
        elif robot_num == 2:
            # Overwrite the joint targets for the right arm
            self.robot2_dof_targets[:] = joint_targets

    def get_ee_poses(self):
        """
        Returns the current and target end-effector poses for both robots.
        This data would be sent to the external GUI.
        """
        ee1_current_pose = self._robot.data.body_pose_w[0, self.robot1_entity_cfg.body_ids[0]].tolist()
        ee2_current_pose = self._robot2.data.body_pose_w[0, self.robot2_entity_cfg.body_ids[0]].tolist()
        
        # Get the target poses based on the current grasp phase
        if self.grasp_phase == "approach":
            ee1_target_pose = self.robot1_approach_pose[0].tolist()
            ee2_target_pose = self.robot2_approach_pose[0].tolist()
        elif self.grasp_phase == "descend":
            ee1_target_pose = self.robot1_descend_pose[0].tolist()
            ee2_target_pose = self.robot2_descend_pose[0].tolist()
        elif self.grasp_phase == "grasp":
            ee1_target_pose = self.robot1_grasp_pose[0].tolist()
            ee2_target_pose = self.robot2_grasp_pose[0].tolist()
        elif self.grasp_phase == "lift":
            ee1_target_pose = self.robot1_lift_pose[0].tolist()
            ee2_target_pose = self.robot2_lift_pose[0].tolist()
        else: # hold phase
            ee1_target_pose = self.robot1_lift_pose[0].tolist()
            ee2_target_pose = self.robot2_lift_pose[0].tolist()
            
        return {
            "ee1_current": ee1_current_pose,
            "ee2_current": ee2_current_pose,
            "ee1_target": ee1_target_pose,
            "ee2_target": ee2_target_pose,
        }

    # -- Required RL Environment Methods --
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        return torch.ones(self.num_envs, device=self.device)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        self._robot.reset(env_ids)
        self._robot2.reset(env_ids)
        self._rod.reset(env_ids=env_ids)
        
        # Reset state machine for the new episode
        self.grasp_phase = "approach"
        self.phase_timer = 0
        self.rod_grasped = False
        self.target_poses_calculated = False
        
        if env_ids is None:
            self.robot_dof_targets[:] = self._robot.data.default_joint_pos
            self.robot2_dof_targets[:] = self._robot2.data.default_joint_pos
            self.diff_ik_controller_1.reset()
            self.diff_ik_controller_2.reset()

    def _get_observations(self) -> dict:
        obs = torch.cat([
            self._robot.data.joint_pos,
            self._robot2.data.joint_pos
        ], dim=-1)
        
        if obs.shape[-1] > 18:
            obs = obs[:, :18]
        elif obs.shape[-1] < 18:
            padding = torch.zeros(obs.shape[0], 18 - obs.shape[-1], device=self.device)
            obs = torch.cat([obs, padding], dim=-1)

        return {"policy": obs}
