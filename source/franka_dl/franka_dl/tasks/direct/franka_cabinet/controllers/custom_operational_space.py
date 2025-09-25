from __future__ import annotations
from typing import TYPE_CHECKING
import torch

# Import base OSC from Isaac Lab package
from isaaclab.controllers.operational_space import OperationalSpaceController

if TYPE_CHECKING:
    from .custom_operational_space_cfg import CustomOperationalSpaceControllerCfg


class CustomOperationalSpaceController(OperationalSpaceController):
    """Custom operational-space controller with joint-space control modes.

    This controller extends the base OperationalSpaceController to provide additional
    control modes including joint-space PD control, joint-space impedance control,
    and custom control equations. It provides access to robot parameters like
    Jacobian and inertia matrices for implementing custom control laws.

    Supported control modes:
    - operational_space: Standard operational space control (inherited)
    - joint_pd: Joint-space PD controller
    - joint_impedance: Joint-space impedance controller
    - custom: Custom control equations with access to all robot parameters
    """

    def __init__(self, cfg: CustomOperationalSpaceControllerCfg, num_envs: int, device: str):
        """Initialize custom operational-space controller.

        Args:
            cfg: The configuration for custom operational-space controller.
            num_envs: The number of environments.
            device: The device to use for computations.

        Raises:
            ValueError: When invalid control mode is provided.
        """
        # Initialize the parent operational space controller
        parent_cfg = cfg  # The config inherits from OperationalSpaceControllerCfg
        super().__init__(parent_cfg, num_envs, device)
        
        # Store custom configuration
        self.custom_cfg = cfg
        
        # Validate control mode
        valid_modes = ["operational_space", "joint_pd", "joint_impedance", "custom"]
        if cfg.control_mode not in valid_modes:
            raise ValueError(f"Invalid control mode: {cfg.control_mode}. Valid modes: {valid_modes}")
        
        # Initialize joint-space control parameters
        self._init_joint_control_parameters()
        
        # Buffers for robot parameters (accessible for custom equations)
        self.jacobian_b = None
        self.mass_matrix = None
        self.gravity_vector = None
        self.current_joint_positions = None
        self.current_joint_velocities = None
        self.current_ee_pose_b = None
        self.current_ee_vel_b = None
        
        # Target commands for joint-space control
        self._joint_pos_target = torch.zeros(self.num_envs, self.action_dim, device=self._device)
        self._joint_vel_target = torch.zeros(self.num_envs, self.action_dim, device=self._device)

    def _init_joint_control_parameters(self):
        """Initialize joint-space control parameters."""
        # For joint control, we need to know the number of DoF
        # This will be set when compute is first called
        self.num_dof = None
        
        # Joint-space PD gains (will be initialized when num_dof is known)
        self._joint_p_gains = None
        self._joint_d_gains = None
        
        # Joint impedance parameters
        self._joint_impedance_p_gains = None
        self._joint_impedance_d_gains = None
        
        # Custom control weights
        self._custom_weights = torch.tensor(
            self.custom_cfg.custom_equation_weights, 
            dtype=torch.float, 
            device=self._device
        )

    def _initialize_joint_gains(self, num_dof: int):
        """Initialize joint gains once the number of DoF is known."""
        if self.num_dof is not None:
            return  # Already initialized
            
        self.num_dof = num_dof
        
        # Joint-space PD gains
        self._joint_p_gains = torch.zeros(self.num_envs, num_dof, device=self._device)
        self._joint_d_gains = torch.zeros(self.num_envs, num_dof, device=self._device)

        # Resolve controller gains: pd_kp/pd_kd override legacy fields
        kp_src = self.custom_cfg.pd_kp if hasattr(self.custom_cfg, "pd_kp") else None
        kd_src = self.custom_cfg.pd_kd if hasattr(self.custom_cfg, "pd_kd") else None

        # Kp
        if kp_src is not None:
            if isinstance(kp_src, (int, float)):
                self._joint_p_gains[:] = kp_src
            else:
                self._joint_p_gains[:] = torch.tensor(kp_src, device=self._device)
        else:
            # Fallback to legacy alias (interpreted as Kp)
            legacy_kp = self.custom_cfg.joint_stiffness
            if isinstance(legacy_kp, (int, float)):
                self._joint_p_gains[:] = legacy_kp
            else:
                self._joint_p_gains[:] = torch.tensor(legacy_kp, device=self._device)

        # Kd
        if kd_src is not None:
            if isinstance(kd_src, (int, float)):
                self._joint_d_gains[:] = kd_src
            else:
                self._joint_d_gains[:] = torch.tensor(kd_src, device=self._device)
        else:
            # Compute from damping ratio if kd not provided
            joint_damping_ratio = self.custom_cfg.joint_damping_ratio
            if isinstance(joint_damping_ratio, (int, float)):
                damping_ratios = torch.full((num_dof,), joint_damping_ratio, device=self._device)
            else:
                damping_ratios = torch.tensor(joint_damping_ratio, device=self._device)
            self._joint_d_gains[:] = 2 * torch.sqrt(self._joint_p_gains) * damping_ratios
        
        # Joint impedance gains (same initialization for now)
        self._joint_impedance_p_gains = self._joint_p_gains.clone()
        self._joint_impedance_d_gains = self._joint_d_gains.clone()
        
        # Resize target commands
        self._joint_pos_target = torch.zeros(self.num_envs, num_dof, device=self._device)
        self._joint_vel_target = torch.zeros(self.num_envs, num_dof, device=self._device)

    @property
    def action_dim(self) -> int:
        """Dimension of the action space of controller."""
        if self.custom_cfg.control_mode == "operational_space":
            return super().action_dim
        elif self.custom_cfg.control_mode in ["joint_pd", "joint_impedance"]:
            # For joint control, action dimension depends on the number of joints
            # This will be determined at runtime
            if self.num_dof is not None:
                if self.custom_cfg.joint_impedance_mode == "fixed":
                    return self.num_dof  # Joint positions only
                elif self.custom_cfg.joint_impedance_mode == "variable_kp":
                    return self.num_dof * 2  # Joint positions + stiffness
                elif self.custom_cfg.joint_impedance_mode == "variable":
                    return self.num_dof * 3  # Joint positions + stiffness + damping
            return 6  # Default fallback (will be updated)
        elif self.custom_cfg.control_mode == "custom":
            # Custom mode uses operational space action dimensions by default
            return super().action_dim
        else:
            return super().action_dim

    def reset(self):
        """Reset the internals."""
        super().reset()
        if self.num_dof is not None:
            self._joint_pos_target.zero_()
            self._joint_vel_target.zero_()

    def set_joint_command(
        self,
        joint_positions: torch.Tensor,
        joint_velocities: torch.Tensor | None = None,
        joint_stiffness: torch.Tensor | None = None,
        joint_damping: torch.Tensor | None = None,
    ):
        """Set joint-space command for joint PD or impedance control.

        Args:
            joint_positions: Target joint positions of shape (num_envs, num_dof).
            joint_velocities: Target joint velocities of shape (num_envs, num_dof). Defaults to zero.
            joint_stiffness: Joint stiffness gains of shape (num_envs, num_dof). Optional.
            joint_damping: Joint damping gains of shape (num_envs, num_dof). Optional.
        """
        if self.num_dof is None and joint_positions is not None:
            # Initialize gains/targets based on provided command shape
            self._initialize_joint_gains(joint_positions.shape[1])
            
        self._joint_pos_target[:] = joint_positions
        
        if joint_velocities is not None:
            self._joint_vel_target[:] = joint_velocities
        else:
            self._joint_vel_target.zero_()
        
        if joint_stiffness is not None and self._joint_p_gains is not None:
            self._joint_p_gains[:] = joint_stiffness
            
        if joint_damping is not None and self._joint_d_gains is not None:
            self._joint_d_gains[:] = joint_damping

    def compute_joint_pd(
        self,
        current_joint_pos: torch.Tensor,
        current_joint_vel: torch.Tensor,
        mass_matrix: torch.Tensor | None = None,
        gravity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute joint torques using PD control.

        Args:
            current_joint_pos: Current joint positions of shape (num_envs, num_dof).
            current_joint_vel: Current joint velocities of shape (num_envs, num_dof).
            mass_matrix: Joint-space mass matrix of shape (num_envs, num_dof, num_dof). Optional.
            gravity: Gravity compensation torques of shape (num_envs, num_dof). Optional.

        Returns:
            Joint torques of shape (num_envs, num_dof).
        """
        # Compute errors
        pos_error = self._joint_pos_target - current_joint_pos
        vel_error = self._joint_vel_target - current_joint_vel
        
        # PD control
        desired_acc = self._joint_p_gains * pos_error + self._joint_d_gains * vel_error

        if self.custom_cfg.enable_debug_logging:
            if not hasattr(self, "_debug_step"):
                self._debug_step = 0
            self._debug_step += 1
            if self._debug_step % self.custom_cfg.debug_log_interval == 0:
                try:
                    idx = 0
                    print(
                        "[Controller Debug] step={}, joint_pos={}, pos_err={}, vel_err={}".format(
                            self._debug_step,
                            current_joint_pos[idx].detach().cpu().numpy(),
                            pos_error[idx].detach().cpu().numpy(),
                            vel_error[idx].detach().cpu().numpy(),
                        )
                    )
                except Exception:
                    pass
        
        # Convert to torques
        if self.custom_cfg.joint_inertial_compensation and mass_matrix is not None:
            # Inverse dynamics
            joint_torques = torch.bmm(mass_matrix, desired_acc.unsqueeze(-1)).squeeze(-1)
        else:
            # Direct acceleration command
            joint_torques = desired_acc
        
        # # Add gravity compensation
        # if self.custom_cfg.joint_gravity_compensation and gravity is not None:
        #     joint_torques += gravity
            
        return joint_torques

    def compute_joint_impedance(
        self,
        current_joint_pos: torch.Tensor,
        current_joint_vel: torch.Tensor,
        mass_matrix: torch.Tensor | None = None,
        gravity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute joint torques using impedance control.

        Args:
            current_joint_pos: Current joint positions of shape (num_envs, num_dof).
            current_joint_vel: Current joint velocities of shape (num_envs, num_dof).
            mass_matrix: Joint-space mass matrix of shape (num_envs, num_dof, num_dof). Optional.
            gravity: Gravity compensation torques of shape (num_envs, num_dof). Optional.

        Returns:
            Joint torques of shape (num_envs, num_dof).
        """
        # Compute errors
        pos_error = self._joint_pos_target - current_joint_pos
        vel_error = self._joint_vel_target - current_joint_vel
        
        # Impedance control (similar to PD but with different interpretation)
        desired_acc = self._joint_impedance_p_gains * pos_error + self._joint_impedance_d_gains * vel_error
        
        # Convert to torques with inertia shaping
        if mass_matrix is not None:
            # Full impedance with inertia matrix
            joint_torques = torch.bmm(mass_matrix, desired_acc.unsqueeze(-1)).squeeze(-1)
        else:
            # Simplified impedance
            joint_torques = desired_acc
        
        # Add gravity compensation
        if self.custom_cfg.joint_gravity_compensation and gravity is not None:
            joint_torques += gravity
            
        return joint_torques

    def compute_custom_control(
        self,
        jacobian_b: torch.Tensor,
        current_joint_pos: torch.Tensor,
        current_joint_vel: torch.Tensor,
        mass_matrix: torch.Tensor | None = None,
        gravity: torch.Tensor | None = None,
        current_ee_pose_b: torch.Tensor | None = None,
        current_ee_vel_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute joint torques using custom control equations.
        
        This method provides access to all robot parameters and can be overridden
        to implement custom control laws.

        Args:
            jacobian_b: End-effector Jacobian matrix of shape (num_envs, 6, num_dof).
            current_joint_pos: Current joint positions of shape (num_envs, num_dof).
            current_joint_vel: Current joint velocities of shape (num_envs, num_dof).
            mass_matrix: Joint-space mass matrix of shape (num_envs, num_dof, num_dof). Optional.
            gravity: Gravity compensation torques of shape (num_envs, num_dof). Optional.
            current_ee_pose_b: Current end-effector pose of shape (num_envs, 7). Optional.
            current_ee_vel_b: Current end-effector velocity of shape (num_envs, 6). Optional.

        Returns:
            Joint torques of shape (num_envs, num_dof).
        """
        # Default implementation combines PD control with operational space control
        
        # Compute joint PD component
        joint_pd_torques = self.compute_joint_pd(current_joint_pos, current_joint_vel, mass_matrix, gravity)
        
        # Compute operational space component (if desired pose is set)
        os_torques = torch.zeros_like(joint_pd_torques)
        if self.desired_ee_pose_b is not None and current_ee_pose_b is not None and current_ee_vel_b is not None:
            # Use parent class operational space computation
            os_torques = super().compute(
                jacobian_b=jacobian_b,
                current_ee_pose_b=current_ee_pose_b,
                current_ee_vel_b=current_ee_vel_b,
                mass_matrix=mass_matrix,
                gravity=None,  # Avoid double gravity compensation
                current_joint_pos=current_joint_pos,
                current_joint_vel=current_joint_vel,
            )
        
        # Combine components with weights
        w1, w2 = 0.5, 0.5  # Default equal weighting
        if len(self._custom_weights) >= 2:
            w1, w2 = self._custom_weights[0], self._custom_weights[1]
        
        combined_torques = w1 * joint_pd_torques + w2 * os_torques
        
        return combined_torques

    def compute(
        self,
        jacobian_b: torch.Tensor,
        current_ee_pose_b: torch.Tensor | None = None,
        current_ee_vel_b: torch.Tensor | None = None,
        current_ee_force_b: torch.Tensor | None = None,
        mass_matrix: torch.Tensor | None = None,
        gravity: torch.Tensor | None = None,
        current_joint_pos: torch.Tensor | None = None,
        current_joint_vel: torch.Tensor | None = None,
        nullspace_joint_pos_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Performs inference with the controller.

        Args:
            jacobian_b: The Jacobian matrix of the end-effector in root frame. It is a tensor of shape
                (``num_envs``, 6, ``num_DoF``).
            current_ee_pose_b: The current end-effector pose in root frame. It is a tensor of shape
                (``num_envs``, 7), which contains the position and quaternion ``(w, x, y, z)``. Defaults to ``None``.
            current_ee_vel_b: The current end-effector velocity in root frame. It is a tensor of shape
                (``num_envs``, 6), which contains the linear and angular velocities. Defaults to None.
            current_ee_force_b: The current external force on the end-effector in root frame. It is a tensor of
                shape (``num_envs``, 3), which contains the linear force. Defaults to ``None``.
            mass_matrix: The joint-space mass/inertia matrix. It is a tensor of shape (``num_envs``, ``num_DoF``,
                ``num_DoF``). Defaults to ``None``.
            gravity: The joint-space gravity vector. It is a tensor of shape (``num_envs``, ``num_DoF``). Defaults
                to ``None``.
            current_joint_pos: The current joint positions. It is a tensor of shape (``num_envs``, ``num_DoF``).
                Defaults to ``None``.
            current_joint_vel: The current joint velocities. It is a tensor of shape (``num_envs``, ``num_DoF``).
                Defaults to ``None``.
            nullspace_joint_pos_target: The target joint positions the null space controller is trying to enforce, when
                possible. It is a tensor of shape (``num_envs``, ``num_DoF``).

        Returns:
            Tensor: The joint efforts computed by the controller. It is a tensor of shape (``num_envs``, ``num_DoF``).
        """
        # Store robot parameters for access in custom equations
        self.jacobian_b = jacobian_b
        self.mass_matrix = mass_matrix
        self.gravity_vector = gravity
        self.current_joint_positions = current_joint_pos
        self.current_joint_velocities = current_joint_vel
        self.current_ee_pose_b = current_ee_pose_b
        self.current_ee_vel_b = current_ee_vel_b
        
        # Initialize joint gains if not done yet
        num_dof = jacobian_b.shape[2]
        self._initialize_joint_gains(num_dof)
        
        # Route to appropriate control method based on mode
        if self.custom_cfg.control_mode == "operational_space":
            # Use parent class operational space control
            return super().compute(
                jacobian_b=jacobian_b,
                current_ee_pose_b=current_ee_pose_b,
                current_ee_vel_b=current_ee_vel_b,
                current_ee_force_b=current_ee_force_b,
                mass_matrix=mass_matrix,
                gravity=gravity,
                current_joint_pos=current_joint_pos,
                current_joint_vel=current_joint_vel,
                nullspace_joint_pos_target=nullspace_joint_pos_target,
            )
        
        elif self.custom_cfg.control_mode == "joint_pd":
            if current_joint_pos is None or current_joint_vel is None:
                raise ValueError("Joint positions and velocities are required for joint PD control.")
            return self.compute_joint_pd(current_joint_pos, current_joint_vel, mass_matrix, gravity)
        
        elif self.custom_cfg.control_mode == "joint_impedance":
            if current_joint_pos is None or current_joint_vel is None:
                raise ValueError("Joint positions and velocities are required for joint impedance control.")
            return self.compute_joint_impedance(current_joint_pos, current_joint_vel, mass_matrix, gravity)
        
        elif self.custom_cfg.control_mode == "custom":
            if current_joint_pos is None or current_joint_vel is None:
                raise ValueError("Joint positions and velocities are required for custom control.")
            return self.compute_custom_control(
                jacobian_b=jacobian_b,
                current_joint_pos=current_joint_pos,
                current_joint_vel=current_joint_vel,
                mass_matrix=mass_matrix,
                gravity=gravity,
                current_ee_pose_b=current_ee_pose_b,
                current_ee_vel_b=current_ee_vel_b,
            )
        
        else:
            raise ValueError(f"Invalid control mode: {self.custom_cfg.control_mode}")

    def get_robot_parameters(self) -> dict:
        """Get access to current robot parameters for custom control equations.
        
        Returns:
            Dictionary containing current robot parameters:
            - jacobian_b: End-effector Jacobian matrix
            - mass_matrix: Joint-space mass matrix
            - gravity_vector: Gravity compensation vector
            - current_joint_positions: Current joint positions
            - current_joint_velocities: Current joint velocities
            - current_ee_pose_b: Current end-effector pose
            - current_ee_vel_b: Current end-effector velocity
        """
        return {
            "jacobian_b": self.jacobian_b,
            "mass_matrix": self.mass_matrix,
            "gravity_vector": self.gravity_vector,
            "current_joint_positions": self.current_joint_positions,
            "current_joint_velocities": self.current_joint_velocities,
            "current_ee_pose_b": self.current_ee_pose_b,
            "current_ee_vel_b": self.current_ee_vel_b,
        }

    def update_joint_gains(
        self,
        stiffness: torch.Tensor | None = None,
        damping_ratio: torch.Tensor | None = None,
        impedance_stiffness: torch.Tensor | None = None,
        impedance_damping: torch.Tensor | None = None,
    ):
        """Update joint control gains during runtime.
        
        Args:
            stiffness: New joint stiffness gains of shape (num_envs, num_dof).
            damping_ratio: New joint damping ratios of shape (num_envs, num_dof).
            impedance_stiffness: New impedance stiffness gains of shape (num_envs, num_dof).
            impedance_damping: New impedance damping gains of shape (num_envs, num_dof).
        """
        if self._joint_p_gains is None:
            return  # Not initialized yet
            
        if stiffness is not None:
            self._joint_p_gains[:] = stiffness
            # Update damping gains to maintain damping ratio
            if damping_ratio is not None and self._joint_d_gains is not None:
                self._joint_d_gains[:] = 2 * torch.sqrt(self._joint_p_gains) * damping_ratio
            
        if impedance_stiffness is not None and self._joint_impedance_p_gains is not None:
            self._joint_impedance_p_gains[:] = impedance_stiffness
            
        if impedance_damping is not None and self._joint_impedance_d_gains is not None:
            self._joint_impedance_d_gains[:] = impedance_damping
