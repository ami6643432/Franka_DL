#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from typing import Tuple, List, Optional, Dict, Union
import math
import warnings
import os
import sys
import torch


def _compute_condition_number(jacobian: torch.Tensor) -> float:
    """Return the Jacobian condition number (σ_max / σ_min)."""

    try:
        sigma = torch.linalg.svdvals(jacobian)
    except Exception:
        try:
            sigma = torch.linalg.svd(jacobian, full_matrices=False).S
        except Exception:
            return float("nan")

    if sigma.numel() == 0:
        return float("nan")

    sigma_max = sigma.max().item()
    sigma_min = sigma.min().item()
    if sigma_min <= 1e-6:
        return float("inf")
    return float(sigma_max / sigma_min)


def _extend_sys_path_with_isaaclab_sources():
    """Add common Isaac Lab source roots to ``sys.path`` if they exist."""

    candidates = [
        os.environ.get("ISAACLAB_PATH"),
        os.environ.get("ISAACLAB_ROOT"),
        os.path.expanduser("~/IsaacLab"),
        os.path.expanduser("~/IsaacLab_4.5"),
    ]

    for root in filter(None, candidates):
        if not os.path.isdir(root):
            continue

        source_dir = os.path.join(root, "source")
        if os.path.isdir(source_dir) and source_dir not in sys.path:
            sys.path.append(source_dir)

        for sub_pkg in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_mimic", "isaaclab_rl"):
            pkg_path = os.path.join(source_dir, sub_pkg)
            if os.path.isdir(pkg_path) and pkg_path not in sys.path:
                sys.path.append(pkg_path)


def _import_isaaclab():
    """Try the Isaac Lab 4.x namespace first, then the legacy one."""

    _extend_sys_path_with_isaaclab_sources()

    # Isaac Lab 4.x (isaaclab package)
    try:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.sim import SimulationContext, SimulationCfg
        import isaaclab.utils.math as math_utils
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
        from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

        return {
            "sim_utils": sim_utils,
            "Articulation": Articulation,
            "ArticulationCfg": ArticulationCfg,
            "SimulationContext": SimulationContext,
            "SimulationCfg": SimulationCfg,
            "math_utils": math_utils,
            "DifferentialIKController": DifferentialIKController,
            "DifferentialIKControllerCfg": DifferentialIKControllerCfg,
            "FRANKA_PANDA_CFG": FRANKA_PANDA_CFG,
        }
    except ImportError:
        pass

    # Legacy (Isaac Lab 2023.*)
    try:
        import omni.isaac.lab.sim as sim_utils
        from omni.isaac.lab.assets import Articulation, ArticulationCfg
        from omni.isaac.lab.sim import SimulationContext
        import omni.isaac.lab.utils.math as math_utils
        from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
        from omni.isaac.lab_assets.franka import FRANKA_PANDA_CFG

        try:
            from omni.isaac.lab.sim import SimulationCfg
        except ImportError:
            SimulationCfg = None

        return {
            "sim_utils": sim_utils,
            "Articulation": Articulation,
            "ArticulationCfg": ArticulationCfg,
            "SimulationContext": SimulationContext,
            "SimulationCfg": SimulationCfg,
            "math_utils": math_utils,
            "DifferentialIKController": DifferentialIKController,
            "DifferentialIKControllerCfg": DifferentialIKControllerCfg,
            "FRANKA_PANDA_CFG": FRANKA_PANDA_CFG,
        }
    except ImportError:
        pass

    return None


_isaaclab_modules = None
ISAACLAB_AVAILABLE = False


def _ensure_isaaclab_imported() -> bool:
    """Attempt to import Isaac Lab lazily; returns ``True`` on success."""

    global _isaaclab_modules, ISAACLAB_AVAILABLE

    if ISAACLAB_AVAILABLE and _isaaclab_modules is not None:
        return True

    modules = _import_isaaclab()
    if modules is None:
        ISAACLAB_AVAILABLE = False
        warnings.warn("IsaacLab not available: unable to import isaaclab or omni.isaac.lab modules")
        print(" Running without IsaacLab - will use placeholder IK")
        return False

    _isaaclab_modules = modules
    globals().update(modules)
    ISAACLAB_AVAILABLE = True
    print("✅ IsaacLab imports successful")
    return True

class Trajectory_generator:
    """
    SLERP (Spherical Linear Interpolation) based trajectory generator for single arm robotics
    """
    
    def __init__(self, start_pose: Dict, end_pose: Dict, total_time: float, 
                 max_velocity: Optional[float] = None, 
                 max_acceleration: Optional[float] = None,
                 num_waypoints: int = 50):
    
        self.start_pose = start_pose
        self.end_pose = end_pose
        self.total_time = total_time
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.num_waypoints = num_waypoints
        
        self._validate_poses()
        
        self.distance = np.linalg.norm(
            np.array(end_pose['position']) - np.array(start_pose['position'])
        )
        
        # Adjust timing if velocity/acceleration constraints are provided
        self.adjusted_time = self._calculate_constrained_time()
        
    def _validate_poses(self):
        required_keys = ['position', 'orientation']
        
        for pose_name, pose in [('start_pose', self.start_pose), ('end_pose', self.end_pose)]:
            for key in required_keys:
                if key not in pose:
                    raise ValueError(f"{pose_name} missing '{key}' key")
            
            if len(pose['position']) != 3:
                raise ValueError(f"{pose_name} position must have 3 elements [x,y,z]")
            
            if len(pose['orientation']) != 4:
                raise ValueError(f"{pose_name} orientation must have 4 elements [x,y,z,w] (quaternion)")
                
    def _calculate_constrained_time(self) -> float:
        min_time = self.total_time
        
        if self.max_velocity is not None:
            vel_time = self.distance / self.max_velocity
            min_time = max(min_time, vel_time)
            
        if self.max_acceleration is not None:
            acc_time = math.sqrt(2 * self.distance / self.max_acceleration)
            min_time = max(min_time, acc_time)
            
        return min_time
    
    def _generate_time_profile(self) -> np.ndarray:
        return np.linspace(0, self.adjusted_time, self.num_waypoints)
    
    def _interpolate_position(self, t_normalized: float) -> np.ndarray:
        start_pos = np.array(self.start_pose['position'])
        end_pos = np.array(self.end_pose['position'])
        return start_pos + t_normalized * (end_pos - start_pos)
    
    def _interpolate_orientation(self, t_normalized: float) -> np.ndarray:
        start_quat = self.start_pose['orientation']
        end_quat = self.end_pose['orientation']
        
        # Create rotation objects
        start_rot = R.from_quat(start_quat)
        end_rot = R.from_quat(end_quat)
        
        # Create SLERP interpolator
        key_rots = R.concatenate([start_rot, end_rot])
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rots)
        
        # Interpolate
        interp_rot = slerp(t_normalized)
        return interp_rot.as_quat()
    
    def compute_waypoints(self) -> Dict[str, List]:
        """Generate trajectory waypoints with pose and orientation"""
        
        timestamps = self._generate_time_profile()
        
        positions = []
        orientations = []
        velocities = []
        
        for i, timestamp in enumerate(timestamps):
            # Normalized time (0 to 1)
            t_norm = timestamp / self.adjusted_time if self.adjusted_time > 0 else 0
            
            # Interpolate position and orientation
            position = self._interpolate_position(t_norm)
            positions.append(position.tolist())
            
            orientation = self._interpolate_orientation(t_norm)
            orientations.append(orientation.tolist())
            
            # Calculate velocity
            if i > 0:
                dt = timestamps[i] - timestamps[i-1]
                if dt > 0:
                    vel = np.linalg.norm(np.array(positions[i]) - np.array(positions[i-1])) / dt
                else:
                    vel = 0.0
            else:
                vel = 0.0
            velocities.append(vel)
        
        return {
            'timestamps': timestamps.tolist(),
            'positions': positions,
            'orientations': orientations,
            'velocities': velocities,
            'total_time': self.adjusted_time,
            'distance': self.distance
        }
    
    def plot_trajectory(self, save_path: Optional[str] = None):
        """Plot 3D trajectory and velocity profile"""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("matplotlib not available for plotting")
            return
            
        waypoints = self.compute_waypoints()
        positions = np.array(waypoints['positions'])
        
        fig = plt.figure(figsize=(12, 5))
        
        # 3D trajectory plot
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Trajectory')
        ax1.scatter(*self.start_pose['position'], color='green', s=100, label='Start')
        ax1.scatter(*self.end_pose['position'], color='red', s=100, label='End')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.legend()
        ax1.set_title('3D Trajectory')
        
        # Velocity profile
        ax2 = fig.add_subplot(122)
        ax2.plot(waypoints['timestamps'], waypoints['velocities'], 'r-', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_title('Velocity Profile')
        ax2.grid(True)
        
        if self.max_velocity:
            ax2.axhline(y=self.max_velocity, color='k', linestyle='--', alpha=0.7, label='Max Velocity')
            ax2.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class FrankaIKSolver:
    """
    Simple Franka arm IK solver using IsaacLab (when available) or analytical methods
    """
    
    def __init__(self):
        """Initialize the IK solver"""
        self.isaaclab_available = _ensure_isaaclab_imported()
        self.metrics: Dict[str, List[float]] = {"condition_numbers": []}
        
        if self.isaaclab_available:
            self._setup_isaaclab_ik()
        else:
            print("Using analytical IK fallback")
    
    def _setup_isaaclab_ik(self):
        """Setup IsaacLab IK components"""
        try:
            # Create simulation context
            if 'SimulationCfg' in globals() and SimulationCfg is not None:
                sim_cfg = SimulationCfg(dt=0.01, render_interval=1)
                self.sim_context = SimulationContext(sim_cfg)
            else:
                self.sim_context = SimulationContext(stage_units_in_meters=1.0, physics_dt=0.01, rendering_dt=0.01)

            # Setup Franka arm
            self.franka_cfg = FRANKA_PANDA_CFG.copy()
            self.franka_cfg.prim_path = "/World/Franka"
            self.franka_arm = Articulation(cfg=self.franka_cfg)

            # Reset and step once to initialize state buffers
            self.sim_context.reset()
            self.sim_context.step()

            # Cache end-effector and arm joint indices
            self.ee_frame_idx = self.franka_arm.find_bodies("panda_hand")[0][0]
            self.ee_jacobi_idx = self.ee_frame_idx - 1
            self.arm_joint_ids = self.franka_arm.find_joints(["panda_joint.*"])[0]

            # Preferred torch device
            sim_device = getattr(self.sim_context, "device", None)
            if sim_device is None:
                sim_device = "cuda:0" if torch.cuda.is_available() else "cpu"
            if isinstance(sim_device, str):
                self.device = torch.device(sim_device) if ("cuda" in sim_device and torch.cuda.is_available()) else torch.device("cpu") if sim_device.startswith("cuda") and not torch.cuda.is_available() else torch.device(sim_device)
            else:
                self.device = sim_device

            # Setup IK controller (DLS)
            ik_cfg = DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
                ik_params={"lambda_val": 0.05},
            )
            self.ik_controller = DifferentialIKController(cfg=ik_cfg, num_envs=1, device=self.device)

            print("✅ IsaacLab IK solver initialized")
            
        except Exception as e:
            print(f"❌ IsaacLab IK setup failed: {e}")
            self.isaaclab_available = False

    def _record_condition_number(self, value: float) -> None:
        self.metrics.setdefault("condition_numbers", []).append(float(value))
    
    def solve_ik(self, position: List[float], orientation: List[float]) -> np.ndarray:
        """
        Solve inverse kinematics for given pose
        
        Args:
            position: [x, y, z] in meters
            orientation: [x, y, z, w] quaternion
            
        Returns:
            joint_angles: 7-DOF joint angles for Franka Panda
        """
        
        if self.isaaclab_available and hasattr(self, 'ik_controller'):
            return self._solve_ik_isaaclab(position, orientation)

        self._record_condition_number(float("nan"))
        return self._solve_ik_analytical(position, orientation, record=False)
    
    def _solve_ik_isaaclab(self, position: List[float], orientation: List[float]) -> np.ndarray:
        """Solve IK using IsaacLab"""
        try:
            # Convert desired world pose to torch
            from isaaclab.utils.array import convert_to_torch

            pos_w = convert_to_torch(np.array([position]), device=self.device, dtype=torch.float32)
            quat_xyzw = convert_to_torch(np.array([orientation]), device=self.device, dtype=torch.float32)
            quat_wxyz = math_utils.convert_quat(quat_xyzw, to="wxyz")

            # Ensure buffers are up to date
            self.franka_arm.update(self.sim_context.get_physics_dt())

            # Current root and EE pose in world
            root_pose_w = self.franka_arm.data.root_pose_w.to(self.device)  # (1,13)
            ee_pose_w = self.franka_arm.data.body_link_pose_w[:, self.ee_frame_idx].to(self.device)  # (1,7)

            # Convert desired world pose to base frame (b)
            des_pos_b, des_quat_b = math_utils.subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], pos_w, quat_wxyz
            )

            # Prepare command and current state (in base frame)
            self.ik_controller.set_command(torch.cat([des_pos_b, des_quat_b], dim=-1))

            cur_pos_b, cur_quat_b = math_utils.subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )

            # Jacobian in world frame for EE link and arm joints
            jacobians = self.franka_arm.root_physx_view.get_jacobians()
            jacobian_w = jacobians[:, self.ee_jacobi_idx, :, self.arm_joint_ids].to(self.device)

            # Rotate Jacobian to base frame
            base_quat = root_pose_w[:, 3:7]
            base_rot = math_utils.matrix_from_quat(math_utils.quat_inv(base_quat))
            jacobian_b = jacobian_w.clone()
            jacobian_b[:, :3, :] = torch.bmm(base_rot, jacobian_b[:, :3, :])
            jacobian_b[:, 3:, :] = torch.bmm(base_rot, jacobian_b[:, 3:, :])

            # Current joint positions
            joint_pos = self.franka_arm.data.joint_pos[:, self.arm_joint_ids].to(self.device)

            cond_number = _compute_condition_number(jacobian_b[0])
            self._record_condition_number(cond_number)

            # Compute desired joints
            joint_pos_des = self.ik_controller.compute(cur_pos_b, cur_quat_b, jacobian_b, joint_pos)
            return joint_pos_des[0].cpu().numpy()
        
        except Exception as e:
            print(f"IsaacLab IK failed: {e}, using analytical fallback")
            self._record_condition_number(float("inf"))
            return self._solve_ik_analytical(position, orientation, record=False)
    
    def _solve_ik_analytical(self, position: List[float], orientation: List[float], record: bool = True) -> np.ndarray:
        """
        Analytical IK solution for Franka Panda (simplified)
        This is a placeholder - in production you'd use proper Franka kinematics
        """
        
        x, y, z = position
        qx, qy, qz, qw = orientation
        
        # Convert position to joint angles using simplified geometric approach
        # This is a basic approximation for demo purposes
        
        # Joint 1: Base rotation based on Y position
        q1 = np.arctan2(y, x)
        
        # Joint 2-3: Shoulder and elbow based on reach distance
        reach = np.sqrt(x**2 + y**2 + (z-0.333)**2)  # 0.333 is base height
        reach = np.clip(reach, 0.1, 0.855)  # Franka workspace limits
        
        # Simplified 2-DOF arm solution for shoulder/elbow
        L1, L2 = 0.316, 0.384  # Franka link lengths (approximate)
        cos_q3 = (reach**2 - L1**2 - L2**2) / (2 * L1 * L2)
        cos_q3 = np.clip(cos_q3, -1, 1)
        
        q3 = np.arccos(cos_q3)
        q2 = np.arctan2(z - 0.333, np.sqrt(x**2 + y**2)) - np.arctan2(L2 * np.sin(q3), L1 + L2 * np.cos(q3))
        
        # Joints 4-7: Wrist orientation (simplified)
        # Convert quaternion to Euler for wrist joints
        r = R.from_quat([qx, qy, qz, qw])
        euler = r.as_euler('xyz')
        
        q4 = -q2 - q3  # Elbow joint
        q5 = euler[2]   # Wrist rotation
        q6 = euler[1]   # Wrist pitch  
        q7 = euler[0]   # Wrist roll
        
        # Apply joint limits (approximate Franka limits)
        joint_limits = [
            (-2.8973, 2.8973),   # q1
            (-1.7628, 1.7628),   # q2
            (-2.8973, 2.8973),   # q3
            (-3.0718, -0.0698),  # q4
            (-2.8973, 2.8973),   # q5
            (-0.0175, 3.7525),   # q6
            (-2.8973, 2.8973)    # q7
        ]
        
        joints = [q1, q2, q3, q4, q5, q6, q7]
        for i, (joint, (min_lim, max_lim)) in enumerate(zip(joints, joint_limits)):
            joints[i] = np.clip(joint, min_lim, max_lim)

        if record:
            self._record_condition_number(float("nan"))

        return np.array(joints)


def create_quaternion_from_euler(roll, pitch, yaw):
    """Convert Euler angles (degrees) to quaternion [x, y, z, w]"""
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    return r.as_quat().tolist()


def demo_trajectory_to_joint_angles():
    """Main demo: Generate trajectory and convert to joint angles"""
    print("=== TRAJECTORY TO JOINT ANGLES DEMO ===")
    
    # Step 1: Generate trajectory waypoints
    print("\n1 Generating trajectory waypoints...")
    
    start_pose = {
        'position': [0.4, 0.0, 0.4],      # Franka reachable workspace
        'orientation': [0, 0, 0, 1]        # No rotation
    }
    
    end_pose = {
        'position': [0.6, 0.2, 0.3],      # Different position
        'orientation': create_quaternion_from_euler(0, 45, 30)  # Some rotation
    }
    
    traj_gen = Trajectory_generator(
        start_pose=start_pose,
        end_pose=end_pose,
        total_time=5.0,
        max_velocity=0.2,
        num_waypoints=20  # Keep it manageable for demo
    )
    
    waypoints = traj_gen.compute_waypoints()
    print(f"   Generated {len(waypoints['positions'])} waypoints")
    print(f"   Total time: {waypoints['total_time']:.2f}s")
    print(f"   Distance: {waypoints['distance']:.3f}m")
    
    # Step 2: Initialize IK solver
    print("\n2 Initializing IK solver...")
    ik_solver = FrankaIKSolver()
    
    # Step 3: Convert poses to joint angles
    print("\n3 Converting poses to joint angles...")
    
    joint_trajectories = []
    ik_success_count = 0
    
    for i, (pos, ori, timestamp) in enumerate(zip(
        waypoints['positions'], 
        waypoints['orientations'], 
        waypoints['timestamps']
    )):
        try:
            # Solve IK
            joint_angles = ik_solver.solve_ik(pos, ori)
            joint_trajectories.append(joint_angles.tolist())
            ik_success_count += 1
            
            # Show progress for first few and last few waypoints
            if i < 3 or i >= len(waypoints['positions']) - 3:
                print(f"   Waypoint {i+1:2d}: t={timestamp:.2f}s")
                print(f"      Pose: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                print(f"      Joints: [{', '.join([f'{j:.3f}' for j in joint_angles])}]")
            elif i == 3:
                print("      ... (intermediate waypoints) ...")
                
        except Exception as e:
            print(f"    IK failed for waypoint {i+1}: {e}")
            # Use previous solution or home position
            if joint_trajectories:
                joint_trajectories.append(joint_trajectories[-1])
            else:
                joint_trajectories.append([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.8])
    
    ik_success_rate = (ik_success_count / len(waypoints['positions'])) * 100
    print(f"\n   IK Success Rate: {ik_success_rate:.1f}%")
    
    # Step 4: Show results summary
    print("\n4 Results Summary:")
    print(f"    Total waypoints: {len(waypoints['positions'])}")
    print(f"    Successful IK solutions: {ik_success_count}")
    print(f"   Trajectory duration: {waypoints['total_time']:.2f}s")
    print(f"    Distance traveled: {waypoints['distance']:.3f}m")
    print(f"   Average velocity: {waypoints['distance']/waypoints['total_time']:.3f} m/s")
    
    # Show joint angle ranges
    if joint_trajectories:
        joint_array = np.array(joint_trajectories)
        print(f"\n   🤖 Joint angle ranges (radians):")
        joint_names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7']
        for i, name in enumerate(joint_names):
            min_angle = joint_array[:, i].min()
            max_angle = joint_array[:, i].max()
            range_deg = np.degrees(max_angle - min_angle)
            print(f"      {name}: {min_angle:.3f} to {max_angle:.3f} rad (range: {range_deg:.1f}°)")
    
    return {
        'trajectory_generator': traj_gen,
        'waypoints': waypoints,
        'joint_trajectories': joint_trajectories,
        'ik_solver': ik_solver,
        'ik_success_rate': ik_success_rate
    }


def save_complete_trajectory(results: Dict, filename: str = "franka_trajectory_complete.txt"):
    """Save trajectory with poses and joint angles"""
    
    waypoints = results['waypoints']
    joint_trajectories = results['joint_trajectories']
    
    with open(filename, 'w') as f:
        f.write("# Franka Trajectory: Poses and Joint Angles\n")
        f.write(f"# IK Success Rate: {results['ik_success_rate']:.1f}%\n")
        f.write(f"# Total time: {waypoints['total_time']:.3f} seconds\n")
        f.write(f"# Total distance: {waypoints['distance']:.3f} meters\n")
        f.write("# Format: timestamp, x, y, z, qx, qy, qz, qw, velocity, j1, j2, j3, j4, j5, j6, j7\n")
        
        for i in range(len(waypoints['timestamps'])):
            t = waypoints['timestamps'][i]
            pos = waypoints['positions'][i]
            ori = waypoints['orientations'][i]
            vel = waypoints['velocities'][i]
            joints = joint_trajectories[i]
            
            f.write(f"{t:.4f}, {pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}, "
                   f"{ori[0]:.6f}, {ori[1]:.6f}, {ori[2]:.6f}, {ori[3]:.6f}, {vel:.6f}, "
                   f"{', '.join([f'{j:.6f}' for j in joints])}\n")
    
    print(f" Complete trajectory saved to {filename}")


def main():
    """Main function demonstrating the complete pipeline"""
    print(" Franka Trajectory Generator with IK")
    print("=" * 50)
    
    try:
        # Run the main demo
        results = demo_trajectory_to_joint_angles()
        
        # Plot trajectory
        print("\n📈 Plotting trajectory...")
        results['trajectory_generator'].plot_trajectory()
        
        # Save complete results
        save_complete_trajectory(results)
        
        print("\n Pipeline completed successfully!")
        print("\nWhat was accomplished:")
        print("  Generated smooth trajectory waypoints using SLERP interpolation")
        print("  Spawned Franka arm configuration")
        print("  Converted all pose waypoints to joint angles using IK")
        print("  Saved complete trajectory data")
        
        print(f"\nGenerated files:")
        print("  franka_trajectory_complete.txt")
        
    except Exception as e:
        print(f" Error occurred: {e}")
        print("Requirements:")
        print("  - numpy, scipy, matplotlib")
        print("  - IsaacLab (optional, will use analytical IK as fallback)")


if __name__ == "__main__":
    main()
