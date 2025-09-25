#!/usr/bin/env python3
"""Replay a Cartesian end-effector trajectory on the Franka Panda inside Isaac Lab."""

import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R, Slerp

try:
    from isaaclab.app import AppLauncher
except ImportError as import_exc:  # pragma: no cover - helpful guidance when Isaac Lab missing
    raise ImportError(
        "Could not import isaaclab. Activate the Isaac Lab Conda environment (env_isaaclab) before "
        "running this script."
    ) from import_exc


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Cartesian waypoints with franka_traj.Trajectory_generator, convert them to joint "
            "trajectories via FrankaIKSolver, and replay the motion inside a minimal Isaac Lab scene."
        )
    )

    # Cartesian trajectory configuration
    parser.add_argument(
        "--start-pos",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=[0.0, 0.0, 1.486],
        help="Start position in metres (Frankas nominal home pose).",
    )
    parser.add_argument(
        "--start-rpy",
        type=float,
        nargs=3,
        metavar=("ROLL", "PITCH", "YAW"),
        default=[0.0, 180.0, 0.0],
        help="Start orientation as XYZ Euler angles in degrees (tool pointing down).",
    )
    parser.add_argument(
        "--end-pos",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=[0.0, 0.0, 1.486],
        help="Goal position in metres (defaults to Frankas home pose).",
    )
    parser.add_argument(
        "--end-rpy",
        type=float,
        nargs=3,
        metavar=("ROLL", "PITCH", "YAW"),
        default=[0.0, 180.0, 0.0],
        help="Goal orientation as XYZ Euler angles in degrees (defaults to home pose).",
    )
    parser.add_argument(
        "--total-time", type=float, default=6.0,
        help="Duration of the Cartesian motion in seconds."
    )
    parser.add_argument(
        "--num-waypoints", type=int, default=10,
        help="Number of Cartesian waypoints sampled along the path."
    )
    parser.add_argument(
        "--max-vel", type=float, default=0.25,
        help="Optional translational velocity limit in m/s (set <=0 to disable)."
    )
    parser.add_argument(
        "--max-acc", type=float, default=0.0,
        help="Optional translational acceleration limit in m/s^2 (set <=0 to disable)."
    )

    # Playback options
    parser.add_argument(
        "--physics-dt", type=float, default=1.0 / 120.0,
        help="Physics step used for SimulationContext (seconds)."
    )
    parser.add_argument(
        "--render-dt", type=float, default=1.0 / 60.0,
        help="Render step used for SimulationContext (seconds)."
    )
    parser.add_argument(
        "--interpolation", choices=["linear", "hold"], default="linear",
        help="How joint targets are sampled between Cartesian waypoints."
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="Replay the motion repeatedly until the window closes."
    )
    parser.add_argument(
        "--gripper-open", type=float, default=0.04,
        help="Finger joint target during playback (metres)."
    )
    parser.add_argument(
        "--debug-print", action="store_true",
        help="Print joint targets at the start/mid/end of execution."
    )

    # Control mode for playback
    parser.add_argument(
        "--control-mode", choices=["position", "effort"], default="position",
        help="Playback using position targets or torque (effort) via simple joint PD."
    )
    parser.add_argument("--kp", type=float, default=50.0, help="PD Kp for effort control (Nm/rad)")
    parser.add_argument("--kd", type=float, default=5.0, help="PD Kd for effort control (Nms/rad)")

    # Visualization and logging helpers
    parser.add_argument(
        "--viz-waypoints",
        choices=["none", "endpoints", "all"],
        default="endpoints",
        help="Toggle frame markers at trajectory waypoints (endpoints-only or full set).",
    )
    parser.add_argument(
        "--viz-frame-scale",
        type=float,
        default=0.14,
        help="Scale applied to visualization frame markers.",
    )
    parser.add_argument(
        "--viz-ee-frame",
        action="store_true",
        help="Render a live marker for the Franka end-effector while tracking the trajectory.",
    )
    parser.add_argument(
        "--save-track",
        type=str,
        default=None,
        help="Optional path to save joint/EE tracking data (.npz).",
    )

    # Let Isaac Lab expose its own CLI flags (e.g. --headless, --/kit options, etc.).
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _orientation_from_rpy(rpy_deg: List[float]):
    from franka_traj import create_quaternion_from_euler

    return create_quaternion_from_euler(*rpy_deg)


def _generate_cartesian_path(args: argparse.Namespace):
    from franka_traj import Trajectory_generator

    start_pose = {
        "position": args.start_pos,
        "orientation": _orientation_from_rpy(args.start_rpy),
    }
    end_pose = {
        "position": args.end_pos,
        "orientation": _orientation_from_rpy(args.end_rpy),
    }

    max_velocity = args.max_vel if args.max_vel > 0.0 else None
    max_acceleration = args.max_acc if args.max_acc > 0.0 else None

    generator = Trajectory_generator(
        start_pose=start_pose,
        end_pose=end_pose,
        total_time=args.total_time,
        max_velocity=max_velocity,
        max_acceleration=max_acceleration,
        num_waypoints=args.num_waypoints,
    )

    waypoints = generator.compute_waypoints()
    return generator, waypoints


def _compute_joint_trajectory(waypoints: dict) -> Tuple[np.ndarray, float, object, Dict[str, List[float]]]:
    from franka_traj import FrankaIKSolver

    ik_solver = FrankaIKSolver()
    if not getattr(ik_solver, "isaaclab_available", False):
        raise RuntimeError("Isaac Lab is required for this script; install omni.isaac.lab/isaaclab packages.")

    joints: List[np.ndarray] = []
    success_count = 0

    for pos, quat in zip(waypoints["positions"], waypoints["orientations"]):
        try:
            joint_angles = ik_solver.solve_ik(pos, quat)
            joints.append(np.asarray(joint_angles, dtype=np.float32))
            success_count += 1
        except Exception as exc:  # pragma: no cover - informative log when IK fails
            print(f"[WARN] IK failed for pose {pos}: {exc}")
            if joints:
                joints.append(joints[-1].copy())
            else:
                joints.append(np.zeros(7, dtype=np.float32))

    success_rate = 100.0 * success_count / max(1, len(joints))
    franka_cfg = getattr(ik_solver, "franka_cfg", None)
    metrics = getattr(ik_solver, "metrics", {})
    del ik_solver

    if franka_cfg is None:
        raise RuntimeError("Franka IK solver did not expose its articulation config; update franka_traj.py")

    return np.vstack(joints), success_rate, franka_cfg, metrics


def _resample_joint_targets(
    timestamps: np.ndarray,
    joint_targets: np.ndarray,
    target_dt: float,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    dense_times = np.arange(0.0, timestamps[-1] + 1e-9, target_dt)

    if mode == "hold":
        sample_indices = np.searchsorted(timestamps, dense_times, side="right") - 1
        sample_indices = np.clip(sample_indices, 0, len(joint_targets) - 1)
        dense_targets = joint_targets[sample_indices]
    else:
        dense_targets = np.empty((dense_times.size, joint_targets.shape[1]), dtype=np.float32)
        for joint_id in range(joint_targets.shape[1]):
            dense_targets[:, joint_id] = np.interp(dense_times, timestamps, joint_targets[:, joint_id])

    return dense_times, dense_targets


def _resample_cartesian_waypoints(waypoints: dict, target_times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate Cartesian waypoints onto the dense playback timeline."""

    key_times = np.asarray(waypoints["timestamps"], dtype=np.float64)
    positions = np.asarray(waypoints["positions"], dtype=np.float64)
    orientations = np.asarray(waypoints["orientations"], dtype=np.float64)

    if positions.shape[0] == 0:
        return np.zeros((target_times.size, 3), dtype=np.float32), np.tile([0.0, 0.0, 0.0, 1.0], (target_times.size, 1)).astype(np.float32)

    clipped_times = np.clip(target_times.astype(np.float64), key_times[0], key_times[-1])

    dense_positions = np.empty((clipped_times.size, 3), dtype=np.float32)
    for axis in range(3):
        dense_positions[:, axis] = np.interp(clipped_times, key_times, positions[:, axis])

    if orientations.shape[0] == 1:
        dense_orientations = np.repeat(orientations.astype(np.float32), clipped_times.size, axis=0)
    else:
        rotation_keys = R.from_quat(orientations)
        slerp = Slerp(key_times, rotation_keys)
        dense_orientations = slerp(clipped_times).as_quat().astype(np.float32)

    return dense_positions, dense_orientations


def _setup_scene(franka_cfg, physics_dt: float, render_dt: float):
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext, SimulationCfg
    from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
    from isaaclab.assets import Articulation

    # The FrankaIKSolver already spawns a SimulationContext when Isaac Lab is present, but we recreate
    # it with the requested time-steps to keep the demo predictable.
    render_interval = max(1, int(round(render_dt / physics_dt)))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sim_cfg = SimulationCfg(dt=physics_dt, render_interval=render_interval, device=device)
    sim_context = SimulationContext(sim_cfg)

    robot_cfg = franka_cfg.copy()
    robot_cfg.prim_path = "/World/Franka"
    franka_arm = Articulation(cfg=robot_cfg)

    spawn_ground_plane("/World/Ground", GroundPlaneCfg())

    # Simple dome light to keep the scene visible when rendering is enabled
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0))
    light_cfg.func("/World/Lighting", light_cfg)

    sim_context.reset()
    sim_context.step()
    return sim_context, franka_arm


def _create_frame_marker(sim_context, position, orientation, prim_path="/World/Frame", scale=0.12):
    """Create a single static visualization frame."""

    from isaaclab.markers import VisualizationMarkers
    from isaaclab.markers.config import FRAME_MARKER_CFG
    from isaaclab.utils import math as math_utils

    device = torch.device(sim_context.device) if hasattr(sim_context, "device") else torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    frame_cfg = FRAME_MARKER_CFG.copy()
    frame_cfg.prim_path = prim_path
    frame_cfg.markers["frame"].scale = (scale, scale, scale)
    frame_cfg.markers.pop("connecting_line", None)
    frame_cfg.markers.pop("frame_secondary", None)
    marker = VisualizationMarkers(frame_cfg)

    pos_tensor = torch.tensor([position], device=device, dtype=torch.float32)
    quat_xyzw = torch.tensor([orientation], device=device, dtype=torch.float32)
    quat_wxyz = math_utils.convert_quat(quat_xyzw, to="wxyz")
    marker.visualize(pos_tensor, quat_wxyz)
    return marker


def _create_waypoint_markers(sim_context, waypoints: Dict[str, List], mode: str, scale: float):
    """Spawn frame markers for the requested subset of waypoints."""

    markers = []
    if mode == "none":
        return markers

    positions = waypoints["positions"]
    orientations = waypoints["orientations"]

    if not positions:
        return markers

    # Always include endpoints when visualization is enabled
    markers.append(
        _create_frame_marker(
            sim_context,
            positions[0],
            orientations[0],
            prim_path="/World/Waypoints/Start",
            scale=scale,
        )
    )

    if len(positions) > 1:
        markers.append(
            _create_frame_marker(
                sim_context,
                positions[-1],
                orientations[-1],
                prim_path="/World/Waypoints/End",
                scale=scale,
            )
        )

    if mode == "all" and len(positions) > 2:
        for idx in range(1, len(positions) - 1):
            markers.append(
                _create_frame_marker(
                    sim_context,
                    positions[idx],
                    orientations[idx],
                    prim_path=f"/World/Waypoints/WP_{idx:03d}",
                    scale=scale,
                )
            )

    return markers


def _playback(
    sim_context,
    simulation_app,
    franka_arm,
    dense_targets: np.ndarray,
    dense_times: np.ndarray,
    desired_cartesian: Optional[Tuple[np.ndarray, np.ndarray]],
    gripper_open: float,
    loop_forever: bool,
    debug_print: bool,
    control_mode: str,
    kp: float,
    kd: float,
    ee_marker=None,
    log_tracking: bool = False,
):
    import torch
    from isaaclab.utils import math as math_utils

    sim_dt = sim_context.get_physics_dt()

    if hasattr(franka_arm, "data"):
        franka_arm.update(sim_dt)
        num_dof = franka_arm.data.joint_pos.shape[-1]
        device = franka_arm.data.joint_pos.device
    else:
        num_dof = dense_targets.shape[1] + 2
        device = torch.device("cpu")

    ee_body_idx = None
    if hasattr(franka_arm, "find_bodies"):
        try:
            ee_body_idx = franka_arm.find_bodies("panda_hand")[0][0]
        except Exception:
            ee_body_idx = None

    # Build full joint targets (append finger DOFs).
    full_targets = torch.zeros((dense_targets.shape[0], num_dof), device=device, dtype=torch.float32)
    arm_dof = min(dense_targets.shape[1], num_dof)
    full_targets[:, :arm_dof] = torch.from_numpy(dense_targets[:, :arm_dof]).to(device=device, dtype=torch.float32)
    if num_dof > arm_dof:
        full_targets[:, arm_dof:] = gripper_open

    frame_count = full_targets.shape[0]
    step = 0

    if debug_print:
        mid_idx = frame_count // 2
        print("[INFO] First joint target:", full_targets[0, :arm_dof].cpu().numpy())
        print("[INFO] Mid joint target:", full_targets[mid_idx, :arm_dof].cpu().numpy())
        print("[INFO] Last joint target:", full_targets[-1, :arm_dof].cpu().numpy())

    # Arm joint count (exclude gripper if present)
    arm_dof = min(7, num_dof)

    tracking_data = [] if log_tracking else None

    while simulation_app.is_running():
        if control_mode == "position":
            franka_arm.set_joint_position_target(full_targets[step : step + 1])
        else:
            q = franka_arm.data.joint_pos[:, :arm_dof]
            qd = franka_arm.data.joint_vel[:, :arm_dof]
            q_des = full_targets[step : step + 1, :arm_dof]
            tau = kp * (q_des - q) - kd * qd
            if num_dof > arm_dof:
                tau_full = torch.zeros_like(franka_arm.data.joint_pos)
                tau_full[:, :arm_dof] = tau
            else:
                tau_full = tau
            franka_arm.set_joint_effort_target(tau_full)

        franka_arm.write_data_to_sim()
        sim_context.step()
        franka_arm.update(sim_dt)

        if ee_marker is not None and ee_body_idx is not None:
            ee_pose = franka_arm.data.body_link_pose_w[:, ee_body_idx]
            ee_pos = ee_pose[0, 0:3].detach().to(device)
            ee_quat = ee_pose[0, 3:7].detach().to(device)
            quat_wxyz = math_utils.convert_quat(ee_quat.unsqueeze(0), to="wxyz")
            ee_marker.visualize(ee_pos.unsqueeze(0), quat_wxyz)

        if tracking_data is not None and ee_body_idx is not None:
            joint_actual = franka_arm.data.joint_pos[0, :arm_dof].detach().cpu().numpy()
            ee_pose = franka_arm.data.body_link_pose_w[0, ee_body_idx].detach().cpu().numpy()
            time_val = dense_times[step] if step < dense_times.size else step * sim_dt
            entry: Dict[str, np.ndarray] = {
                "time": float(time_val),
                "joint_target": full_targets[step, :arm_dof].detach().cpu().numpy(),
                "joint_actual": joint_actual,
                "ee_pos": ee_pose[:3].astype(np.float32),
                "ee_quat_xyzw": ee_pose[3:7].astype(np.float32),
            }
            if desired_cartesian is not None:
                des_pos, des_quat = desired_cartesian
                if step < des_pos.shape[0]:
                    entry["ee_pos_des"] = des_pos[step]
                if step < des_quat.shape[0]:
                    entry["ee_quat_des_xyzw"] = des_quat[step]
            tracking_data.append(entry)

        step += 1
        if step >= frame_count:
            if loop_forever:
                step = 0
            else:
                break

    return tracking_data


def _save_tracking_data(
    path: str,
    samples: List[Dict[str, np.ndarray]],
    ik_metrics: Optional[Dict[str, List[float]]] = None,
    waypoint_timestamps: Optional[List[float]] = None,
):
    if not samples:
        print("[WARN] No tracking samples were captured; skipping save.")
        return

    data: Dict[str, np.ndarray] = {}

    data["time"] = np.asarray([entry["time"] for entry in samples], dtype=np.float32)
    for key in (
        "joint_target",
        "joint_actual",
        "ee_pos",
        "ee_quat_xyzw",
        "ee_pos_des",
        "ee_quat_des_xyzw",
    ):
        if key in samples[0]:
            data[key] = np.stack([entry[key] for entry in samples], axis=0)

    if ik_metrics:
        conds = ik_metrics.get("condition_numbers", [])
        if conds:
            data["ik_condition_numbers"] = np.asarray(conds, dtype=np.float32)
            if waypoint_timestamps is not None:
                limit = min(len(conds), len(waypoint_timestamps))
                data["ik_waypoint_timestamps"] = np.asarray(waypoint_timestamps[:limit], dtype=np.float32)

    np.savez(path, **data)
    print(f"[INFO] Saved tracking log: {path}")


def _plot_condition_numbers(
    timestamps: np.ndarray,
    condition_numbers: np.ndarray,
    show: bool = False,
    save_path: Optional[str] = None,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available; cannot plot condition numbers.")
        return

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.plot(timestamps, condition_numbers, marker="o", linestyle="-", color="tab:blue")
    ax.set_xlabel("Waypoint time (s)")
    ax.set_ylabel("Jacobian condition number")
    ax.set_title("Franka IK Jacobian conditioning")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    finite_vals = condition_numbers[np.isfinite(condition_numbers)]
    if finite_vals.size:
        ax.set_ylim([max(1.0, finite_vals.min() * 0.8), finite_vals.max() * 1.2])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved condition plot: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    _, waypoints = _generate_cartesian_path(args)
    joint_targets, success_rate, franka_cfg, ik_metrics = _compute_joint_trajectory(waypoints)
    print(f"[INFO] Generated {len(waypoints['positions'])} Cartesian waypoints.")
    print(f"[INFO] IK success rate: {success_rate:.1f}%")

    condition_numbers = np.asarray(ik_metrics.get("condition_numbers", []), dtype=np.float64)
    if condition_numbers.size:
        finite_mask = np.isfinite(condition_numbers)
        if finite_mask.any():
            finite_vals = condition_numbers[finite_mask]
            max_val = float(finite_vals.max())
            print(f"[INFO] Max Jacobian condition number: {max_val:.2f}")
            if max_val > 200.0:
                print("[WARN] Jacobian conditioning exceeded 200 — trajectory may approach a singular configuration.")
        nonfinite = np.logical_not(np.isfinite(condition_numbers))
        if nonfinite.any():
            inf_count = np.isinf(condition_numbers).sum()
            nan_count = np.isnan(condition_numbers).sum()
            print(
                f"[WARN] Condition log contains {inf_count} inf and {nan_count} nan entries — check for singular or unreachable poses."
            )

    sim_context, franka_arm = _setup_scene(franka_cfg, args.physics_dt, args.render_dt)

    waypoint_markers = _create_waypoint_markers(
        sim_context,
        waypoints,
        mode=args.viz_waypoints,
        scale=args.viz_frame_scale,
    )
    _ = waypoint_markers

    ee_marker = None
    if args.viz_ee_frame:
        ee_marker = _create_frame_marker(
            sim_context,
            waypoints["positions"][0],
            waypoints["orientations"][0],
            prim_path="/World/Tracking/EndEffector",
            scale=args.viz_frame_scale,
        )

    dense_times, dense_targets = _resample_joint_targets(
        np.asarray(waypoints["timestamps"], dtype=np.float32),
        joint_targets,
        args.physics_dt,
        args.interpolation,
    )

    desired_cartesian = _resample_cartesian_waypoints(waypoints, dense_times)

    print(f"[INFO] Playback frames: {dense_targets.shape[0]} (dt={args.physics_dt:.4f}s)")

    save_track_path = args.save_track
    if args.loop and save_track_path:
        print("[WARN] Ignoring --save-track because --loop repeats indefinitely.")
        save_track_path = None

    try:
        tracking_samples = _playback(
            sim_context=sim_context,
            simulation_app=simulation_app,
            franka_arm=franka_arm,
            dense_targets=dense_targets,
            dense_times=dense_times,
            desired_cartesian=desired_cartesian,
            gripper_open=args.gripper_open,
            loop_forever=args.loop,
            debug_print=args.debug_print,
            control_mode=args.control_mode,
            kp=args.kp,
            kd=args.kd,
            ee_marker=ee_marker,
            log_tracking=bool(save_track_path),
        )
        if save_track_path and tracking_samples is not None:
            _save_tracking_data(
                save_track_path,
                tracking_samples,
                ik_metrics=ik_metrics,
                waypoint_timestamps=waypoints.get("timestamps"),
            )
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
