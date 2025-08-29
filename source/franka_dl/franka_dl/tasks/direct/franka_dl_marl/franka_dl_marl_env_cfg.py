# Copyright (c) 2022-2025, The Isaac Lab Project
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
from isaaclab.utils import configclass

# Core env cfg base
from isaaclab.envs.direct_marl_env_cfg import DirectMARLEnvCfg

# Sim & Scene - NEW (Isaac Lab 4.5)
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.scene import InteractiveSceneCfg

# Assets
from isaaclab.assets import ArticulationCfg
from isaaclab.assets import RigidObjectCfg

# Spawning configurations
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg, CollisionPropertiesCfg
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg

# Franka template (you can swap to a custom one if needed)
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


# -----------------------------
# Robot configs (two Frankas)
# -----------------------------
# Note: Using FRANKA_PANDA_HIGH_PD_CFG directly with replace() method
# for simpler configuration without custom inheritance


# -----------------------------
# Object / goal configs
# -----------------------------
@configclass
class AssemblyObjectCfg(RigidObjectCfg):
    """Dynamic object the robots must co-manipulate."""
    prim_path: str = "/World/envs/env_.*/AssemblyObject"
    spawn: UsdFileCfg = UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        scale=(1.25, 0.2, 0.1),  # Make it beam-like based on size_xyz
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
        ),
        mass_props=MassPropertiesCfg(mass=0.8),
        # Remove collision_props to avoid warnings with instanced USD prims
        visual_material=PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
    )
    # Dimensions the env can use for spawning (meters)
    size_xyz: tuple[float, float, float] = (0.50, 0.08, 0.04)  # e.g., a short beam
    tabletop_height: float = 0.76  # meters


@configclass
class GoalFixtureCfg(RigidObjectCfg):
    """Static goal/fixture indicating the target placement pose."""
    prim_path: str = "/World/envs/env_.*/GoalFixture"
    spawn: UsdFileCfg = UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        scale=(1.25, 0.2, 0.1),  # Same size as assembly object
        rigid_props=RigidBodyPropertiesCfg(
            kinematic_enabled=True, 
            disable_gravity=True
        ),
        # Remove collision_props to avoid warnings with instanced USD prims  
        visual_material=PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.9)),
    )
    pose_w: tuple[tuple[float, float, float], tuple[float, float, float, float]] = (
        (0.0, 0.60, 0.80),  # position in world (example)
        (1.0, 0.0, 0.0, 0.0),  # orientation (w,x,y,z)
    )


# -----------------------------
# Env config (4 agents)
# -----------------------------
@configclass
class FrankaDlMarlEnvCfg(DirectMARLEnvCfg):
    """Dual-arm Franka collaborative assembly (4 agents: motion+stiffness per robot)."""

    # --- Sim & Scene ---
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=2)  # Match decimation
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=3.5,   # more space for 2 robots + object
    )

    # --- Robots ---
    robot1: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot1",
        init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(
            pos=(0.0, 0.50, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0)  # (w,x,y,z) identity
        )
    )
    robot2: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot2",
        init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(
            pos=(0.0, -0.50, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0)  # 180° about Z: (w,x,y,z) = (0,0,0,1)
        )
    )

    # --- Objects ---
    assembly_object: AssemblyObjectCfg = AssemblyObjectCfg()
    goal_fixture: GoalFixtureCfg = GoalFixtureCfg()

    # --- Episode / control cadence ---
    decimation: int = 2
    episode_length_s: float = 12.0

    # --- Agents (4) ---
    # Actions: motion = 13D (7 pose + 6 wrench), stiffness = 6D (variable Kp)
    possible_agents = ["robot1_motion", "robot1_stiffness", "robot2_motion", "robot2_stiffness"]

    # Observation schema (same for all agents)
    #   - joints q, qd both robots: 14
    #   - EE poses both robots: 14
    #   - object pose (7) + goal pose (7): 14
    #   - EE wrenches both robots: 12
    # -> total = 14 + 14 + 14 + 12 = 54  (add margins: frame-relative, extras -> set 68 to be safe)
    _OBS_DIM = 68

    observation_spaces = {
        "robot1_motion": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(_OBS_DIM,), dtype=np.float32),
        "robot1_stiffness": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(_OBS_DIM,), dtype=np.float32),
        "robot2_motion": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(_OBS_DIM,), dtype=np.float32),
        "robot2_stiffness": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(_OBS_DIM,), dtype=np.float32),
    }

    action_spaces = {
        "robot1_motion":     gym.spaces.Box(low=-1.0, high=1.0, shape=(13,), dtype=np.float32),
        "robot1_stiffness":  gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
        "robot2_motion":     gym.spaces.Box(low=-1.0, high=1.0, shape=(13,), dtype=np.float32),
        "robot2_stiffness":  gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
    }

    # State space: concatenate all observations for centralized critic (MAPPO)
    state_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4 * _OBS_DIM,), dtype=np.float32)

    # --- OSC / stiffness mapping (used by env when combining per-robot actions) ---
    # Per-axis Kp bounds (XYZ then RPY)
    kp_min_task: tuple[float, float, float, float, float, float] = (50.0, 50.0, 50.0, 5.0, 5.0, 5.0)
    kp_max_task: tuple[float, float, float, float, float, float] = (1500.0, 1500.0, 1500.0, 300.0, 300.0, 300.0)

    # Wrench limits for action scaling / safety (env should clamp/mask by these)
    max_force: float = 50.0      # N (per axis)
    max_torque: float = 10.0     # Nm (per axis)

    # --- Reward knobs (grouped per robot; role-specific terms added in env) ---
    distance_reward_scale: float = 2.0
    orientation_reward_scale: float = 0.5
    coordination_reward_scale: float = 0.5        # keep object level / stable / synchronized motion
    action_penalty_scale_motion: float = 0.01
    action_penalty_scale_stiffness: float = 0.005
    conflict_force_penalty_scale: float = 0.5     # penalize opposing commanded forces
    compliance_contact_bonus: float = 0.1         # small bonus for low Kp under contact

    # Shared sparse bonus when object placed & stable
    task_completion_reward: float = 10.0

    # --- Success / termination ---
    # Success if object pose within thresholds and stable for >= hold time
    success_position_threshold: float = 0.03      # meters
    success_orientation_threshold: float = 0.10   # radians (approx. angle error)
    success_hold_time_s: float = 1.0

    # Failure if object dropped or unsafe forces exceeded
    object_drop_height: float = 0.60              # below this z (m) -> dropped (tune to your table)
    safety_force_limit: float = 75.0              # hard stop (N) for measured EE force magnitude
    safety_torque_limit: float = 15.0             # hard stop (Nm)

    # --- Misc / visualization ---
    enable_visualization_markers: bool = True