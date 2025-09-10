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
from isaaclab.assets import AssetBaseCfg

# Actuators
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg

# Terrains
from isaaclab.terrains import TerrainImporterCfg

# Spawning configurations
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg, CollisionPropertiesCfg
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
import isaaclab.sim as sim_utils

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
class RodCfg(RigidObjectCfg):
    """Dynamic rod object the robots must co-manipulate."""
    prim_path: str = "/World/envs/env_.*/Rod"
    init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.5, 0.5, 0.5),
        rot=(0.7071068, 0.0, 0.7071068, 0.0),  # Horizontal orientation
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
    )
    spawn = sim_utils.CapsuleCfg(
        height=0.5,
        radius=0.02,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.25),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 1.0)),
    )
    # Dimensions the env can use for spawning (meters)
    size_xyz: tuple[float, float, float] = (0.50, 0.04, 0.04)  # height, diameter, diameter for capsule
    tabletop_height: float = 0.30  # meters (table surface height)


@configclass
class GoalFixtureCfg(RigidObjectCfg):
    """Static goal/fixture indicating the target placement pose."""
    prim_path: str = "/World/envs/env_.*/GoalFixture"
    spawn = sim_utils.CapsuleCfg(
        height=0.5,
        radius=0.02,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.9)),
    )
    init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.75, 0.0, 0.32),  # On table: x=0.75 (table), y=0.0 (center), z=0.32 (table surface at 0.30 + radius 0.02)
        rot=(0.7071068, 0.0, 0.7071068, 0.0),  # Horizontal orientation
    )
    
    # Goal pose for reward computation (horizontal orientation)
    pose_w = ((0.75, 0.0, 0.32), (0.7071068, 0.0, 0.7071068, 0.0))


# -----------------------------
# Env config (4 agents)
# -----------------------------
@configclass
class FrankaDlMarlCustomEnvCfg(DirectMARLEnvCfg):
    """Custom dual-arm Franka collaborative assembly (4 agents: motion+stiffness per robot)."""

    # --- Sim & Scene ---
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=3.5,   # more space for 2 robots + object
    )

    # --- Robots ---
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd"
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.157, "panda_joint2": -1.066, "panda_joint3": -0.155,
                "panda_joint4": -2.239, "panda_joint5": -1.841, "panda_joint6": 1.003,
                "panda_joint7": 0.469, "panda_finger_joint.*": 0.035,
            },
            pos=(0.0, 0.50, 0.0),
        ),
        actuators={
            "arm": ImplicitActuatorCfg(joint_names_expr=["panda_joint.*"], stiffness=80.0, damping=8.0),
            "hand": ImplicitActuatorCfg(joint_names_expr=["panda_finger_joint.*"], stiffness=2e3, damping=1e2)
        }
    )
    robot2: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd"
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.157, "panda_joint2": -1.066, "panda_joint3": -0.155,
                "panda_joint4": -2.239, "panda_joint5": -1.841, "panda_joint6": 1.003,
                "panda_joint7": 0.469, "panda_finger_joint.*": 0.035,
            },
            pos=(0.0, -0.50, 0.0),
        ),
        actuators={
            "arm": ImplicitActuatorCfg(joint_names_expr=["panda_joint.*"], stiffness=80.0, damping=8.0),
            "hand": ImplicitActuatorCfg(joint_names_expr=["panda_finger_joint.*"], stiffness=2e3, damping=1e2)
        }
    )

    # --- Table ---
    table = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 1.0, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.8, 0.8)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.75, 0.0, 0.25),
        ),
    )

    # --- Terrain ---
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        env_spacing=3.5,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # --- Objects ---
    rod: RodCfg = RodCfg()
    goal_fixture: GoalFixtureCfg = GoalFixtureCfg()

    # --- Episode / control cadence ---
    decimation: int = 2
    episode_length_s: float = 12.0

    # --- Agents (4) ---
    # Actions: motion = 13D (7 pose + 6 wrench), stiffness = 6D (variable Kp)
    possible_agents = ["robot_motion", "robot_stiffness", "robot2_motion", "robot2_stiffness"]

    # Observation schema (same for all agents)
    #   - joints q, qd both robots: 14
    #   - EE poses both robots: 14
    #   - object pose (7) + goal pose (7): 14
    #   - EE wrenches both robots: 12
    # -> total = 14 + 14 + 14 + 12 = 54  (add margins: frame-relative, extras -> set 68 to be safe)
    _OBS_DIM = 68

    observation_spaces = {
        "robot_motion": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(_OBS_DIM,), dtype=np.float32),
        "robot_stiffness": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(_OBS_DIM,), dtype=np.float32),
        "robot2_motion": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(_OBS_DIM,), dtype=np.float32),
        "robot2_stiffness": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(_OBS_DIM,), dtype=np.float32),
    }

    action_spaces = {
        "robot_motion":     gym.spaces.Box(low=-1.0, high=1.0, shape=(13,), dtype=np.float32),
        "robot_stiffness":  gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
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
    object_drop_height: float = 0.20              # below this z (m) -> dropped (below table surface at 0.30)
    safety_force_limit: float = 75.0              # hard stop (N) for measured EE force magnitude
    safety_torque_limit: float = 15.0             # hard stop (Nm)

    # --- Misc / visualization ---
    enable_visualization_markers: bool = True

    # --- Joint PD controller gains (controller-side, not physical drives) ---
    # Scalars apply to all 7 arm joints; provide a length-7 tuple to set per-joint gains.
    pd_kp_joint: float | tuple[float, float, float, float, float, float, float] = 150.0
    pd_kd_joint: float | tuple[float, float, float, float, float, float, float] = 25.0
