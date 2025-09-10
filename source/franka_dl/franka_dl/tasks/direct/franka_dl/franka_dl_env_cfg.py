# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils

from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.assets import AssetBaseCfg


@configclass
class FrankaDlEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s: float = 8.3333
    decimation: int = 2
    action_space: int = 14  # 7D Cartesian control for each arm
    # Corrected to match the minimal observation (9 joint pos per robot)
    observation_space: int = 18
    state_space: int = 0

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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=True)


    # mount
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
    )

    # robot
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
            pos=(1.0, 0.0, 0.0),
        ),
        actuators={
            "arm": ImplicitActuatorCfg(joint_names_expr=["panda_joint.*"], stiffness=80.0, damping=4.0),
            "hand": ImplicitActuatorCfg(joint_names_expr=["panda_finger_joint.*"], stiffness=2e3, damping=1e2)
        }
    )

    # robot2
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
            pos=(1.0, 1.0, 0.0),
        ),
        actuators={
            "arm": ImplicitActuatorCfg(joint_names_expr=["panda_joint.*"], stiffness=80.0, damping=4.0),
            "hand": ImplicitActuatorCfg(joint_names_expr=["panda_finger_joint.*"], stiffness=2e3, damping=1e2)
        }
    )

    # rod
    rod: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Rod",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.5, 0.0, 2.0),
            rot=(0.7071, 2.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CapsuleCfg(
            height=3.0,
            radius=0.02,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.25),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 1.0)),
        ),
    )

    # ground plane
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # rewards
    action_penalty_scale: float = 0.05
