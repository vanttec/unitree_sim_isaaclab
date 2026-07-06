# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
public base scene configuration module (diagonal stick on green base)
provides a scene with the G1 robot facing a packing table, a static green
cylindrical base with two support posts (a cradle) sitting on the table, and a
black stick mounted diagonally across the cradle, elevated so the hand fits in
the gap below it.
"""
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from tasks.common_config import CameraBaseCfg  # isort: skip
import os
project_root = os.environ.get("PROJECT_ROOT")


@configclass
class TableStickSceneCfg(InteractiveSceneCfg):
    """scene: G1 robot + packing table + green base + cradle + diagonal stick."""

    # 1. room walls
    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0],
            rot=[1.0, 0.0, 0.0, 0.0]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
        ),
    )

    # 2. tables
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, -0.2],
                                                rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/PackingTable/PackingTable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    packing_table_2 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable_2",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-3.5, 0.55, -0.2],
                                                rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/PackingTable/PackingTable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    packing_table_3 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable_3",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[3.5, 0.55, -0.2],
                                                rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/PackingTable/PackingTable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    packing_table_4 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable_4",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[3.5, -5, -0.2],
                                                rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/PackingTable/PackingTable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    packing_table_5 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable_5",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-3.5, -5, -0.2],
                                                rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/PackingTable/PackingTable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    packing_table_6 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable_6",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, -5, -0.2],
                                                rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/PackingTable/PackingTable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # 3. green base (static pedestal on the table)
    # Robot spawns at x=-0.15; align base with robot midline on the table.
    # short wide cylinder; bottom sits on the table surface (~0.794)
    # center = table_top(0.794) + height/2(0.025) = 0.819
    base = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Base",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-0.15, 0.40, 0.819],
                                                rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.CylinderCfg(
            radius=0.055,   # base radius
            height=0.05,    # base height
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.33, 0.40, 0.18)),  # olive green
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=3.0,
                dynamic_friction=3.0,
                restitution=0.0,
            ),
        ),
    )

    # 3b. cradle: two static green posts of different heights on the base top (0.844).
    # left post taller, right post shorter -> the stick laid across sits diagonally.
    # short post on the robot (humanoid) side -> the stick's low end faces the robot
    support_left = AssetBaseCfg(
        prim_path="/World/envs/env_.*/SupportLeft",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-0.15, 0.31, 0.874],  # robot side; bottom on table (0.794)
                                                rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.02, 0.16),   # short: bottom 0.794, top 0.954
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.33, 0.40, 0.18)),  # olive green
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max", restitution_combine_mode="min",
                static_friction=3.0, dynamic_friction=3.0, restitution=0.0,
            ),
        ),
    )
    # tall post on the far side -> the stick's high end
    support_right = AssetBaseCfg(
        prim_path="/World/envs/env_.*/SupportRight",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-0.15, 0.49, 0.894],  # far side; bottom on table (0.794)
                                                rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.02, 0.20),   # tall: bottom 0.794, top 0.994
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.33, 0.40, 0.18)),  # olive green
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max", restitution_combine_mode="min",
                static_friction=3.0, dynamic_friction=3.0, restitution=0.0,
            ),
        ),
    )

    # 4. stick (graspable object) mounted diagonally across the cradle
    # black cylinder, axis Y (spans robot<->far posts), tilted ~12.5 deg about X
    # short post top 0.954 @ y=0.31 (robot side, low), tall post top 0.994 @ y=0.49 (far, high)
    # mid (y=0.40) line height 0.974; stick center z = 0.974 + radius(0.007695) = 0.9817
    # tilt ~12.5 deg about X (+Y end higher, low end toward robot) -> quat [cos6.25, sin6.25, 0, 0]
    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.15, 0.40, 0.9817],
                                                  rot=[0.9940, 0.1089, 0.0, 0.0]),  # tilt 12.5 deg about X
        spawn=sim_utils.CylinderCfg(
            radius=0.007695,  # 5% thinner than 0.0081 (~14.5% vs original 9 mm)
            height=0.24,    # stick length
            axis="Y",       # long axis spans the two posts
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.059),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05)),  # black
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=3.0,
                dynamic_friction=3.0,
                restitution=0.0,
            ),
        ),
    )

    # 5. light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75),
                                     intensity=3000.0),
    )

    world_camera = CameraBaseCfg.get_camera_config(prim_path="/World/PerspectiveCamera",
                                                    pos_offset=(-0.1, 3.6, 1.6),
                                                    rot_offset=(-0.00617, 0.00617, 0.70708, -0.70708),
                                                    focal_length=16.5)
