# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
public base scene configuration module (deck of cards on green base)
provides a scene with the G1 robot facing a packing table, a static green
cylindrical base sitting on the table, and a white rectangular deck of cards
resting on top of it.
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
class TableCardsDeckSceneCfg(InteractiveSceneCfg):
    """scene: G1 robot + packing table + green base + deck of cards."""

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
    # Rectangular pedestal aligned under the deck; bottom sits on the table (~0.794).
    # center = table_top(0.794) + height/2(0.0175) = 0.8115  (30% smaller than prior)
    base = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Base",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-0.152, 0.380, 0.8115],
                                                rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.063, 0.021, 0.035),
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

    # 4. deck of cards (graspable object) resting flat on top of the base
    # base top = 0.829; deck center z = 0.829 + thickness/2
    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.152, 0.380, 0.8412],
                                                  rot=[0.70710678, 0.0, 0.0, 0.70710678]),  # 90 deg yaw: long axis perpendicular to robot
        spawn=sim_utils.CuboidCfg(
            size=(0.0353, 0.0546, 0.0244),   # deck (W x L x thickness), 30% smaller
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.087),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 0.0),
                metallic=0.0,
                roughness=1.0,
            ),  # vanta black
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
