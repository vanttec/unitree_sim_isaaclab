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
    # short wide cylinder; bottom sits on the table surface (~0.794)
    # center = table_top(0.794) + height/2(0.025) = 0.819
    base = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Base",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-0.35, 0.40, 0.819],
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
                static_friction=1.5,
                dynamic_friction=1.5,
                restitution=0.0,
            ),
        ),
    )

    # 4. deck of cards (graspable object) resting flat on top of the base
    # box lying flat: size = (0.063 x 0.088 x 0.02) ~ poker deck
    # base top = 0.819 + 0.025 = 0.844; deck center = 0.844 + 0.02/2 = 0.854
    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.35, 0.40, 0.854],
                                                  rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.063, 0.088, 0.02),   # deck of cards (W x L x thickness)
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.095),  # ~95 g full deck
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95)),  # white
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=1.5,
                dynamic_friction=1.5,
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
