# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""State-based residual-grasp env: tennisball, NO cameras.

Bring-up / validation env that sidesteps the upstream camera annotator crash
(IsaacLab issue #3312). Policy observes privileged object pose (state) instead of
pixels — enough to validate that the residual + reward + PPO learn to beat the
base-only success rate under placement jitter. Swap to the vision env once the
camera bug is resolved; only the obs group changes.
"""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from tasks.common_curriculum.jitter_curriculum import object_jitter_range

from tasks.common_config import G1RobotPresets
from tasks.common_scene.base_scene_pickplace_tennisball import TableTennisBallSceneCfg
from tasks.common_actions import ResidualArmActionCfg
from tasks.common_observations import residual_obs as robs

from tasks.g1_tasks.residual_grasp_coin_g1_29dof_inspire.residual_grasp_coin_env_cfg import (
    RewardsCfg, TerminationsCfg,
)
from tasks.g1_tasks.residual_grasp_tennisball_g1_29dof_inspire.residual_grasp_tennisball_env_cfg import (
    ActionsCfg, EventCfg,
)

_ACTION_NAME = "residual"


@configclass
class ObjectTableSceneCfg(TableTennisBallSceneCfg):
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_inspire_base_fix()
    # no cameras (avoids IsaacLab #3312); drop the base scene's world_camera too
    world_camera = None


@configclass
class ObservationsCfg:
    """State-based asymmetric AC. Policy sees privileged object pose (placeholder
    for vision). Same critic. Concatenated vectors — no images."""

    @configclass
    class PolicyCfg(ObsGroup):
        arm_pos = ObsTerm(func=robs.arm_joint_pos)
        arm_vel = ObsTerm(func=robs.arm_joint_vel)
        phase = ObsTerm(func=robs.grasp_phase, params={"action_name": _ACTION_NAME})
        object_offset = ObsTerm(func=robs.object_offset)   # clean jitter signal
        object_pose = ObsTerm(func=robs.object_pose_b)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        arm_pos = ObsTerm(func=robs.arm_joint_pos)
        arm_vel = ObsTerm(func=robs.arm_joint_vel)
        phase = ObsTerm(func=robs.grasp_phase, params={"action_name": _ACTION_NAME})
        object_offset = ObsTerm(func=robs.object_offset)   # clean jitter signal
        object_pose = ObsTerm(func=robs.object_pose_b)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class CurriculumCfg:
    # grow placement jitter +-1cm -> +-3cm over training so the residual learns
    # the centering mapping on graspable cases first, then transfers to hard ones
    jitter = CurriculumTermCfg(
        func=object_jitter_range,
        params={"start_mag": 0.01, "end_mag": 0.03, "num_steps": 8000},
    )


@configclass
class ResidualGraspTennisBallStateEnvCfg(ManagerBasedRLEnvCfg):
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(
        num_envs=256, env_spacing=2.5, replicate_physics=True
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    commands = None
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        # ~25 s covers approach + grasp + full lift (trajectory ~38 s total; the
        # place-down tail is not needed for grasp-robustness training)
        self.episode_length_s = 25.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        # buffers sized for many parallel envs — the default 16K total pairs
        # overflows at 256 envs (dropped contacts -> objects explode)
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 1024 * 1024
        self.sim.physx.gpu_max_rigid_contact_count = 2 ** 23
        self.sim.physx.gpu_max_rigid_patch_count = 2 ** 20
        self.sim.physx.friction_correlation_distance = 0.00625
