# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Residual-grasp training env: tennisball (slot 3), right-arm residual over a fixed base.

Round object, power grasp — real contact holds without cradle clamping (unlike the
coin), so this is the object to prove the residual loop on. Obs/reward/termination/
events are reused from the coin env (object-agnostic); only the scene + trajectory
slot change.
"""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from tasks.common_config import G1RobotPresets, CameraPresets
from tasks.common_scene.base_scene_pickplace_tennisball import TableTennisBallSceneCfg
from tasks.common_actions import ResidualArmActionCfg
from tasks.common_observations import residual_obs as robs
from tasks.common_event.reset_to_trajectory_start import reset_robot_to_trajectory_start

# reuse the object-agnostic managers from the coin env (reward/termination)
from tasks.g1_tasks.residual_grasp_coin_g1_29dof_inspire.residual_grasp_coin_env_cfg import (
    RewardsCfg, TerminationsCfg,
)

_ACTION_NAME = "residual"
_CAM_H, _CAM_W = 96, 96   # low-res for RL (proven CameraCfg path; tiled later)


@configclass
class ObjectTableSceneCfg(TableTennisBallSceneCfg):
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_inspire_base_fix()

    # cameras (approach = front, contact = right wrist): proven presets (640x480);
    # camera_rgb obs downsamples to _CAM_H x _CAM_W for the policy.
    front_camera = CameraPresets.g1_front_camera()
    right_wrist_camera = CameraPresets.right_inspire_wrist_camera()


@configclass
class ObservationsCfg:
    """Asymmetric actor-critic. Policy sees pixels (deployable, transfers);
    critic additionally sees privileged object pose (sim-only, trains faster)."""

    @configclass
    class PolicyCfg(ObsGroup):
        arm_pos = ObsTerm(func=robs.arm_joint_pos)
        arm_vel = ObsTerm(func=robs.arm_joint_vel)
        phase = ObsTerm(func=robs.grasp_phase, params={"action_name": _ACTION_NAME})
        front_img = ObsTerm(func=robs.camera_rgb, params={"sensor_name": "front_camera"})
        wrist_img = ObsTerm(func=robs.camera_rgb, params={"sensor_name": "right_wrist_camera"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False   # keep images as images (dict obs)

    @configclass
    class CriticCfg(ObsGroup):
        arm_pos = ObsTerm(func=robs.arm_joint_pos)
        arm_vel = ObsTerm(func=robs.arm_joint_vel)
        phase = ObsTerm(func=robs.grasp_phase, params={"action_name": _ACTION_NAME})
        object_pose = ObsTerm(func=robs.object_pose_b)   # privileged, sim-only

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ActionsCfg:
    residual = ResidualArmActionCfg(
        asset_name="robot",
        slot=3,                 # tennisball
        arm="right",
        # CARTESIAN residual: policy outputs a 3D wrist offset -> IK -> Δq. Joint-
        # space collapsed to a constant (couldn't learn inverse kinematics); the
        # Cartesian map is ~identity. Analytic feed-forward proved 79-82% ceiling.
        # RPL on the analytic base: base = IK shift-by-jitter (~82%, stable, does
        # the heavy lift), policy learns a SMALL residual on top (fingertip offset,
        # model error). No discovery, no exploration cliff -> RL is stable here.
        # This is the learned policy for the DR-robustness story.
        cartesian_action=True,
        cartesian_ff_base=True,
        cartesian_gain=1.3,            # analytic base strength (1.4 peaked ~82%)
        cartesian_action_scale=0.03,   # 0.05 hurt (66% vs 74%) -> not authority-limited
        ee_body_name="right_hand_base_link",
        freeze_phase=0.05,             # one-shot latch (open-loop): 78%. Closed-loop
                                       # live tried @0.7 -> 44% (clamp-saturation
                                       # instability); delicate grasp rejects live.
        playback_hz=100.0,
        hand_close_boost=0.5,   # real physical grip (no kinematic attach)
    )


@configclass
class EventCfg:
    """Real-physics grasp: no attach. Object placement jitter is the variance the
    residual must absorb; the fixed base grip alone will miss/slip under it."""
    reset_object = EventTermCfg(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "pose_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
            "velocity_range": {},
        },
    )
    # teleport robot to trajectory frame 0 -> no PD start-snap that sweeps the
    # hand through the ball and topples it (the reset de-sync)
    reset_robot_start = EventTermCfg(func=reset_robot_to_trajectory_start, mode="reset")

    # --- domain randomization -------------------------------------------------
    # Gives the learned residual a JOB the analytic base can't do: actuator-gain
    # error makes the nominal-model IK correction wrong -> residual compensates.
    # Friction/mass vary contact. This is the robustness the paper claims + the
    # sim2real bridge.
    dr_object_friction = EventTermCfg(
        func=base_mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (1.5, 4.0),
            "dynamic_friction_range": (1.5, 4.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    dr_object_mass = EventTermCfg(
        func=base_mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.7, 1.3),
            "operation": "scale",
        },
    )
    dr_arm_gains = EventTermCfg(
        func=base_mdp.randomize_actuator_gains,
        mode="startup",   # fixed-but-unknown model error per env
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["right_.*_joint"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )


@configclass
class ResidualGraspTennisBallEnvCfg(ManagerBasedRLEnvCfg):
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(
        num_envs=64, env_spacing=2.5, replicate_physics=True
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    commands = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
