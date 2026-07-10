# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Residual-grasp training env: coin (slot 1), right-arm residual over a fixed base.

Base = recorded semantic grasp (grasp_bridge slot 1). Policy outputs a small,
bounded residual on the 7 right-arm joints only; the hand is byte-exact from the
recording, so grasp identity is preserved. Trained to be robust to a few-cm
object placement jitter.
"""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from tasks.common_config import G1RobotPresets
from tasks.common_scene.base_scene_pickplace_coin import TableCoinSceneCfg
from tasks.common_actions import ResidualArmActionCfg
from tasks.common_rewards import residual_grasp_reward as rgr
from tasks.common_observations import residual_obs as robs
from tasks.common_termination.residual_grasp_termination import object_dropped
from tasks.common_event.grasp_attach import grasp_attach_step, grasp_attach_reset

# name the action term "residual" so reward/obs terms can find it
_ACTION_NAME = "residual"


@configclass
class ObjectTableSceneCfg(TableCoinSceneCfg):
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_inspire_base_fix()
    # cameras added at the TiledCamera step; state-based bring-up first.


@configclass
class ActionsCfg:
    residual = ResidualArmActionCfg(
        asset_name="robot",
        slot=1,                 # coin
        arm="right",
        residual_scale=0.08,    # ~few-cm jitter budget at the wrist
        playback_hz=100.0,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Deployable obs. Object pose here is a privileged placeholder — replace
        with a camera term once TiledCamera is wired, then drop it from policy."""
        arm_pos = ObsTerm(func=robs.arm_joint_pos)
        arm_vel = ObsTerm(func=robs.arm_joint_vel)
        phase = ObsTerm(func=robs.grasp_phase, params={"action_name": _ACTION_NAME})
        object_pose = ObsTerm(func=robs.object_pose_b)  # TODO: -> critic-only w/ vision

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged critic (asymmetric actor-critic)."""
        arm_pos = ObsTerm(func=robs.arm_joint_pos)
        arm_vel = ObsTerm(func=robs.arm_joint_vel)
        phase = ObsTerm(func=robs.grasp_phase, params={"action_name": _ACTION_NAME})
        object_pose = ObsTerm(func=robs.object_pose_b)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    # held-gated lift: only rewarded while the object is IN the hand (anti-hack).
    # dominates the objective — reach is only a hint to bootstrap the approach.
    lift = RewTerm(func=rgr.object_lift, weight=5.0)
    # dense: aim the hand at the jittered ball (gradient the sparse lift lacks)
    # tight std -> sharp gradient to CENTER the hand on the jittered ball at the
    # grasp moment (0.15 was too soft: a 3cm miss scored ~0.82 ~= perfect, so no
    # push to center; that's why large jitters stayed unrecovered).
    reach = RewTerm(func=rgr.reach_align, weight=1.0, params={"std": 0.04})
    # punish flinging/knocking the object (a real grasp lifts it slowly)
    speed_penalty = RewTerm(func=rgr.object_speed, weight=-0.3)
    residual_penalty = RewTerm(
        # keep the residual LAZY: no gratuitous perturbation of the (good) analytic
        # base -> stays ~0 where base is already right, activates only where DR
        # opens a gap. Without this the residual hurt (82% base -> 69% RPL).
        func=rgr.residual_l2, weight=-0.5, params={"action_name": _ACTION_NAME}
    )
    residual_rate = RewTerm(
        func=rgr.residual_rate_l2, weight=-0.01, params={"action_name": _ACTION_NAME}
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    dropped = DoneTerm(func=object_dropped)


@configclass
class EventCfg:
    # placement jitter: the variance the residual must absorb
    reset_object = EventTermCfg(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "pose_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
            "velocity_range": {},
        },
    )
    # clear kinematic attach on episode reset
    attach_reset = EventTermCfg(func=grasp_attach_reset, mode="reset")
    # kinematic grasp-attach every control step (grasp-success physics)
    attach_step = EventTermCfg(
        func=grasp_attach_step,
        mode="interval",
        interval_range_s=(0.01, 0.01),   # = decimation * sim.dt -> every step
        is_global_time=False,
    )


@configclass
class ResidualGraspCoinEnvCfg(ManagerBasedRLEnvCfg):
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
        # 100 Hz control matches the recorded trajectory (playback_hz) 1:1.
        self.decimation = 2
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
