# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Observation terms for residual-grasp training.

Kept deliberately small and RL-friendly (plain tensors, not the DDS shared-memory
teleop path). Used to build an asymmetric actor/critic:

* policy group  : arm proprio + grasp phase (+ vision later)  -> deployable
* critic group  : the above + privileged object pose          -> trains faster

The object-pose term is privileged (sim ground truth). During bring-up it may
also sit in the policy group; once a camera term is added it moves to critic-only
so the deployed policy localizes from pixels.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_RIGHT_ARM_JOINT_NAMES = (
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


def _arm_ids(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, arm: str):
    robot: Articulation = env.scene[asset_cfg.name]
    names = _RIGHT_ARM_JOINT_NAMES if arm == "right" else tuple(
        n.replace("right_", "left_") for n in _RIGHT_ARM_JOINT_NAMES
    )
    ids, _ = robot.find_joints(list(names), preserve_order=True)
    return robot, torch.tensor(ids, device=env.device, dtype=torch.long)


def arm_joint_pos(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    arm: str = "right",
) -> torch.Tensor:
    """Working-arm joint positions, (num_envs, 7)."""
    robot, ids = _arm_ids(env, asset_cfg, arm)
    return robot.data.joint_pos[:, ids]


def arm_joint_vel(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    arm: str = "right",
) -> torch.Tensor:
    """Working-arm joint velocities, (num_envs, 7)."""
    robot, ids = _arm_ids(env, asset_cfg, arm)
    return robot.data.joint_vel[:, ids]


def arm_joint_pos_error(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    arm: str = "right",
) -> torch.Tensor:
    """Arm joint tracking error = commanded target - measured, (num_envs, 7).

    THE closed-loop model-error signal: under actuator-gain DR the PD tracks the
    same target differently, so this error directly exposes the gain randomization
    the residual must compensate. Feedforward obs (object_offset) can't reveal it."""
    robot, ids = _arm_ids(env, asset_cfg, arm)
    return robot.data.joint_pos_target[:, ids] - robot.data.joint_pos[:, ids]


def fingertip_to_object(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Grasp-point -> object vector (world), (num_envs, 3).

    Live grasp-misalignment feedback: where the fingers are vs where the ball
    actually is, updated every step -> the residual can close the loop on it."""
    robot: Articulation = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    gi = getattr(env, "_rgr_grasp_idx", None)
    if gi is None:
        names = list(robot.data.body_names)
        gi = next(i for i, n in enumerate(names) if "R_index_proximal" in n)
        env._rgr_grasp_idx = gi
    return obj.data.root_pos_w - robot.data.body_pos_w[:, gi]


def grasp_phase(
    env: "ManagerBasedRLEnv",
    action_name: str = "residual",
) -> torch.Tensor:
    """Normalized base-trajectory phase, (num_envs, 1)."""
    am = env.action_manager
    try:
        term = am.get_term(action_name)
    except Exception:
        term = am._terms[action_name]  # noqa: SLF001
    return term.phase_fraction.unsqueeze(-1)


def camera_rgb(
    env: "ManagerBasedRLEnv",
    sensor_name: str = "front_camera",
) -> torch.Tensor:
    """Normalized RGB from a (tiled) camera, (num_envs, H, W, 3) in [0,1].

    This is the deployable, real-transferable signal — the policy localizes the
    object from pixels instead of privileged pose. Keep the group's
    ``concatenate_terms=False`` so the image passes through as an image.
    """
    cam = env.scene[sensor_name]
    rgb = cam.data.output["rgb"][..., :3].float() / 255.0
    return rgb


def object_pose_b(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object pose relative to its env origin (privileged), (num_envs, 7).

    position (xyz) made env-relative so it is comparable across parallel envs;
    orientation (wxyz) passed through.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    pos = obj.data.root_pos_w - env.scene.env_origins
    quat = obj.data.root_quat_w
    return torch.cat([pos, quat], dim=-1)


# nominal (un-jittered) object position, env-relative — the pedestal top center.
_NOMINAL_OBJ_POS = (-0.15, 0.40, 0.87535)


def object_offset(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object displacement from its nominal spawn = the JITTER itself, (num_envs, 3).

    Zero-mean +-3cm signal. Far cleaner to condition on than the absolute pose
    (which is a big constant with a tiny jitter riding on it) — the policy maps
    this directly to the arm correction Δq."""
    obj: RigidObject = env.scene[object_cfg.name]
    pos = obj.data.root_pos_w - env.scene.env_origins
    nominal = torch.tensor(_NOMINAL_OBJ_POS, device=pos.device, dtype=pos.dtype)
    return pos - nominal
