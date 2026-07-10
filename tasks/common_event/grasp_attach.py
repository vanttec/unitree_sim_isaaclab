# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Vectorized kinematic grasp-attach for RL training (DDS-free).

Physical grasping of small objects (coin, etc.) is too weak to hold in sim, so —
mirroring the teleop ``ObjectGraspController`` — when the working hand is near the
object AND closed, the object is welded to the wrist (its pose follows the wrist)
and released when the fingers open.

Runs as an interval event every control step across all envs. This is the
grasp-success physics the residual policy learns to trigger under placement
jitter: base alone may leave the hand too far to attach; the residual corrects.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# right-hand finger joints + per-joint closed-angle (rad) for a normalized closure
_FINGER_MAX = {
    "R_index_proximal_joint": 1.7,
    "R_middle_proximal_joint": 1.7,
    "R_ring_proximal_joint": 1.7,
    "R_pinky_proximal_joint": 1.7,
    "R_thumb_proximal_pitch_joint": 0.5,
}


class _AttachState:
    def __init__(self, env, wrist_body: str):
        robot: Articulation = env.scene["robot"]
        body_names = list(robot.data.body_names)
        self.wrist_idx = _find(body_names, (wrist_body, "right_wrist_yaw", "right_wrist"))
        # grasp point: prefer index finger link, fall back to wrist
        self.grasp_idx = _find_opt(body_names, ("R_index_proximal", "right_index")) or self.wrist_idx
        names = list(_FINGER_MAX.keys())
        ids, _ = robot.find_joints(names, preserve_order=True)
        self.finger_ids = torch.tensor(ids, device=env.device, dtype=torch.long)
        self.finger_max = torch.tensor([_FINGER_MAX[n] for n in names], device=env.device)
        n = env.num_envs
        self.active = torch.zeros(n, dtype=torch.bool, device=env.device)
        self.off_pos = torch.zeros(n, 3, device=env.device)
        self.off_quat = torch.zeros(n, 4, device=env.device)
        self.off_quat[:, 0] = 1.0


def _find(names, cands):
    for tok in cands:
        for i, nm in enumerate(names):
            if tok in nm:
                return i
    raise ValueError(f"body not found among {names}")


def _find_opt(names, cands):
    try:
        return _find(names, cands)
    except ValueError:
        return None


def grasp_attach_reset(env: "ManagerBasedRLEnv", env_ids: torch.Tensor,
                       wrist_body: str = "right_wrist_yaw_link") -> None:
    """Clear attach state for the given envs (mode='reset')."""
    st = getattr(env, "_grasp_attach", None)
    if st is None:
        env._grasp_attach = st = _AttachState(env, wrist_body)
    st.active[env_ids] = False


def grasp_attach_step(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    wrist_body: str = "right_wrist_yaw_link",
    grasp_distance: float = 0.08,
    grasp_closure: float = 0.5,
    release_closure: float = 0.25,
) -> None:
    """Attach/follow/release, vectorized over all envs (mode='interval')."""
    st = getattr(env, "_grasp_attach", None)
    if st is None:
        env._grasp_attach = st = _AttachState(env, wrist_body)

    robot: Articulation = env.scene["robot"]
    obj: RigidObject = env.scene[object_cfg.name]

    wrist_pos = robot.data.body_pos_w[:, st.wrist_idx]
    wrist_quat = robot.data.body_quat_w[:, st.wrist_idx]
    grasp_pos = robot.data.body_pos_w[:, st.grasp_idx]
    obj_pos = obj.data.root_pos_w
    obj_quat = obj.data.root_quat_w

    dist = torch.norm(grasp_pos - obj_pos, dim=-1)                      # (n,)
    fq = robot.data.joint_pos[:, st.finger_ids] / st.finger_max        # (n,5) normalized
    power = fq[:, :4].mean(dim=-1)
    pinch = 0.5 * (fq[:, 0] + fq[:, 4])                                # index + thumb
    closure = torch.maximum(power, pinch)                             # (n,)

    near_closing = (dist <= grasp_distance) & (closure >= grasp_closure)
    released = closure < release_closure

    # newly attach: record wrist->object offset
    new_attach = (~st.active) & near_closing
    if new_attach.any():
        rp, rq = subtract_frame_transforms(
            wrist_pos[new_attach], wrist_quat[new_attach],
            obj_pos[new_attach], obj_quat[new_attach],
        )
        st.off_pos[new_attach] = rp
        st.off_quat[new_attach] = rq

    st.active = (st.active | new_attach) & (~released)

    # follow: object pose = wrist ⊕ offset, for attached envs
    if st.active.any():
        ids = st.active.nonzero(as_tuple=False).squeeze(-1)
        np_, nq = combine_frame_transforms(
            wrist_pos[ids], wrist_quat[ids], st.off_pos[ids], st.off_quat[ids]
        )
        pose = torch.cat([np_, nq], dim=-1)
        obj.write_root_pose_to_sim(pose, env_ids=ids)
        obj.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=env.device), env_ids=ids)
