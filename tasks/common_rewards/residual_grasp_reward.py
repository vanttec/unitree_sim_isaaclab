# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Dense outcome reward for residual-grasp training.

The base trajectory already performs the reach + close, so the residual only has
to make the grasp *succeed under placement jitter*. Reward is therefore on the
outcome (object lifted, then placed in the target box) plus a penalty that keeps
the residual small — which is what protects the semantic grasp shape.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _grasp_body(env: "ManagerBasedRLEnv", robot_cfg: SceneEntityCfg):
    """Right-hand grasp point (index proximal), resolved + cached by name."""
    robot = env.scene[robot_cfg.name]
    idx = getattr(env, "_grasp_body_idx", None)
    if idx is None:
        names = list(robot.data.body_names)
        idx = next(i for i, n in enumerate(names) if "R_index_proximal" in n)
        env._grasp_body_idx = idx
    return robot, idx


def object_lift(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    table_height: float = 0.79,
    target_height: float = 0.90,
    hold_dist: float = 0.09,
    speed_max: float = 0.5,
) -> torch.Tensor:
    """Normalized [0,1] lift, GATED by a STABLE hold (in-hand AND slow).

    Reward counts only if the object is within ``hold_dist`` of the grasp point
    AND moving slower than ``speed_max``. A whack is briefly near+high but fast,
    so it earns nothing; only a genuine slow lift-and-hold is rewarded.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    robot, gi = _grasp_body(env, robot_cfg)
    dist = torch.norm(robot.data.body_pos_w[:, gi] - obj.data.root_pos_w, dim=-1)
    speed = torch.norm(obj.data.root_lin_vel_w, dim=-1)
    held = ((dist < hold_dist) & (speed < speed_max)).float()
    h = obj.data.root_pos_w[:, 2]
    denom = max(target_height - table_height, 1e-3)
    lift = torch.clamp((h - table_height) / denom, 0.0, 1.0)
    return lift * held


def object_speed(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object linear speed — pass a negative weight to punish flinging/knocking."""
    obj: RigidObject = env.scene[object_cfg.name]
    return torch.norm(obj.data.root_lin_vel_w, dim=-1)


def reach_align(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.15,
    action_name: str = "residual",
    phase_lo: float = 0.28,
    phase_hi: float = 0.45,
) -> torch.Tensor:
    """Dense [0,1] hand->object alignment, ACTIVE ONLY around the grasp moment.

    exp(-dist/std), gated to phase in [phase_lo, phase_hi] (fingers-closing
    window). This rewards correcting the approach aim at the decisive instant —
    the jitter-correction signal — instead of just being near the ball during the
    hold (which the base already does). Gives the hard/large-jitter cases a
    gradient even when the lift ultimately fails.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    robot, gi = _grasp_body(env, robot_cfg)
    dist = torch.norm(robot.data.body_pos_w[:, gi] - obj.data.root_pos_w, dim=-1)
    align = torch.exp(-dist / std)
    pf = _residual_term(env, action_name).phase_fraction
    in_window = ((pf >= phase_lo) & (pf <= phase_hi)).float()
    return align * in_window


def object_in_target(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    min_x: float = 0.28,
    max_x: float = 0.96,
    min_y: float = 0.24,
    max_y: float = 0.57,
    min_height: float = 0.81,
    max_height: float = 0.90,
) -> torch.Tensor:
    """1.0 when the object sits inside the target placement box, else 0.0."""
    obj: RigidObject = env.scene[object_cfg.name]
    p = obj.data.root_pos_w
    inside = (
        (p[:, 0] > min_x) & (p[:, 0] < max_x)
        & (p[:, 1] > min_y) & (p[:, 1] < max_y)
        & (p[:, 2] > min_height) & (p[:, 2] < max_height)
    )
    return inside.float()


def _residual_term(env: "ManagerBasedRLEnv", action_name: str):
    """Fetch the residual action term (processed_actions = Δq)."""
    am = env.action_manager
    try:
        return am.get_term(action_name)
    except Exception:
        return am._terms[action_name]  # noqa: SLF001  (older Isaac Lab)


def residual_l2(
    env: "ManagerBasedRLEnv",
    action_name: str = "residual",
) -> torch.Tensor:
    """Sum of squared residual per env — pass a negative weight to penalize."""
    term = _residual_term(env, action_name)
    dq = term.processed_actions
    return torch.sum(dq * dq, dim=-1)


def residual_rate_l2(
    env: "ManagerBasedRLEnv",
    action_name: str = "residual",
) -> torch.Tensor:
    """Squared change in residual between steps — smoothness (negative weight)."""
    term = _residual_term(env, action_name)
    dq = term.processed_actions
    prev = getattr(env, "_residual_prev", None)
    if prev is None or prev.shape != dq.shape:
        prev = torch.zeros_like(dq)
    rate = torch.sum((dq - prev) * (dq - prev), dim=-1)
    env._residual_prev = dq.detach().clone()
    return rate
