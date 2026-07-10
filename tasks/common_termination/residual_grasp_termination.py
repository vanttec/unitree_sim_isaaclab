# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Terminations for residual-grasp training."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_dropped(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    min_height: float = 0.40,
) -> torch.Tensor:
    """True (per env) when the object has fallen below ``min_height``."""
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_pos_w[:, 2] < min_height
